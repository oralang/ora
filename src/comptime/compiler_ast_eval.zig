const std = @import("std");
const ast = @import("../ast/mod.zig");
const abi_type_names = @import("../abi/type_names.zig");
const bridge = @import("compiler_const_bridge.zig");
const comptime_mod = @import("mod.zig");
const ora_types = @import("ora_types");
const compiler_query = @import("../compiler_query.zig");
const type_builtin = ora_types.builtin;
const diagnostics = @import("../diagnostics/mod.zig");
const stage_mod = @import("stage.zig");
const model = @import("../sema/model.zig");
const lookup_index = @import("../sema/lookup.zig");
const abi_layout_provider = @import("../sema/abi_layout_provider.zig");
const type_descriptors = @import("../sema/type_descriptors.zig");
const refinements = @import("ora_types").refinement_semantics;
const source = @import("../source/mod.zig");
const error_mod = @import("error.zig");
const hir_abi = @import("../hir/abi.zig");
const abi_layout_context = @import("../abi/layout_context.zig");
const abi_comptime_encoder = @import("../abi/comptime_encoder.zig");
const abi_comptime_decoder = @import("../abi/comptime_decoder.zig");
const compile_options = @import("../compile_options.zig");
const module_graph = @import("../sema/module_graph.zig");

const ConstEvalResult = model.ConstEvalResult;
const ConstValue = ora_types.ConstValue;
const Type = ora_types.SemanticType;
const TypeKind = ora_types.TypeKind;
const IntegerType = ora_types.IntegerType;
const AnonymousStructField = ora_types.AnonymousStructField;
const CtAggregate = comptime_mod.CtAggregate;
const CtEnum = comptime_mod.CtEnum;
const CtErrorUnion = comptime_mod.CtErrorUnion;
const CtEnv = bridge.CtEnv;
const CtValue = bridge.CtValue;
const SourceSpan = error_mod.SourceSpan;
const Stage = stage_mod.Stage;
const LimitCheck = comptime_mod.LimitCheck;
const constEquals = bridge.constEquals;
const ctValueToConstValue = bridge.ctValueToConstValue;
const constToCtValue = bridge.constToCtValue;
const evalBinary = bridge.evalBinary;
const evalUnary = bridge.evalUnary;
const EvalConfig = comptime_mod.EvalConfig;
const parseIntegerLiteral = bridge.parseIntegerLiteral;
const wrapIntegerConstToType = bridge.wrapIntegerConstToType;
const named_type_id_module_stride: u32 = 100_000;

fn selectorFixedBytes(allocator: std.mem.Allocator, selector: u32) ![]u8 {
    const bytes = try allocator.alloc(u8, 4);
    std.mem.writeInt(u32, bytes[0..4], selector, .big);
    return bytes;
}

fn keccakFixedBytes(allocator: std.mem.Allocator, bytes: []const u8) ![]const u8 {
    const hash = hir_abi.keccak256(bytes);
    return allocator.dupe(u8, hash[0..]);
}

pub const ConstEvalOptions = struct {
    module_id: ?source.ModuleId = null,
    type_query: ?compiler_query.ComptimeView = null,
    config: EvalConfig = .default,
    chain_id: u64 = compile_options.default_chain_id,
};

/// Compiler-AST constant evaluator.
///
/// This is the migration boundary for moving the refactored compiler onto the
/// shared comptime subsystem. It preserves the current `ConstEvalResult` shape
/// used by sema/HIR while relocating the AST walker into `src/comptime/`.
///
/// The immediate goal is architectural:
/// - compiler DB should query comptime through `src/comptime/`
/// - the AST walker remains isolated in `ast_eval.zig`
/// - the lightweight compiler walker can later be upgraded to use the full
///   shared environment/value engine without moving the compiler call sites again
pub fn constEval(allocator: std.mem.Allocator, file: *const ast.AstFile, options: ConstEvalOptions) !ConstEvalResult {
    var result = ConstEvalResult{
        .arena = std.heap.ArenaAllocator.init(allocator),
        .values = &[_]?ConstValue{},
        .diagnostics = diagnostics.DiagnosticList.init(allocator),
    };
    errdefer result.deinit();

    const arena = result.arena.allocator();
    const values = try arena.alloc(?ConstValue, file.expressions.len);
    @memset(values, null);

    var fallback_item_index: ?model.ItemIndexResult = null;
    defer if (fallback_item_index) |*item_index| item_index.deinit();
    if (options.type_query == null) {
        fallback_item_index = try module_graph.buildItemIndex(arena, file);
    }
    const fallback_item_index_ptr: ?*const model.ItemIndexResult = if (fallback_item_index) |*item_index| item_index else null;

    var evaluator = ConstEvaluator{
        .allocator = arena,
        .file = file,
        .values = values,
        .env = CtEnv.init(arena, options.config),
        .module_id = options.module_id,
        .type_query = options.type_query,
        .fallback_item_index = fallback_item_index_ptr,
        .chain_id = options.chain_id,
    };
    defer evaluator.env.deinit();
    for (file.root_items) |item_id| {
        evaluator.visitItem(item_id);
    }

    if (evaluator.last_error) |ct_error| {
        try appendCtDiagnostic(&result.diagnostics, file.file_id, ct_error);
    }

    result.values = values;
    return result;
}

fn appendCtDiagnostic(
    list: *diagnostics.DiagnosticList,
    file_id: source.FileId,
    ct_error: error_mod.CtError,
) !void {
    var buffer: [256]u8 = undefined;
    const message = if (ct_error.reason) |reason|
        try std.fmt.bufPrint(&buffer, "{s}: {s}", .{ ct_error.message, reason })
    else
        ct_error.message;

    const end = ct_error.span.byte_offset + ct_error.span.length;
    try list.appendError(message, .{
        .file_id = file_id,
        .range = .{
            .start = ct_error.span.byte_offset,
            .end = end,
        },
    });
}

const ConstEvaluator = struct {
    allocator: std.mem.Allocator,
    file: *const ast.AstFile,
    values: []?ConstValue,
    env: CtEnv,
    module_id: ?source.ModuleId = null,
    type_query: ?compiler_query.ComptimeView = null,
    fallback_item_index: ?*const model.ItemIndexResult = null,
    chain_id: u64,
    current_typecheck_key: ?model.TypeCheckKey = null,
    current_contract: ?ast.ItemId = null,
    call_depth: u32 = 0,
    required_comptime_depth: u32 = 0,
    last_error: ?error_mod.CtError = null,

    const BodyControl = union(enum) {
        value: ?ConstValue,
        return_value: ?ConstValue,
        break_loop,
        continue_loop,
        indeterminate,
    };

    const CtBodyControl = union(enum) {
        value: ?CtValue,
        return_value: ?CtValue,
        break_loop,
        continue_loop,
        indeterminate,
    };

    const ValueConstructionTarget = enum {
        none,
        slice,
        map,
    };

    fn visitItem(self: *ConstEvaluator, item_id: ast.ItemId) void {
        const previous_key = self.current_typecheck_key;
        self.current_typecheck_key = .{ .item = item_id };
        defer self.current_typecheck_key = previous_key;

        switch (self.file.item(item_id).*) {
            .Contract => |contract| {
                const previous_contract = self.current_contract;
                self.current_contract = item_id;
                defer self.current_contract = previous_contract;
                for (contract.invariants) |expr_id| _ = self.evalExpr(expr_id) catch null;
                for (contract.members) |member_id| self.visitItem(member_id);
            },
            .Function => |function| {
                for (function.clauses) |clause| _ = self.evalExpr(clause.expr) catch null;
                self.visitBody(function.body);
            },
            .Enum => |enum_item| {
                self.required_comptime_depth += 1;
                defer self.required_comptime_depth -= 1;
                self.env.pushScope(false) catch return;
                defer self.env.popScope();
                var next_value: i64 = 0;
                for (enum_item.variants) |variant| {
                    var bound_value: ?ConstValue = null;
                    if (variant.value) |expr_id| {
                        const value = self.evalExpr(expr_id) catch null;
                        self.values[expr_id.index()] = value;
                        bound_value = value;
                    } else {
                        bound_value = .{ .integer = std.math.big.int.Managed.initSet(self.allocator, next_value) catch return };
                    }
                    if (bound_value) |value| {
                        var exported_value = value;
                        switch (value) {
                            .integer => |integer| {
                                if (integer.toInt(i64)) |small| {
                                    next_value = small + 1;
                                } else |_| {}
                            },
                            .boolean => |boolean| {
                                const integer_value: i64 = if (boolean) 1 else 0;
                                exported_value = .{ .integer = std.math.big.int.Managed.initSet(self.allocator, integer_value) catch return };
                                next_value = integer_value + 1;
                            },
                            else => {},
                        }
                        self.bindName(variant.name, exported_value) catch {};
                    }
                }
            },
            .Field => |field| {
                if (field.value) |expr_id| {
                    self.required_comptime_depth += 1;
                    defer self.required_comptime_depth -= 1;
                    const value = self.evalExpr(expr_id) catch null;
                    self.bindName(field.name, value) catch {};
                    self.values[expr_id.index()] = value;
                }
            },
            .Constant => |constant| {
                self.required_comptime_depth += 1;
                defer self.required_comptime_depth -= 1;
                const value = self.evalExpr(constant.value) catch null;
                self.bindName(constant.name, value) catch {};
                self.values[constant.value.index()] = value;
            },
            .GhostBlock => |ghost_block| self.visitBody(ghost_block.body),
            else => {},
        }
    }

    fn visitBody(self: *ConstEvaluator, body_id: ast.BodyId) void {
        const previous_key = self.current_typecheck_key;
        self.current_typecheck_key = .{ .body = body_id };
        defer self.current_typecheck_key = previous_key;

        self.env.pushScope(false) catch return;
        defer self.env.popScope();

        const body = self.file.body(body_id).*;
        for (body.statements) |statement_id| {
            switch (self.file.statement(statement_id).*) {
                .VariableDecl => |decl| {
                    const value = if (decl.value) |expr_id|
                        self.normalizeWrappingValueForDecl(decl, expr_id, self.evalExpr(expr_id) catch null) catch null
                    else
                        null;
                    self.bindPattern(decl.pattern, value) catch {};
                    if (decl.value) |expr_id| self.values[expr_id.index()] = value;
                },
                .Return => |ret| {
                    if (ret.value) |expr_id| {
                        const value = self.evalExpr(expr_id) catch null;
                        self.values[expr_id.index()] = value;
                    }
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
                        self.visitSwitchPattern(arm.pattern);
                        self.visitBody(arm.body);
                    }
                    if (switch_stmt.else_body) |else_body| self.visitBody(else_body);
                },
                .Try => |try_stmt| {
                    self.visitBody(try_stmt.try_body);
                    if (try_stmt.catch_clause) |catch_clause| self.visitBody(catch_clause.body);
                },
                .Expr => |expr_stmt| switch (self.file.expression(expr_stmt.expr).*) {
                    .Comptime => |comptime_expr| {
                        self.required_comptime_depth += 1;
                        defer self.required_comptime_depth -= 1;
                        _ = self.evalComptimeBody(comptime_expr.body) catch null;
                    },
                    else => _ = self.evalExpr(expr_stmt.expr) catch null,
                },
                .Assign => |assign| _ = self.evalComptimeAssign(assign, true) catch null,
                .Log => |log_stmt| {
                    for (log_stmt.args) |arg| _ = self.evalExpr(arg) catch null;
                },
                .Lock, .Unlock, .Break, .Continue, .Havoc, .CallHint => {},
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

    fn visitSwitchPattern(self: *ConstEvaluator, pattern: ast.SwitchPattern) void {
        switch (pattern) {
            .Expr => |expr_id| _ = self.evalExpr(expr_id) catch null,
            .Range => |range_pattern| {
                _ = self.evalExpr(range_pattern.start) catch null;
                _ = self.evalExpr(range_pattern.end) catch null;
            },
            .NamedError => |named_error| _ = self.evalExpr(named_error.callee) catch null,
            .Or => |or_pattern| for (or_pattern.alternatives) |alternative| self.visitSwitchPattern(alternative),
            .Ok, .Err, .Else => {},
        }
    }

    fn evalSwitchPatternExprs(self: *ConstEvaluator, pattern: ast.SwitchPattern, comptime use_cache: bool) anyerror!void {
        switch (pattern) {
            .Expr => |expr_id| _ = try self.evalExprImpl(expr_id, use_cache),
            .Range => |range_pattern| {
                _ = try self.evalExprImpl(range_pattern.start, use_cache);
                _ = try self.evalExprImpl(range_pattern.end, use_cache);
            },
            .NamedError => |named_error| _ = try self.evalExprImpl(named_error.callee, use_cache),
            .Or => |or_pattern| for (or_pattern.alternatives) |alternative| try self.evalSwitchPatternExprs(alternative, use_cache),
            .Ok, .Err, .Else => {},
        }
    }

    fn evalExprImpl(self: *ConstEvaluator, expr_id: ast.ExprId, comptime use_cache: bool) anyerror!?ConstValue {
        if (use_cache) {
            if (self.values[expr_id.index()]) |cached| return cached;
        }

        if (self.consumeStep(self.exprRange(expr_id))) return null;

        if (!self.exprUsesConstFallbackForCtValue(expr_id)) {
            if (try self.evalExprCtValueImpl(expr_id, use_cache, false)) |ct_value| {
                const const_value = try ctValueToConstValue(self.allocator, &self.env.heap, ct_value);
                if (const_value != null) {
                    if (use_cache) self.values[expr_id.index()] = const_value;
                    return const_value;
                }
            }
        }

        const value: ?ConstValue = switch (self.file.expression(expr_id).*) {
            .IntegerLiteral => |literal| try parseIntegerLiteral(self.allocator, literal.text),
            .StringLiteral => |literal| ConstValue{ .string = literal.text },
            .BoolLiteral => |literal| ConstValue{ .boolean = literal.value },
            .AddressLiteral, .BytesLiteral => null,
            .TypeValue => |type_value| blk: {
                try self.ensureTypeExprTypeChecked(type_value.type_expr);
                if (self.current_typecheck_key) |key| try self.ensureTypeChecked(key);
                break :blk null;
            },
            .Tuple => |tuple| blk: {
                for (tuple.elements) |element| _ = try self.evalExprImpl(element, use_cache);
                break :blk null;
            },
            .ArrayLiteral => |array| blk: {
                for (array.elements) |element| _ = try self.evalExprImpl(element, use_cache);
                break :blk null;
            },
            .StructLiteral => |struct_literal| blk: {
                if (struct_literal.type_expr) |type_expr|
                    try self.ensureTypeExprTypeChecked(type_expr)
                else
                    try self.ensureNamedItemTypeChecked(struct_literal.type_name);
                for (struct_literal.fields) |field| _ = try self.evalExprImpl(field.value, use_cache);
                break :blk null;
            },
            .ExternalProxy => null,
            .Switch => |switch_expr| blk: {
                if (try self.evalExprCtValueImpl(switch_expr.condition, use_cache, true)) |condition_ct| {
                    for (switch_expr.arms) |arm| {
                        if (!(try self.patternMatchesCt(condition_ct, arm.pattern))) continue;
                        self.env.pushScope(false) catch break :blk null;
                        defer self.env.popScope();
                        try self.bindSwitchPatternCtValue(condition_ct, arm.pattern);
                        break :blk try self.evalExprImpl(arm.value, use_cache);
                    }
                    if (switch_expr.else_expr) |else_expr| break :blk try self.evalExprImpl(else_expr, use_cache);
                    break :blk null;
                }

                const condition = (try self.evalExprImpl(switch_expr.condition, use_cache)) orelse {
                    for (switch_expr.arms) |arm| {
                        try self.evalSwitchPatternExprs(arm.pattern, use_cache);
                        _ = try self.evalExprImpl(arm.value, use_cache);
                    }
                    if (switch_expr.else_expr) |else_expr| _ = try self.evalExprImpl(else_expr, use_cache);
                    break :blk null;
                };

                for (switch_expr.arms) |arm| {
                    try self.evalSwitchPatternExprs(arm.pattern, use_cache);
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
                self.required_comptime_depth += 1;
                defer self.required_comptime_depth -= 1;
                const ct_value = try self.evalComptimeBodyCtValue(comptime_expr.body, use_cache);
                var value = if (ct_value) |inner|
                    try ctValueToConstValue(self.allocator, &self.env.heap, inner)
                else
                    null;
                if (value == null) {
                    value = try self.evalComptimeBody(comptime_expr.body);
                }
                if (value == null) {
                    self.recordMissingComptimeValue(
                        self.sourceSpan(comptime_expr.range),
                        "comptime block did not produce a value",
                        null,
                    );
                }
                break :blk value;
            },
            .ErrorReturn => |error_return| blk: {
                for (error_return.args) |arg| _ = try self.evalExprImpl(arg, use_cache);
                break :blk null;
            },
            .Name => |name| blk: {
                const value = self.env.lookupValue(name.name) orelse break :blk null;
                break :blk try ctValueToConstValue(self.allocator, &self.env.heap, value);
            },
            .Result => null,
            .Unary => |unary| try evalUnary(self.allocator, unary.op, try self.evalExprImpl(unary.operand, use_cache)),
            .Binary => |binary| try self.evalBinaryExpr(binary, use_cache),
            .Call => |call| blk: {
                if (try self.evalCallCtValue(call, use_cache)) |ct_value| {
                    break :blk try ctValueToConstValue(self.allocator, &self.env.heap, ct_value);
                }
                if (self.last_error != null) break :blk null;
                break :blk try self.evalCall(call, use_cache);
            },
            .Builtin => |builtin| blk: {
                if (self.exprStage(expr_id) == .runtime_only) {
                    self.recordCtError(error_mod.CtError.stageViolation(
                        self.sourceSpan(builtin.range),
                        builtin.name,
                    ));
                    break :blk null;
                }
                break :blk try self.evalBuiltin(builtin);
            },
            .Field => |field| blk: {
                _ = try self.evalExprImpl(field.base, use_cache);
                if ((try self.importedModuleForExpr(field.base))) |target_module_id| {
                    const target_item_id = (try self.lookupNamedItemInModule(target_module_id, field.name)) orelse break :blk null;
                    const target_file = try self.astFileForModule(target_module_id);
                    const target_const_eval = (try self.constEvalForModule(target_module_id)) orelse break :blk null;
                    switch (target_file.item(target_item_id).*) {
                        .Constant => |constant| break :blk target_const_eval.values[constant.value.index()],
                        .Field => |decl| {
                            const value_expr = decl.value orelse break :blk null;
                            break :blk target_const_eval.values[value_expr.index()];
                        },
                        else => {},
                    }
                }
                if (try self.evalExprCtValueImpl(expr_id, use_cache, false)) |ct_value| {
                    break :blk try ctValueToConstValue(self.allocator, &self.env.heap, ct_value);
                }
                break :blk null;
            },
            .Index => |index| blk: {
                _ = try self.evalExprImpl(index.base, use_cache);
                _ = try self.evalExprImpl(index.index, use_cache);
                if (try self.evalExprCtValueImpl(expr_id, use_cache, false)) |ct_value| {
                    break :blk try ctValueToConstValue(self.allocator, &self.env.heap, ct_value);
                }
                break :blk null;
            },
            .Group => |group| try self.evalExprImpl(group.expr, use_cache),
            // old(...) is a verification-time pre-state operator, not a comptime
            // expression. Folding it here can erase the pre-state reference and
            // incorrectly collapse postconditions like old(x) + 1 to constants.
            .Old => null,
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

    fn evalBinaryExpr(self: *ConstEvaluator, binary: ast.BinaryExpr, comptime use_cache: bool) anyerror!?ConstValue {
        const lhs = try self.evalExprImpl(binary.lhs, use_cache);
        const rhs = try self.evalExprImpl(binary.rhs, use_cache);
        if (self.rejectInvalidBinaryComptime(binary.op, lhs, rhs, binary.range)) return null;
        return try evalBinary(self.allocator, binary.op, lhs, rhs);
    }

    fn evalBinaryValue(
        self: *ConstEvaluator,
        op: ast.BinaryOp,
        lhs: ?ConstValue,
        rhs: ?ConstValue,
        range: source.TextRange,
    ) anyerror!?ConstValue {
        if (self.rejectInvalidBinaryComptime(op, lhs, rhs, range)) return null;
        return try evalBinary(self.allocator, op, lhs, rhs);
    }

    fn rejectInvalidBinaryComptime(
        self: *ConstEvaluator,
        op: ast.BinaryOp,
        lhs: ?ConstValue,
        rhs: ?ConstValue,
        range: source.TextRange,
    ) bool {
        const left = lhs orelse return false;
        const right = rhs orelse return false;
        const left_integer = switch (left) {
            .integer => true,
            else => false,
        };
        const right_integer = switch (right) {
            .integer => |integer| integer,
            else => return false,
        };
        if (!left_integer) return false;

        switch (op) {
            .div, .mod => {
                if (right_integer.eqlZero()) {
                    self.recordRequiredBinaryError(
                        .division_by_zero,
                        range,
                        "comptime division by zero",
                        "division and modulo by zero are invalid in required comptime evaluation",
                    );
                    return true;
                }
            },
            .shl, .wrapping_shl, .shr, .wrapping_shr => {
                if (bridge.positiveShiftAmount(right_integer) == null) {
                    self.recordRequiredBinaryError(
                        .invalid_cast,
                        range,
                        "invalid comptime shift amount",
                        "shift amount must be non-negative and fit in usize",
                    );
                    return true;
                }
            },
            .pow, .wrapping_pow => {
                const amount = bridge.positiveShiftAmount(right_integer) orelse {
                    self.recordRequiredBinaryError(
                        .invalid_cast,
                        range,
                        "invalid comptime exponent",
                        "exponent must be non-negative and fit in usize",
                    );
                    return true;
                };
                const amount_u64: u64 = @intCast(amount);
                if (amount_u64 > self.env.config.max_loop_iterations or
                    amount_u64 > LimitCheck.init(self.env.config, &self.env.stats).remainingSteps())
                {
                    self.recordRequiredBinaryError(
                        .step_limit,
                        range,
                        "comptime exponentiation exceeds evaluation step budget",
                        "exponent exceeds the configured comptime evaluation budget",
                    );
                    return true;
                }
            },
            else => {},
        }
        return false;
    }

    fn recordRequiredBinaryError(
        self: *ConstEvaluator,
        kind: error_mod.CtErrorKind,
        range: source.TextRange,
        message: []const u8,
        reason: []const u8,
    ) void {
        self.recordCtError(error_mod.CtError.withReason(
            kind,
            self.sourceSpan(range),
            message,
            reason,
        ));
    }

    fn evalExprCtValue(self: *ConstEvaluator, expr_id: ast.ExprId) anyerror!?CtValue {
        return self.evalExprCtValueImpl(expr_id, true, true);
    }

    fn evalExprCtValueImpl(
        self: *ConstEvaluator,
        expr_id: ast.ExprId,
        comptime use_cache: bool,
        comptime charge_step: bool,
    ) anyerror!?CtValue {
        if (charge_step and self.consumeStep(self.exprRange(expr_id))) return null;

        return switch (self.file.expression(expr_id).*) {
            .IntegerLiteral => |literal| blk: {
                const value = (try parseIntegerLiteral(self.allocator, literal.text)) orelse break :blk null;
                break :blk try constToCtValue(value);
            },
            .StringLiteral => |literal| blk: {
                const heap_id = try self.env.heap.allocString(literal.text);
                break :blk CtValue{ .string_ref = heap_id };
            },
            .BoolLiteral => |literal| CtValue{ .boolean = literal.value },
            .AddressLiteral => |literal| blk: {
                const value = self.parseAddressLiteral(literal.text) orelse break :blk null;
                break :blk CtValue{ .address = value };
            },
            .BytesLiteral => |literal| blk: {
                const bytes = try self.decodeHexBytesLiteral(literal.text);
                const heap_id = try self.env.heap.allocBytesOwned(bytes);
                break :blk CtValue{ .bytes_ref = heap_id };
            },
            .Name => |name| self.env.lookupValue(name.name) orelse blk: {
                if (self.pathTypeId(name.name)) |type_id| break :blk CtValue{ .type_val = type_id };
                break :blk null;
            },
            .Group => |group| try self.evalExprCtValueImpl(group.expr, use_cache, true),
            .Call => |call| if (try self.evalResultConstructorCallCtValue(call, use_cache)) |result_value|
                result_value
            else if (try self.evalEnumConstructorCallCtValue(expr_id, call, use_cache)) |enum_value|
                enum_value
            else if (try self.evalErrorDeclCallCtValue(call, use_cache)) |error_value|
                error_value
            else
                try self.evalCallCtValue(call, use_cache),
            .Unary => blk: {
                const const_value = (try self.evalExprImpl(expr_id, use_cache)) orelse break :blk null;
                break :blk (try self.constValueToCtValue(const_value)) orelse null;
            },
            .Switch => |switch_expr| try self.evalSwitchExprCtValue(switch_expr, use_cache),
            .Comptime => |comptime_expr| blk: {
                self.required_comptime_depth += 1;
                defer self.required_comptime_depth -= 1;
                break :blk try self.evalComptimeBodyCtValue(comptime_expr.body, use_cache);
            },
            .ArrayLiteral => |array| blk: {
                const elems = try self.allocator.alloc(CtValue, array.elements.len);
                for (array.elements, 0..) |element_id, idx| {
                    _ = try self.evalExprImpl(element_id, use_cache);
                    elems[idx] = (try self.evalExprCtValueImpl(element_id, use_cache, true)) orelse break :blk null;
                }
                const heap_id = try self.env.heap.allocArrayOwned(elems);
                break :blk CtValue{ .array_ref = heap_id };
            },
            .Tuple => |tuple| blk: {
                const elems = try self.allocator.alloc(CtValue, tuple.elements.len);
                for (tuple.elements, 0..) |element_id, idx| {
                    _ = try self.evalExprImpl(element_id, use_cache);
                    elems[idx] = (try self.evalExprCtValueImpl(element_id, use_cache, true)) orelse break :blk null;
                }
                const heap_id = try self.env.heap.allocTupleOwned(elems);
                break :blk CtValue{ .tuple_ref = heap_id };
            },
            .StructLiteral => |struct_literal| blk: {
                if (try self.evalEnumStructLiteralCtValue(expr_id, struct_literal, use_cache)) |enum_value| break :blk enum_value;
                const type_id = (try self.structTypeIdForExpr(expr_id, struct_literal)) orelse break :blk null;
                const fields = try self.allocator.alloc(CtAggregate.StructField, struct_literal.fields.len);
                for (struct_literal.fields, 0..) |field, idx| {
                    _ = try self.evalExprImpl(field.value, use_cache);
                    const field_index = (try self.structFieldIndex(type_id, field.name)) orelse break :blk null;
                    const value = (try self.evalExprCtValueImpl(field.value, use_cache, true)) orelse break :blk null;
                    if (try self.structLiteralFieldType(expr_id, field.name)) |field_type| {
                        if (!try self.validateCtValueForType(value, field_type, field.range)) break :blk null;
                    }
                    fields[idx] = .{
                        .field_id = @intCast(field_index),
                        .value = value,
                    };
                }
                const heap_id = try self.env.heap.allocStructOwned(type_id, fields);
                break :blk CtValue{ .struct_ref = heap_id };
            },
            .Index => |index| blk: {
                _ = try self.evalExprImpl(index.base, use_cache);
                _ = try self.evalExprImpl(index.index, use_cache);
                const base = (try self.evalExprCtValueImpl(index.base, use_cache, true)) orelse break :blk null;
                const index_value = (try self.evalExprCtValueImpl(index.index, use_cache, true)) orelse break :blk null;

                break :blk switch (base) {
                    .array_ref => |heap_id| blk_elem: {
                        const idx = self.ctIndexValue(index_value) orelse break :blk_elem null;
                        const elems = self.env.heap.getArray(heap_id).elems;
                        if (idx >= elems.len) break :blk_elem null;
                        break :blk_elem elems[idx];
                    },
                    .slice_ref => |heap_id| blk_elem: {
                        const idx = self.ctIndexValue(index_value) orelse break :blk_elem null;
                        const elems = self.env.heap.getSlice(heap_id).elems;
                        if (idx >= elems.len) break :blk_elem null;
                        break :blk_elem elems[idx];
                    },
                    .tuple_ref => |heap_id| blk_elem: {
                        const idx = self.ctIndexValue(index_value) orelse break :blk_elem null;
                        const elems = self.env.heap.getTuple(heap_id).elems;
                        if (idx >= elems.len) break :blk_elem null;
                        break :blk_elem elems[idx];
                    },
                    .map_ref => |heap_id| blk_elem: {
                        const entries = self.env.heap.getMap(heap_id).entries;
                        const key = index_value;
                        for (entries) |entry| {
                            if (self.ctValuesEqual(entry.key, key)) break :blk_elem entry.value;
                        }
                        break :blk_elem null;
                    },
                    .string_ref => |heap_id| blk_elem: {
                        const idx = self.ctIndexValue(index_value) orelse break :blk_elem null;
                        const bytes = self.env.heap.getString(heap_id);
                        if (idx >= bytes.len) break :blk_elem null;
                        break :blk_elem CtValue{ .integer = bytes[idx] };
                    },
                    .bytes_ref => |heap_id| blk_elem: {
                        const idx = self.ctIndexValue(index_value) orelse break :blk_elem null;
                        const bytes = self.env.heap.getBytes(heap_id);
                        if (idx >= bytes.len) break :blk_elem null;
                        break :blk_elem CtValue{ .integer = bytes[idx] };
                    },
                    else => null,
                };
            },
            .Builtin => |builtin| blk: {
                if (std.mem.eql(u8, builtin.name, "structFields")) {
                    if (builtin.args.len != 1) break :blk null;
                    const struct_item = (try self.resolveReflectionStructReference(builtin.args[0])) orelse break :blk null;
                    break :blk try self.buildStructFieldsCtValue(struct_item);
                }
                if (std.mem.eql(u8, builtin.name, "traitMethods")) {
                    if (builtin.args.len != 1) break :blk null;
                    const trait_item = (try self.resolveReflectionTraitReference(builtin.args[0])) orelse break :blk null;
                    break :blk try self.buildTraitMethodsCtValue(trait_item);
                }
                if (std.mem.eql(u8, builtin.name, "abiDecode") or std.mem.eql(u8, builtin.name, "abiDecodePermissive")) {
                    break :blk try self.evalAbiDecodeBuiltinCtValue(builtin, use_cache);
                }
                if (std.mem.eql(u8, builtin.name, "cast") and builtin.type_arg != null and builtin.args.len > 0) {
                    const target = self.valueConstructionTarget(builtin.type_arg.?);
                    if (target != .none) {
                        break :blk try self.evalExprCtValueAsImpl(builtin.args[0], target, use_cache);
                    }
                }
                const const_value = (try self.evalBuiltin(builtin)) orelse break :blk null;
                break :blk (try self.constValueToCtValue(const_value)) orelse null;
            },
            .Field => |field| blk: {
                _ = try self.evalExprImpl(field.base, use_cache);
                switch (self.file.expression(field.base).*) {
                    .Name => |name| {
                        if (self.lookupNamedEnumVariant(name.name, field.name)) |enum_value| break :blk enum_value;
                    },
                    else => {},
                }
                const base = (try self.evalExprCtValueImpl(field.base, use_cache, true)) orelse break :blk null;
                break :blk switch (base) {
                    .struct_ref => |heap_id| blk_field: {
                        const struct_data = self.env.heap.getStruct(heap_id);
                        const field_index = (try self.structFieldIndex(struct_data.type_id, field.name)) orelse break :blk_field null;
                        break :blk_field self.structFieldValue(struct_data, field_index);
                    },
                    .tuple_ref => |heap_id| blk_field: {
                        const elems = self.env.heap.getTuple(heap_id).elems;
                        const field_index = (try self.anonymousStructFieldIndexForExpr(field.base, field.name)) orelse break :blk_field null;
                        if (field_index >= elems.len) break :blk_field null;
                        break :blk_field elems[field_index];
                    },
                    .string_ref => |heap_id| blk_field: {
                        if (!(std.mem.eql(u8, field.name, "length") or std.mem.eql(u8, field.name, "len"))) break :blk_field null;
                        break :blk_field CtValue{ .integer = @intCast(self.env.heap.getString(heap_id).len) };
                    },
                    .bytes_ref => |heap_id| blk_field: {
                        if (!(std.mem.eql(u8, field.name, "length") or std.mem.eql(u8, field.name, "len"))) break :blk_field null;
                        break :blk_field CtValue{ .integer = @intCast(self.env.heap.getBytes(heap_id).len) };
                    },
                    .slice_ref => |heap_id| blk_field: {
                        if (!(std.mem.eql(u8, field.name, "length") or std.mem.eql(u8, field.name, "len"))) break :blk_field null;
                        break :blk_field CtValue{ .integer = @intCast(self.env.heap.getSlice(heap_id).elems.len) };
                    },
                    .map_ref => |heap_id| blk_field: {
                        if (!(std.mem.eql(u8, field.name, "length") or std.mem.eql(u8, field.name, "len"))) break :blk_field null;
                        break :blk_field CtValue{ .integer = @intCast(self.env.heap.getMap(heap_id).entries.len) };
                    },
                    else => null,
                };
            },
            .Binary => |binary| blk: {
                if (try self.evalExprCtValueImpl(binary.lhs, use_cache, true)) |lhs| {
                    if (try self.evalExprCtValueImpl(binary.rhs, use_cache, true)) |rhs| {
                        break :blk switch (binary.op) {
                            .eq => CtValue{ .boolean = self.ctValuesEqual(lhs, rhs) },
                            .ne => CtValue{ .boolean = !self.ctValuesEqual(lhs, rhs) },
                            else => null,
                        };
                    }
                }

                const lhs_const = (try self.evalExprImpl(binary.lhs, use_cache)) orelse break :blk null;
                const rhs_const = (try self.evalExprImpl(binary.rhs, use_cache)) orelse break :blk null;
                break :blk switch (binary.op) {
                    .eq => CtValue{ .boolean = constEquals(lhs_const, rhs_const) },
                    .ne => CtValue{ .boolean = !constEquals(lhs_const, rhs_const) },
                    else => null,
                };
            },
            .ErrorReturn => |error_return| try self.evalErrorReturnCtValue(error_return, use_cache),
            else => null,
        };
    }

    fn valueConstructionTarget(self: *ConstEvaluator, type_expr_id: ast.TypeExprId) ValueConstructionTarget {
        return switch (self.file.typeExpr(type_expr_id).*) {
            .Slice => .slice,
            .Generic => |generic| if (std.mem.eql(u8, generic.name, "map")) .map else .none,
            else => .none,
        };
    }

    fn exprUsesConstFallbackForCtValue(self: *ConstEvaluator, expr_id: ast.ExprId) bool {
        return switch (self.file.expression(expr_id).*) {
            .Call, .Unary, .Comptime => true,
            else => false,
        };
    }

    fn evalSwitchExprCtValue(self: *ConstEvaluator, switch_expr: ast.SwitchExpr, comptime use_cache: bool) anyerror!?CtValue {
        if (try self.evalExprAsCtValue(switch_expr.condition, use_cache)) |condition_ct| {
            for (switch_expr.arms) |arm| {
                const matches = try self.patternMatchesCt(condition_ct, arm.pattern);
                if (!matches) continue;
                self.env.pushScope(false) catch return null;
                defer self.env.popScope();
                try self.bindSwitchPatternCtValue(condition_ct, arm.pattern);
                return try self.evalExprAsCtValue(arm.value, use_cache);
            }
            if (switch_expr.else_expr) |else_expr| return try self.evalExprAsCtValue(else_expr, use_cache);
            return null;
        }

        const condition = (try self.evalExprImpl(switch_expr.condition, use_cache)) orelse return null;
        for (switch_expr.arms) |arm| {
            if (self.patternMatches(condition, arm.pattern)) {
                return try self.evalExprAsCtValue(arm.value, use_cache);
            }
        }
        if (switch_expr.else_expr) |else_expr| return try self.evalExprAsCtValue(else_expr, use_cache);
        return null;
    }

    fn evalExprCtValueAs(self: *ConstEvaluator, expr_id: ast.ExprId, target: ValueConstructionTarget) anyerror!?CtValue {
        return self.evalExprCtValueAsImpl(expr_id, target, true);
    }

    fn evalExprCtValueAsImpl(self: *ConstEvaluator, expr_id: ast.ExprId, target: ValueConstructionTarget, comptime use_cache: bool) anyerror!?CtValue {
        return switch (target) {
            .none => try self.evalExprCtValueImpl(expr_id, use_cache, true),
            .slice => switch (self.file.expression(expr_id).*) {
                .ArrayLiteral => |array| blk: {
                    const elems = try self.allocator.alloc(CtValue, array.elements.len);
                    for (array.elements, 0..) |element_id, idx| {
                        _ = try self.evalExprImpl(element_id, use_cache);
                        elems[idx] = (try self.evalExprCtValueImpl(element_id, use_cache, true)) orelse break :blk null;
                    }
                    const heap_id = try self.env.heap.allocSliceOwned(elems);
                    break :blk CtValue{ .slice_ref = heap_id };
                },
                else => try self.evalExprCtValueImpl(expr_id, use_cache, true),
            },
            .map => switch (self.file.expression(expr_id).*) {
                .ArrayLiteral, .Tuple => blk: {
                    const entries = (try self.evalMapEntries(expr_id)) orelse break :blk null;
                    const heap_id = try self.env.heap.allocMapOwned(entries);
                    break :blk CtValue{ .map_ref = heap_id };
                },
                else => try self.evalExprCtValueImpl(expr_id, use_cache, true),
            },
        };
    }

    fn evalMapEntries(self: *ConstEvaluator, expr_id: ast.ExprId) anyerror!?[]CtAggregate.MapEntry {
        const items: []const ast.ExprId = switch (self.file.expression(expr_id).*) {
            .ArrayLiteral => |array| array.elements,
            .Tuple => |tuple| tuple.elements,
            else => return null,
        };
        const entries = try self.allocator.alloc(CtAggregate.MapEntry, items.len);
        for (items, 0..) |item_expr, idx| {
            entries[idx] = (try self.evalMapEntry(item_expr)) orelse return null;
        }
        return entries;
    }

    fn evalMapEntry(self: *ConstEvaluator, expr_id: ast.ExprId) anyerror!?CtAggregate.MapEntry {
        return switch (self.file.expression(expr_id).*) {
            .Tuple => |tuple| blk: {
                if (tuple.elements.len != 2) break :blk null;
                _ = try self.evalExpr(tuple.elements[0]);
                _ = try self.evalExpr(tuple.elements[1]);
                const key = (try self.evalExprCtValue(tuple.elements[0])) orelse break :blk null;
                const value = (try self.evalExprCtValue(tuple.elements[1])) orelse break :blk null;
                break :blk .{ .key = key, .value = value };
            },
            else => null,
        };
    }

    fn structTypeId(self: *ConstEvaluator, type_name: []const u8) ?u32 {
        const item_id = self.lookupNamedItem(type_name) orelse return null;
        if (self.file.item(item_id).* != .Struct) return null;
        return self.namedTypeId(item_id);
    }

    fn structTypeIdForExpr(self: *ConstEvaluator, expr_id: ast.ExprId, struct_literal: ast.StructLiteralExpr) !?u32 {
        if (try self.currentTypeCheckResult()) |typecheck| {
            const expr_type = typecheck.exprType(expr_id);
            if (expr_type == .struct_) {
                if (self.lookupNamedItem(expr_type.struct_.name)) |item_id| {
                    if (self.file.item(item_id).* == .Struct) return self.namedTypeId(item_id);
                }
                if (typecheck.instantiatedStructByName(expr_type.struct_.name)) |instantiated| {
                    return self.namedTypeId(instantiated.template_item_id);
                }
            }
        }
        return self.structTypeId(struct_literal.type_name);
    }

    fn lookupNamedEnumVariant(self: *ConstEvaluator, enum_name: []const u8, variant_name: []const u8) ?CtValue {
        const item_id = self.lookupNamedItem(enum_name) orelse return null;
        const item = self.file.item(item_id).*;
        if (item != .Enum) return null;
        const variant_index = self.enumVariantIndex(item_id, item.Enum.variants, variant_name) orelse return null;
        return CtValue{ .adt_val = CtEnum{
            .type_id = self.namedTypeId(item_id),
            .variant_id = @intCast(variant_index),
            .payload = null,
        } };
    }

    const EnumVariantRef = struct {
        item_id: ast.ItemId,
        variant_id: u32,
    };

    fn enumVariantRefFromExpr(self: *ConstEvaluator, expr_id: ast.ExprId) ?EnumVariantRef {
        const field = switch (self.file.expression(expr_id).*) {
            .Field => |field| field,
            .Group => |group| return self.enumVariantRefFromExpr(group.expr),
            else => return null,
        };
        const enum_name = switch (self.file.expression(field.base).*) {
            .Name => |name| name.name,
            .Group => |group| switch (self.file.expression(group.expr).*) {
                .Name => |name| name.name,
                else => return null,
            },
            else => return null,
        };
        const item_id = self.lookupNamedItem(enum_name) orelse return null;
        const item = self.file.item(item_id).*;
        if (item != .Enum) return null;
        const variant_index = self.enumVariantIndex(item_id, item.Enum.variants, field.name) orelse return null;
        return .{ .item_id = item_id, .variant_id = @intCast(variant_index) };
    }

    fn enumVariantRefFromStructLiteral(self: *ConstEvaluator, struct_literal: ast.StructLiteralExpr) ?EnumVariantRef {
        const dot_index = std.mem.lastIndexOfScalar(u8, struct_literal.type_name, '.') orelse return null;
        if (dot_index == 0 or dot_index + 1 >= struct_literal.type_name.len) return null;
        const enum_name = struct_literal.type_name[0..dot_index];
        const variant_name = struct_literal.type_name[dot_index + 1 ..];
        const item_id = self.lookupNamedItem(enum_name) orelse return null;
        const item = self.file.item(item_id).*;
        if (item != .Enum) return null;
        const variant_index = self.enumVariantIndex(item_id, item.Enum.variants, variant_name) orelse return null;
        return .{ .item_id = item_id, .variant_id = @intCast(variant_index) };
    }

    fn enumVariantIndex(self: *ConstEvaluator, item_id: ast.ItemId, variants: []const ast.EnumVariant, name: []const u8) ?usize {
        if (self.currentItemIndex() catch null) |item_index| {
            return item_index.lookupEnumVariantIndex(item_id, name);
        }
        for (variants, 0..) |variant, index| {
            if (std.mem.eql(u8, variant.name, name)) return index;
        }
        return null;
    }

    fn enumVariantRefFromPattern(self: *ConstEvaluator, pattern: ast.SwitchPattern) ?EnumVariantRef {
        return switch (pattern) {
            .Expr => |expr_id| blk: {
                switch (self.file.expression(expr_id).*) {
                    .Call => |call| break :blk self.enumVariantRefFromExpr(call.callee),
                    else => break :blk self.enumVariantRefFromExpr(expr_id),
                }
            },
            .NamedError => |named| self.enumVariantRefFromExpr(named.callee),
            else => null,
        };
    }

    fn errorVariantRefFromExpr(self: *ConstEvaluator, expr_id: ast.ExprId) ?EnumVariantRef {
        const name = self.errorDeclNameFromExpr(expr_id) orelse return null;
        const item_id = self.lookupNamedItem(name) orelse return null;
        if (self.file.item(item_id).* != .ErrorDecl) return null;
        return .{ .item_id = item_id, .variant_id = 0 };
    }

    fn errorDeclNameFromExpr(self: *ConstEvaluator, expr_id: ast.ExprId) ?[]const u8 {
        return switch (self.file.expression(expr_id).*) {
            .Name => |name| name.name,
            .Field => |field| field.name,
            .Group => |group| self.errorDeclNameFromExpr(group.expr),
            else => null,
        };
    }

    fn sumVariantRefFromPattern(self: *ConstEvaluator, pattern: ast.SwitchPattern) ?EnumVariantRef {
        return self.enumVariantRefFromPattern(pattern) orelse switch (pattern) {
            .Expr => |expr_id| blk: {
                switch (self.file.expression(expr_id).*) {
                    .Call => |call| break :blk self.errorVariantRefFromExpr(call.callee),
                    else => break :blk self.errorVariantRefFromExpr(expr_id),
                }
            },
            .NamedError => |named| self.errorVariantRefFromExpr(named.callee),
            else => null,
        };
    }

    fn evalResultConstructorCallCtValue(self: *ConstEvaluator, call: ast.CallExpr, comptime use_cache: bool) anyerror!?CtValue {
        const name = switch (self.file.expression(call.callee).*) {
            .Name => |callee| callee.name,
            .Group => |group| switch (self.file.expression(group.expr).*) {
                .Name => |callee| callee.name,
                else => return null,
            },
            else => return null,
        };
        if (!std.mem.eql(u8, name, "Ok") and !std.mem.eql(u8, name, "Err")) return null;
        if (call.args.len != 1) return null;

        const payload_value = (try self.evalExprAsCtValue(call.args[0], use_cache)) orelse return null;
        const payload_id = try self.env.heap.allocTupleOwned(try self.allocator.dupe(CtValue, &.{payload_value}));
        return CtValue{ .error_union_val = CtErrorUnion{
            .is_error = std.mem.eql(u8, name, "Err"),
            .payload = payload_id,
        } };
    }

    fn evalErrorReturnCtValue(self: *ConstEvaluator, error_return: ast.ErrorReturnExpr, comptime use_cache: bool) anyerror!?CtValue {
        const item_id = self.lookupNamedItem(error_return.name) orelse return null;
        const item = self.file.item(item_id).*;
        if (item != .ErrorDecl) return null;
        const payload_id: ?comptime_mod.HeapId = if (error_return.args.len == 0) null else blk: {
            const elems = try self.allocator.alloc(CtValue, error_return.args.len);
            for (error_return.args, 0..) |arg, index| {
                elems[index] = (try self.evalExprAsCtValue(arg, use_cache)) orelse return null;
            }
            break :blk try self.env.heap.allocTupleOwned(elems);
        };
        return CtValue{ .adt_val = CtEnum{
            .type_id = self.namedTypeId(item_id),
            .variant_id = 0,
            .payload = payload_id,
        } };
    }

    fn evalErrorDeclCallCtValue(self: *ConstEvaluator, call: ast.CallExpr, comptime use_cache: bool) anyerror!?CtValue {
        const name = self.errorDeclNameFromExpr(call.callee) orelse return null;
        const item_id = self.lookupNamedItem(name) orelse return null;
        const item = self.file.item(item_id).*;
        if (item != .ErrorDecl) return null;

        const payload_id: ?comptime_mod.HeapId = if (call.args.len == 0) null else blk: {
            const elems = try self.allocator.alloc(CtValue, call.args.len);
            for (call.args, 0..) |arg, index| {
                elems[index] = (try self.evalExprAsCtValue(arg, use_cache)) orelse return null;
            }
            break :blk try self.env.heap.allocTupleOwned(elems);
        };
        return CtValue{ .adt_val = .{
            .type_id = self.namedTypeId(item_id),
            .variant_id = 0,
            .payload = payload_id,
        } };
    }

    fn evalEnumStructLiteralCtValue(self: *ConstEvaluator, expr_id: ast.ExprId, struct_literal: ast.StructLiteralExpr, comptime use_cache: bool) anyerror!?CtValue {
        const typecheck = (try self.currentTypeCheckResult()) orelse return null;
        if (typecheck.exprType(expr_id).kind() != .enum_) return null;
        const variant_ref = self.enumVariantRefFromStructLiteral(struct_literal) orelse return null;

        const payload_fields = try self.enumVariantNamedPayloadFields(expr_id, variant_ref);
        if (payload_fields == null) {
            if (struct_literal.fields.len != 0) return null;
            return CtValue{ .adt_val = .{
                .type_id = self.namedTypeId(variant_ref.item_id),
                .variant_id = variant_ref.variant_id,
                .payload = null,
            } };
        }

        const fields = payload_fields.?;
        const init_lookup = try lookup_index.buildNamed(ast.StructFieldInit, self.allocator, struct_literal.fields, "name");
        defer self.allocator.free(init_lookup);
        const elems = try self.allocator.alloc(CtValue, fields.len);
        for (fields, 0..) |payload_field, index| {
            const init = lookup_index.findNamedItem(ast.StructFieldInit, struct_literal.fields, init_lookup, payload_field.name) orelse return null;
            _ = try self.evalExprImpl(init.value, use_cache);
            const value = (try self.evalExprCtValueImpl(init.value, use_cache, true)) orelse return null;
            if (!try self.validateCtValueForType(value, payload_field.ty, init.range)) return null;
            elems[index] = value;
        }

        const payload_id = try self.env.heap.allocTupleOwned(elems);
        return CtValue{ .adt_val = .{
            .type_id = self.namedTypeId(variant_ref.item_id),
            .variant_id = variant_ref.variant_id,
            .payload = payload_id,
        } };
    }

    fn evalEnumConstructorCallCtValue(self: *ConstEvaluator, expr_id: ast.ExprId, call: ast.CallExpr, comptime use_cache: bool) anyerror!?CtValue {
        const variant_ref = self.enumVariantRefFromExpr(call.callee) orelse return null;
        const payload_id: ?comptime_mod.HeapId = if (call.args.len == 0) null else blk: {
            const elems = try self.allocator.alloc(CtValue, call.args.len);
            for (call.args, 0..) |arg, index| {
                const value = (try self.evalExprAsCtValue(arg, use_cache)) orelse return null;
                if (try self.enumVariantPayloadArgType(expr_id, variant_ref, index)) |arg_type| {
                    if (!try self.validateCtValueForType(value, arg_type, self.exprRange(arg))) return null;
                }
                elems[index] = value;
            }
            break :blk try self.env.heap.allocTupleOwned(elems);
        };
        return CtValue{ .adt_val = .{
            .type_id = self.namedTypeId(variant_ref.item_id),
            .variant_id = variant_ref.variant_id,
            .payload = payload_id,
        } };
    }

    fn parseAddressLiteral(self: *ConstEvaluator, text: []const u8) ?u160 {
        _ = self;
        if (!std.mem.startsWith(u8, text, "0x")) return null;
        return std.fmt.parseInt(u160, text[2..], 16) catch null;
    }

    fn ctValuesEqual(self: *ConstEvaluator, lhs: CtValue, rhs: CtValue) bool {
        return switch (lhs) {
            .integer => |value| switch (rhs) {
                .integer => |other| value == other,
                else => false,
            },
            .boolean => |value| switch (rhs) {
                .boolean => |other| value == other,
                else => false,
            },
            .address => |value| switch (rhs) {
                .address => |other| value == other,
                else => false,
            },
            .string_ref => |heap_id| switch (rhs) {
                .string_ref => |other| std.mem.eql(u8, self.env.heap.getString(heap_id), self.env.heap.getString(other)),
                else => false,
            },
            .bytes_ref => |heap_id| switch (rhs) {
                .bytes_ref => |other| std.mem.eql(u8, self.env.heap.getBytes(heap_id), self.env.heap.getBytes(other)),
                else => false,
            },
            .adt_val => |value| switch (rhs) {
                .adt_val => |other| value.type_id == other.type_id and value.variant_id == other.variant_id and value.payload == other.payload,
                else => false,
            },
            .error_union_val => |value| switch (rhs) {
                .error_union_val => |other| value.is_error == other.is_error and self.ctValuesEqual(
                    self.env.heap.getTuple(value.payload).elems[0],
                    self.env.heap.getTuple(other.payload).elems[0],
                ),
                else => false,
            },
            .type_val => |value| switch (rhs) {
                .type_val => |other| value == other,
                else => false,
            },
            .void_val => rhs == .void_val,
            else => false,
        };
    }

    fn ctIndexValue(self: *ConstEvaluator, value: CtValue) ?usize {
        _ = self;
        return switch (value) {
            .integer => |integer| blk: {
                if (integer > std.math.maxInt(usize)) break :blk null;
                break :blk @as(usize, @intCast(integer));
            },
            else => null,
        };
    }

    fn setMapEntryValue(self: *ConstEvaluator, heap_id: comptime_mod.HeapId, key: CtValue, value: CtValue) !comptime_mod.HeapId {
        const unique_id = try self.env.heap.ensureUnique(heap_id);
        const map = &self.env.heap.get(unique_id).data.map;
        for (map.entries) |*entry| {
            if (self.ctValuesEqual(entry.key, key)) {
                entry.value = value;
                return unique_id;
            }
        }

        const old_entries = map.entries;
        const grown = try self.allocator.alloc(CtAggregate.MapEntry, old_entries.len + 1);
        @memcpy(grown[0..old_entries.len], old_entries);
        grown[old_entries.len] = .{ .key = key, .value = value };
        self.allocator.free(old_entries);
        map.entries = grown;
        return unique_id;
    }

    fn decodeHexBytesLiteral(self: *ConstEvaluator, text: []const u8) ![]u8 {
        if (text.len % 2 != 0) return error.InvalidHexLiteral;

        const out = try self.allocator.alloc(u8, text.len / 2);
        var i: usize = 0;
        while (i < out.len) : (i += 1) {
            const hi = std.fmt.charToDigit(text[i * 2], 16) catch return error.InvalidHexLiteral;
            const lo = std.fmt.charToDigit(text[i * 2 + 1], 16) catch return error.InvalidHexLiteral;
            out[i] = @intCast((hi << 4) | lo);
        }
        return out;
    }

    fn structFieldIndex(self: *ConstEvaluator, type_id: u32, field_name: []const u8) !?usize {
        const item_id = self.itemIdForNamedTypeId(type_id) orelse return null;
        const item_index = (try self.currentItemIndex()) orelse return null;
        return item_index.lookupStructFieldIndex(item_id, field_name);
    }

    fn structLiteralFieldType(self: *ConstEvaluator, expr_id: ast.ExprId, field_name: []const u8) !?Type {
        const typecheck = (try self.currentTypeCheckResult()) orelse return null;
        const expr_type = typecheck.exprType(expr_id);
        if (expr_type != .struct_) return null;

        if (typecheck.instantiatedStructByName(expr_type.struct_.name)) |instantiated| {
            if (instantiated.fieldByName(field_name)) |field| return field.ty;
            return null;
        }

        const item_id = self.lookupNamedItem(expr_type.struct_.name) orelse return null;
        const item_index = (try self.currentItemIndex()) orelse return null;
        const field = item_index.lookupStructField(self.file, item_id, field_name) orelse return null;
        return try self.modelTypeFromTypeExpr(field.type_expr);
    }

    fn enumVariantNamedPayloadFields(self: *ConstEvaluator, expr_id: ast.ExprId, variant_ref: EnumVariantRef) !?[]const AnonymousStructField {
        if (try self.currentTypeCheckResult()) |typecheck| {
            const expr_type = typecheck.exprType(expr_id);
            if (expr_type == .enum_) {
                if (typecheck.instantiatedEnumByName(expr_type.enum_.name)) |instantiated| {
                    if (variant_ref.variant_id >= instantiated.variants.len) return null;
                    const payload = instantiated.variants[variant_ref.variant_id].payload_type orelse return null;
                    return switch (payload) {
                        .anonymous_struct => |struct_type| struct_type.fields,
                        else => null,
                    };
                }
            }
        }

        const item = self.file.item(variant_ref.item_id).*;
        if (item != .Enum or variant_ref.variant_id >= item.Enum.variants.len) return null;
        return try self.enumNamedPayloadFieldsFromAst(item.Enum.variants[@intCast(variant_ref.variant_id)].payload);
    }

    fn enumNamedPayloadFieldsFromAst(self: *ConstEvaluator, payload: ast.EnumVariantPayload) !?[]const AnonymousStructField {
        return switch (payload) {
            .named => |fields| blk: {
                const result = try self.allocator.alloc(AnonymousStructField, fields.len);
                for (fields, 0..) |field, index| {
                    result[index] = .{
                        .name = field.name,
                        .ty = (try self.modelTypeFromTypeExpr(field.type_expr)) orelse .{ .unknown = {} },
                    };
                }
                break :blk result;
            },
            else => null,
        };
    }

    fn enumVariantPayloadArgType(self: *ConstEvaluator, expr_id: ast.ExprId, variant_ref: EnumVariantRef, arg_index: usize) !?Type {
        const typecheck = (try self.currentTypeCheckResult()) orelse return null;
        const expr_type = typecheck.exprType(expr_id);
        if (expr_type == .enum_) {
            if (typecheck.instantiatedEnumByName(expr_type.enum_.name)) |instantiated| {
                if (variant_ref.variant_id >= instantiated.variants.len) return null;
                return enumPayloadArgTypeFromModel(instantiated.variants[variant_ref.variant_id].payload_type, arg_index);
            }
        }

        const item = self.file.item(variant_ref.item_id).*;
        if (item != .Enum or variant_ref.variant_id >= item.Enum.variants.len) return null;
        return try self.enumPayloadArgTypeFromAst(item.Enum.variants[variant_ref.variant_id].payload, arg_index);
    }

    fn enumPayloadArgTypeFromModel(payload_type: ?Type, arg_index: usize) ?Type {
        const payload = payload_type orelse return null;
        return switch (payload) {
            .tuple => |elements| if (arg_index < elements.len) elements[arg_index] else null,
            .anonymous_struct => |struct_type| if (arg_index < struct_type.fields.len) struct_type.fields[arg_index].ty else null,
            else => if (arg_index == 0) payload else null,
        };
    }

    fn enumPayloadArgTypeFromAst(self: *ConstEvaluator, payload: ast.EnumVariantPayload, arg_index: usize) !?Type {
        return switch (payload) {
            .none => null,
            .positional => |types| if (arg_index < types.len) try self.modelTypeFromTypeExpr(types[arg_index]) else null,
            .named => |fields| if (arg_index < fields.len) try self.modelTypeFromTypeExpr(fields[arg_index].type_expr) else null,
        };
    }

    fn modelTypeFromTypeExpr(self: *ConstEvaluator, type_expr_id: ast.TypeExprId) !?Type {
        return switch (self.file.typeExpr(type_expr_id).*) {
            .Path => |path| blk: {
                if (integerTypeFromName(path.name)) |integer| break :blk Type{ .integer = integer };
                if (std.mem.eql(u8, std.mem.trim(u8, path.name, " \t\n\r"), "address")) break :blk Type{ .address = {} };
                if (std.mem.eql(u8, std.mem.trim(u8, path.name, " \t\n\r"), "bool")) break :blk Type{ .bool = {} };
                break :blk null;
            },
            .Generic => |generic| blk: {
                if (!refinements.isKnownName(generic.name) or generic.args.len == 0 or generic.args[0] != .Type) break :blk null;
                const base_type = (try self.modelTypeFromTypeExpr(generic.args[0].Type)) orelse break :blk null;
                const args = try type_descriptors.refinementArgsFromAst(self.allocator, self.file, generic.args);
                break :blk try refinements.refinementType(self.allocator, generic.name, base_type, args);
            },
            else => null,
        };
    }

    fn validateCtValueForType(self: *ConstEvaluator, value: CtValue, ty: Type, range: source.TextRange) !bool {
        return switch (ty) {
            .refinement => |refinement| try self.validateCtValueForRefinement(value, refinement, range),
            else => true,
        };
    }

    fn validateCtValueForRefinement(self: *ConstEvaluator, value: CtValue, refinement: model.RefinementType, range: source.TextRange) !bool {
        const valid = if (refinements.bounds(refinement)) |info| blk: {
            const integer = switch (value) {
                .integer => |integer| integer,
                else => break :blk true,
            };
            if (parseU256Text(info.min_text)) |min| {
                if (integer < min) break :blk false;
            }
            if (parseU256Text(info.max_text)) |max| {
                if (integer > max) break :blk false;
            }
            break :blk true;
        } else if (refinements.kindForName(refinement.name) == .non_zero_address) blk: {
            const address = switch (value) {
                .address => |address| address,
                else => break :blk true,
            };
            break :blk address != 0;
        } else true;

        if (valid) return true;
        self.recordCtError(error_mod.CtError.withReason(
            .not_comptime,
            self.sourceSpan(range),
            "comptime refinement violation",
            try refinements.expectationText(self.allocator, refinement),
        ));
        return false;
    }

    fn parseU256Text(text: ?[]const u8) ?u256 {
        const raw = text orelse return null;
        return std.fmt.parseInt(u256, raw, 10) catch null;
    }

    fn structFieldValue(self: *ConstEvaluator, struct_data: CtAggregate.StructData, field_index: usize) ?CtValue {
        _ = self;
        const field_id: comptime_mod.FieldId = @intCast(field_index);
        for (struct_data.fields) |field| {
            if (field.field_id == field_id) return field.value;
        }
        return null;
    }

    fn readPatternCtValue(self: *ConstEvaluator, pattern_id: ast.PatternId) anyerror!?CtValue {
        return switch (self.file.pattern(pattern_id).*) {
            .Name => |name| self.env.lookupValue(name.name),
            .Field => |field| blk: {
                const base = (try self.readPatternCtValue(field.base)) orelse break :blk null;
                break :blk switch (base) {
                    .struct_ref => |heap_id| blk_field: {
                        const struct_data = self.env.heap.getStruct(heap_id);
                        const field_index = (try self.structFieldIndex(struct_data.type_id, field.name)) orelse break :blk_field null;
                        break :blk_field self.structFieldValue(struct_data, field_index);
                    },
                    else => null,
                };
            },
            else => null,
        };
    }

    fn writePatternCtValue(self: *ConstEvaluator, pattern_id: ast.PatternId, value: CtValue) anyerror!bool {
        switch (self.file.pattern(pattern_id).*) {
            .Name => |name| {
                try self.env.set(name.name, value);
                return true;
            },
            .Field => |field| {
                const base = (try self.readPatternCtValue(field.base)) orelse return false;
                const updated = try self.updateStructFieldCtValue(base, field.name, value) orelse return false;
                return try self.writePatternCtValue(field.base, updated);
            },
            else => return false,
        }
    }

    fn updateStructFieldCtValue(self: *ConstEvaluator, base: CtValue, field_name: []const u8, value: CtValue) anyerror!?CtValue {
        return switch (base) {
            .struct_ref => |heap_id| blk: {
                const struct_data = self.env.heap.getStruct(heap_id);
                const field_index = (try self.structFieldIndex(struct_data.type_id, field_name)) orelse break :blk null;
                break :blk CtValue{ .struct_ref = try self.env.heap.setStructField(heap_id, @intCast(field_index), value) };
            },
            else => null,
        };
    }

    fn lookupNamedItem(self: *ConstEvaluator, name: []const u8) ?ast.ItemId {
        if (self.current_contract) |contract_id| {
            switch (self.file.item(contract_id).*) {
                .Contract => |contract| for (contract.members) |member_id| {
                    const member_name = self.itemName(member_id) orelse continue;
                    if (std.mem.eql(u8, member_name, name)) return member_id;
                },
                else => {},
            }
        }
        for (self.file.root_items) |item_id| {
            if (self.itemName(item_id)) |item_name| {
                if (std.mem.eql(u8, item_name, name)) return item_id;
            }
        }
        return null;
    }

    fn patternName(self: *ConstEvaluator, pattern_id: ast.PatternId) ?[]const u8 {
        return switch (self.file.pattern(pattern_id).*) {
            .Name => |name| name.name,
            else => null,
        };
    }

    const CallableFunction = struct {
        module_id: source.ModuleId,
        file: *const ast.AstFile,
        item_id: ast.ItemId,
        function: ast.FunctionItem,
        contract_id: ?ast.ItemId = null,
        synthetic_self_arg: ?ast.ExprId = null,
    };

    const AbiFunctionReference = struct {
        name: []const u8,
        param_types: []const Type,
        has_self: bool = false,
        signature: ?[]const u8 = null,
    };

    const AbiEventReference = struct {
        signature: []const u8,
    };

    const AbiStructReference = struct {
        item: ast.StructItem,
    };

    const ReflectionTraitReference = struct {
        module_id: source.ModuleId,
        file: *const ast.AstFile,
        item: ast.TraitItem,
    };

    const NamedTypeRef = struct {
        module_id: ?source.ModuleId,
        item_id: ast.ItemId,
    };

    fn signatureForTraitMethod(self: *ConstEvaluator, method: anytype) !?[]const u8 {
        var abi_types: std.ArrayList([]const u8) = .empty;
        defer abi_types.deinit(self.allocator);

        for (method.parameters) |parameter| {
            const abi_type = (try self.typeExprAbiName(parameter.type_expr)) orelse return null;
            try abi_types.append(self.allocator, abi_type);
        }

        return try hir_abi.signatureForAbiTypes(self.allocator, method.name, abi_types.items);
    }

    fn signatureForLogDecl(self: *ConstEvaluator, log_decl: ast.LogDeclItem) !?[]const u8 {
        var abi_types: std.ArrayList([]const u8) = .empty;
        defer abi_types.deinit(self.allocator);

        for (log_decl.fields) |field| {
            const abi_type = (try self.typeExprAbiName(field.type_expr)) orelse return null;
            try abi_types.append(self.allocator, abi_type);
        }

        const event_name = hir_abi.eventWireNameFromLogDecl(self.file, log_decl) orelse return null;
        return try hir_abi.signatureForAbiTypes(self.allocator, event_name, abi_types.items);
    }

    fn ensureTypeChecked(self: *ConstEvaluator, key: model.TypeCheckKey) !void {
        const module_id = self.module_id orelse return;
        const type_query = self.type_query orelse return;
        _ = try type_query.ensureTypeCheck(module_id, key);
    }

    fn currentTypeCheckResult(self: *ConstEvaluator) !?*const model.TypeCheckResult {
        const key = self.current_typecheck_key orelse return null;
        const module_id = self.module_id orelse return null;
        const type_query = self.type_query orelse return null;
        return try type_query.ensureTypeCheck(module_id, key);
    }

    fn currentModuleTypeCheckResult(self: *ConstEvaluator) !?*const model.TypeCheckResult {
        const module_id = self.module_id orelse return null;
        const type_query = self.type_query orelse return null;
        return try type_query.moduleTypeCheck(module_id);
    }

    fn currentItemIndex(self: *ConstEvaluator) !?*const model.ItemIndexResult {
        const type_query = self.type_query orelse return self.fallback_item_index;
        const module_id = self.module_id orelse return self.fallback_item_index;
        return try type_query.itemIndex(module_id);
    }

    fn currentLayoutContext(self: *ConstEvaluator) !?abi_layout_context.LayoutContext {
        const typecheck = (try self.currentTypeCheckResult()) orelse (try self.currentModuleTypeCheckResult()) orelse return null;
        const item_index = (try self.currentItemIndex()) orelse return null;
        return .{
            .allocator = self.allocator,
            .provider = abi_layout_provider.abiLayoutProvider(self.file, item_index, typecheck),
        };
    }

    fn constEvalForModule(self: *ConstEvaluator, module_id: source.ModuleId) !?*const ConstEvalResult {
        const type_query = self.type_query orelse return null;
        return try type_query.constEval(module_id);
    }

    fn callableFunctionIsPure(self: *ConstEvaluator, item_id: ast.ItemId) !bool {
        const typecheck = (try self.currentModuleTypeCheckResult()) orelse return true;
        return typecheck.itemEffect(item_id) == .pure;
    }

    fn ensureNamedItemTypeChecked(self: *ConstEvaluator, name: []const u8) !void {
        const item_id = self.lookupNamedItem(name) orelse return;
        try self.ensureTypeChecked(.{ .item = item_id });
    }

    fn astFileForModule(self: *ConstEvaluator, module_id: source.ModuleId) !*const ast.AstFile {
        const type_query = self.type_query orelse return error.MissingTypeQuery;
        return try type_query.astFile(module_id);
    }

    fn lookupNamedItemInModule(self: *ConstEvaluator, module_id: source.ModuleId, name: []const u8) !?ast.ItemId {
        const type_query = self.type_query orelse return null;
        return try type_query.lookupItem(module_id, name);
    }

    fn resolveImportAlias(self: *ConstEvaluator, alias: []const u8) !?source.ModuleId {
        const module_id = self.module_id orelse return null;
        const type_query = self.type_query orelse return null;
        return try type_query.resolveImportAlias(module_id, alias);
    }

    fn importedModuleForExpr(self: *ConstEvaluator, expr_id: ast.ExprId) !?source.ModuleId {
        return switch (self.file.expression(expr_id).*) {
            .Name => |name| try self.resolveImportAlias(name.name),
            .Field => |field| blk: {
                const base_module_id = (try self.importedModuleForExpr(field.base)) orelse break :blk null;
                const type_query = self.type_query orelse break :blk null;
                break :blk try type_query.resolveImportAlias(base_module_id, field.name);
            },
            .Group => |group| try self.importedModuleForExpr(group.expr),
            else => null,
        };
    }

    fn functionRuntimeSelfParameterIndex(self: *ConstEvaluator, function: ast.FunctionItem) ?usize {
        return model.functionRuntimeSelfParameterIndex(self.file, function);
    }

    fn functionRuntimeSelfParameterIndexInFile(self: *ConstEvaluator, file: *const ast.AstFile, function: ast.FunctionItem) ?usize {
        _ = self;
        return model.functionRuntimeSelfParameterIndex(file, function);
    }

    fn typeNameForTypeId(self: *ConstEvaluator, type_id: u32) ?[]const u8 {
        if (type_builtin.fixedBytesLenForTypeId(type_id)) |len| return type_builtin.fixedBytesName(len);
        if (type_builtin.lookupBuiltinByComptimeTypeId(type_id)) |spec| return spec.source_name;
        return if (self.itemIdForNamedTypeId(type_id)) |item_id| self.itemName(item_id) else null;
    }

    fn abiTypeNameForTypeId(self: *ConstEvaluator, type_id: u32) ?[]const u8 {
        if (type_builtin.fixedBytesLenForTypeId(type_id)) |len| return abi_type_names.fixedBytesAbiName(len);
        if (type_builtin.lookupBuiltinByComptimeTypeId(type_id)) |spec| return abi_type_names.builtinSpecAbiName(spec);
        return if (self.itemIdForNamedTypeId(type_id)) |item_id| self.itemName(item_id) else null;
    }

    fn typeByteSizeForTypeId(self: *ConstEvaluator, type_id: u32) ?u256 {
        if (type_builtin.fixedBytesLenForTypeId(type_id)) |len| return len;
        if (type_builtin.lookupBuiltinByComptimeTypeId(type_id)) |spec| {
            return if (spec.byte_width) |width| @as(u256, width) else null;
        }
        return if (self.itemIdForNamedTypeId(type_id)) |item_id|
            self.itemByteSize(item_id)
        else
            null;
    }

    fn itemByteSize(self: *ConstEvaluator, item_id: ast.ItemId) ?u256 {
        return switch (self.file.item(item_id).*) {
            .Struct => |struct_item| blk: {
                if (struct_item.is_generic) break :blk null;
                var total: u256 = 0;
                for (struct_item.fields) |field| {
                    total += self.typeExprByteSize(field.type_expr) orelse break :blk null;
                }
                break :blk total;
            },
            .Bitfield => |bitfield_item| if (bitfield_item.base_type) |base_type|
                self.typeExprByteSize(base_type)
            else
                null,
            .TypeAlias => |type_alias| self.typeExprByteSize(type_alias.target_type),
            else => null,
        };
    }

    fn arraySizeValue(self: *ConstEvaluator, size: ast.TypeArraySize) ?u256 {
        return switch (size) {
            .Integer => |literal| std.fmt.parseInt(u256, literal.text, 10) catch null,
            .Name => |name| blk: {
                const value = self.env.lookupValue(name.name) orelse break :blk null;
                break :blk switch (value) {
                    .integer => |integer| integer,
                    else => null,
                };
            },
        };
    }

    fn typeExprByteSize(self: *ConstEvaluator, type_expr_id: ast.TypeExprId) ?u256 {
        return switch (self.file.typeExpr(type_expr_id).*) {
            .Path => |path| if (self.pathTypeId(path.name)) |type_id|
                self.typeByteSizeForTypeId(type_id)
            else
                null,
            .Generic => |generic| blk: {
                if (refinements.isKnownName(generic.name)) {
                    if (generic.args.len > 0 and generic.args[0] == .Type) break :blk self.typeExprByteSize(generic.args[0].Type);
                }
                if (self.pathTypeId(generic.name)) |type_id| break :blk self.typeByteSizeForTypeId(type_id);
                break :blk null;
            },
            .Tuple => |tuple| blk: {
                var total: u256 = 0;
                for (tuple.elements) |element| {
                    total += self.typeExprByteSize(element) orelse break :blk null;
                }
                break :blk total;
            },
            .AnonymousStruct => |struct_type| blk: {
                var total: u256 = 0;
                for (struct_type.fields) |field| {
                    total += self.typeExprByteSize(field.type_expr) orelse break :blk null;
                }
                break :blk total;
            },
            .Array => |array| blk: {
                const element_size = self.typeExprByteSize(array.element) orelse break :blk null;
                const len = self.arraySizeValue(array.size) orelse break :blk null;
                break :blk element_size * len;
            },
            .Slice, .ErrorUnion, .Error => null,
        };
    }

    fn typeExprAbiName(self: *ConstEvaluator, type_expr_id: ast.TypeExprId) anyerror!?[]const u8 {
        return switch (self.file.typeExpr(type_expr_id).*) {
            .Path => |path| blk: {
                if (type_builtin.parseFixedBytesName(path.name)) |_| break :blk try self.allocator.dupe(u8, path.name);
                if (self.pathTypeId(path.name)) |type_id| break :blk self.abiTypeNameForTypeId(type_id);
                break :blk null;
            },
            .Generic => |generic| blk: {
                if (refinements.isKnownName(generic.name)) {
                    if (generic.args.len > 0 and generic.args[0] == .Type) break :blk self.typeExprAbiName(generic.args[0].Type);
                }
                var rendered_args: std.ArrayList([]const u8) = .empty;
                defer rendered_args.deinit(self.allocator);
                for (generic.args) |arg| switch (arg) {
                    .Type => |nested| {
                        const name = (try self.typeExprAbiName(nested)) orelse break :blk null;
                        try rendered_args.append(self.allocator, name);
                    },
                    .Integer => |integer| try rendered_args.append(self.allocator, integer.text),
                };
                const joined = try std.mem.join(self.allocator, ",", rendered_args.items);
                return if (rendered_args.items.len == 0)
                    try self.allocator.dupe(u8, generic.name)
                else
                    try std.fmt.allocPrint(self.allocator, "{s}<{s}>", .{ generic.name, joined });
            },
            .Tuple => |tuple| blk: {
                var rendered_elements: std.ArrayList([]const u8) = .empty;
                defer rendered_elements.deinit(self.allocator);
                for (tuple.elements) |element| {
                    const name = (try self.typeExprAbiName(element)) orelse break :blk null;
                    try rendered_elements.append(self.allocator, name);
                }
                const joined = try std.mem.join(self.allocator, ",", rendered_elements.items);
                break :blk try std.fmt.allocPrint(self.allocator, "({s})", .{joined});
            },
            .AnonymousStruct => |struct_type| blk: {
                var rendered_fields: std.ArrayList([]const u8) = .empty;
                defer rendered_fields.deinit(self.allocator);
                for (struct_type.fields) |field| {
                    const name = (try self.typeExprAbiName(field.type_expr)) orelse break :blk null;
                    try rendered_fields.append(self.allocator, name);
                }
                const joined = try std.mem.join(self.allocator, ",", rendered_fields.items);
                break :blk try std.fmt.allocPrint(self.allocator, "({s})", .{joined});
            },
            .Array => |array| blk: {
                const element = (try self.typeExprAbiName(array.element)) orelse break :blk null;
                const len = self.arraySizeValue(array.size) orelse break :blk null;
                break :blk try std.fmt.allocPrint(self.allocator, "{s}[{d}]", .{ element, len });
            },
            .Slice => |slice| blk: {
                const element = (try self.typeExprAbiName(slice.element)) orelse break :blk null;
                break :blk try std.fmt.allocPrint(self.allocator, "{s}[]", .{element});
            },
            .ErrorUnion, .Error => null,
        };
    }

    fn concreteTypeNameForCtValue(self: *ConstEvaluator, value: CtValue) ?[]const u8 {
        return switch (value) {
            .type_val => |type_id| self.typeNameForTypeId(type_id),
            .struct_ref => |heap_id| self.typeNameForTypeId(self.env.heap.getStruct(heap_id).type_id),
            else => null,
        };
    }

    fn resolveConcreteTraitMethodCall(self: *ConstEvaluator, field: ast.FieldExpr) !?CallableFunction {
        const typecheck = (try self.currentModuleTypeCheckResult()) orelse return null;
        const base_value = (self.evalExprCtValue(field.base) catch null) orelse self.typeExprCtValue(field.base);
        const target_name = if (base_value) |value|
            self.concreteTypeNameForCtValue(value)
        else
            typecheck.exprType(field.base).name();
        const concrete_name = target_name orelse return null;
        var matched_impl_item_id: ?ast.ItemId = null;
        var matched_method_item_id: ?ast.ItemId = null;
        var matched_function: ?ast.FunctionItem = null;

        for (typecheck.impl_interfaces) |impl_interface| {
            if (!std.mem.eql(u8, impl_interface.target_name, concrete_name)) continue;
            const trait_interface = typecheck.traitInterfaceByName(impl_interface.trait_name) orelse continue;
            const trait_method = trait_interface.methodByName(field.name) orelse continue;
            if (!trait_method.is_comptime) continue;

            const item_index = (try self.currentItemIndex()) orelse return null;
            const method_count = item_index.countImplMethods(self.file, impl_interface.impl_item_id, field.name);
            if (method_count == 0) continue;
            if (matched_impl_item_id != null or method_count > 1) return null;
            const method_item_id = item_index.lookupImplMethod(self.file, impl_interface.impl_item_id, field.name) orelse continue;
            const item = self.file.item(method_item_id).*;
            if (item != .Function) continue;
            matched_impl_item_id = impl_interface.impl_item_id;
            matched_method_item_id = method_item_id;
            matched_function = item.Function;
        }

        _ = matched_impl_item_id orelse return null;
        const method_item_id = matched_method_item_id orelse return null;
        const function = matched_function orelse return null;
        return .{
            .module_id = self.module_id orelse return null,
            .file = self.file,
            .item_id = method_item_id,
            .function = function,
            .synthetic_self_arg = if (self.functionRuntimeSelfParameterIndex(function) != null) field.base else null,
        };
    }

    fn ensureTypeExprTypeChecked(self: *ConstEvaluator, type_expr_id: ast.TypeExprId) !void {
        switch (self.file.typeExpr(type_expr_id).*) {
            .Path => |path| try self.ensureNamedItemTypeChecked(path.name),
            .Generic => |generic| {
                try self.ensureNamedItemTypeChecked(generic.name);
                for (generic.args) |arg| switch (arg) {
                    .Type => |nested| try self.ensureTypeExprTypeChecked(nested),
                    .Integer => {},
                };
            },
            .Tuple => |tuple| for (tuple.elements) |element| try self.ensureTypeExprTypeChecked(element),
            .AnonymousStruct => |struct_type| for (struct_type.fields) |field| try self.ensureTypeExprTypeChecked(field.type_expr),
            .Array => |array| try self.ensureTypeExprTypeChecked(array.element),
            .Slice => |slice| try self.ensureTypeExprTypeChecked(slice.element),
            .ErrorUnion => |error_union| {
                try self.ensureTypeExprTypeChecked(error_union.payload);
                for (error_union.errors) |error_type| try self.ensureTypeExprTypeChecked(error_type);
            },
            .Error => {},
        }
    }

    fn lookupCallableFunction(self: *ConstEvaluator, callee: ast.ExprId) !?CallableFunction {
        switch (self.file.expression(callee).*) {
            .Name => |name| {
                const function_item_id = self.lookupNamedItem(name.name) orelse return null;
                const item = self.file.item(function_item_id).*;
                if (item != .Function) return null;
                return .{
                    .module_id = self.module_id orelse return null,
                    .file = self.file,
                    .item_id = function_item_id,
                    .function = item.Function,
                    .contract_id = self.current_contract,
                };
            },
            .Field => |field| {
                if (try self.importedModuleForExpr(field.base)) |target_module_id| {
                    const target_file = try self.astFileForModule(target_module_id);
                    const function_item_id = (try self.lookupNamedItemInModule(target_module_id, field.name)) orelse return null;
                    const item = target_file.item(function_item_id).*;
                    if (item == .Function) {
                        return .{
                            .module_id = target_module_id,
                            .file = target_file,
                            .item_id = function_item_id,
                            .function = item.Function,
                            .contract_id = null,
                        };
                    }
                }
                return try self.resolveConcreteTraitMethodCall(field);
            },
            .Group => |group| return try self.lookupCallableFunction(group.expr),
            else => return null,
        }
    }

    fn functionReferenceFromItemType(
        self: *ConstEvaluator,
        name: []const u8,
        ty: Type,
        self_param_index: ?usize,
    ) !?AbiFunctionReference {
        if (ty.kind() != .function) return null;
        const params = ty.function.param_types;
        const filtered = if (self_param_index) |skip_index| blk: {
            if (skip_index >= params.len) break :blk params;
            var out = try self.allocator.alloc(Type, params.len - 1);
            var out_index: usize = 0;
            for (params, 0..) |param, index| {
                if (index == skip_index) continue;
                out[out_index] = param;
                out_index += 1;
            }
            break :blk out;
        } else params;
        return .{
            .name = ty.function.name orelse name,
            .param_types = filtered,
            .has_self = self_param_index != null,
        };
    }

    fn resolveAbiFunctionReference(self: *ConstEvaluator, expr_id: ast.ExprId) !?AbiFunctionReference {
        const typecheck = try self.currentModuleTypeCheckResult();
        return switch (self.file.expression(expr_id).*) {
            .Group => |group| try self.resolveAbiFunctionReference(group.expr),
            .Name => |name| blk: {
                const module_typecheck = typecheck orelse break :blk null;
                const item_id = self.lookupNamedItem(name.name) orelse break :blk null;
                const item = self.file.item(item_id).*;
                if (item != .Function) break :blk null;
                break :blk try self.functionReferenceFromItemType(
                    name.name,
                    module_typecheck.itemLocatedType(item_id).type,
                    self.functionRuntimeSelfParameterIndex(item.Function),
                );
            },
            .Field => |field| blk: {
                if (switch (self.file.expression(field.base).*) {
                    .Name => |name| name.name,
                    else => null,
                }) |base_name| {
                    if (try self.resolveImportAlias(base_name)) |target_module_id| {
                        const target_file = try self.astFileForModule(target_module_id);
                        const target_typecheck = if (self.type_query) |query|
                            try query.moduleTypeCheck(target_module_id)
                        else
                            break :blk null;
                        const item_id = (try self.lookupNamedItemInModule(target_module_id, field.name)) orelse break :blk null;
                        const item = target_file.item(item_id).*;
                        if (item != .Function) break :blk null;
                        break :blk try self.functionReferenceFromItemType(
                            field.name,
                            target_typecheck.itemLocatedType(item_id).type,
                            self.functionRuntimeSelfParameterIndex(item.Function),
                        );
                    }

                    if (self.lookupNamedItem(base_name)) |base_item_id| {
                        switch (self.file.item(base_item_id).*) {
                            .Trait => |trait_item| {
                                if (try self.currentItemIndex()) |item_index| {
                                    if (item_index.lookupTraitMethod(self.file, base_item_id, field.name)) |method| {
                                        const signature = (try self.signatureForTraitMethod(method)) orelse break :blk null;
                                        break :blk AbiFunctionReference{
                                            .name = method.name,
                                            .param_types = &.{},
                                            .has_self = method.receiver_kind != .none,
                                            .signature = signature,
                                        };
                                    }
                                }

                                const module_typecheck = typecheck orelse break :blk null;
                                if (module_typecheck.traitInterfaceByName(trait_item.name)) |trait_interface| {
                                    const method = trait_interface.methodByName(field.name) orelse break :blk null;
                                    break :blk AbiFunctionReference{
                                        .name = method.name,
                                        .param_types = method.param_types,
                                        .has_self = method.receiver_kind != .none,
                                    };
                                } else {
                                    break :blk null;
                                }
                            },
                            .Contract => {
                                const module_typecheck = typecheck orelse break :blk null;
                                const item_index = (try self.currentItemIndex()) orelse break :blk null;
                                const member_id = item_index.lookupContractMemberWithRoles(self.file, base_item_id, field.name, .{ .function = true }) orelse break :blk null;
                                const member = self.file.item(member_id).*;
                                break :blk try self.functionReferenceFromItemType(
                                    field.name,
                                    module_typecheck.itemLocatedType(member_id).type,
                                    self.functionRuntimeSelfParameterIndex(member.Function),
                                );
                            },
                            else => {},
                        }
                    }
                }

                const module_typecheck = typecheck orelse break :blk null;
                const ty = module_typecheck.exprType(expr_id);
                break :blk try self.functionReferenceFromItemType(field.name, ty, null);
            },
            else => null,
        };
    }

    fn signatureForAbiFunctionReference(self: *ConstEvaluator, function_ref: AbiFunctionReference) !?[]const u8 {
        const layout_ctx = (try self.currentLayoutContext()) orelse return null;
        return try layout_ctx.signatureForMethod(
            function_ref.name,
            function_ref.has_self,
            function_ref.param_types,
        );
    }

    /// Resolves `Name` or `Contract.Member` paths to an item id, recursing
    /// through `.Group` wrappers. Returns null for anything else (imports,
    /// traits, expressions). Used by the event/struct ABI reference resolvers.
    fn resolveContractMemberPath(self: *ConstEvaluator, expr_id: ast.ExprId) !?ast.ItemId {
        return switch (self.file.expression(expr_id).*) {
            .Group => |group| try self.resolveContractMemberPath(group.expr),
            .Name => |name| self.lookupNamedItem(name.name),
            .Field => |field| blk: {
                const base_item_id = (try self.resolveContractMemberPath(field.base)) orelse break :blk null;
                const item_index = (try self.currentItemIndex()) orelse break :blk null;
                break :blk item_index.lookupContractMemberWithRoles(self.file, base_item_id, field.name, .{
                    .function = true,
                    .struct_ = true,
                    .bitfield = true,
                    .enum_ = true,
                    .trait_ = true,
                    .field = true,
                    .constant = true,
                    .log_decl = true,
                    .error_decl = true,
                });
            },
            else => null,
        };
    }

    fn resolveAbiEventReference(self: *ConstEvaluator, expr_id: ast.ExprId) !?AbiEventReference {
        const item_id = (try self.resolveContractMemberPath(expr_id)) orelse return null;
        const item = self.file.item(item_id).*;
        if (item != .LogDecl) return null;
        const signature = (try self.signatureForLogDecl(item.LogDecl)) orelse return null;
        return .{ .signature = signature };
    }

    fn resolveAbiStructReference(self: *ConstEvaluator, expr_id: ast.ExprId) !?AbiStructReference {
        const item_id = (try self.resolveContractMemberPath(expr_id)) orelse return null;
        const item = self.file.item(item_id).*;
        if (item != .Struct) return null;
        return .{ .item = item.Struct };
    }

    fn resolveReflectionStructReference(self: *ConstEvaluator, expr_id: ast.ExprId) !?ast.StructItem {
        const item_id = (try self.resolveContractMemberPath(expr_id)) orelse return null;
        const item = self.file.item(item_id).*;
        return if (item == .Struct) item.Struct else null;
    }

    fn resolveReflectionTraitReference(self: *ConstEvaluator, expr_id: ast.ExprId) !?ReflectionTraitReference {
        if (try self.resolveContractMemberPath(expr_id)) |item_id| {
            const item = self.file.item(item_id).*;
            return if (item == .Trait) .{
                .module_id = self.module_id orelse return null,
                .file = self.file,
                .item = item.Trait,
            } else null;
        }

        const type_value = self.typeExprCtValue(expr_id) orelse return null;
        const type_id = switch (type_value) {
            .type_val => |value| value,
            else => return null,
        };
        const named_ref = self.namedTypeRefForTypeId(type_id) orelse return null;
        const module_id = named_ref.module_id orelse self.module_id orelse return null;
        const file = try self.astFileForModule(module_id);
        const item = file.item(named_ref.item_id).*;
        return if (item == .Trait) .{
            .module_id = module_id,
            .file = file,
            .item = item.Trait,
        } else null;
    }

    fn buildStructFieldsCtValue(self: *ConstEvaluator, struct_item: ast.StructItem) !CtValue {
        const elems = try self.allocator.alloc(CtValue, struct_item.fields.len);
        for (struct_item.fields, 0..) |field, index| {
            const type_id = self.typeExprTypeId(field.type_expr) orelse return error.NotComptime;
            elems[index] = try self.reflectionTuple(&.{
                CtValue{ .string_ref = try self.env.heap.allocString(field.name) },
                CtValue{ .type_val = type_id },
            });
        }
        return .{ .slice_ref = try self.env.heap.allocSliceOwned(elems) };
    }

    fn buildTraitMethodsCtValue(self: *ConstEvaluator, trait_ref: ReflectionTraitReference) !CtValue {
        const type_query = self.type_query orelse return error.NotComptime;
        const typecheck = try type_query.moduleTypeCheck(trait_ref.module_id);
        const item_index = try type_query.itemIndex(trait_ref.module_id);
        const layout_ctx = abi_layout_context.LayoutContext{
            .allocator = self.allocator,
            .provider = abi_layout_provider.abiLayoutProvider(trait_ref.file, item_index, typecheck),
        };
        const trait_interface = typecheck.traitInterfaceByName(trait_ref.item.name) orelse return error.NotComptime;
        const elems = try self.allocator.alloc(CtValue, trait_interface.methods.len);
        for (trait_interface.methods, 0..) |method, index| {
            const params = try self.allocator.alloc(CtValue, method.param_types.len);
            for (method.param_types, 0..) |param_type, param_index| {
                params[param_index] = .{ .type_val = self.typeIdForModelType(param_type) orelse return error.NotComptime };
            }

            const declared_errors = try self.allocator.alloc(CtValue, method.errors.len);
            for (method.errors, 0..) |err_name, err_index| {
                declared_errors[err_index] = .{ .string_ref = try self.env.heap.allocString(err_name) };
            }

            const signature = try layout_ctx.signatureForMethod(
                method.name,
                method.receiver_kind != .none,
                method.param_types,
            );
            defer self.allocator.free(signature);
            const selector = hir_abi.keccakSelectorValue(signature);

            // Keep this tuple order in sync with traitMethodReflectionType.
            elems[index] = try self.reflectionTuple(&.{
                CtValue{ .string_ref = try self.env.heap.allocString(method.name) },
                CtValue{ .slice_ref = try self.env.heap.allocSliceOwned(params) },
                CtValue{ .type_val = self.typeIdForModelType(method.return_type) orelse return error.NotComptime },
                CtValue{ .boolean = method.receiver_kind != .none },
                CtValue{ .string_ref = try self.env.heap.allocString(externCallKindName(method.extern_call_kind)) },
                CtValue{ .slice_ref = try self.env.heap.allocSliceOwned(declared_errors) },
                CtValue{ .bytes_ref = try self.env.heap.allocBytesOwned(try selectorFixedBytes(self.allocator, selector)) },
            });
        }
        return .{ .slice_ref = try self.env.heap.allocSliceOwned(elems) };
    }

    fn reflectionTuple(self: *ConstEvaluator, values: []const CtValue) !CtValue {
        const copied = try self.allocator.dupe(CtValue, values);
        return .{ .tuple_ref = try self.env.heap.allocTupleOwned(copied) };
    }

    fn anonymousStructFieldIndexForExpr(self: *ConstEvaluator, expr_id: ast.ExprId, field_name: []const u8) !?usize {
        const fields = (try self.anonymousStructFieldsForExpr(expr_id)) orelse return null;
        return model.anonymousStructFieldIndex(fields, field_name);
    }

    fn anonymousStructFieldsForExpr(self: *ConstEvaluator, expr_id: ast.ExprId) !?[]const AnonymousStructField {
        if (try self.currentTypeCheckResult()) |typecheck| {
            const ty = typecheck.exprType(expr_id);
            if (ty.kind() == .anonymous_struct) return ty.anonymous_struct.fields;
        }
        if (try self.currentModuleTypeCheckResult()) |typecheck| {
            const ty = typecheck.exprType(expr_id);
            if (ty.kind() == .anonymous_struct) return ty.anonymous_struct.fields;
        }
        return null;
    }

    fn typeIdForModelType(self: *ConstEvaluator, ty: Type) ?u32 {
        return switch (ty) {
            .integer => |integer| blk: {
                const spec = integer.builtinSpec() orelse break :blk null;
                break :blk spec.comptime_type_id;
            },
            .bool => type_builtin.lookupBuiltinById(.bool).comptime_type_id,
            .address => type_builtin.lookupBuiltinById(.address).comptime_type_id,
            .fixed_bytes => |fixed_bytes| type_builtin.fixedBytesTypeId(fixed_bytes.len),
            .string => type_builtin.lookupBuiltinById(.string).comptime_type_id,
            .bytes => type_builtin.lookupBuiltinById(.bytes).comptime_type_id,
            .void => type_builtin.lookupBuiltinById(.void).comptime_type_id,
            .struct_ => |named| self.pathTypeId(named.name),
            .contract => |named| self.pathTypeId(named.name),
            .bitfield => |named| self.pathTypeId(named.name),
            .enum_ => |named| self.pathTypeId(named.name),
            .named => |named| self.pathTypeId(named.name),
            .refinement => |refinement| self.typeIdForModelType(refinement.base_type.*),
            else => null,
        };
    }

    fn externCallKindName(kind: ast.ExternCallKind) []const u8 {
        // TODO: replace this string representation once synthetic reflection
        // records can carry real enum values through CtValue/ConstValue.
        return switch (kind) {
            .none => "none",
            .call => "call",
            .staticcall => "staticcall",
        };
    }

    fn itemName(self: *ConstEvaluator, item_id: ast.ItemId) ?[]const u8 {
        return switch (self.file.item(item_id).*) {
            .Contract => |contract| contract.name,
            .Function => |function| function.name,
            .Struct => |struct_item| struct_item.name,
            .Bitfield => |bitfield_item| bitfield_item.name,
            .Enum => |enum_item| enum_item.name,
            .Trait => |trait_item| trait_item.name,
            .Field => |field| field.name,
            .Constant => |constant| constant.name,
            .LogDecl => |log_decl| log_decl.name,
            .ErrorDecl => |error_decl| error_decl.name,
            else => null,
        };
    }

    fn patternMatches(self: *ConstEvaluator, condition: ConstValue, pattern: ast.SwitchPattern) bool {
        return switch (pattern) {
            .Expr => |expr_id| if (self.evalExpr(expr_id) catch null) |value| constEquals(condition, value) else false,
            .NamedError => false,
            .Or => |or_pattern| blk: {
                for (or_pattern.alternatives) |alternative| {
                    if (self.patternMatches(condition, alternative)) break :blk true;
                }
                break :blk false;
            },
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
            .Ok, .Err => false,
            .Else => true,
        };
    }

    fn patternMatchesCt(self: *ConstEvaluator, condition: CtValue, pattern: ast.SwitchPattern) !bool {
        if (pattern == .Or) {
            for (pattern.Or.alternatives) |alternative| {
                if (try self.patternMatchesCt(condition, alternative)) return true;
            }
            return false;
        }
        switch (condition) {
            .adt_val => |enum_value| {
                if (self.sumVariantRefFromPattern(pattern)) |variant_ref| {
                    return enum_value.type_id == self.namedTypeId(variant_ref.item_id) and
                        enum_value.variant_id == variant_ref.variant_id;
                }
                return pattern == .Else;
            },
            .error_union_val => |error_union| {
                return switch (pattern) {
                    .Ok => !error_union.is_error,
                    .Err => error_union.is_error,
                    .NamedError, .Expr => blk: {
                        if (!error_union.is_error) break :blk false;
                        const payload = self.env.heap.getTuple(error_union.payload);
                        if (payload.elems.len == 0) break :blk false;
                        const error_value = switch (payload.elems[0]) {
                            .adt_val => |value| value,
                            else => break :blk false,
                        };
                        const variant_ref = self.sumVariantRefFromPattern(pattern) orelse break :blk false;
                        break :blk error_value.type_id == self.namedTypeId(variant_ref.item_id) and
                            error_value.variant_id == variant_ref.variant_id;
                    },
                    .Else => true,
                    else => false,
                };
            },
            else => {
                const const_value = (try ctValueToConstValue(self.allocator, &self.env.heap, condition)) orelse return false;
                return self.patternMatches(const_value, pattern);
            },
        }
    }

    fn bindSwitchPatternCtValue(self: *ConstEvaluator, condition: CtValue, pattern: ast.SwitchPattern) !void {
        switch (pattern) {
            .Or => |or_pattern| {
                for (or_pattern.alternatives) |alternative| {
                    if (try self.patternMatchesCt(condition, alternative)) {
                        try self.bindSwitchPatternCtValue(condition, alternative);
                        return;
                    }
                }
                return;
            },
            .Ok => |pattern_id| {
                const result = switch (condition) {
                    .error_union_val => |value| value,
                    else => return,
                };
                if (result.is_error) return;
                const payload = self.env.heap.getTuple(result.payload);
                if (payload.elems.len == 0) return;
                try self.bindPatternCtValue(pattern_id, payload.elems[0]);
                return;
            },
            .Err => |pattern_id| {
                const result = switch (condition) {
                    .error_union_val => |value| value,
                    else => return,
                };
                if (!result.is_error) return;
                const payload = self.env.heap.getTuple(result.payload);
                if (payload.elems.len == 0) return;
                const error_value = payload.elems[0];
                switch (error_value) {
                    .adt_val => |enum_value| if (enum_value.payload) |payload_id| {
                        const error_payload = self.env.heap.getTuple(payload_id);
                        if (error_payload.elems.len == 1) {
                            try self.bindPatternCtValue(pattern_id, error_payload.elems[0]);
                            return;
                        }
                    },
                    else => {},
                }
                try self.bindPatternCtValue(pattern_id, error_value);
                return;
            },
            else => {},
        }

        const named = switch (pattern) {
            .NamedError => |named| named,
            .Expr => |expr_id| {
                const call = switch (self.file.expression(expr_id).*) {
                    .Call => |call| call,
                    else => return,
                };
                const enum_value = switch (condition) {
                    .adt_val => |enum_value| enum_value,
                    .error_union_val => |result| result_blk: {
                        if (!result.is_error) return;
                        const result_payload = self.env.heap.getTuple(result.payload);
                        if (result_payload.elems.len == 0) return;
                        break :result_blk switch (result_payload.elems[0]) {
                            .adt_val => |enum_value| enum_value,
                            else => return,
                        };
                    },
                    else => return,
                };
                const payload_id = enum_value.payload orelse return;
                const payload = self.env.heap.getTuple(payload_id);
                for (call.args, 0..) |binding_expr, index| {
                    if (index >= payload.elems.len) return;
                    switch (self.file.expression(binding_expr).*) {
                        .Name => |name| try self.bindNameCtValue(name.name, payload.elems[index]),
                        else => return,
                    }
                }
                return;
            },
            else => return,
        };
        const enum_value = switch (condition) {
            .adt_val => |enum_value| enum_value,
            .error_union_val => |result| blk: {
                if (!result.is_error) return;
                const result_payload = self.env.heap.getTuple(result.payload);
                if (result_payload.elems.len == 0) return;
                break :blk switch (result_payload.elems[0]) {
                    .adt_val => |enum_value| enum_value,
                    else => return,
                };
            },
            else => return,
        };
        const payload_id = enum_value.payload orelse return;
        const payload = self.env.heap.getTuple(payload_id);
        if (named.bindings.len == 1 and try self.bindEnumNamedPayloadDestructureCtValue(enum_value, named.bindings[0], payload)) {
            return;
        }
        for (named.bindings, 0..) |binding, index| {
            if (index >= payload.elems.len) return;
            try self.bindPatternCtValue(binding, payload.elems[index]);
        }
    }

    fn bindEnumNamedPayloadDestructureCtValue(self: *ConstEvaluator, enum_value: CtEnum, pattern_id: ast.PatternId, payload: CtAggregate.TupleData) !bool {
        const destructure = switch (self.file.pattern(pattern_id).*) {
            .StructDestructure => |destructure| destructure,
            else => return false,
        };
        const item_id = self.itemIdForNamedTypeId(enum_value.type_id) orelse return false;
        const item = self.file.item(item_id).*;
        if (item != .Enum or enum_value.variant_id >= item.Enum.variants.len) return false;
        const fields = (try self.enumNamedPayloadFieldsFromAst(item.Enum.variants[@intCast(enum_value.variant_id)].payload)) orelse return false;

        for (destructure.fields) |field| {
            const index = model.anonymousStructFieldIndex(fields, field.name) orelse return false;
            if (index >= payload.elems.len) return false;
            try self.bindPatternCtValue(field.binding, payload.elems[index]);
        }
        return true;
    }

    fn evalAbiEncodeBuiltin(self: *ConstEvaluator, builtin: ast.BuiltinExpr) anyerror!?ConstValue {
        if (builtin.args.len != 1) return null;
        const arg_id = builtin.args[0];
        if (self.isEmptyTupleLiteral(arg_id)) {
            return .{ .fixed_bytes = try self.allocator.alloc(u8, 0) };
        }
        if (try self.evalAbiEncodeVoidCallArgument(arg_id)) {
            return .{ .fixed_bytes = try self.allocator.alloc(u8, 0) };
        }
        const typecheck = (try self.currentTypeCheckResult()) orelse (try self.currentModuleTypeCheckResult()) orelse return null;
        const arg_type = typecheck.exprType(arg_id);
        const layout_context = (try self.currentLayoutContext()) orelse return null;
        var layout = layout_context.layoutForType(arg_type) catch return null;
        defer layout.deinit(self.allocator);
        if (arg_type.kind() == .void or layout.staticWordCount() == 0) {
            return .{ .fixed_bytes = try self.allocator.alloc(u8, 0) };
        }

        const value: abi_comptime_encoder.ComptimeAbiValue = if (try self.evalExprCtValue(arg_id)) |ct_value|
            .{ .ct = ct_value }
        else if (try self.evalExpr(arg_id)) |const_value|
            .{ .constant = const_value }
        else
            return null;

        return .{ .fixed_bytes = try abi_comptime_encoder.encodeComptimeValue(self.allocator, &self.env.heap, layout, value) };
    }

    fn evalAbiDecodeBuiltinCtValue(self: *ConstEvaluator, builtin: ast.BuiltinExpr, comptime use_cache: bool) anyerror!?CtValue {
        const expected_value_args: usize = if (builtin.type_arg != null) 1 else 2;
        if (builtin.args.len != expected_value_args) return null;
        const typecheck = (try self.currentTypeCheckResult()) orelse (try self.currentModuleTypeCheckResult()) orelse return null;
        // Source-level @abiDecode(T, bytes) always arrives with type_arg set.
        // The expression path is retained for defensive direct-AST callers.
        const target_type = if (builtin.type_arg) |type_arg|
            try self.abiDecodeTypeFromTypeArg(type_arg)
        else
            typecheck.exprType(builtin.args[0]);
        const bytes_arg = if (builtin.type_arg != null) builtin.args[0] else builtin.args[1];
        const bytes_value = (try self.evalExprCtValueImpl(bytes_arg, use_cache, true)) orelse return null;
        const bytes = switch (bytes_value) {
            .bytes_ref => |heap_id| self.env.heap.getBytes(heap_id),
            else => return null,
        };

        const layout_context = (try self.currentLayoutContext()) orelse return null;
        var layout = layout_context.layoutForType(target_type) catch return error.AbiDecoderInternalShapeMismatch;
        defer layout.deinit(self.allocator);

        const decoded = if (std.mem.eql(u8, builtin.name, "abiDecodePermissive"))
            try abi_comptime_decoder.decodeComptimeValuePermissive(
                self.allocator,
                &self.env.heap,
                self.abiDecodeTypeResolver(),
                layout,
                target_type,
                bytes,
            )
        else
            try abi_comptime_decoder.decodeComptimeValue(
                self.allocator,
                &self.env.heap,
                self.abiDecodeTypeResolver(),
                layout,
                target_type,
                bytes,
            );
        return switch (decoded) {
            .ok => |value| try self.abiDecodeOk(value),
            .err => |err| try self.abiDecodeErr(err),
        };
    }

    fn abiDecodeTypeFromTypeArg(self: *ConstEvaluator, type_arg: ast.TypeExprId) !Type {
        const item_index = (try self.currentItemIndex()) orelse return error.AbiDecoderInternalShapeMismatch;
        const raw = try type_descriptors.descriptorFromTypeExpr(self.allocator, self.file, item_index, type_arg);
        return try self.resolveAbiDecodeTypeAliases(raw);
    }

    fn resolveAbiDecodeTypeAliases(self: *ConstEvaluator, ty: Type) !Type {
        return switch (ty) {
            .named => |named| blk: {
                const item_index = (try self.currentItemIndex()) orelse return error.AbiDecoderInternalShapeMismatch;
                const item_id = item_index.lookup(named.name) orelse break :blk ty;
                const item = self.file.item(item_id).*;
                if (item != .TypeAlias) break :blk ty;
                const target = try type_descriptors.descriptorFromTypeExpr(self.allocator, self.file, item_index, item.TypeAlias.target_type);
                break :blk try self.resolveAbiDecodeTypeAliases(target);
            },
            .refinement => |refinement| blk: {
                var copy = refinement;
                copy.base_type = try self.storeAbiDecodeType(try self.resolveAbiDecodeTypeAliases(refinement.base_type.*));
                break :blk Type{ .refinement = copy };
            },
            .array => |array| .{ .array = .{
                .element_type = try self.storeAbiDecodeType(try self.resolveAbiDecodeTypeAliases(array.element_type.*)),
                .len = array.len,
            } },
            .slice => |slice| .{ .slice = .{
                .element_type = try self.storeAbiDecodeType(try self.resolveAbiDecodeTypeAliases(slice.element_type.*)),
            } },
            .tuple => |elements| blk: {
                const resolved = try self.allocator.alloc(Type, elements.len);
                for (elements, 0..) |element, index| {
                    resolved[index] = try self.resolveAbiDecodeTypeAliases(element);
                }
                break :blk Type{ .tuple = resolved };
            },
            .anonymous_struct => |struct_type| blk: {
                const fields = try self.allocator.alloc(AnonymousStructField, struct_type.fields.len);
                for (struct_type.fields, 0..) |field, index| {
                    fields[index] = .{
                        .name = field.name,
                        .ty = try self.resolveAbiDecodeTypeAliases(field.ty),
                    };
                }
                break :blk Type{ .anonymous_struct = .{ .fields = fields } };
            },
            .error_union => |error_union| blk: {
                const errors = try self.allocator.alloc(Type, error_union.error_types.len);
                for (error_union.error_types, 0..) |err, index| {
                    errors[index] = try self.resolveAbiDecodeTypeAliases(err);
                }
                break :blk Type{ .error_union = .{
                    .payload_type = try self.storeAbiDecodeType(try self.resolveAbiDecodeTypeAliases(error_union.payload_type.*)),
                    .error_types = errors,
                } };
            },
            else => ty,
        };
    }

    fn storeAbiDecodeType(self: *ConstEvaluator, ty: Type) !*Type {
        const ptr = try self.allocator.create(Type);
        ptr.* = ty;
        return ptr;
    }

    fn abiDecodeOk(self: *ConstEvaluator, value: CtValue) !CtValue {
        const payload_id = try self.env.heap.allocTupleOwned(try self.allocator.dupe(CtValue, &.{value}));
        return .{ .error_union_val = .{
            .is_error = false,
            .payload = payload_id,
        } };
    }

    fn abiDecodeErr(self: *ConstEvaluator, err: abi_comptime_decoder.DecodeError) !CtValue {
        const err_value = CtValue{ .adt_val = .{
            .type_id = type_builtin.abi_decode_error_type_id,
            .variant_id = @intFromEnum(err),
            .payload = null,
        } };
        const payload_id = try self.env.heap.allocTupleOwned(try self.allocator.dupe(CtValue, &.{err_value}));
        return .{ .error_union_val = .{
            .is_error = true,
            .payload = payload_id,
        } };
    }

    fn abiDecodeTypeResolver(self: *ConstEvaluator) abi_comptime_decoder.TypeResolver {
        return .{
            .context = self,
            .typeIdForType = abiDecodeTypeIdForType,
            .structFields = abiDecodeStructFields,
            .enumVariantCount = abiDecodeEnumVariantCount,
        };
    }

    fn abiDecodeTypeIdForType(context: *anyopaque, ty: Type) anyerror!?u32 {
        const self: *ConstEvaluator = @ptrCast(@alignCast(context));
        return self.typeIdForModelType(ty);
    }

    fn abiDecodeStructFields(context: *anyopaque, name: []const u8) anyerror!?[]const AnonymousStructField {
        const self: *ConstEvaluator = @ptrCast(@alignCast(context));
        const typecheck = (try self.currentModuleTypeCheckResult()) orelse return null;
        if (typecheck.instantiatedStructByName(name)) |instantiated| {
            const fields = try self.allocator.alloc(AnonymousStructField, instantiated.fields.len);
            for (instantiated.fields, 0..) |field, index| {
                fields[index] = .{ .name = field.name, .ty = field.ty };
            }
            return fields;
        }
        const item_id = self.lookupNamedItem(name) orelse return null;
        const struct_item = switch (self.file.item(item_id).*) {
            .Struct => |struct_item| struct_item,
            else => return null,
        };
        const item_index = (try self.currentItemIndex()) orelse return null;
        const fields = try self.allocator.alloc(AnonymousStructField, struct_item.fields.len);
        for (struct_item.fields, 0..) |field, index| {
            fields[index] = .{
                .name = field.name,
                .ty = try type_descriptors.descriptorFromTypeExpr(self.allocator, self.file, item_index, field.type_expr),
            };
        }
        return fields;
    }

    fn abiDecodeEnumVariantCount(context: *anyopaque, name: []const u8) anyerror!?usize {
        const self: *ConstEvaluator = @ptrCast(@alignCast(context));
        const typecheck = (try self.currentModuleTypeCheckResult()) orelse return null;
        if (typecheck.instantiatedEnumByName(name)) |instantiated| return instantiated.variants.len;
        const item_id = self.lookupNamedItem(name) orelse return null;
        return switch (self.file.item(item_id).*) {
            .Enum => |enum_item| enum_item.variants.len,
            else => null,
        };
    }

    fn evalAbiEncodeVoidCallArgument(self: *ConstEvaluator, expr_id: ast.ExprId) anyerror!bool {
        return switch (self.file.expression(expr_id).*) {
            .Group => |group| try self.evalAbiEncodeVoidCallArgument(group.expr),
            .Call => |call| blk: {
                const callable = (try self.lookupCallableFunction(call.callee)) orelse break :blk false;
                const function = callable.function;
                if (function.return_type != null) break :blk false;

                const arg_values = (try self.materializeCallArgumentCtValues(callable, call, false)) orelse break :blk true;
                const previous_file = self.file;
                const previous_module_id = self.module_id;
                const previous_key = self.current_typecheck_key;
                const previous_contract = self.current_contract;
                self.file = callable.file;
                self.module_id = callable.module_id;
                self.current_typecheck_key = .{ .item = callable.item_id };
                self.current_contract = callable.contract_id;
                defer {
                    self.file = previous_file;
                    self.module_id = previous_module_id;
                    self.current_typecheck_key = previous_key;
                    self.current_contract = previous_contract;
                }

                try self.ensureTypeChecked(.{ .item = callable.item_id });
                if (!(try self.callableFunctionIsPure(callable.item_id))) break :blk true;
                if (self.functionStage(function) == .runtime_only) {
                    self.recordCtError(error_mod.CtError.stageViolation(
                        self.sourceSpan(call.range),
                        function.name,
                    ));
                    break :blk true;
                }
                if (self.call_depth >= self.env.config.max_recursion_depth) {
                    self.recordCtError(error_mod.CtError.init(
                        .recursion_limit,
                        self.sourceSpan(call.range),
                        "comptime recursion depth exceeded",
                    ));
                    break :blk true;
                }

                self.env.pushScope(false) catch break :blk true;
                defer self.env.popScope();
                try self.bindCallArguments(function, arg_values);

                self.call_depth += 1;
                defer self.call_depth -= 1;

                for (function.clauses) |clause| {
                    if (clause.kind != .requires and clause.kind != .guard) continue;
                    const condition = (try self.evalExprUncached(clause.expr)) orelse break :blk true;
                    const truthy = self.constConditionTruthy(condition) orelse break :blk true;
                    if (!truthy) break :blk true;
                }

                _ = try self.evalComptimeBodyControlCtValue(function.body, false);
                break :blk true;
            },
            else => false,
        };
    }

    fn isEmptyTupleLiteral(self: *ConstEvaluator, expr_id: ast.ExprId) bool {
        return switch (self.file.expression(expr_id).*) {
            .Tuple => |tuple| tuple.elements.len == 0,
            .Group => |group| self.isEmptyTupleLiteral(group.expr),
            else => false,
        };
    }

    fn evalBuiltin(self: *ConstEvaluator, builtin: ast.BuiltinExpr) anyerror!?ConstValue {
        if (std.mem.eql(u8, builtin.name, "chainId")) {
            if (builtin.args.len != 0) return null;
            return .{ .integer = try std.math.big.int.Managed.initSet(self.allocator, self.chain_id) };
        }

        if (std.mem.eql(u8, builtin.name, "compileError")) {
            if (builtin.args.len != 1) {
                self.recordInternalBuiltinError(builtin.range, "@compileError reached evaluator with invalid arity");
                return null;
            }
            const message: []const u8 = if (try self.evalExprCtValue(builtin.args[0])) |value|
                switch (value) {
                    .string_ref => |heap_id| self.env.heap.getString(heap_id),
                    else => {
                        self.recordInternalBuiltinError(builtin.range, "@compileError reached evaluator with non-string argument");
                        return null;
                    },
                }
            else if (try self.evalExpr(builtin.args[0])) |value|
                switch (value) {
                    .string => |string| string,
                    else => {
                        self.recordInternalBuiltinError(builtin.range, "@compileError reached evaluator with non-string argument");
                        return null;
                    },
                }
            else {
                self.recordInternalBuiltinError(builtin.range, "@compileError message was not compile-time known");
                return null;
            };

            self.recordCtError(error_mod.CtError.init(
                .compile_error,
                self.sourceSpan(builtin.range),
                message,
            ));
            return null;
        }

        if (std.mem.eql(u8, builtin.name, "selector") or std.mem.eql(u8, builtin.name, "abiSignature")) {
            if (builtin.args.len != 1) return null;
            const function_ref = (try self.resolveAbiFunctionReference(builtin.args[0])) orelse return null;
            const signature = function_ref.signature orelse (try self.signatureForAbiFunctionReference(function_ref)) orelse return null;
            if (std.mem.eql(u8, builtin.name, "abiSignature")) return .{ .string = signature };

            const selector = hir_abi.keccakSelectorValue(signature);
            return .{ .fixed_bytes = try selectorFixedBytes(self.allocator, selector) };
        }

        if (std.mem.eql(u8, builtin.name, "eventTopic")) {
            if (builtin.args.len != 1) return null;
            const event_ref = (try self.resolveAbiEventReference(builtin.args[0])) orelse return null;
            return .{ .fixed_bytes = try keccakFixedBytes(self.allocator, event_ref.signature) };
        }

        if (std.mem.eql(u8, builtin.name, "abiEncode")) {
            return try self.evalAbiEncodeBuiltin(builtin);
        }

        if (std.mem.eql(u8, builtin.name, "cast")) {
            if (builtin.args.len == 0) return null;
            if (builtin.type_arg) |type_arg| {
                const target = self.valueConstructionTarget(type_arg);
                if (target != .none) {
                    if (try self.evalExprCtValueAs(builtin.args[0], target)) |ct_value| {
                        return try ctValueToConstValue(self.allocator, &self.env.heap, ct_value);
                    }
                }
            }
            return try self.evalExpr(builtin.args[0]);
        }

        if (std.mem.eql(u8, builtin.name, "sizeOf")) {
            const type_arg = builtin.type_arg orelse return null;
            const size = self.typeExprByteSize(type_arg) orelse return null;
            return .{ .integer = try std.math.big.int.Managed.initSet(self.allocator, size) };
        }

        if (std.mem.eql(u8, builtin.name, "typeName")) {
            const type_arg = builtin.type_arg orelse return null;
            const name = (try self.typeExprAbiName(type_arg)) orelse return null;
            return .{ .string = name };
        }

        if (std.mem.eql(u8, builtin.name, "keccak256")) {
            if (builtin.args.len == 0) return null;
            const bytes: []const u8 = if (try self.evalExprCtValue(builtin.args[0])) |value|
                switch (value) {
                    .string_ref => |heap_id| self.env.heap.getString(heap_id),
                    .bytes_ref => |heap_id| self.env.heap.getBytes(heap_id),
                    else => return null,
                }
            else if (try self.evalExpr(builtin.args[0])) |value|
                switch (value) {
                    .string => |string| string,
                    else => return null,
                }
            else
                return null;
            const hash = hir_abi.keccak256(bytes);
            const value = std.mem.readInt(u256, &hash, .big);
            return .{ .integer = try std.math.big.int.Managed.initSet(self.allocator, value) };
        }

        if (builtin.args.len >= 2 and (std.mem.eql(u8, builtin.name, "divTrunc") or
            std.mem.eql(u8, builtin.name, "divFloor") or
            std.mem.eql(u8, builtin.name, "divCeil") or
            std.mem.eql(u8, builtin.name, "divExact") or
            std.mem.eql(u8, builtin.name, "divmod")))
        {
            const lhs = try self.evalExpr(builtin.args[0]);
            const rhs = try self.evalExpr(builtin.args[1]);
            if (lhs == null or rhs == null) return null;
            return switch (lhs.?) {
                .integer => |a| switch (rhs.?) {
                    .integer => |b| blk: {
                        if (b.eqlZero()) {
                            self.recordRequiredBinaryError(
                                .division_by_zero,
                                builtin.range,
                                "comptime division by zero",
                                "division builtin divisor must be nonzero in required comptime evaluation",
                            );
                            break :blk null;
                        }
                        var quotient = try std.math.big.int.Managed.init(self.allocator);
                        var remainder = try std.math.big.int.Managed.init(self.allocator);
                        if (std.mem.eql(u8, builtin.name, "divFloor")) {
                            try std.math.big.int.Managed.divFloor(&quotient, &remainder, &a, &b);
                        } else {
                            try std.math.big.int.Managed.divTrunc(&quotient, &remainder, &a, &b);
                            if (std.mem.eql(u8, builtin.name, "divCeil") and !remainder.eqlZero()) {
                                const signs_differ = a.toConst().positive != b.toConst().positive;
                                if (!signs_differ) {
                                    try quotient.addScalar(&quotient, 1);
                                }
                            } else if (std.mem.eql(u8, builtin.name, "divExact") and !remainder.eqlZero()) {
                                self.recordRequiredBinaryError(
                                    .invalid_cast,
                                    builtin.range,
                                    "comptime exact division has nonzero remainder",
                                    "@divExact requires an exactly divisible numerator and denominator in required comptime evaluation",
                                );
                                break :blk null;
                            }
                        }
                        if (std.mem.eql(u8, builtin.name, "divmod")) {
                            const elems = try self.allocator.alloc(ConstValue, 2);
                            elems[0] = .{ .integer = quotient };
                            elems[1] = .{ .integer = remainder };
                            break :blk .{ .tuple = elems };
                        }
                        break :blk .{ .integer = quotient };
                    },
                    else => null,
                },
                else => null,
            };
        }

        if (std.mem.eql(u8, builtin.name, "truncate")) {
            if (builtin.args.len == 0) return null;
            return try self.evalExpr(builtin.args[0]);
        }

        for (builtin.args) |arg| _ = try self.evalExpr(arg);
        return null;
    }

    fn materializeCallArgumentCtValues(
        self: *ConstEvaluator,
        callable: CallableFunction,
        call: ast.CallExpr,
        comptime use_cache: bool,
    ) anyerror!?[]CtValue {
        const function = callable.function;
        const self_param_index = self.functionRuntimeSelfParameterIndexInFile(callable.file, function);
        const expected_args = function.parameters.len - @intFromBool(callable.synthetic_self_arg != null and self_param_index != null);
        if (expected_args != call.args.len) return null;

        var arg_values = try self.allocator.alloc(CtValue, function.parameters.len);
        var user_arg_index: usize = 0;
        for (function.parameters, 0..) |parameter, idx| {
            const arg_expr = if (callable.synthetic_self_arg != null and self_param_index != null and idx == self_param_index.?)
                callable.synthetic_self_arg.?
            else blk: {
                const arg = call.args[user_arg_index];
                user_arg_index += 1;
                break :blk arg;
            };
            arg_values[idx] = (try self.evalCallArgumentCtValue(callable.file, parameter, arg_expr, use_cache)) orelse return null;
        }
        return arg_values;
    }

    fn bindCallArguments(self: *ConstEvaluator, function: ast.FunctionItem, arg_values: []const CtValue) !void {
        for (function.parameters, 0..) |parameter, idx| {
            try self.bindPatternCtValue(parameter.pattern, arg_values[idx]);
        }
    }

    fn evalCall(self: *ConstEvaluator, call: ast.CallExpr, comptime use_cache: bool) anyerror!?ConstValue {
        const callable = (try self.lookupCallableFunction(call.callee)) orelse {
            _ = try self.evalExprImpl(call.callee, use_cache);
            for (call.args) |arg| _ = try self.evalExprImpl(arg, use_cache);
            return null;
        };
        const function = callable.function;
        const arg_values = (try self.materializeCallArgumentCtValues(callable, call, use_cache)) orelse return null;
        const previous_file = self.file;
        const previous_module_id = self.module_id;
        const previous_key = self.current_typecheck_key;
        const previous_contract = self.current_contract;
        self.file = callable.file;
        self.module_id = callable.module_id;
        self.current_typecheck_key = .{ .item = callable.item_id };
        self.current_contract = callable.contract_id;
        defer {
            self.file = previous_file;
            self.module_id = previous_module_id;
            self.current_typecheck_key = previous_key;
            self.current_contract = previous_contract;
        }
        try self.ensureTypeChecked(.{ .item = callable.item_id });
        if (!(try self.callableFunctionIsPure(callable.item_id))) {
            return null;
        }

        if (self.functionStage(function) == .runtime_only) {
            self.recordCtError(error_mod.CtError.stageViolation(
                self.sourceSpan(call.range),
                function.name,
            ));
            return null;
        }

        if (self.call_depth >= self.env.config.max_recursion_depth) {
            self.recordCtError(error_mod.CtError.init(
                .recursion_limit,
                self.sourceSpan(call.range),
                "comptime recursion depth exceeded",
            ));
            return null;
        }

        self.env.pushScope(false) catch return null;
        defer self.env.popScope();
        try self.bindCallArguments(function, arg_values);

        self.call_depth += 1;
        defer self.call_depth -= 1;

        for (function.clauses) |clause| {
            if (clause.kind != .requires and clause.kind != .guard) continue;
            const condition = (try self.evalExprUncached(clause.expr)) orelse return null;
            const truthy = self.constConditionTruthy(condition) orelse return null;
            if (!truthy) return null;
        }

        const value = try self.evalComptimeBody(function.body);
        if (value == null) {
            self.recordMissingComptimeValue(
                self.sourceSpan(call.range),
                "comptime call did not produce a value",
                function.name,
            );
        }
        return value;
    }

    fn evalCallCtValue(self: *ConstEvaluator, call: ast.CallExpr, comptime use_cache: bool) anyerror!?CtValue {
        const callable = (try self.lookupCallableFunction(call.callee)) orelse {
            _ = try self.evalExprImpl(call.callee, use_cache);
            for (call.args) |arg| _ = try self.evalExprImpl(arg, use_cache);
            return null;
        };
        const function = callable.function;
        const arg_values = (try self.materializeCallArgumentCtValues(callable, call, use_cache)) orelse return null;
        const previous_file = self.file;
        const previous_module_id = self.module_id;
        const previous_key = self.current_typecheck_key;
        const previous_contract = self.current_contract;
        self.file = callable.file;
        self.module_id = callable.module_id;
        self.current_typecheck_key = .{ .item = callable.item_id };
        self.current_contract = callable.contract_id;
        defer {
            self.file = previous_file;
            self.module_id = previous_module_id;
            self.current_typecheck_key = previous_key;
            self.current_contract = previous_contract;
        }
        try self.ensureTypeChecked(.{ .item = callable.item_id });
        if (!(try self.callableFunctionIsPure(callable.item_id))) {
            return null;
        }

        if (self.functionStage(function) == .runtime_only) {
            self.recordCtError(error_mod.CtError.stageViolation(
                self.sourceSpan(call.range),
                function.name,
            ));
            return null;
        }

        if (self.call_depth >= self.env.config.max_recursion_depth) {
            self.recordCtError(error_mod.CtError.init(
                .recursion_limit,
                self.sourceSpan(call.range),
                "comptime recursion depth exceeded",
            ));
            return null;
        }

        self.env.pushScope(false) catch return null;
        defer self.env.popScope();
        try self.bindCallArguments(function, arg_values);

        self.call_depth += 1;
        defer self.call_depth -= 1;

        for (function.clauses) |clause| {
            if (clause.kind != .requires and clause.kind != .guard) continue;
            const condition = (try self.evalExprUncached(clause.expr)) orelse return null;
            const truthy = self.constConditionTruthy(condition) orelse return null;
            if (!truthy) return null;
        }

        // Do not persist callee-body expression values into the module-global
        // const-eval cache. Those expression ids belong to the generic function
        // body and can otherwise be polluted by one concrete call context.
        const value = try self.evalComptimeBodyCtValue(function.body, false);
        if (value == null) {
            self.recordMissingComptimeValue(
                self.sourceSpan(call.range),
                "comptime call did not produce a value",
                function.name,
            );
        }
        return value;
    }

    fn evalExprAsCtValue(self: *ConstEvaluator, expr_id: ast.ExprId, comptime use_cache: bool) anyerror!?CtValue {
        switch (self.file.expression(expr_id).*) {
            .Call => |call| {
                if (try self.evalResultConstructorCallCtValue(call, use_cache)) |ct_value| return ct_value;
                if (try self.evalEnumConstructorCallCtValue(expr_id, call, use_cache)) |ct_value| return ct_value;
                if (try self.evalErrorDeclCallCtValue(call, use_cache)) |ct_value| return ct_value;
                if (try self.evalCallCtValue(call, use_cache)) |ct_value| return ct_value;
                if (self.last_error != null) return null;
            },
            else => {
                if (try self.evalExprCtValueImpl(expr_id, use_cache, true)) |ct_value| return ct_value;
            },
        }

        const const_value = (try self.evalExprImpl(expr_id, use_cache)) orelse return null;
        return (try self.constValueToCtValue(const_value)) orelse null;
    }

    fn evalCallArgumentCtValue(self: *ConstEvaluator, parameter_file: *const ast.AstFile, parameter: ast.Parameter, arg: ast.ExprId, comptime use_cache: bool) anyerror!?CtValue {
        if (parameter.is_comptime and self.parameterExpectsTypeValueInFile(parameter_file, parameter)) {
            return self.typeExprCtValue(arg);
        }

        _ = try self.evalExprImpl(arg, use_cache);
        return (try self.evalExprAsCtValue(arg, use_cache)) orelse blk: {
            const const_value = (try self.evalExprImpl(arg, use_cache)) orelse return null;
            break :blk (try self.constValueToCtValue(const_value)) orelse return null;
        };
    }

    fn parameterExpectsTypeValue(self: *ConstEvaluator, parameter: ast.Parameter) bool {
        return self.parameterExpectsTypeValueInFile(self.file, parameter);
    }

    fn parameterExpectsTypeValueInFile(self: *ConstEvaluator, file: *const ast.AstFile, parameter: ast.Parameter) bool {
        _ = self;
        return switch (file.typeExpr(parameter.type_expr).*) {
            .Path => |path| std.mem.eql(u8, std.mem.trim(u8, path.name, " \t\n\r"), "type"),
            else => false,
        };
    }

    fn typeExprCtValue(self: *ConstEvaluator, expr_id: ast.ExprId) ?CtValue {
        return switch (self.file.expression(expr_id).*) {
            .TypeValue => |type_value| if (self.typeExprTypeId(type_value.type_expr)) |type_id|
                CtValue{ .type_val = type_id }
            else
                null,
            .Name => |name| if (self.pathTypeId(name.name)) |type_id|
                CtValue{ .type_val = type_id }
            else
                null,
            .Group => |group| self.typeExprCtValue(group.expr),
            else => blk: {
                const value = (self.evalExprCtValue(expr_id) catch null) orelse break :blk null;
                break :blk switch (value) {
                    .type_val => value,
                    else => null,
                };
            },
        };
    }

    fn typeExprTypeId(self: *ConstEvaluator, type_expr_id: ast.TypeExprId) ?u32 {
        return switch (self.file.typeExpr(type_expr_id).*) {
            .Path => |path| self.pathTypeId(path.name),
            .Generic => |generic| self.pathTypeId(generic.name),
            .AnonymousStruct => null,
            else => null,
        };
    }

    fn pathTypeId(self: *ConstEvaluator, name: []const u8) ?u32 {
        const trimmed = std.mem.trim(u8, name, " \t\n\r");
        if (self.env.lookupValue(trimmed)) |value| {
            return switch (value) {
                .type_val => |type_id| type_id,
                else => null,
            };
        }
        if (type_builtin.lookupBuiltinByName(trimmed)) |spec| return spec.comptime_type_id;
        if (type_builtin.parseFixedBytesName(trimmed)) |len| return type_builtin.fixedBytesTypeId(len);
        if (self.lookupNamedItem(trimmed)) |item_id| return self.namedTypeId(item_id);
        return null;
    }

    fn namedTypeId(self: *ConstEvaluator, item_id: ast.ItemId) u32 {
        const module_component: u32 = if (self.module_id) |module_id| @intCast(module_id.index() + 1) else 0;
        return type_builtin.named_type_id_base +
            module_component * named_type_id_module_stride +
            @as(u32, @intCast(item_id.index()));
    }

    fn namedTypeRefForTypeId(self: *ConstEvaluator, type_id: u32) ?NamedTypeRef {
        _ = self;
        if (type_id < type_builtin.named_type_id_base) return null;
        const offset = type_id - type_builtin.named_type_id_base;
        const module_component = offset / named_type_id_module_stride;
        const item_component = offset % named_type_id_module_stride;
        return .{
            .module_id = if (module_component == 0) null else source.ModuleId.fromIndex(module_component - 1),
            .item_id = ast.ItemId.fromIndex(item_component),
        };
    }

    fn itemIdForNamedTypeId(self: *ConstEvaluator, type_id: u32) ?ast.ItemId {
        const named_ref = self.namedTypeRefForTypeId(type_id) orelse return null;
        if (named_ref.module_id) |module_id| {
            if (self.module_id == null or module_id != self.module_id.?) return null;
        }
        return named_ref.item_id;
    }

    fn functionStage(self: *ConstEvaluator, function: ast.FunctionItem) Stage {
        return self.bodyStage(function.body);
    }

    fn bodyStage(self: *ConstEvaluator, body_id: ast.BodyId) Stage {
        const body = self.file.body(body_id).*;
        for (body.statements) |statement_id| {
            if (self.statementStage(statement_id) == .runtime_only) return .runtime_only;
        }
        return .comptime_ok;
    }

    fn statementStage(self: *ConstEvaluator, statement_id: ast.StmtId) Stage {
        return switch (self.file.statement(statement_id).*) {
            .VariableDecl => |decl| if (decl.value) |expr_id| self.exprStage(expr_id) else .comptime_ok,
            .Return => |ret| if (ret.value) |expr_id| self.exprStage(expr_id) else .comptime_ok,
            .If => |if_stmt| self.mergeStages(.{
                self.exprStage(if_stmt.condition),
                self.bodyStage(if_stmt.then_body),
                if (if_stmt.else_body) |else_body| self.bodyStage(else_body) else .comptime_ok,
            }),
            .While => |while_stmt| self.mergeStages(.{
                self.exprStage(while_stmt.condition),
                self.bodyStage(while_stmt.body),
            }),
            .For => |for_stmt| self.mergeStages(.{
                self.exprStage(for_stmt.iterable),
                self.bodyStage(for_stmt.body),
            }),
            .Switch => |switch_stmt| blk: {
                if (self.exprStage(switch_stmt.condition) == .runtime_only) break :blk .runtime_only;
                for (switch_stmt.arms) |arm| {
                    const pattern_stage = self.switchPatternStage(arm.pattern);
                    if (pattern_stage == .runtime_only or self.bodyStage(arm.body) == .runtime_only) break :blk .runtime_only;
                }
                if (switch_stmt.else_body) |else_body| {
                    if (self.bodyStage(else_body) == .runtime_only) break :blk .runtime_only;
                }
                break :blk .comptime_ok;
            },
            .Try => |try_stmt| blk: {
                if (self.bodyStage(try_stmt.try_body) == .runtime_only) break :blk .runtime_only;
                if (try_stmt.catch_clause) |catch_clause| {
                    if (self.bodyStage(catch_clause.body) == .runtime_only) break :blk .runtime_only;
                }
                break :blk .comptime_ok;
            },
            .Assign => |assign| self.exprStage(assign.value),
            .Expr => |expr_stmt| self.exprStage(expr_stmt.expr),
            .Block => |block_stmt| self.bodyStage(block_stmt.body),
            .LabeledBlock => |block_stmt| self.bodyStage(block_stmt.body),
            .Assert => |assert_stmt| self.exprStage(assert_stmt.condition),
            .Assume => |assume_stmt| self.exprStage(assume_stmt.condition),
            .Log, .Lock, .Unlock => .runtime_only,
            // A dispatch-layout hint: no value, no effect, valid anywhere
            // comptime looks at it.
            .CallHint => .comptime_ok,
            .Havoc, .Break, .Continue, .Error => .comptime_ok,
        };
    }

    fn exprStage(self: *ConstEvaluator, expr_id: ast.ExprId) Stage {
        return switch (self.file.expression(expr_id).*) {
            .TypeValue => .comptime_only,
            .Builtin => |builtin| blk: {
                if (stage_mod.isRuntimeOnlyIntrinsic(builtin.name)) break :blk .runtime_only;
                if (stage_mod.isComptimeOnlyIntrinsic(builtin.name)) break :blk .comptime_only;
                break :blk .comptime_ok;
            },
            .Unary => |unary| self.exprStage(unary.operand),
            .Binary => |binary| self.mergeStages(.{ self.exprStage(binary.lhs), self.exprStage(binary.rhs) }),
            .Call => |call| self.mergeStages(.{
                self.exprStage(call.callee),
                self.argsStage(call.args),
            }),
            .ExternalProxy => .runtime_only,
            .Field => |field| self.exprStage(field.base),
            .Index => |index| self.mergeStages(.{
                self.exprStage(index.base),
                self.exprStage(index.index),
            }),
            .Group => |group| self.exprStage(group.expr),
            .Old => |old| self.exprStage(old.expr),
            .Quantified => |quantified| self.mergeStages(.{
                if (quantified.condition) |condition| self.exprStage(condition) else .comptime_ok,
                self.exprStage(quantified.body),
            }),
            .Tuple => |tuple| self.argsStage(tuple.elements),
            .ArrayLiteral => |array| self.argsStage(array.elements),
            .StructLiteral => |struct_literal| blk: {
                for (struct_literal.fields) |field| {
                    if (self.exprStage(field.value) == .runtime_only) break :blk .runtime_only;
                }
                break :blk .comptime_ok;
            },
            .Switch => |switch_expr| blk: {
                if (self.exprStage(switch_expr.condition) == .runtime_only) break :blk .runtime_only;
                for (switch_expr.arms) |arm| {
                    const pattern_stage = self.switchPatternStage(arm.pattern);
                    if (pattern_stage == .runtime_only or self.exprStage(arm.value) == .runtime_only) break :blk .runtime_only;
                }
                if (switch_expr.else_expr) |else_expr| {
                    if (self.exprStage(else_expr) == .runtime_only) break :blk .runtime_only;
                }
                break :blk .comptime_ok;
            },
            .Comptime => |comptime_expr| self.bodyStage(comptime_expr.body),
            .ErrorReturn => |error_return| self.argsStage(error_return.args),
            .IntegerLiteral, .StringLiteral, .BoolLiteral, .AddressLiteral, .BytesLiteral, .Name, .Result, .Error => .comptime_ok,
        };
    }

    fn namedErrorPatternStage(self: *ConstEvaluator, pattern: ast.SwitchPattern) Stage {
        return if (self.sumVariantRefFromPattern(pattern) != null) .comptime_ok else .runtime_only;
    }

    fn switchPatternStage(self: *ConstEvaluator, pattern: ast.SwitchPattern) Stage {
        return switch (pattern) {
            .Expr => |expr_id| self.exprStage(expr_id),
            .Range => |range_pattern| self.mergeStages(.{
                self.exprStage(range_pattern.start),
                self.exprStage(range_pattern.end),
            }),
            .NamedError => self.namedErrorPatternStage(pattern),
            .Or => |or_pattern| blk: {
                var stage: Stage = .comptime_ok;
                for (or_pattern.alternatives) |alternative| {
                    switch (self.switchPatternStage(alternative)) {
                        .runtime_only => break :blk .runtime_only,
                        .comptime_only => stage = .comptime_only,
                        .comptime_ok => {},
                    }
                }
                break :blk stage;
            },
            .Ok, .Err, .Else => .comptime_ok,
        };
    }

    fn argsStage(self: *ConstEvaluator, args: []const ast.ExprId) Stage {
        var saw_comptime_only = false;
        for (args) |arg| {
            switch (self.exprStage(arg)) {
                .runtime_only => return .runtime_only,
                .comptime_only => saw_comptime_only = true,
                .comptime_ok => {},
            }
        }
        return if (saw_comptime_only) .comptime_only else .comptime_ok;
    }

    fn mergeStages(self: *ConstEvaluator, stages: anytype) Stage {
        _ = self;
        var saw_comptime_only = false;
        inline for (stages) |stage| {
            switch (stage) {
                .runtime_only => return .runtime_only,
                .comptime_only => saw_comptime_only = true,
                .comptime_ok => {},
            }
        }
        return if (saw_comptime_only) .comptime_only else .comptime_ok;
    }

    fn sourceSpan(self: *ConstEvaluator, range: source.TextRange) SourceSpan {
        _ = self;
        return .{
            .line = 0,
            .column = 0,
            .length = @intCast(range.end - range.start),
            .byte_offset = @intCast(range.start),
        };
    }

    fn recordMissingComptimeValue(
        self: *ConstEvaluator,
        span: SourceSpan,
        message: []const u8,
        reason: ?[]const u8,
    ) void {
        const ct_error = if (reason) |detail|
            error_mod.CtError.withReason(.not_comptime, span, message, detail)
        else
            error_mod.CtError.init(.not_comptime, span, message);
        self.recordCtError(ct_error);
    }

    fn recordLoopLimitExceeded(self: *ConstEvaluator, range: source.TextRange) void {
        self.recordCtError(error_mod.CtError.withReason(
            .iteration_limit,
            self.sourceSpan(range),
            "comptime loop iteration limit exceeded",
            "evaluation exceeded max_loop_iterations",
        ));
    }

    fn recordStepLimitExceeded(self: *ConstEvaluator, range: source.TextRange) void {
        self.recordCtError(error_mod.CtError.withReason(
            .step_limit,
            self.sourceSpan(range),
            "evaluation step limit exceeded",
            "evaluation exceeded max_steps",
        ));
    }

    fn recordInternalBuiltinError(self: *ConstEvaluator, range: source.TextRange, message: []const u8) void {
        self.recordCtError(error_mod.CtError.init(
            .internal_error,
            self.sourceSpan(range),
            message,
        ));
    }

    fn inRequiredComptime(self: *const ConstEvaluator) bool {
        return self.required_comptime_depth > 0;
    }

    fn recordCtError(self: *ConstEvaluator, ct_error: error_mod.CtError) void {
        if (!self.inRequiredComptime()) return;
        if (self.last_error != null) return;
        self.last_error = ct_error;
    }

    fn consumeStep(self: *ConstEvaluator, range: source.TextRange) bool {
        self.env.stats.recordStep();
        if (LimitCheck.init(self.env.config, &self.env.stats).checkSteps() != null) {
            self.recordStepLimitExceeded(range);
            return true;
        }
        return false;
    }

    fn statementRange(self: *ConstEvaluator, statement_id: ast.StmtId) source.TextRange {
        return switch (self.file.statement(statement_id).*) {
            .VariableDecl => |stmt| stmt.range,
            .Return => |stmt| stmt.range,
            .If => |stmt| stmt.range,
            .While => |stmt| stmt.range,
            .For => |stmt| stmt.range,
            .Switch => |stmt| stmt.range,
            .Try => |stmt| stmt.range,
            .Log => |stmt| stmt.range,
            .Lock => |stmt| stmt.range,
            .CallHint => |stmt| stmt.range,
            .Unlock => |stmt| stmt.range,
            .Assert => |stmt| stmt.range,
            .Assume => |stmt| stmt.range,
            .Havoc => |stmt| stmt.range,
            .Break => |stmt| stmt.range,
            .Continue => |stmt| stmt.range,
            .Assign => |stmt| stmt.range,
            .Expr => |stmt| stmt.range,
            .Block => |stmt| stmt.range,
            .LabeledBlock => |stmt| stmt.range,
            .Error => |stmt| stmt.range,
        };
    }

    fn exprRange(self: *ConstEvaluator, expr_id: ast.ExprId) source.TextRange {
        return switch (self.file.expression(expr_id).*) {
            .IntegerLiteral => |expr| expr.range,
            .StringLiteral => |expr| expr.range,
            .BoolLiteral => |expr| expr.range,
            .AddressLiteral => |expr| expr.range,
            .BytesLiteral => |expr| expr.range,
            .TypeValue => |expr| expr.range,
            .Tuple => |expr| expr.range,
            .ArrayLiteral => |expr| expr.range,
            .StructLiteral => |expr| expr.range,
            .ExternalProxy => |expr| expr.range,
            .Switch => |expr| expr.range,
            .Comptime => |expr| expr.range,
            .ErrorReturn => |expr| expr.range,
            .Name => |expr| expr.range,
            .Result => |expr| expr.range,
            .Unary => |expr| expr.range,
            .Binary => |expr| expr.range,
            .Call => |expr| expr.range,
            .Builtin => |expr| expr.range,
            .Field => |expr| expr.range,
            .Index => |expr| expr.range,
            .Group => |expr| expr.range,
            .Old => |expr| expr.range,
            .Quantified => |expr| expr.range,
            .Error => |expr| expr.range,
        };
    }

    fn bindName(self: *ConstEvaluator, name: []const u8, value: ?ConstValue) !void {
        const const_value = value orelse return;
        const ct_value = (try self.constValueToCtValue(const_value)) orelse return;
        if (self.env.isBoundInCurrentScope(name)) {
            try self.env.set(name, ct_value);
        } else {
            _ = try self.env.bind(name, ct_value);
        }
    }

    fn constValueToCtValue(self: *ConstEvaluator, value: ConstValue) !?CtValue {
        return switch (value) {
            .fixed_bytes => |bytes| CtValue{ .bytes_ref = try self.env.heap.allocBytes(bytes) },
            else => try constToCtValue(value),
        };
    }

    fn bindNameCtValue(self: *ConstEvaluator, name: []const u8, value: ?CtValue) !void {
        const ct_value = value orelse return;
        if (self.env.isBoundInCurrentScope(name)) {
            try self.env.set(name, ct_value);
        } else {
            _ = try self.env.bind(name, ct_value);
        }
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

    fn bindPatternCtValue(self: *ConstEvaluator, pattern_id: ast.PatternId, value: ?CtValue) !void {
        switch (self.file.pattern(pattern_id).*) {
            .Name => |name| try self.bindNameCtValue(name.name, value),
            .StructDestructure => |destructure| {
                for (destructure.fields) |field| {
                    try self.bindPatternCtValue(field.binding, null);
                }
            },
            .Field, .Index, .Error => {},
        }
    }

    fn evalComptimeBody(self: *ConstEvaluator, body_id: ast.BodyId) anyerror!?ConstValue {
        return switch (try self.evalComptimeBodyControl(body_id)) {
            .value => |value| value,
            .return_value => |value| value,
            .break_loop, .continue_loop => null,
            .indeterminate => null,
        };
    }

    fn evalComptimeBodyCtValue(self: *ConstEvaluator, body_id: ast.BodyId, comptime use_cache: bool) anyerror!?CtValue {
        return switch (try self.evalComptimeBodyControlCtValue(body_id, use_cache)) {
            .value => |value| value,
            .return_value => |value| value,
            .break_loop, .continue_loop => null,
            .indeterminate => null,
        };
    }

    fn evalComptimeBodyControl(self: *ConstEvaluator, body_id: ast.BodyId) anyerror!BodyControl {
        self.env.pushScope(false) catch return .{ .value = null };
        defer self.env.popScope();

        const body = self.file.body(body_id).*;
        var last_value: ?ConstValue = null;
        for (body.statements) |statement_id| {
            if (self.consumeStep(self.statementRange(statement_id))) return .{ .value = null };
            if (self.statementStage(statement_id) == .runtime_only) {
                self.recordCtError(error_mod.CtError.stageViolation(
                    self.sourceSpan(self.statementRange(statement_id)),
                    "runtime-only statement",
                ));
                return .{ .value = null };
            }
            switch (self.file.statement(statement_id).*) {
                .VariableDecl => |decl| {
                    if (decl.value) |expr_id| {
                        switch (self.file.expression(expr_id).*) {
                            .Call => |call| if (try self.evalCallCtValue(call, false)) |ct_value| {
                                var persisted = try ctValueToConstValue(self.allocator, &self.env.heap, ct_value);
                                persisted = try self.normalizeWrappingValueForDecl(decl, expr_id, persisted);
                                if (self.shouldBindNormalizedWrappingValue(decl, expr_id, persisted)) {
                                    try self.bindPattern(decl.pattern, persisted);
                                } else {
                                    try self.bindPatternCtValue(decl.pattern, ct_value);
                                }
                                self.values[expr_id.index()] = persisted;
                                last_value = null;
                                continue;
                            },
                            else => {},
                        }
                        if (try self.evalExprCtValue(expr_id)) |ct_value| {
                            var persisted = (try ctValueToConstValue(self.allocator, &self.env.heap, ct_value)) orelse try self.evalExprUncached(expr_id);
                            persisted = try self.normalizeWrappingValueForDecl(decl, expr_id, persisted);
                            if (self.shouldBindNormalizedWrappingValue(decl, expr_id, persisted)) {
                                try self.bindPattern(decl.pattern, persisted);
                            } else {
                                try self.bindPatternCtValue(decl.pattern, ct_value);
                            }
                            self.values[expr_id.index()] = persisted;
                        } else {
                            const persisted = try self.normalizeWrappingValueForDecl(decl, expr_id, try self.evalExprUncached(expr_id));
                            try self.bindPattern(decl.pattern, persisted);
                            self.values[expr_id.index()] = persisted;
                        }
                    }
                    last_value = null;
                },
                .Expr => |expr_stmt| {
                    last_value = try self.evalExprUncached(expr_stmt.expr);
                },
                .Return => |ret| {
                    return .{ .return_value = if (ret.value) |ret_value| try self.evalExprUncached(ret_value) else null };
                },
                .Block => |block_stmt| {
                    switch (try self.evalComptimeBodyControl(block_stmt.body)) {
                        .value => |value| last_value = value,
                        .return_value => |value| return .{ .return_value = value },
                        .break_loop => return .break_loop,
                        .continue_loop => return .continue_loop,
                        .indeterminate => return .indeterminate,
                    }
                },
                .LabeledBlock => |labeled| {
                    switch (try self.evalComptimeBodyControl(labeled.body)) {
                        .value => |value| last_value = value,
                        .return_value => |value| return .{ .return_value = value },
                        .break_loop => return .break_loop,
                        .continue_loop => return .continue_loop,
                        .indeterminate => return .indeterminate,
                    }
                },
                .If => |if_stmt| {
                    switch (try self.evalComptimeIf(if_stmt)) {
                        .value => |value| last_value = value,
                        .return_value => |value| return .{ .return_value = value },
                        .break_loop => return .break_loop,
                        .continue_loop => return .continue_loop,
                        .indeterminate => return .indeterminate,
                    }
                },
                .While => |while_stmt| {
                    switch (try self.evalComptimeWhile(while_stmt)) {
                        .value => |value| last_value = value,
                        .return_value => |value| return .{ .return_value = value },
                        .break_loop => return .break_loop,
                        .continue_loop => return .continue_loop,
                        .indeterminate => return .indeterminate,
                    }
                },
                .For => |for_stmt| {
                    switch (try self.evalComptimeFor(for_stmt)) {
                        .value => |value| last_value = value,
                        .return_value => |value| return .{ .return_value = value },
                        .break_loop => return .break_loop,
                        .continue_loop => return .continue_loop,
                        .indeterminate => return .indeterminate,
                    }
                },
                .Switch => |switch_stmt| {
                    switch (try self.evalComptimeSwitchStmt(switch_stmt)) {
                        .value => |value| last_value = value,
                        .return_value => |value| return .{ .return_value = value },
                        .break_loop => return .break_loop,
                        .continue_loop => return .continue_loop,
                        .indeterminate => return .indeterminate,
                    }
                },
                .Assign => |assign| {
                    last_value = try self.evalComptimeAssign(assign, true);
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

    fn evalComptimeBodyControlCtValue(self: *ConstEvaluator, body_id: ast.BodyId, comptime use_cache: bool) anyerror!CtBodyControl {
        self.env.pushScope(false) catch return .{ .value = null };
        defer self.env.popScope();

        const body = self.file.body(body_id).*;
        var last_value: ?CtValue = null;
        for (body.statements) |statement_id| {
            if (self.consumeStep(self.statementRange(statement_id))) return .{ .value = null };
            if (self.statementStage(statement_id) == .runtime_only) {
                self.recordCtError(error_mod.CtError.stageViolation(
                    self.sourceSpan(self.statementRange(statement_id)),
                    "runtime-only statement",
                ));
                return .{ .value = null };
            }
            switch (self.file.statement(statement_id).*) {
                .VariableDecl => |decl| {
                    if (decl.value) |expr_id| {
                        if (try self.evalExprAsCtValue(expr_id, use_cache)) |ct_value| {
                            var persisted = try ctValueToConstValue(self.allocator, &self.env.heap, ct_value);
                            persisted = try self.normalizeWrappingValueForDecl(decl, expr_id, persisted);
                            if (self.shouldBindNormalizedWrappingValue(decl, expr_id, persisted)) {
                                try self.bindPattern(decl.pattern, persisted);
                            } else {
                                try self.bindPatternCtValue(decl.pattern, ct_value);
                            }
                            self.values[expr_id.index()] = persisted;
                        } else {
                            const persisted = try self.normalizeWrappingValueForDecl(decl, expr_id, try self.evalExprUncached(expr_id));
                            try self.bindPattern(decl.pattern, persisted);
                            self.values[expr_id.index()] = persisted;
                        }
                    }
                    last_value = null;
                },
                .Expr => |expr_stmt| {
                    last_value = if (try self.evalExprAsCtValue(expr_stmt.expr, use_cache)) |ct_value|
                        ct_value
                    else if (try self.evalExprUncached(expr_stmt.expr)) |const_value|
                        (try constToCtValue(const_value)) orelse null
                    else
                        null;
                },
                .Return => |ret| {
                    return .{ .return_value = if (ret.value) |ret_value| try self.evalExprAsCtValue(ret_value, use_cache) else null };
                },
                .Block => |block_stmt| {
                    switch (try self.evalComptimeBodyControlCtValue(block_stmt.body, use_cache)) {
                        .value => |value| last_value = value,
                        .return_value => |value| return .{ .return_value = value },
                        .break_loop => return .break_loop,
                        .continue_loop => return .continue_loop,
                        .indeterminate => return .indeterminate,
                    }
                },
                .LabeledBlock => |labeled| {
                    switch (try self.evalComptimeBodyControlCtValue(labeled.body, use_cache)) {
                        .value => |value| last_value = value,
                        .return_value => |value| return .{ .return_value = value },
                        .break_loop => return .break_loop,
                        .continue_loop => return .continue_loop,
                        .indeterminate => return .indeterminate,
                    }
                },
                .If => |if_stmt| {
                    switch (try self.evalComptimeIfCtValue(if_stmt, use_cache)) {
                        .value => |value| last_value = value,
                        .return_value => |value| return .{ .return_value = value },
                        .break_loop => return .break_loop,
                        .continue_loop => return .continue_loop,
                        .indeterminate => return .indeterminate,
                    }
                },
                .While => |while_stmt| {
                    switch (try self.evalComptimeWhileCtValue(while_stmt, use_cache)) {
                        .value => |value| last_value = value,
                        .return_value => |value| return .{ .return_value = value },
                        .break_loop => return .break_loop,
                        .continue_loop => return .continue_loop,
                        .indeterminate => return .indeterminate,
                    }
                },
                .For => |for_stmt| {
                    switch (try self.evalComptimeForCtValue(for_stmt, use_cache)) {
                        .value => |value| last_value = value,
                        .return_value => |value| return .{ .return_value = value },
                        .break_loop => return .break_loop,
                        .continue_loop => return .continue_loop,
                        .indeterminate => return .indeterminate,
                    }
                },
                .Switch => |switch_stmt| {
                    switch (try self.evalComptimeSwitchStmtCtValue(switch_stmt, use_cache)) {
                        .value => |value| last_value = value,
                        .return_value => |value| return .{ .return_value = value },
                        .break_loop => return .break_loop,
                        .continue_loop => return .continue_loop,
                        .indeterminate => return .indeterminate,
                    }
                },
                .Assign => |assign| {
                    const value = try self.evalComptimeAssign(assign, use_cache);
                    last_value = if (value) |const_value| (try constToCtValue(const_value)) orelse null else null;
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
        const condition = (try self.evalExprUncached(if_stmt.condition)) orelse return .indeterminate;
        const take_then = self.constConditionTruthy(condition) orelse return .indeterminate;
        if (take_then) return try self.evalComptimeBodyControl(if_stmt.then_body);
        if (if_stmt.else_body) |else_body| return try self.evalComptimeBodyControl(else_body);
        return .{ .value = null };
    }

    fn evalComptimeIfCtValue(self: *ConstEvaluator, if_stmt: ast.IfStmt, comptime use_cache: bool) anyerror!CtBodyControl {
        const condition = (try self.evalExprUncached(if_stmt.condition)) orelse return .indeterminate;
        const take_then = self.constConditionTruthy(condition) orelse return .indeterminate;
        if (take_then) return try self.evalComptimeBodyControlCtValue(if_stmt.then_body, use_cache);
        if (if_stmt.else_body) |else_body| return try self.evalComptimeBodyControlCtValue(else_body, use_cache);
        return .{ .value = null };
    }

    fn evalComptimeSwitchStmt(self: *ConstEvaluator, switch_stmt: ast.SwitchStmt) anyerror!BodyControl {
        if (try self.evalExprAsCtValue(switch_stmt.condition, false)) |condition_ct| {
            for (switch_stmt.arms) |arm| {
                if (try self.patternMatchesCt(condition_ct, arm.pattern)) {
                    self.env.pushScope(false) catch return .indeterminate;
                    defer self.env.popScope();
                    try self.bindSwitchPatternCtValue(condition_ct, arm.pattern);
                    return try self.evalComptimeBodyControl(arm.body);
                }
            }
            if (switch_stmt.else_body) |else_body| return try self.evalComptimeBodyControl(else_body);
            return .indeterminate;
        }

        const condition = (try self.evalExprUncached(switch_stmt.condition)) orelse return .indeterminate;
        for (switch_stmt.arms) |arm| {
            if (self.patternMatches(condition, arm.pattern)) {
                return try self.evalComptimeBodyControl(arm.body);
            }
        }
        if (switch_stmt.else_body) |else_body| return try self.evalComptimeBodyControl(else_body);
        return .indeterminate;
    }

    fn evalComptimeSwitchStmtCtValue(self: *ConstEvaluator, switch_stmt: ast.SwitchStmt, comptime use_cache: bool) anyerror!CtBodyControl {
        if (try self.evalExprAsCtValue(switch_stmt.condition, use_cache)) |condition_ct| {
            for (switch_stmt.arms) |arm| {
                const matches = try self.patternMatchesCt(condition_ct, arm.pattern);
                if (matches) {
                    self.env.pushScope(false) catch return .indeterminate;
                    defer self.env.popScope();
                    try self.bindSwitchPatternCtValue(condition_ct, arm.pattern);
                    return try self.evalComptimeBodyControlCtValue(arm.body, use_cache);
                }
            }
            if (switch_stmt.else_body) |else_body| return try self.evalComptimeBodyControlCtValue(else_body, use_cache);
            return .indeterminate;
        }

        const condition = (try self.evalExprUncached(switch_stmt.condition)) orelse return .indeterminate;
        for (switch_stmt.arms) |arm| {
            if (self.patternMatches(condition, arm.pattern)) {
                return try self.evalComptimeBodyControlCtValue(arm.body, use_cache);
            }
        }
        if (switch_stmt.else_body) |else_body| return try self.evalComptimeBodyControlCtValue(else_body, use_cache);
        return .indeterminate;
    }

    fn evalComptimeWhile(self: *ConstEvaluator, while_stmt: ast.WhileStmt) anyerror!BodyControl {
        var iterations: u64 = 0;
        var last_value: ?ConstValue = null;
        while (true) {
            iterations += 1;
            if (iterations > self.env.config.max_loop_iterations) {
                self.recordLoopLimitExceeded(while_stmt.range);
                return .indeterminate;
            }

            const condition = (try self.evalExprUncached(while_stmt.condition)) orelse return .indeterminate;
            const should_continue = self.constConditionTruthy(condition) orelse return .indeterminate;
            if (!should_continue) break;

            switch (try self.evalComptimeBodyControl(while_stmt.body)) {
                .value => |value| last_value = value,
                .return_value => |value| return .{ .return_value = value },
                .break_loop => break,
                .continue_loop => continue,
                .indeterminate => return .indeterminate,
            }
        }
        return .{ .value = last_value };
    }

    fn evalComptimeWhileCtValue(self: *ConstEvaluator, while_stmt: ast.WhileStmt, comptime use_cache: bool) anyerror!CtBodyControl {
        var iterations: u64 = 0;
        var last_value: ?CtValue = null;
        while (true) {
            iterations += 1;
            if (iterations > self.env.config.max_loop_iterations) {
                self.recordLoopLimitExceeded(while_stmt.range);
                return .indeterminate;
            }

            const condition = (try self.evalExprUncached(while_stmt.condition)) orelse return .indeterminate;
            const should_continue = self.constConditionTruthy(condition) orelse return .indeterminate;
            if (!should_continue) break;

            switch (try self.evalComptimeBodyControlCtValue(while_stmt.body, use_cache)) {
                .value => |value| last_value = value,
                .return_value => |value| return .{ .return_value = value },
                .break_loop => break,
                .continue_loop => continue,
                .indeterminate => return .indeterminate,
            }
        }
        return .{ .value = last_value };
    }

    fn evalComptimeFor(self: *ConstEvaluator, for_stmt: ast.ForStmt) anyerror!BodyControl {
        const iterable = (try self.evalIterableCtValue(for_stmt.iterable)) orelse return .{ .value = null };
        var last_value: ?ConstValue = null;

        switch (iterable) {
            .integer => |integer| {
                if (integer > std.math.maxInt(usize)) return .{ .value = null };
                const trip_count: usize = @intCast(integer);
                var iteration: usize = 0;
                while (iteration < trip_count) : (iteration += 1) {
                    if (iteration >= self.env.config.max_loop_iterations) {
                        self.recordLoopLimitExceeded(for_stmt.range);
                        return .{ .value = null };
                    }

                    const item_value = CtValue{ .integer = @intCast(iteration) };
                    try self.bindPatternCtValue(for_stmt.item_pattern, item_value);

                    if (for_stmt.index_pattern) |index_pattern| {
                        const index_value = CtValue{ .integer = @intCast(iteration) };
                        try self.bindPatternCtValue(index_pattern, index_value);
                    }

                    switch (try self.evalComptimeBodyControl(for_stmt.body)) {
                        .value => |value| last_value = value,
                        .return_value => |value| return .{ .return_value = value },
                        .break_loop => break,
                        .continue_loop => continue,
                        .indeterminate => return .indeterminate,
                    }
                }
            },
            .array_ref => |heap_id| {
                const elems = self.env.heap.getArray(heap_id).elems;
                for (elems, 0..) |elem, iteration| {
                    if (iteration >= self.env.config.max_loop_iterations) {
                        self.recordLoopLimitExceeded(for_stmt.range);
                        return .{ .value = null };
                    }

                    try self.bindPatternCtValue(for_stmt.item_pattern, elem);
                    if (for_stmt.index_pattern) |index_pattern| {
                        const index_value = CtValue{ .integer = @intCast(iteration) };
                        try self.bindPatternCtValue(index_pattern, index_value);
                    }

                    switch (try self.evalComptimeBodyControl(for_stmt.body)) {
                        .value => |value| last_value = value,
                        .return_value => |value| return .{ .return_value = value },
                        .break_loop => break,
                        .continue_loop => continue,
                        .indeterminate => return .indeterminate,
                    }
                }
            },
            .slice_ref => |heap_id| {
                const elems = self.env.heap.getSlice(heap_id).elems;
                for (elems, 0..) |elem, iteration| {
                    if (iteration >= self.env.config.max_loop_iterations) {
                        self.recordLoopLimitExceeded(for_stmt.range);
                        return .{ .value = null };
                    }

                    try self.bindPatternCtValue(for_stmt.item_pattern, elem);
                    if (for_stmt.index_pattern) |index_pattern| {
                        const index_value = CtValue{ .integer = @intCast(iteration) };
                        try self.bindPatternCtValue(index_pattern, index_value);
                    }

                    switch (try self.evalComptimeBodyControl(for_stmt.body)) {
                        .value => |value| last_value = value,
                        .return_value => |value| return .{ .return_value = value },
                        .break_loop => break,
                        .continue_loop => continue,
                        .indeterminate => return .indeterminate,
                    }
                }
            },
            .tuple_ref => |heap_id| {
                const elems = self.env.heap.getTuple(heap_id).elems;
                for (elems, 0..) |elem, iteration| {
                    if (iteration >= self.env.config.max_loop_iterations) {
                        self.recordLoopLimitExceeded(for_stmt.range);
                        return .{ .value = null };
                    }

                    try self.bindPatternCtValue(for_stmt.item_pattern, elem);
                    if (for_stmt.index_pattern) |index_pattern| {
                        const index_value = CtValue{ .integer = @intCast(iteration) };
                        try self.bindPatternCtValue(index_pattern, index_value);
                    }

                    switch (try self.evalComptimeBodyControl(for_stmt.body)) {
                        .value => |value| last_value = value,
                        .return_value => |value| return .{ .return_value = value },
                        .break_loop => break,
                        .continue_loop => continue,
                        .indeterminate => return .indeterminate,
                    }
                }
            },
            else => return .{ .value = null },
        }
        return .{ .value = last_value };
    }

    fn evalComptimeForCtValue(self: *ConstEvaluator, for_stmt: ast.ForStmt, comptime use_cache: bool) anyerror!CtBodyControl {
        const iterable = (try self.evalIterableCtValue(for_stmt.iterable)) orelse return .{ .value = null };
        var last_value: ?CtValue = null;

        switch (iterable) {
            .integer => |integer| {
                if (integer > std.math.maxInt(usize)) return .{ .value = null };
                const trip_count: usize = @intCast(integer);
                var iteration: usize = 0;
                while (iteration < trip_count) : (iteration += 1) {
                    if (iteration >= self.env.config.max_loop_iterations) {
                        self.recordLoopLimitExceeded(for_stmt.range);
                        return .{ .value = null };
                    }

                    const item_value = CtValue{ .integer = @intCast(iteration) };
                    try self.bindPatternCtValue(for_stmt.item_pattern, item_value);

                    if (for_stmt.index_pattern) |index_pattern| {
                        const index_value = CtValue{ .integer = @intCast(iteration) };
                        try self.bindPatternCtValue(index_pattern, index_value);
                    }

                    switch (try self.evalComptimeBodyControlCtValue(for_stmt.body, use_cache)) {
                        .value => |value| last_value = value,
                        .return_value => |value| return .{ .return_value = value },
                        .break_loop => break,
                        .continue_loop => continue,
                        .indeterminate => return .indeterminate,
                    }
                }
            },
            .array_ref => |heap_id| {
                const elems = self.env.heap.getArray(heap_id).elems;
                for (elems, 0..) |elem, iteration| {
                    if (iteration >= self.env.config.max_loop_iterations) {
                        self.recordLoopLimitExceeded(for_stmt.range);
                        return .{ .value = null };
                    }

                    try self.bindPatternCtValue(for_stmt.item_pattern, elem);
                    if (for_stmt.index_pattern) |index_pattern| {
                        const index_value = CtValue{ .integer = @intCast(iteration) };
                        try self.bindPatternCtValue(index_pattern, index_value);
                    }

                    switch (try self.evalComptimeBodyControlCtValue(for_stmt.body, use_cache)) {
                        .value => |value| last_value = value,
                        .return_value => |value| return .{ .return_value = value },
                        .break_loop => break,
                        .continue_loop => continue,
                        .indeterminate => return .indeterminate,
                    }
                }
            },
            .slice_ref => |heap_id| {
                const elems = self.env.heap.getSlice(heap_id).elems;
                for (elems, 0..) |elem, iteration| {
                    if (iteration >= self.env.config.max_loop_iterations) {
                        self.recordLoopLimitExceeded(for_stmt.range);
                        return .{ .value = null };
                    }

                    try self.bindPatternCtValue(for_stmt.item_pattern, elem);
                    if (for_stmt.index_pattern) |index_pattern| {
                        const index_value = CtValue{ .integer = @intCast(iteration) };
                        try self.bindPatternCtValue(index_pattern, index_value);
                    }

                    switch (try self.evalComptimeBodyControlCtValue(for_stmt.body, use_cache)) {
                        .value => |value| last_value = value,
                        .return_value => |value| return .{ .return_value = value },
                        .break_loop => break,
                        .continue_loop => continue,
                        .indeterminate => return .indeterminate,
                    }
                }
            },
            .tuple_ref => |heap_id| {
                const elems = self.env.heap.getTuple(heap_id).elems;
                for (elems, 0..) |elem, iteration| {
                    if (iteration >= self.env.config.max_loop_iterations) {
                        self.recordLoopLimitExceeded(for_stmt.range);
                        return .{ .value = null };
                    }

                    try self.bindPatternCtValue(for_stmt.item_pattern, elem);
                    if (for_stmt.index_pattern) |index_pattern| {
                        const index_value = CtValue{ .integer = @intCast(iteration) };
                        try self.bindPatternCtValue(index_pattern, index_value);
                    }

                    switch (try self.evalComptimeBodyControlCtValue(for_stmt.body, use_cache)) {
                        .value => |value| last_value = value,
                        .return_value => |value| return .{ .return_value = value },
                        .break_loop => break,
                        .continue_loop => continue,
                        .indeterminate => return .indeterminate,
                    }
                }
            },
            else => return .{ .value = null },
        }
        return .{ .value = last_value };
    }

    fn evalIterableCtValue(self: *ConstEvaluator, expr_id: ast.ExprId) anyerror!?CtValue {
        return switch (self.file.expression(expr_id).*) {
            .IntegerLiteral => |literal| blk: {
                const value = (try parseIntegerLiteral(self.allocator, literal.text)) orelse break :blk null;
                break :blk try constToCtValue(value);
            },
            .BoolLiteral => |literal| CtValue{ .boolean = literal.value },
            .Name => |name| self.env.lookupValue(name.name),
            .Group => |group| try self.evalIterableCtValue(group.expr),
            .ArrayLiteral => |array| blk: {
                const elems = try self.allocator.alloc(CtValue, array.elements.len);
                for (array.elements, 0..) |element_id, idx| {
                    elems[idx] = (try self.evalIterableCtValue(element_id)) orelse break :blk null;
                }
                const heap_id = try self.env.heap.allocArrayOwned(elems);
                break :blk CtValue{ .array_ref = heap_id };
            },
            .Tuple => |tuple| blk: {
                const elems = try self.allocator.alloc(CtValue, tuple.elements.len);
                for (tuple.elements, 0..) |element_id, idx| {
                    elems[idx] = (try self.evalIterableCtValue(element_id)) orelse break :blk null;
                }
                const heap_id = try self.env.heap.allocTupleOwned(elems);
                break :blk CtValue{ .tuple_ref = heap_id };
            },
            .Builtin => |builtin| blk: {
                if (std.mem.eql(u8, builtin.name, "cast") and builtin.type_arg != null and builtin.args.len > 0) {
                    const target = self.valueConstructionTarget(builtin.type_arg.?);
                    if (target != .none) break :blk try self.evalExprCtValueAs(builtin.args[0], target);
                }
                break :blk null;
            },
            else => blk: {
                const value = (try self.evalExprUncached(expr_id)) orelse break :blk null;
                break :blk try self.constValueToCtValue(value);
            },
        };
    }

    fn compoundAssignBinaryOp(op: ast.AssignmentOp) ?ast.BinaryOp {
        return switch (op) {
            .add_assign => .add,
            .sub_assign => .sub,
            .mul_assign => .mul,
            .div_assign => .div,
            .mod_assign => .mod,
            .bit_and_assign => .bit_and,
            .bit_or_assign => .bit_or,
            .bit_xor_assign => .bit_xor,
            .shl_assign => .shl,
            .shr_assign => .shr,
            .pow_assign => .pow,
            .wrapping_add_assign => .wrapping_add,
            .wrapping_sub_assign => .wrapping_sub,
            .wrapping_mul_assign => .wrapping_mul,
            .assign => null,
        };
    }

    fn evalComptimeAssign(self: *ConstEvaluator, assign: ast.AssignStmt, comptime use_cache: bool) anyerror!?ConstValue {
        const rhs_const = try self.evalExprUncached(assign.value);
        const rhs_ct = (try self.evalExprCtValueImpl(assign.value, use_cache, true)) orelse blk: {
            const rhs = rhs_const orelse break :blk null;
            break :blk (try self.constValueToCtValue(rhs)) orelse break :blk null;
        } orelse return null;
        switch (self.file.pattern(assign.target).*) {
            .Name => |name| {
                if (assign.op == .assign) {
                    try self.env.set(name.name, rhs_ct);
                    return rhs_const;
                }
                const rhs = rhs_const orelse (try ctValueToConstValue(self.allocator, &self.env.heap, rhs_ct)) orelse return null;
                const op = compoundAssignBinaryOp(assign.op) orelse return null;
                const value = (try self.evalBinaryValue(op, try self.readBoundName(name.name), rhs, assign.range)) orelse return null;
                const ct_value = (try self.constValueToCtValue(value)) orelse return null;
                try self.env.set(name.name, ct_value);
                return value;
            },
            .StructDestructure => |destructure| {
                if (assign.op != .assign) return null;
                const heap_id = switch (rhs_ct) {
                    .struct_ref => |heap_id| heap_id,
                    else => return null,
                };
                const struct_data = self.env.heap.getStruct(heap_id);
                for (destructure.fields) |field| {
                    const field_index = (try self.structFieldIndex(struct_data.type_id, field.name)) orelse return null;
                    const field_value = self.structFieldValue(struct_data, field_index) orelse return null;
                    try self.bindPatternCtValue(field.binding, field_value);
                }
                return rhs_const;
            },
            .Index => |index| {
                const rhs = rhs_const orelse return null;
                const base_name = switch (self.file.pattern(index.base).*) {
                    .Name => |name| name.name,
                    else => return null,
                };
                const base_slot = self.env.lookup(base_name) orelse return null;
                const base_value = self.env.read(base_slot);
                const index_value = (try self.evalExprCtValueImpl(index.index, use_cache, true)) orelse return null;
                const maybe_idx = self.ctIndexValue(index_value);
                const updated = switch (base_value) {
                    .array_ref => |heap_id| blk: {
                        const idx = maybe_idx orelse break :blk null;
                        const elems = self.env.heap.getArray(heap_id).elems;
                        if (idx >= elems.len) break :blk null;
                        const next_value = switch (assign.op) {
                            .assign => rhs_ct,
                            else => blk_op: {
                                const current = (try ctValueToConstValue(self.allocator, &self.env.heap, elems[idx])) orelse break :blk_op null;
                                const op = compoundAssignBinaryOp(assign.op) orelse break :blk_op null;
                                const computed = (try self.evalBinaryValue(op, current, rhs, assign.range)) orelse break :blk_op null;
                                break :blk_op (try constToCtValue(computed)) orelse break :blk_op null;
                            },
                        } orelse return null;
                        break :blk CtValue{ .array_ref = try self.env.heap.setArrayElem(heap_id, idx, next_value) };
                    },
                    .slice_ref => |heap_id| blk: {
                        const idx = maybe_idx orelse break :blk null;
                        const elems = self.env.heap.getSlice(heap_id).elems;
                        if (idx >= elems.len) break :blk null;
                        const next_value = switch (assign.op) {
                            .assign => rhs_ct,
                            else => blk_op: {
                                const current = (try ctValueToConstValue(self.allocator, &self.env.heap, elems[idx])) orelse break :blk_op null;
                                const op = compoundAssignBinaryOp(assign.op) orelse break :blk_op null;
                                const computed = (try self.evalBinaryValue(op, current, rhs, assign.range)) orelse break :blk_op null;
                                break :blk_op (try constToCtValue(computed)) orelse break :blk_op null;
                            },
                        } orelse return null;
                        break :blk CtValue{ .slice_ref = try self.env.heap.setSliceElem(heap_id, idx, next_value) };
                    },
                    .map_ref => |heap_id| blk: {
                        const current: ?ConstValue = blk_current: {
                            for (self.env.heap.getMap(heap_id).entries) |entry| {
                                if (self.ctValuesEqual(entry.key, index_value)) {
                                    break :blk_current try ctValueToConstValue(self.allocator, &self.env.heap, entry.value);
                                }
                            }
                            break :blk_current null;
                        };
                        const next_value = switch (assign.op) {
                            .assign => rhs_ct,
                            else => blk_op: {
                                const current_value = current orelse break :blk_op null;
                                const op = compoundAssignBinaryOp(assign.op) orelse break :blk_op null;
                                const computed = (try self.evalBinaryValue(op, current_value, rhs, assign.range)) orelse break :blk_op null;
                                break :blk_op (try constToCtValue(computed)) orelse break :blk_op null;
                            },
                        } orelse return null;
                        break :blk CtValue{ .map_ref = try self.setMapEntryValue(heap_id, index_value, next_value) };
                    },
                    else => return null,
                } orelse return null;
                self.env.update(base_slot, updated);
                return try ctValueToConstValue(self.allocator, &self.env.heap, switch (updated) {
                    .array_ref => |heap_id| self.env.heap.getArray(heap_id).elems[maybe_idx orelse return null],
                    .slice_ref => |heap_id| self.env.heap.getSlice(heap_id).elems[maybe_idx orelse return null],
                    .map_ref => |heap_id| blk: {
                        for (self.env.heap.getMap(heap_id).entries) |entry| {
                            if (self.ctValuesEqual(entry.key, index_value)) break :blk entry.value;
                        }
                        return null;
                    },
                    else => return null,
                });
            },
            .Field => |field| {
                const base_value = (try self.readPatternCtValue(field.base)) orelse return null;

                const struct_data = switch (base_value) {
                    .struct_ref => |heap_id| self.env.heap.getStruct(heap_id),
                    else => return null,
                };
                const field_index = (try self.structFieldIndex(struct_data.type_id, field.name)) orelse return null;
                const current_field = self.structFieldValue(struct_data, field_index) orelse return null;
                const next_value = switch (assign.op) {
                    .assign => rhs_ct,
                    else => blk_op: {
                        const rhs = rhs_const orelse break :blk_op null;
                        const current = (try ctValueToConstValue(self.allocator, &self.env.heap, current_field)) orelse break :blk_op null;
                        const op = compoundAssignBinaryOp(assign.op) orelse break :blk_op null;
                        const computed = (try self.evalBinaryValue(op, current, rhs, assign.range)) orelse break :blk_op null;
                        break :blk_op (try constToCtValue(computed)) orelse break :blk_op null;
                    },
                } orelse return null;

                if (!try self.writePatternCtValue(assign.target, next_value)) return null;
                return try ctValueToConstValue(self.allocator, &self.env.heap, next_value);
            },
            else => return null,
        }
    }

    fn readBoundName(self: *ConstEvaluator, name: []const u8) anyerror!?ConstValue {
        const value = self.env.lookupValue(name) orelse return null;
        return try ctValueToConstValue(self.allocator, &self.env.heap, value);
    }

    fn normalizeWrappingValueForDecl(self: *ConstEvaluator, decl: ast.VariableDeclStmt, expr_id: ast.ExprId, value: ?ConstValue) !?ConstValue {
        const type_expr = decl.type_expr orelse return value;
        if (!self.exprIsWrappingOp(expr_id)) return value;
        const integer = self.typeExprIntegerType(type_expr) orelse return value;
        return if (value) |v| try wrapIntegerConstToType(self.allocator, v, integer) else null;
    }

    fn shouldBindNormalizedWrappingValue(self: *ConstEvaluator, decl: ast.VariableDeclStmt, expr_id: ast.ExprId, value: ?ConstValue) bool {
        _ = value;
        return decl.type_expr != null and self.exprIsWrappingOp(expr_id);
    }

    fn exprIsWrappingOp(self: *ConstEvaluator, expr_id: ast.ExprId) bool {
        return switch (self.file.expression(expr_id).*) {
            .Group => |group| self.exprIsWrappingOp(group.expr),
            .Binary => |binary| switch (binary.op) {
                .wrapping_add, .wrapping_sub, .wrapping_mul, .wrapping_pow, .wrapping_shl, .wrapping_shr => true,
                else => false,
            },
            else => false,
        };
    }

    fn typeExprIntegerType(self: *ConstEvaluator, type_expr_id: ast.TypeExprId) ?IntegerType {
        return switch (self.file.typeExpr(type_expr_id).*) {
            .Path => |path| integerTypeFromName(path.name),
            else => null,
        };
    }

    fn integerTypeFromName(name: []const u8) ?IntegerType {
        const trimmed = std.mem.trim(u8, name, " \t\n\r");
        return type_descriptors.integerTypeFromName(trimmed);
    }

    fn constConditionTruthy(self: *ConstEvaluator, value: ConstValue) ?bool {
        _ = self;
        return switch (value) {
            .boolean => |boolean| boolean,
            .integer => |integer| !integer.eqlZero(),
            .address => null,
            .fixed_bytes => null,
            .string => null,
            .tuple => null,
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
                    self.visitSwitchPattern(arm.pattern);
                    self.visitBody(arm.body);
                }
                if (switch_stmt.else_body) |else_body| self.visitBody(else_body);
            },
            .Try => |try_stmt| {
                self.visitBody(try_stmt.try_body);
                if (try_stmt.catch_clause) |catch_clause| self.visitBody(catch_clause.body);
            },
            .Assign => |assign| _ = self.evalExpr(assign.value) catch null,
            .VariableDecl => |decl| {
                if (decl.value) |expr_id| _ = self.evalExpr(expr_id) catch null;
            },
            .Return => |ret| {
                if (ret.value) |expr_id| _ = self.evalExpr(expr_id) catch null;
            },
            .Expr => |expr_stmt| _ = self.evalExpr(expr_stmt.expr) catch null,
            .Block => |block_stmt| self.visitBody(block_stmt.body),
            .LabeledBlock => |labeled| self.visitBody(labeled.body),
            .Log => |log_stmt| {
                for (log_stmt.args) |arg| _ = self.evalExpr(arg) catch null;
            },
            .Assert => |assert_stmt| _ = self.evalExpr(assert_stmt.condition) catch null,
            .Assume => |assume_stmt| _ = self.evalExpr(assume_stmt.condition) catch null,
            .Lock, .Unlock, .Break, .Continue, .Havoc, .Error, .CallHint => {},
        }
    }
};
