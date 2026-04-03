const std = @import("std");
const ast = @import("../ast/mod.zig");
const bridge = @import("compiler_const_bridge.zig");
const comptime_mod = @import("mod.zig");
const diagnostics = @import("../diagnostics/mod.zig");
const stage_mod = @import("stage.zig");
const model = @import("../sema/model.zig");
const source = @import("../source/mod.zig");
const error_mod = @import("error.zig");

const ConstEvalResult = model.ConstEvalResult;
const ConstValue = model.ConstValue;
const TypeKind = model.TypeKind;
const CtAggregate = comptime_mod.CtAggregate;
const CtEnum = comptime_mod.CtEnum;
const CtEnv = bridge.CtEnv;
const CtValue = bridge.CtValue;
const type_ids = comptime_mod.type_ids;
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
const named_type_id_base: u32 = 1_000_000;

pub const TypeQuery = struct {
    context: *anyopaque,
    ensure_typecheck: *const fn (context: *anyopaque, module_id: source.ModuleId, key: model.TypeCheckKey) anyerror!*const model.TypeCheckResult,
    module_typecheck: *const fn (context: *anyopaque, module_id: source.ModuleId) anyerror!*const model.TypeCheckResult,
    ast_file: *const fn (context: *anyopaque, module_id: source.ModuleId) anyerror!*const ast.AstFile,
    lookup_item: *const fn (context: *anyopaque, module_id: source.ModuleId, name: []const u8) anyerror!?ast.ItemId,
    resolve_import_alias: *const fn (context: *anyopaque, module_id: source.ModuleId, alias: []const u8) anyerror!?source.ModuleId,
};

pub const ConstEvalOptions = struct {
    module_id: ?source.ModuleId = null,
    type_query: ?TypeQuery = null,
    config: EvalConfig = .default,
};

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

    var evaluator = ConstEvaluator{
        .allocator = arena,
        .file = file,
        .values = values,
        .env = CtEnv.init(arena, options.config),
        .module_id = options.module_id,
        .type_query = options.type_query,
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
    type_query: ?TypeQuery = null,
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
    };

    const CtBodyControl = union(enum) {
        value: ?CtValue,
        return_value: ?CtValue,
        break_loop,
        continue_loop,
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
                .Assign => |assign| _ = self.evalComptimeAssign(assign) catch null,
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
                self.required_comptime_depth += 1;
                defer self.required_comptime_depth -= 1;
                const value = try self.evalComptimeBody(comptime_expr.body);
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
            .Binary => |binary| try evalBinary(self.allocator, binary.op, try self.evalExprImpl(binary.lhs, use_cache), try self.evalExprImpl(binary.rhs, use_cache)),
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
                break :blk null;
            },
            .Index => |index| blk: {
                _ = try self.evalExprImpl(index.base, use_cache);
                _ = try self.evalExprImpl(index.index, use_cache);
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
                const heap_id = try self.env.heap.allocBytes(bytes);
                break :blk CtValue{ .bytes_ref = heap_id };
            },
            .Name => |name| self.env.lookupValue(name.name) orelse blk: {
                if (self.pathTypeId(name.name)) |type_id| break :blk CtValue{ .type_val = type_id };
                break :blk null;
            },
            .Group => |group| try self.evalExprCtValueImpl(group.expr, use_cache, true),
            .Call => |call| try self.evalCallCtValue(call, use_cache),
            .Unary, .Switch, .Comptime => blk: {
                const const_value = (try self.evalExprImpl(expr_id, use_cache)) orelse break :blk null;
                break :blk (try constToCtValue(const_value)) orelse null;
            },
            .ArrayLiteral => |array| blk: {
                const elems = try self.allocator.alloc(CtValue, array.elements.len);
                for (array.elements, 0..) |element_id, idx| {
                    _ = try self.evalExprImpl(element_id, use_cache);
                    elems[idx] = (try self.evalExprCtValueImpl(element_id, use_cache, true)) orelse break :blk null;
                }
                const heap_id = try self.env.heap.allocArray(elems);
                break :blk CtValue{ .array_ref = heap_id };
            },
            .Tuple => |tuple| blk: {
                const elems = try self.allocator.alloc(CtValue, tuple.elements.len);
                for (tuple.elements, 0..) |element_id, idx| {
                    _ = try self.evalExprImpl(element_id, use_cache);
                    elems[idx] = (try self.evalExprCtValueImpl(element_id, use_cache, true)) orelse break :blk null;
                }
                const heap_id = try self.env.heap.allocTuple(elems);
                break :blk CtValue{ .tuple_ref = heap_id };
            },
            .StructLiteral => |struct_literal| blk: {
                const fields = try self.allocator.alloc(CtAggregate.StructField, struct_literal.fields.len);
                for (struct_literal.fields, 0..) |field, idx| {
                    _ = try self.evalExprImpl(field.value, use_cache);
                    fields[idx] = .{
                        .field_id = @intCast(idx),
                        .value = (try self.evalExprCtValueImpl(field.value, use_cache, true)) orelse break :blk null,
                    };
                }
                const type_id = (try self.structTypeIdForExpr(expr_id, struct_literal)) orelse break :blk null;
                const heap_id = try self.env.heap.allocStruct(type_id, fields);
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
                if (std.mem.eql(u8, builtin.name, "cast") and builtin.type_arg != null and builtin.args.len > 0) {
                    const target = self.valueConstructionTarget(builtin.type_arg.?);
                    if (target != .none) {
                        break :blk try self.evalExprCtValueAsImpl(builtin.args[0], target, use_cache);
                    }
                }
                const const_value = (try self.evalBuiltin(builtin)) orelse break :blk null;
                break :blk (try constToCtValue(const_value)) orelse null;
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
                        const field_index = self.structFieldIndex(struct_data.type_id, field.name) orelse break :blk_field null;
                        if (field_index >= struct_data.fields.len) break :blk_field null;
                        break :blk_field struct_data.fields[field_index].value;
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
                const lhs = (try self.evalExprCtValueImpl(binary.lhs, use_cache, true)) orelse break :blk null;
                const rhs = (try self.evalExprCtValueImpl(binary.rhs, use_cache, true)) orelse break :blk null;
                break :blk switch (binary.op) {
                    .eq => CtValue{ .boolean = self.ctValuesEqual(lhs, rhs) },
                    .ne => CtValue{ .boolean = !self.ctValuesEqual(lhs, rhs) },
                    else => null,
                };
            },
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
            .Call, .Unary, .Switch, .Comptime => true,
            else => false,
        };
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
                    const heap_id = try self.env.heap.allocSlice(elems);
                    break :blk CtValue{ .slice_ref = heap_id };
                },
                else => try self.evalExprCtValueImpl(expr_id, use_cache, true),
            },
            .map => switch (self.file.expression(expr_id).*) {
                .ArrayLiteral, .Tuple => blk: {
                    const entries = (try self.evalMapEntries(expr_id)) orelse break :blk null;
                    const heap_id = try self.env.heap.allocMap(entries);
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
        for (item.Enum.variants, 0..) |variant, idx| {
            if (std.mem.eql(u8, variant.name, variant_name)) {
                return CtValue{ .enum_val = CtEnum{
                    .type_id = self.namedTypeId(item_id),
                    .variant_id = @intCast(idx),
                    .payload = null,
                } };
            }
        }
        return null;
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
            .enum_val => |value| switch (rhs) {
                .enum_val => |other| value.type_id == other.type_id and value.variant_id == other.variant_id and value.payload == other.payload,
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

    fn structFieldIndex(self: *ConstEvaluator, type_id: u32, field_name: []const u8) ?usize {
        const item_id = self.itemIdForNamedTypeId(type_id) orelse return null;
        const item = self.file.item(item_id).*;
        if (item != .Struct) return null;
        for (item.Struct.fields, 0..) |field, idx| {
            if (std.mem.eql(u8, field.name, field_name)) return idx;
        }
        return null;
    }

    fn lookupNamedItem(self: *ConstEvaluator, name: []const u8) ?ast.ItemId {
        if (self.current_contract) |contract_id| {
            const contract_item = self.file.item(contract_id).*;
            if (contract_item == .Contract) {
                for (contract_item.Contract.members) |member_id| {
                    if (self.itemName(member_id)) |item_name| {
                        if (std.mem.eql(u8, item_name, name)) return member_id;
                    }
                }
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

    fn ensureTypeChecked(self: *ConstEvaluator, key: model.TypeCheckKey) !void {
        const module_id = self.module_id orelse return;
        const type_query = self.type_query orelse return;
        _ = try type_query.ensure_typecheck(type_query.context, module_id, key);
    }

    fn currentTypeCheckResult(self: *ConstEvaluator) !?*const model.TypeCheckResult {
        const key = self.current_typecheck_key orelse return null;
        const module_id = self.module_id orelse return null;
        const type_query = self.type_query orelse return null;
        return try type_query.ensure_typecheck(type_query.context, module_id, key);
    }

    fn currentModuleTypeCheckResult(self: *ConstEvaluator) !?*const model.TypeCheckResult {
        const module_id = self.module_id orelse return null;
        const type_query = self.type_query orelse return null;
        return try type_query.module_typecheck(type_query.context, module_id);
    }

    fn ensureNamedItemTypeChecked(self: *ConstEvaluator, name: []const u8) !void {
        const item_id = self.lookupNamedItem(name) orelse return;
        try self.ensureTypeChecked(.{ .item = item_id });
    }

    fn astFileForModule(self: *ConstEvaluator, module_id: source.ModuleId) !*const ast.AstFile {
        const type_query = self.type_query orelse return error.MissingTypeQuery;
        return try type_query.ast_file(type_query.context, module_id);
    }

    fn lookupNamedItemInModule(self: *ConstEvaluator, module_id: source.ModuleId, name: []const u8) !?ast.ItemId {
        const type_query = self.type_query orelse return null;
        return try type_query.lookup_item(type_query.context, module_id, name);
    }

    fn resolveImportAlias(self: *ConstEvaluator, alias: []const u8) !?source.ModuleId {
        const module_id = self.module_id orelse return null;
        const type_query = self.type_query orelse return null;
        return try type_query.resolve_import_alias(type_query.context, module_id, alias);
    }

    fn functionRuntimeSelfParameterIndex(self: *ConstEvaluator, function: ast.FunctionItem) ?usize {
        for (function.parameters, 0..) |parameter, index| {
            if (parameter.is_comptime) continue;
            return if (std.mem.eql(u8, self.patternName(parameter.pattern) orelse "", "self")) index else null;
        }
        return null;
    }

    fn typeNameForTypeId(self: *ConstEvaluator, type_id: u32) ?[]const u8 {
        return switch (type_id) {
            type_ids.u8_id => "u8",
            type_ids.u16_id => "u16",
            type_ids.u32_id => "u32",
            type_ids.u64_id => "u64",
            type_ids.u128_id => "u128",
            type_ids.u256_id => "u256",
            type_ids.i8_id => "i8",
            type_ids.i16_id => "i16",
            type_ids.i32_id => "i32",
            type_ids.i64_id => "i64",
            type_ids.i128_id => "i128",
            type_ids.i256_id => "i256",
            type_ids.bool_id => "bool",
            type_ids.address_id => "address",
            type_ids.string_id => "string",
            type_ids.bytes_id => "bytes",
            type_ids.void_id => "void",
            else => if (self.itemIdForNamedTypeId(type_id)) |item_id| self.itemName(item_id) else null,
        };
    }

    fn abiTypeNameForTypeId(self: *ConstEvaluator, type_id: u32) ?[]const u8 {
        return switch (type_id) {
            type_ids.u8_id => "uint8",
            type_ids.u16_id => "uint16",
            type_ids.u32_id => "uint32",
            type_ids.u64_id => "uint64",
            type_ids.u128_id => "uint128",
            type_ids.u256_id => "uint256",
            type_ids.i8_id => "int8",
            type_ids.i16_id => "int16",
            type_ids.i32_id => "int32",
            type_ids.i64_id => "int64",
            type_ids.i128_id => "int128",
            type_ids.i256_id => "int256",
            type_ids.bool_id => "bool",
            type_ids.address_id => "address",
            type_ids.string_id => "string",
            type_ids.bytes_id => "bytes",
            type_ids.void_id => "void",
            else => if (self.itemIdForNamedTypeId(type_id)) |item_id| self.itemName(item_id) else null,
        };
    }

    fn typeByteSizeForTypeId(self: *ConstEvaluator, type_id: u32) ?u256 {
        return switch (type_id) {
            type_ids.u8_id, type_ids.i8_id => 1,
            type_ids.u16_id, type_ids.i16_id => 2,
            type_ids.u32_id, type_ids.i32_id => 4,
            type_ids.u64_id, type_ids.i64_id => 8,
            type_ids.u128_id, type_ids.i128_id => 16,
            type_ids.u256_id, type_ids.i256_id => 32,
            type_ids.bool_id => 1,
            type_ids.address_id => 20,
            type_ids.bytes_id, type_ids.string_id => null,
            type_ids.void_id => 0,
            else => if (self.itemIdForNamedTypeId(type_id)) |item_id|
                self.itemByteSize(item_id)
            else
                null,
        };
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
                if (std.mem.eql(u8, generic.name, "MinValue") or
                    std.mem.eql(u8, generic.name, "MaxValue") or
                    std.mem.eql(u8, generic.name, "InRange") or
                    std.mem.eql(u8, generic.name, "Scaled") or
                    std.mem.eql(u8, generic.name, "Exact") or
                    std.mem.eql(u8, generic.name, "NonZero") or
                    std.mem.eql(u8, generic.name, "NonZeroAddress") or
                    std.mem.eql(u8, generic.name, "BasisPoints"))
                {
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
                if (self.pathTypeId(path.name)) |type_id| break :blk self.abiTypeNameForTypeId(type_id);
                break :blk null;
            },
            .Generic => |generic| blk: {
                if (std.mem.eql(u8, generic.name, "MinValue") or
                    std.mem.eql(u8, generic.name, "MaxValue") or
                    std.mem.eql(u8, generic.name, "InRange") or
                    std.mem.eql(u8, generic.name, "Scaled") or
                    std.mem.eql(u8, generic.name, "Exact") or
                    std.mem.eql(u8, generic.name, "NonZero") or
                    std.mem.eql(u8, generic.name, "NonZeroAddress") or
                    std.mem.eql(u8, generic.name, "BasisPoints"))
                {
                    if (generic.args.len > 0 and generic.args[0] == .Type) break :blk self.typeExprAbiName(generic.args[0].Type);
                }
                var rendered_args: std.ArrayList([]const u8) = .{};
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
                var rendered_elements: std.ArrayList([]const u8) = .{};
                defer rendered_elements.deinit(self.allocator);
                for (tuple.elements) |element| {
                    const name = (try self.typeExprAbiName(element)) orelse break :blk null;
                    try rendered_elements.append(self.allocator, name);
                }
                const joined = try std.mem.join(self.allocator, ",", rendered_elements.items);
                break :blk try std.fmt.allocPrint(self.allocator, "({s})", .{joined});
            },
            .AnonymousStruct => |struct_type| blk: {
                var rendered_fields: std.ArrayList([]const u8) = .{};
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

    fn resolveConcreteTraitMethodCall(self: *ConstEvaluator, field: ast.FieldExpr) ?CallableFunction {
        const typecheck = (self.currentModuleTypeCheckResult() catch return null) orelse return null;
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
            var trait_is_comptime = false;
            for (trait_interface.methods) |trait_method| {
                if (!std.mem.eql(u8, trait_method.name, field.name)) continue;
                trait_is_comptime = trait_method.is_comptime;
                break;
            }
            if (!trait_is_comptime) continue;

            const impl_item = self.file.item(impl_interface.impl_item_id).Impl;
            for (impl_item.methods) |method_item_id| {
                const item = self.file.item(method_item_id).*;
                if (item != .Function) continue;
                if (!std.mem.eql(u8, item.Function.name, field.name)) continue;
                if (matched_impl_item_id != null) return null;
                matched_impl_item_id = impl_interface.impl_item_id;
                matched_method_item_id = method_item_id;
                matched_function = item.Function;
            }
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

    fn lookupCallableFunction(self: *ConstEvaluator, callee: ast.ExprId) ?CallableFunction {
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
                if (switch (self.file.expression(field.base).*) {
                    .Name => |name| name.name,
                    else => null,
                }) |base_name| {
                    if ((self.resolveImportAlias(base_name) catch null)) |target_module_id| {
                        const target_file = self.astFileForModule(target_module_id) catch return null;
                        const function_item_id = (self.lookupNamedItemInModule(target_module_id, field.name) catch return null) orelse return null;
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
                }
                return self.resolveConcreteTraitMethodCall(field);
            },
            .Group => |group| return self.lookupCallableFunction(group.expr),
            else => return null,
        }
    }

    fn itemName(self: *ConstEvaluator, item_id: ast.ItemId) ?[]const u8 {
        return switch (self.file.item(item_id).*) {
            .Contract => |contract| contract.name,
            .Function => |function| function.name,
            .Struct => |struct_item| struct_item.name,
            .Bitfield => |bitfield_item| bitfield_item.name,
            .Enum => |enum_item| enum_item.name,
            .Field => |field| field.name,
            .Constant => |constant| constant.name,
            else => null,
        };
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
            var hash: [32]u8 = undefined;
            std.crypto.hash.sha3.Keccak256.hash(bytes, &hash, .{});
            var text: [64]u8 = undefined;
            for (hash, 0..) |byte, index| {
                text[index * 2] = std.fmt.hex_charset[byte >> 4];
                text[index * 2 + 1] = std.fmt.hex_charset[byte & 0x0f];
            }
            var integer = try std.math.big.int.Managed.init(self.allocator);
            try integer.setString(16, text[0..]);
            return .{ .integer = integer };
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

    fn materializeCallArgumentCtValues(
        self: *ConstEvaluator,
        callable: CallableFunction,
        call: ast.CallExpr,
        comptime use_cache: bool,
    ) anyerror!?[]CtValue {
        const function = callable.function;
        const self_param_index = self.functionRuntimeSelfParameterIndex(function);
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
            arg_values[idx] = (try self.evalCallArgumentCtValue(parameter, arg_expr, use_cache)) orelse return null;
        }
        return arg_values;
    }

    fn bindCallArguments(self: *ConstEvaluator, function: ast.FunctionItem, arg_values: []const CtValue) !void {
        for (function.parameters, 0..) |parameter, idx| {
            try self.bindPatternCtValue(parameter.pattern, arg_values[idx]);
        }
    }

    fn evalCall(self: *ConstEvaluator, call: ast.CallExpr, comptime use_cache: bool) anyerror!?ConstValue {
        const callable = self.lookupCallableFunction(call.callee) orelse {
            _ = try self.evalExprImpl(call.callee, use_cache);
            for (call.args) |arg| _ = try self.evalExprImpl(arg, use_cache);
            return null;
        };
        const function = callable.function;
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

        const arg_values = (try self.materializeCallArgumentCtValues(callable, call, use_cache)) orelse return null;

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
        const callable = self.lookupCallableFunction(call.callee) orelse {
            _ = try self.evalExprImpl(call.callee, use_cache);
            for (call.args) |arg| _ = try self.evalExprImpl(arg, use_cache);
            return null;
        };
        const function = callable.function;
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

        const arg_values = (try self.materializeCallArgumentCtValues(callable, call, use_cache)) orelse return null;

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
                if (try self.evalCallCtValue(call, use_cache)) |ct_value| return ct_value;
                if (self.last_error != null) return null;
            },
            else => {
                if (try self.evalExprCtValueImpl(expr_id, use_cache, true)) |ct_value| return ct_value;
            },
        }

        const const_value = (try self.evalExprImpl(expr_id, use_cache)) orelse return null;
        return (try constToCtValue(const_value)) orelse null;
    }

    fn evalCallArgumentCtValue(self: *ConstEvaluator, parameter: ast.Parameter, arg: ast.ExprId, comptime use_cache: bool) anyerror!?CtValue {
        if (parameter.is_comptime and self.parameterExpectsTypeValue(parameter)) {
            return self.typeExprCtValue(arg);
        }

        _ = try self.evalExprImpl(arg, use_cache);
        return (try self.evalExprAsCtValue(arg, use_cache)) orelse blk: {
            const const_value = (try self.evalExprImpl(arg, use_cache)) orelse return null;
            break :blk (try constToCtValue(const_value)) orelse return null;
        };
    }

    fn parameterExpectsTypeValue(self: *ConstEvaluator, parameter: ast.Parameter) bool {
        return switch (self.file.typeExpr(parameter.type_expr).*) {
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
            else => null,
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
        if (std.mem.eql(u8, trimmed, "u8")) return type_ids.u8_id;
        if (std.mem.eql(u8, trimmed, "u16")) return type_ids.u16_id;
        if (std.mem.eql(u8, trimmed, "u32")) return type_ids.u32_id;
        if (std.mem.eql(u8, trimmed, "u64")) return type_ids.u64_id;
        if (std.mem.eql(u8, trimmed, "u128")) return type_ids.u128_id;
        if (std.mem.eql(u8, trimmed, "u256")) return type_ids.u256_id;
        if (std.mem.eql(u8, trimmed, "i8")) return type_ids.i8_id;
        if (std.mem.eql(u8, trimmed, "i16")) return type_ids.i16_id;
        if (std.mem.eql(u8, trimmed, "i32")) return type_ids.i32_id;
        if (std.mem.eql(u8, trimmed, "i64")) return type_ids.i64_id;
        if (std.mem.eql(u8, trimmed, "i128")) return type_ids.i128_id;
        if (std.mem.eql(u8, trimmed, "i256")) return type_ids.i256_id;
        if (std.mem.eql(u8, trimmed, "bool")) return type_ids.bool_id;
        if (std.mem.eql(u8, trimmed, "address")) return type_ids.address_id;
        if (std.mem.eql(u8, trimmed, "string")) return type_ids.string_id;
        if (std.mem.eql(u8, trimmed, "bytes")) return type_ids.bytes_id;
        if (std.mem.eql(u8, trimmed, "void")) return type_ids.void_id;
        if (self.lookupNamedItem(trimmed)) |item_id| return self.namedTypeId(item_id);
        return null;
    }

    fn namedTypeId(self: *ConstEvaluator, item_id: ast.ItemId) u32 {
        _ = self;
        return named_type_id_base + @as(u32, @intCast(item_id.index()));
    }

    fn itemIdForNamedTypeId(self: *ConstEvaluator, type_id: u32) ?ast.ItemId {
        _ = self;
        if (type_id < named_type_id_base) return null;
        return ast.ItemId.fromIndex(type_id - named_type_id_base);
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
                    const pattern_stage = switch (arm.pattern) {
                        .Expr => |expr_id| self.exprStage(expr_id),
                        .Range => |range_pattern| self.mergeStages(.{
                            self.exprStage(range_pattern.start),
                            self.exprStage(range_pattern.end),
                        }),
                        .Else => .comptime_ok,
                    };
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
                    const pattern_stage = switch (arm.pattern) {
                        .Expr => |pattern_expr| self.exprStage(pattern_expr),
                        .Range => |range_pattern| self.mergeStages(.{
                            self.exprStage(range_pattern.start),
                            self.exprStage(range_pattern.end),
                        }),
                        .Else => .comptime_ok,
                    };
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
        const ct_value = (try constToCtValue(const_value)) orelse return;
        if (self.env.isBoundInCurrentScope(name)) {
            try self.env.set(name, ct_value);
        } else {
            _ = try self.env.bind(name, ct_value);
        }
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
        };
    }

    fn evalComptimeBodyCtValue(self: *ConstEvaluator, body_id: ast.BodyId, comptime use_cache: bool) anyerror!?CtValue {
        return switch (try self.evalComptimeBodyControlCtValue(body_id, use_cache)) {
            .value => |value| value,
            .return_value => |value| value,
            .break_loop, .continue_loop => null,
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
                    }
                },
                .LabeledBlock => |labeled| {
                    switch (try self.evalComptimeBodyControl(labeled.body)) {
                        .value => |value| last_value = value,
                        .return_value => |value| return .{ .return_value = value },
                        .break_loop => return .break_loop,
                        .continue_loop => return .continue_loop,
                    }
                },
                .If => |if_stmt| {
                    switch (try self.evalComptimeIf(if_stmt)) {
                        .value => |value| last_value = value,
                        .return_value => |value| return .{ .return_value = value },
                        .break_loop => return .break_loop,
                        .continue_loop => return .continue_loop,
                    }
                },
                .While => |while_stmt| {
                    switch (try self.evalComptimeWhile(while_stmt)) {
                        .value => |value| last_value = value,
                        .return_value => |value| return .{ .return_value = value },
                        .break_loop => return .break_loop,
                        .continue_loop => return .continue_loop,
                    }
                },
                .For => |for_stmt| {
                    switch (try self.evalComptimeFor(for_stmt)) {
                        .value => |value| last_value = value,
                        .return_value => |value| return .{ .return_value = value },
                        .break_loop => return .break_loop,
                        .continue_loop => return .continue_loop,
                    }
                },
                .Switch => |switch_stmt| {
                    switch (try self.evalComptimeSwitchStmt(switch_stmt)) {
                        .value => |value| last_value = value,
                        .return_value => |value| return .{ .return_value = value },
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
                    last_value = try self.evalExprAsCtValue(expr_stmt.expr, use_cache);
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
                    }
                },
                .LabeledBlock => |labeled| {
                    switch (try self.evalComptimeBodyControlCtValue(labeled.body, use_cache)) {
                        .value => |value| last_value = value,
                        .return_value => |value| return .{ .return_value = value },
                        .break_loop => return .break_loop,
                        .continue_loop => return .continue_loop,
                    }
                },
                .If => |if_stmt| {
                    switch (try self.evalComptimeIfCtValue(if_stmt, use_cache)) {
                        .value => |value| last_value = value,
                        .return_value => |value| return .{ .return_value = value },
                        .break_loop => return .break_loop,
                        .continue_loop => return .continue_loop,
                    }
                },
                .While => |while_stmt| {
                    switch (try self.evalComptimeWhileCtValue(while_stmt, use_cache)) {
                        .value => |value| last_value = value,
                        .return_value => |value| return .{ .return_value = value },
                        .break_loop => return .break_loop,
                        .continue_loop => return .continue_loop,
                    }
                },
                .For => |for_stmt| {
                    switch (try self.evalComptimeForCtValue(for_stmt, use_cache)) {
                        .value => |value| last_value = value,
                        .return_value => |value| return .{ .return_value = value },
                        .break_loop => return .break_loop,
                        .continue_loop => return .continue_loop,
                    }
                },
                .Switch => |switch_stmt| {
                    switch (try self.evalComptimeSwitchStmtCtValue(switch_stmt, use_cache)) {
                        .value => |value| last_value = value,
                        .return_value => |value| return .{ .return_value = value },
                        .break_loop => return .break_loop,
                        .continue_loop => return .continue_loop,
                    }
                },
                .Assign => |assign| {
                    const value = (try self.evalComptimeAssign(assign)) orelse return .{ .value = null };
                    last_value = (try constToCtValue(value)) orelse null;
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

    fn evalComptimeIfCtValue(self: *ConstEvaluator, if_stmt: ast.IfStmt, comptime use_cache: bool) anyerror!CtBodyControl {
        const condition = (try self.evalExprUncached(if_stmt.condition)) orelse return .{ .value = null };
        const take_then = self.constConditionTruthy(condition) orelse return .{ .value = null };
        if (take_then) return try self.evalComptimeBodyControlCtValue(if_stmt.then_body, use_cache);
        if (if_stmt.else_body) |else_body| return try self.evalComptimeBodyControlCtValue(else_body, use_cache);
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

    fn evalComptimeSwitchStmtCtValue(self: *ConstEvaluator, switch_stmt: ast.SwitchStmt, comptime use_cache: bool) anyerror!CtBodyControl {
        const condition = (try self.evalExprUncached(switch_stmt.condition)) orelse return .{ .value = null };
        for (switch_stmt.arms) |arm| {
            if (self.patternMatches(condition, arm.pattern)) {
                return try self.evalComptimeBodyControlCtValue(arm.body, use_cache);
            }
        }
        if (switch_stmt.else_body) |else_body| return try self.evalComptimeBodyControlCtValue(else_body, use_cache);
        return .{ .value = null };
    }

    fn evalComptimeWhile(self: *ConstEvaluator, while_stmt: ast.WhileStmt) anyerror!BodyControl {
        var iterations: u64 = 0;
        var last_value: ?ConstValue = null;
        while (true) {
            iterations += 1;
            if (iterations > self.env.config.max_loop_iterations) {
                self.recordLoopLimitExceeded(while_stmt.range);
                return .{ .value = null };
            }

            const condition = (try self.evalExprUncached(while_stmt.condition)) orelse return .{ .value = null };
            const should_continue = self.constConditionTruthy(condition) orelse return .{ .value = null };
            if (!should_continue) break;

            switch (try self.evalComptimeBodyControl(while_stmt.body)) {
                .value => |value| last_value = value,
                .return_value => |value| return .{ .return_value = value },
                .break_loop => break,
                .continue_loop => continue,
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
                return .{ .value = null };
            }

            const condition = (try self.evalExprUncached(while_stmt.condition)) orelse return .{ .value = null };
            const should_continue = self.constConditionTruthy(condition) orelse return .{ .value = null };
            if (!should_continue) break;

            switch (try self.evalComptimeBodyControlCtValue(while_stmt.body, use_cache)) {
                .value => |value| last_value = value,
                .return_value => |value| return .{ .return_value = value },
                .break_loop => break,
                .continue_loop => continue,
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
                const heap_id = try self.env.heap.allocArray(elems);
                break :blk CtValue{ .array_ref = heap_id };
            },
            .Tuple => |tuple| blk: {
                const elems = try self.allocator.alloc(CtValue, tuple.elements.len);
                for (tuple.elements, 0..) |element_id, idx| {
                    elems[idx] = (try self.evalIterableCtValue(element_id)) orelse break :blk null;
                }
                const heap_id = try self.env.heap.allocTuple(elems);
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
                break :blk try constToCtValue(value);
            },
        };
    }

    fn evalComptimeAssign(self: *ConstEvaluator, assign: ast.AssignStmt) anyerror!?ConstValue {
        const rhs_const = try self.evalExprUncached(assign.value);
        const rhs_ct = (try self.evalExprCtValue(assign.value)) orelse blk: {
            const rhs = rhs_const orelse break :blk null;
            break :blk (try constToCtValue(rhs)) orelse break :blk null;
        } orelse return null;
        switch (self.file.pattern(assign.target).*) {
            .Name => |name| {
                const rhs = rhs_const orelse return null;
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
                const ct_value = (try constToCtValue(value)) orelse return null;
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
                    const field_index = self.structFieldIndex(struct_data.type_id, field.name) orelse return null;
                    if (field_index >= struct_data.fields.len) return null;
                    try self.bindPatternCtValue(field.binding, struct_data.fields[field_index].value);
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
                const index_value = (try self.evalExprCtValue(index.index)) orelse return null;
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
                                const computed = switch (assign.op) {
                                    .add_assign => try evalBinary(self.allocator, .add, current, rhs),
                                    .sub_assign => try evalBinary(self.allocator, .sub, current, rhs),
                                    .mul_assign => try evalBinary(self.allocator, .mul, current, rhs),
                                    .div_assign => try evalBinary(self.allocator, .div, current, rhs),
                                    .mod_assign => try evalBinary(self.allocator, .mod, current, rhs),
                                    .bit_and_assign => try evalBinary(self.allocator, .bit_and, current, rhs),
                                    .bit_or_assign => try evalBinary(self.allocator, .bit_or, current, rhs),
                                    .bit_xor_assign => try evalBinary(self.allocator, .bit_xor, current, rhs),
                                    .shl_assign => try evalBinary(self.allocator, .shl, current, rhs),
                                    .shr_assign => try evalBinary(self.allocator, .shr, current, rhs),
                                    .pow_assign => try evalBinary(self.allocator, .pow, current, rhs),
                                    .wrapping_add_assign => try evalBinary(self.allocator, .wrapping_add, current, rhs),
                                    .wrapping_sub_assign => try evalBinary(self.allocator, .wrapping_sub, current, rhs),
                                    .wrapping_mul_assign => try evalBinary(self.allocator, .wrapping_mul, current, rhs),
                                    .assign => unreachable,
                                } orelse break :blk_op null;
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
                                const computed = switch (assign.op) {
                                    .add_assign => try evalBinary(self.allocator, .add, current, rhs),
                                    .sub_assign => try evalBinary(self.allocator, .sub, current, rhs),
                                    .mul_assign => try evalBinary(self.allocator, .mul, current, rhs),
                                    .div_assign => try evalBinary(self.allocator, .div, current, rhs),
                                    .mod_assign => try evalBinary(self.allocator, .mod, current, rhs),
                                    .bit_and_assign => try evalBinary(self.allocator, .bit_and, current, rhs),
                                    .bit_or_assign => try evalBinary(self.allocator, .bit_or, current, rhs),
                                    .bit_xor_assign => try evalBinary(self.allocator, .bit_xor, current, rhs),
                                    .shl_assign => try evalBinary(self.allocator, .shl, current, rhs),
                                    .shr_assign => try evalBinary(self.allocator, .shr, current, rhs),
                                    .pow_assign => try evalBinary(self.allocator, .pow, current, rhs),
                                    .wrapping_add_assign => try evalBinary(self.allocator, .wrapping_add, current, rhs),
                                    .wrapping_sub_assign => try evalBinary(self.allocator, .wrapping_sub, current, rhs),
                                    .wrapping_mul_assign => try evalBinary(self.allocator, .wrapping_mul, current, rhs),
                                    .assign => unreachable,
                                } orelse break :blk_op null;
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
                                const computed = switch (assign.op) {
                                    .add_assign => try evalBinary(self.allocator, .add, current_value, rhs),
                                    .sub_assign => try evalBinary(self.allocator, .sub, current_value, rhs),
                                    .mul_assign => try evalBinary(self.allocator, .mul, current_value, rhs),
                                    .div_assign => try evalBinary(self.allocator, .div, current_value, rhs),
                                    .mod_assign => try evalBinary(self.allocator, .mod, current_value, rhs),
                                    .bit_and_assign => try evalBinary(self.allocator, .bit_and, current_value, rhs),
                                    .bit_or_assign => try evalBinary(self.allocator, .bit_or, current_value, rhs),
                                    .bit_xor_assign => try evalBinary(self.allocator, .bit_xor, current_value, rhs),
                                    .shl_assign => try evalBinary(self.allocator, .shl, current_value, rhs),
                                    .shr_assign => try evalBinary(self.allocator, .shr, current_value, rhs),
                                    .pow_assign => try evalBinary(self.allocator, .pow, current_value, rhs),
                                    .wrapping_add_assign => try evalBinary(self.allocator, .wrapping_add, current_value, rhs),
                                    .wrapping_sub_assign => try evalBinary(self.allocator, .wrapping_sub, current_value, rhs),
                                    .wrapping_mul_assign => try evalBinary(self.allocator, .wrapping_mul, current_value, rhs),
                                    .assign => unreachable,
                                } orelse break :blk_op null;
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
                    else => unreachable,
                });
            },
            .Field => |field| {
                const rhs = rhs_const orelse return null;
                const base_name = switch (self.file.pattern(field.base).*) {
                    .Name => |name| name.name,
                    else => return null,
                };
                const base_slot = self.env.lookup(base_name) orelse return null;
                const base_value = self.env.read(base_slot);

                const updated = switch (base_value) {
                    .struct_ref => |heap_id| blk: {
                        const struct_data = self.env.heap.getStruct(heap_id);
                        const field_index = self.structFieldIndex(struct_data.type_id, field.name) orelse break :blk null;
                        if (field_index >= struct_data.fields.len) break :blk null;

                        const next_value = switch (assign.op) {
                            .assign => rhs_ct,
                            else => blk_op: {
                                const current = (try ctValueToConstValue(self.allocator, &self.env.heap, struct_data.fields[field_index].value)) orelse break :blk_op null;
                                const computed = switch (assign.op) {
                                    .add_assign => try evalBinary(self.allocator, .add, current, rhs),
                                    .sub_assign => try evalBinary(self.allocator, .sub, current, rhs),
                                    .mul_assign => try evalBinary(self.allocator, .mul, current, rhs),
                                    .div_assign => try evalBinary(self.allocator, .div, current, rhs),
                                    .mod_assign => try evalBinary(self.allocator, .mod, current, rhs),
                                    .bit_and_assign => try evalBinary(self.allocator, .bit_and, current, rhs),
                                    .bit_or_assign => try evalBinary(self.allocator, .bit_or, current, rhs),
                                    .bit_xor_assign => try evalBinary(self.allocator, .bit_xor, current, rhs),
                                    .shl_assign => try evalBinary(self.allocator, .shl, current, rhs),
                                    .shr_assign => try evalBinary(self.allocator, .shr, current, rhs),
                                    .pow_assign => try evalBinary(self.allocator, .pow, current, rhs),
                                    .wrapping_add_assign => try evalBinary(self.allocator, .wrapping_add, current, rhs),
                                    .wrapping_sub_assign => try evalBinary(self.allocator, .wrapping_sub, current, rhs),
                                    .wrapping_mul_assign => try evalBinary(self.allocator, .wrapping_mul, current, rhs),
                                    .assign => unreachable,
                                } orelse break :blk_op null;
                                break :blk_op (try constToCtValue(computed)) orelse break :blk_op null;
                            },
                        } orelse return null;

                        break :blk CtValue{ .struct_ref = try self.env.heap.setStructField(heap_id, @intCast(field_index), next_value) };
                    },
                    else => return null,
                } orelse return null;

                self.env.update(base_slot, updated);
                return try ctValueToConstValue(self.allocator, &self.env.heap, switch (updated) {
                    .struct_ref => |heap_id| blk: {
                        const struct_data = self.env.heap.getStruct(heap_id);
                        const field_index = self.structFieldIndex(struct_data.type_id, field.name) orelse return null;
                        break :blk struct_data.fields[field_index].value;
                    },
                    else => unreachable,
                });
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

    fn typeExprIntegerType(self: *ConstEvaluator, type_expr_id: ast.TypeExprId) ?model.IntegerType {
        return switch (self.file.typeExpr(type_expr_id).*) {
            .Path => |path| integerTypeFromName(path.name),
            else => null,
        };
    }

    fn integerTypeFromName(name: []const u8) ?model.IntegerType {
        const trimmed = std.mem.trim(u8, name, " \t\n\r");
        if (trimmed.len < 2) return null;
        const signed = switch (trimmed[0]) {
            'u' => false,
            'i' => true,
            else => return null,
        };
        const bits = std.fmt.parseInt(u16, trimmed[1..], 10) catch return null;
        return .{ .bits = bits, .signed = signed, .spelling = trimmed };
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
