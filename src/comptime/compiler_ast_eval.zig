const std = @import("std");
const ast = @import("../compiler/ast/mod.zig");
const bridge = @import("compiler_const_bridge.zig");
const comptime_mod = @import("mod.zig");
const diagnostics = @import("../compiler/diagnostics/mod.zig");
const stage_mod = @import("stage.zig");
const model = @import("../compiler/sema/model.zig");
const source = @import("../compiler/source/mod.zig");
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
const constEquals = bridge.constEquals;
const ctValueToConstValue = bridge.ctValueToConstValue;
const constToCtValue = bridge.constToCtValue;
const evalBinary = bridge.evalBinary;
const evalUnary = bridge.evalUnary;
const parseIntegerLiteral = bridge.parseIntegerLiteral;

pub const TypeQuery = struct {
    context: *anyopaque,
    ensure_typecheck: *const fn (context: *anyopaque, module_id: source.ModuleId, key: model.TypeCheckKey) anyerror!*const model.TypeCheckResult,
};

pub const ConstEvalOptions = struct {
    module_id: ?source.ModuleId = null,
    type_query: ?TypeQuery = null,
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
        .env = CtEnv.init(arena, .{}),
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
    call_depth: u32 = 0,
    max_call_depth: u32 = 64,
    last_error: ?error_mod.CtError = null,

    const BodyControl = union(enum) {
        value: ?ConstValue,
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
        const previous_key = self.current_typecheck_key;
        self.current_typecheck_key = .{ .body = body_id };
        defer self.current_typecheck_key = previous_key;

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

        if (try self.evalExprCtValue(expr_id)) |ct_value| {
            const const_value = try ctValueToConstValue(self.allocator, ct_value);
            if (const_value != null) {
                if (use_cache) self.values[expr_id.index()] = const_value;
                return const_value;
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
                break :blk try self.evalCall(call, use_cache);
            },
            .Builtin => |builtin| blk: {
                if (self.exprStage(expr_id) == .runtime_only) {
                    self.last_error = error_mod.CtError.stageViolation(
                        self.sourceSpan(builtin.range),
                        builtin.name,
                    );
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

    fn evalExprCtValue(self: *ConstEvaluator, expr_id: ast.ExprId) anyerror!?CtValue {
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
            .Name => |name| self.env.lookupValue(name.name),
            .Group => |group| try self.evalExprCtValue(group.expr),
            .ArrayLiteral => |array| blk: {
                const elems = try self.allocator.alloc(CtValue, array.elements.len);
                for (array.elements, 0..) |element_id, idx| {
                    _ = try self.evalExpr(element_id);
                    elems[idx] = (try self.evalExprCtValue(element_id)) orelse break :blk null;
                }
                const heap_id = try self.env.heap.allocArray(elems);
                break :blk CtValue{ .array_ref = heap_id };
            },
            .Tuple => |tuple| blk: {
                const elems = try self.allocator.alloc(CtValue, tuple.elements.len);
                for (tuple.elements, 0..) |element_id, idx| {
                    _ = try self.evalExpr(element_id);
                    elems[idx] = (try self.evalExprCtValue(element_id)) orelse break :blk null;
                }
                const heap_id = try self.env.heap.allocTuple(elems);
                break :blk CtValue{ .tuple_ref = heap_id };
            },
            .StructLiteral => |struct_literal| blk: {
                const fields = try self.allocator.alloc(CtAggregate.StructField, struct_literal.fields.len);
                for (struct_literal.fields, 0..) |field, idx| {
                    _ = try self.evalExpr(field.value);
                    fields[idx] = .{
                        .field_id = @intCast(idx),
                        .value = (try self.evalExprCtValue(field.value)) orelse break :blk null,
                    };
                }
                const type_id = (try self.structTypeIdForExpr(expr_id, struct_literal)) orelse break :blk null;
                const heap_id = try self.env.heap.allocStruct(type_id, fields);
                break :blk CtValue{ .struct_ref = heap_id };
            },
            .Index => |index| blk: {
                _ = try self.evalExpr(index.base);
                _ = try self.evalExpr(index.index);
                const base = (try self.evalExprCtValue(index.base)) orelse break :blk null;
                const index_value = (try self.evalExprCtValue(index.index)) orelse break :blk null;

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
                        break :blk try self.evalExprCtValueAs(builtin.args[0], target);
                    }
                }
                break :blk null;
            },
            .Field => |field| blk: {
                _ = try self.evalExpr(field.base);
                switch (self.file.expression(field.base).*) {
                    .Name => |name| {
                        if (self.lookupNamedEnumVariant(name.name, field.name)) |enum_value| break :blk enum_value;
                    },
                    else => {},
                }
                const base = (try self.evalExprCtValue(field.base)) orelse break :blk null;
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
                const lhs = (try self.evalExprCtValue(binary.lhs)) orelse break :blk null;
                const rhs = (try self.evalExprCtValue(binary.rhs)) orelse break :blk null;
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

    fn evalExprCtValueAs(self: *ConstEvaluator, expr_id: ast.ExprId, target: ValueConstructionTarget) anyerror!?CtValue {
        return switch (target) {
            .none => try self.evalExprCtValue(expr_id),
            .slice => switch (self.file.expression(expr_id).*) {
                .ArrayLiteral => |array| blk: {
                    const elems = try self.allocator.alloc(CtValue, array.elements.len);
                    for (array.elements, 0..) |element_id, idx| {
                        _ = try self.evalExpr(element_id);
                        elems[idx] = (try self.evalExprCtValue(element_id)) orelse break :blk null;
                    }
                    const heap_id = try self.env.heap.allocSlice(elems);
                    break :blk CtValue{ .slice_ref = heap_id };
                },
                else => try self.evalExprCtValue(expr_id),
            },
            .map => switch (self.file.expression(expr_id).*) {
                .ArrayLiteral, .Tuple => blk: {
                    const entries = (try self.evalMapEntries(expr_id)) orelse break :blk null;
                    const heap_id = try self.env.heap.allocMap(entries);
                    break :blk CtValue{ .map_ref = heap_id };
                },
                else => try self.evalExprCtValue(expr_id),
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
        return @intCast(item_id.index());
    }

    fn structTypeIdForExpr(self: *ConstEvaluator, expr_id: ast.ExprId, struct_literal: ast.StructLiteralExpr) !?u32 {
        if (try self.currentTypeCheckResult()) |typecheck| {
            const expr_type = typecheck.exprType(expr_id);
            if (expr_type == .struct_) {
                if (self.lookupNamedItem(expr_type.struct_.name)) |item_id| {
                    if (self.file.item(item_id).* == .Struct) return @intCast(item_id.index());
                }
                if (typecheck.instantiatedStructByName(expr_type.struct_.name)) |instantiated| {
                    return @intCast(instantiated.template_item_id.index());
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
                    .type_id = @intCast(item_id.index()),
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
        const item_id = ast.ItemId.fromIndex(type_id);
        const item = self.file.item(item_id).*;
        if (item != .Struct) return null;
        for (item.Struct.fields, 0..) |field, idx| {
            if (std.mem.eql(u8, field.name, field_name)) return idx;
        }
        return null;
    }

    fn lookupNamedItem(self: *ConstEvaluator, name: []const u8) ?ast.ItemId {
        for (self.file.root_items) |item_id| {
            if (self.itemName(item_id)) |item_name| {
                if (std.mem.eql(u8, item_name, name)) return item_id;
            }
        }
        return null;
    }

    const CallableFunction = struct {
        item_id: ast.ItemId,
        function: ast.FunctionItem,
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

    fn ensureNamedItemTypeChecked(self: *ConstEvaluator, name: []const u8) !void {
        const item_id = self.lookupNamedItem(name) orelse return;
        try self.ensureTypeChecked(.{ .item = item_id });
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
        const function_item_id = switch (self.file.expression(callee).*) {
            .Name => |name| self.lookupNamedItem(name.name) orelse return null,
            .Group => |group| return self.lookupCallableFunction(group.expr),
            else => return null,
        };

        const item = self.file.item(function_item_id).*;
        if (item != .Function) return null;
        return .{ .item_id = function_item_id, .function = item.Function };
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
        if (builtin.args.len == 0) return null;

        if (std.mem.eql(u8, builtin.name, "cast")) {
            if (builtin.type_arg) |type_arg| {
                const target = self.valueConstructionTarget(type_arg);
                if (target != .none) {
                    if (try self.evalExprCtValueAs(builtin.args[0], target)) |ct_value| {
                        return try ctValueToConstValue(self.allocator, ct_value);
                    }
                }
            }
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

    fn evalCall(self: *ConstEvaluator, call: ast.CallExpr, comptime use_cache: bool) anyerror!?ConstValue {
        const callable = self.lookupCallableFunction(call.callee) orelse {
            _ = try self.evalExprImpl(call.callee, use_cache);
            for (call.args) |arg| _ = try self.evalExprImpl(arg, use_cache);
            return null;
        };
        const function = callable.function;
        try self.ensureTypeChecked(.{ .item = callable.item_id });

        if (self.functionStage(function) == .runtime_only) {
            self.last_error = error_mod.CtError.stageViolation(
                self.sourceSpan(call.range),
                function.name,
            );
            return null;
        }

        if (function.parameters.len != call.args.len) return null;
        if (self.call_depth >= self.max_call_depth) {
            self.last_error = error_mod.CtError.init(
                .recursion_limit,
                self.sourceSpan(call.range),
                "comptime recursion depth exceeded",
            );
            return null;
        }

        var arg_values = try self.allocator.alloc(CtValue, call.args.len);
        for (call.args, function.parameters, 0..) |arg, parameter, idx| {
            arg_values[idx] = (try self.evalCallArgumentCtValue(parameter, arg, use_cache)) orelse return null;
        }

        self.env.pushScope(false) catch return null;
        defer self.env.popScope();

        for (function.parameters, 0..) |parameter, idx| {
            try self.bindPatternCtValue(parameter.pattern, arg_values[idx]);
        }

        self.call_depth += 1;
        defer self.call_depth -= 1;

        for (function.clauses) |clause| {
            if (clause.kind != .requires) continue;
            const condition = (try self.evalExprUncached(clause.expr)) orelse return null;
            const truthy = self.constConditionTruthy(condition) orelse return null;
            if (!truthy) return null;
        }

        return try self.evalComptimeBody(function.body);
    }

    fn evalCallCtValue(self: *ConstEvaluator, call: ast.CallExpr, comptime use_cache: bool) anyerror!?CtValue {
        const callable = self.lookupCallableFunction(call.callee) orelse {
            _ = try self.evalExprImpl(call.callee, use_cache);
            for (call.args) |arg| _ = try self.evalExprImpl(arg, use_cache);
            return null;
        };
        const function = callable.function;
        try self.ensureTypeChecked(.{ .item = callable.item_id });

        if (self.functionStage(function) == .runtime_only) {
            self.last_error = error_mod.CtError.stageViolation(
                self.sourceSpan(call.range),
                function.name,
            );
            return null;
        }

        if (function.parameters.len != call.args.len) return null;
        if (self.call_depth >= self.max_call_depth) {
            self.last_error = error_mod.CtError.init(
                .recursion_limit,
                self.sourceSpan(call.range),
                "comptime recursion depth exceeded",
            );
            return null;
        }

        var arg_values = try self.allocator.alloc(CtValue, call.args.len);
        for (call.args, function.parameters, 0..) |arg, parameter, idx| {
            arg_values[idx] = (try self.evalCallArgumentCtValue(parameter, arg, use_cache)) orelse return null;
        }

        self.env.pushScope(false) catch return null;
        defer self.env.popScope();

        for (function.parameters, 0..) |parameter, idx| {
            try self.bindPatternCtValue(parameter.pattern, arg_values[idx]);
        }

        self.call_depth += 1;
        defer self.call_depth -= 1;

        for (function.clauses) |clause| {
            if (clause.kind != .requires) continue;
            const condition = (try self.evalExprUncached(clause.expr)) orelse return null;
            const truthy = self.constConditionTruthy(condition) orelse return null;
            if (!truthy) return null;
        }

        const body = self.file.body(function.body).*;
        if (body.statements.len == 1) {
            switch (self.file.statement(body.statements[0]).*) {
                .Return => |ret| if (ret.value) |ret_value| {
                    _ = try self.evalExprImpl(ret_value, use_cache);
                    return (try self.evalExprCtValue(ret_value)) orelse blk: {
                        const const_value = (try self.evalExprImpl(ret_value, use_cache)) orelse return null;
                        break :blk (try constToCtValue(const_value)) orelse return null;
                    };
                },
                else => {},
            }
        }

        return null;
    }

    fn evalCallArgumentCtValue(self: *ConstEvaluator, parameter: ast.Parameter, arg: ast.ExprId, comptime use_cache: bool) anyerror!?CtValue {
        if (parameter.is_comptime and self.parameterExpectsTypeValue(parameter)) {
            return self.typeExprCtValue(arg);
        }

        _ = try self.evalExprImpl(arg, use_cache);
        return (try self.evalExprCtValue(arg)) orelse blk: {
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
        if (self.lookupNamedItem(trimmed)) |item_id| return @intCast(item_id.index());
        return null;
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

    fn bindName(self: *ConstEvaluator, name: []const u8, value: ?ConstValue) !void {
        const const_value = value orelse return;
        const ct_value = (try constToCtValue(const_value)) orelse return;
        try self.env.set(name, ct_value);
    }

    fn bindNameCtValue(self: *ConstEvaluator, name: []const u8, value: ?CtValue) !void {
        const ct_value = value orelse return;
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
            .break_loop, .continue_loop => null,
        };
    }

    fn evalComptimeBodyControl(self: *ConstEvaluator, body_id: ast.BodyId) anyerror!BodyControl {
        self.env.pushScope(false) catch return .{ .value = null };
        defer self.env.popScope();

        const body = self.file.body(body_id).*;
        var last_value: ?ConstValue = null;
        for (body.statements) |statement_id| {
            if (self.statementStage(statement_id) == .runtime_only) {
                self.last_error = error_mod.CtError.stageViolation(
                    self.sourceSpan(self.statementRange(statement_id)),
                    "runtime-only statement",
                );
                return .{ .value = null };
            }
            switch (self.file.statement(statement_id).*) {
                .VariableDecl => |decl| {
                    if (decl.value) |expr_id| {
                        switch (self.file.expression(expr_id).*) {
                            .Call => |call| if (try self.evalCallCtValue(call, false)) |ct_value| {
                                try self.bindPatternCtValue(decl.pattern, ct_value);
                                self.values[expr_id.index()] = try ctValueToConstValue(self.allocator, ct_value);
                                last_value = null;
                                continue;
                            },
                            else => {},
                        }
                        if (try self.evalExprCtValue(expr_id)) |ct_value| {
                            try self.bindPatternCtValue(decl.pattern, ct_value);
                            const persisted = (try ctValueToConstValue(self.allocator, ct_value)) orelse try self.evalExprUncached(expr_id);
                            self.values[expr_id.index()] = persisted;
                        } else {
                            const persisted = try self.evalExprUncached(expr_id);
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
        const iterable = (try self.evalIterableCtValue(for_stmt.iterable)) orelse return null;
        var last_value: ?ConstValue = null;

        switch (iterable) {
            .integer => |integer| {
                if (integer > std.math.maxInt(usize)) return null;
                const trip_count: usize = @intCast(integer);
                var iteration: usize = 0;
                while (iteration < trip_count) : (iteration += 1) {
                    if (iteration >= self.env.config.max_loop_iterations) return null;

                    const item_value = CtValue{ .integer = @intCast(iteration) };
                    try self.bindPatternCtValue(for_stmt.item_pattern, item_value);

                    if (for_stmt.index_pattern) |index_pattern| {
                        const index_value = CtValue{ .integer = @intCast(iteration) };
                        try self.bindPatternCtValue(index_pattern, index_value);
                    }

                    switch (try self.evalComptimeBodyControl(for_stmt.body)) {
                        .value => |value| last_value = value,
                        .break_loop => break,
                        .continue_loop => continue,
                    }
                }
            },
            .array_ref => |heap_id| {
                const elems = self.env.heap.getArray(heap_id).elems;
                for (elems, 0..) |elem, iteration| {
                    if (iteration >= self.env.config.max_loop_iterations) return null;

                    try self.bindPatternCtValue(for_stmt.item_pattern, elem);
                    if (for_stmt.index_pattern) |index_pattern| {
                        const index_value = CtValue{ .integer = @intCast(iteration) };
                        try self.bindPatternCtValue(index_pattern, index_value);
                    }

                    switch (try self.evalComptimeBodyControl(for_stmt.body)) {
                        .value => |value| last_value = value,
                        .break_loop => break,
                        .continue_loop => continue,
                    }
                }
            },
            .slice_ref => |heap_id| {
                const elems = self.env.heap.getSlice(heap_id).elems;
                for (elems, 0..) |elem, iteration| {
                    if (iteration >= self.env.config.max_loop_iterations) return null;

                    try self.bindPatternCtValue(for_stmt.item_pattern, elem);
                    if (for_stmt.index_pattern) |index_pattern| {
                        const index_value = CtValue{ .integer = @intCast(iteration) };
                        try self.bindPatternCtValue(index_pattern, index_value);
                    }

                    switch (try self.evalComptimeBodyControl(for_stmt.body)) {
                        .value => |value| last_value = value,
                        .break_loop => break,
                        .continue_loop => continue,
                    }
                }
            },
            .tuple_ref => |heap_id| {
                const elems = self.env.heap.getTuple(heap_id).elems;
                for (elems, 0..) |elem, iteration| {
                    if (iteration >= self.env.config.max_loop_iterations) return null;

                    try self.bindPatternCtValue(for_stmt.item_pattern, elem);
                    if (for_stmt.index_pattern) |index_pattern| {
                        const index_value = CtValue{ .integer = @intCast(iteration) };
                        try self.bindPatternCtValue(index_pattern, index_value);
                    }

                    switch (try self.evalComptimeBodyControl(for_stmt.body)) {
                        .value => |value| last_value = value,
                        .break_loop => break,
                        .continue_loop => continue,
                    }
                }
            },
            else => return null,
        }
        return last_value;
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
        const rhs = (try self.evalExprUncached(assign.value)) orelse return null;
        const rhs_ct = (try self.evalExprCtValue(assign.value)) orelse (try constToCtValue(rhs)) orelse return null;
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
            .Index => |index| {
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
                                const current = (try ctValueToConstValue(self.allocator, elems[idx])) orelse break :blk_op null;
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
                                const current = (try ctValueToConstValue(self.allocator, elems[idx])) orelse break :blk_op null;
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
                                    break :blk_current try ctValueToConstValue(self.allocator, entry.value);
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
                return try ctValueToConstValue(self.allocator, switch (updated) {
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
                                const current = (try ctValueToConstValue(self.allocator, struct_data.fields[field_index].value)) orelse break :blk_op null;
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
                return try ctValueToConstValue(self.allocator, switch (updated) {
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
