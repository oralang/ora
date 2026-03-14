const std = @import("std");
const ast = @import("../ast/mod.zig");
const const_bridge = @import("../../comptime/compiler_const_bridge.zig");
const diagnostics = @import("../diagnostics/mod.zig");
const model = @import("model.zig");
const source = @import("../source/mod.zig");
const descriptors = @import("type_descriptors.zig");
const region_rules = @import("region.zig");

const ItemIndexResult = model.ItemIndexResult;
const NameResolutionResult = model.NameResolutionResult;
const ResolvedBinding = model.ResolvedBinding;
const TypeCheckKey = model.TypeCheckKey;
const TypeCheckResult = model.TypeCheckResult;
const Type = model.Type;
const LocatedType = model.LocatedType;
const Region = model.Region;
const Effect = model.Effect;
const ConstEvalResult = model.ConstEvalResult;
const ConstValue = model.ConstValue;
const BigInt = std.math.big.int.Managed;
const descriptorFromTypeExpr = descriptors.descriptorFromTypeExpr;
const inferItemType = descriptors.inferItemType;
const mergeExprType = descriptors.mergeExprType;
const typeEql = descriptors.typeEql;

fn declarationRegion(storage_class: ast.StorageClass) Region {
    return switch (storage_class) {
        .none => .none,
        .storage => .storage,
        .memory => .memory,
        .tstore => .transient,
    };
}

pub fn typeCheck(
    allocator: std.mem.Allocator,
    file_id: source.FileId,
    file: *const ast.AstFile,
    item_index: *const ItemIndexResult,
    resolution: *const NameResolutionResult,
    const_eval: *const ConstEvalResult,
    key: TypeCheckKey,
) !TypeCheckResult {
    var result = TypeCheckResult{
        .arena = std.heap.ArenaAllocator.init(allocator),
        .key = key,
        .item_types = &.{},
        .item_regions = &.{},
        .item_effects = &.{},
        .pattern_types = &.{},
        .expr_types = &.{},
        .body_types = &.{},
        .diagnostics = diagnostics.DiagnosticList.init(allocator),
    };
    errdefer result.deinit();

    const arena = result.arena.allocator();
    var item_types = try arena.alloc(Type, file.items.len);
    var item_regions = try arena.alloc(Region, file.items.len);
    const item_effects = try arena.alloc(Effect, file.items.len);
    var pattern_types = try arena.alloc(LocatedType, file.patterns.len);
    const expr_types = try arena.alloc(Type, file.expressions.len);
    var body_types = try arena.alloc(Type, file.bodies.len);
    @memset(item_types, .{ .unknown = {} });
    @memset(item_regions, .none);
    @memset(item_effects, .pure);
    @memset(pattern_types, LocatedType.unlocated(.{ .unknown = {} }));
    @memset(expr_types, .{ .unknown = {} });
    @memset(body_types, .{ .void = {} });

    for (file.items, 0..) |item, index| {
        item_types[index] = try inferItemType(arena, file, item_index, item);
        switch (item) {
            .Function => |function| {
                for (function.parameters) |parameter| {
                    pattern_types[parameter.pattern.index()] = LocatedType.withRegion(
                        try descriptorFromTypeExpr(arena, file, item_index, parameter.type_expr),
                        .calldata,
                    );
                }
                body_types[function.body.index()] = if (function.return_type) |type_expr| try descriptorFromTypeExpr(arena, file, item_index, type_expr) else .{ .void = {} };
            },
            .Field => |field| {
                item_regions[index] = declarationRegion(field.storage_class);
                if (field.type_expr) |type_expr| item_types[index] = try descriptorFromTypeExpr(arena, file, item_index, type_expr);
            },
            .Constant => |constant| {
                if (constant.type_expr) |type_expr| item_types[index] = try descriptorFromTypeExpr(arena, file, item_index, type_expr);
            },
            else => {},
        }
    }

    for (file.statements) |statement| {
        if (statement == .VariableDecl) {
            const decl = statement.VariableDecl;
            if (decl.type_expr) |type_expr| {
                pattern_types[decl.pattern.index()] = LocatedType.unlocated(
                    try descriptorFromTypeExpr(arena, file, item_index, type_expr),
                );
            }
        }
    }

    var typechecker = TypeChecker{
        .arena = arena,
        .file_id = file_id,
        .file = file,
        .item_index = item_index,
        .resolution = resolution,
        .const_eval = const_eval,
        .item_types = item_types,
        .item_regions = item_regions,
        .item_effects = item_effects,
        .pattern_types = pattern_types,
        .expr_types = expr_types,
        .diagnostics = &result.diagnostics,
    };

    try result.diagnostics.appendList(&const_eval.diagnostics);

    for (file.root_items) |item_id| {
        try typechecker.visitItem(item_id);
    }

    result.item_types = item_types;
    result.item_regions = item_regions;
    result.item_effects = item_effects;
    result.pattern_types = pattern_types;
    result.expr_types = expr_types;
    result.body_types = body_types;
    return result;
}

const TypeChecker = struct {
    arena: std.mem.Allocator,
    file_id: source.FileId,
    file: *const ast.AstFile,
    item_index: *const ItemIndexResult,
    resolution: *const NameResolutionResult,
    const_eval: *const ConstEvalResult,
    item_types: []Type,
    item_regions: []Region,
    item_effects: []Effect,
    pattern_types: []LocatedType,
    expr_types: []Type,
    current_return_type: ?Type = null,
    current_contract: ?ast.ItemId = null,
    diagnostics: *diagnostics.DiagnosticList,

    fn visitItem(self: *TypeChecker, item_id: ast.ItemId) anyerror!void {
        switch (self.file.item(item_id).*) {
            .Contract => |contract| {
                const previous_contract = self.current_contract;
                self.current_contract = item_id;
                defer self.current_contract = previous_contract;
                for (contract.invariants) |expr_id| try self.visitExpr(expr_id);
                for (contract.members) |member_id| try self.visitItem(member_id);
            },
            .Function => |function| {
                const previous_return_type = self.current_return_type;
                self.current_return_type = if (function.return_type) |type_expr| try descriptorFromTypeExpr(self.arena, self.file, self.item_index, type_expr) else .{ .void = {} };
                defer self.current_return_type = previous_return_type;
                for (function.clauses) |clause| try self.visitExpr(clause.expr);
                try self.visitBody(function.body);
                self.item_effects[item_id.index()] = try self.summarizeFunctionEffects(function);
            },
            .Field => |field| if (field.value) |expr_id| {
                try self.visitExpr(expr_id);
                if (field.type_expr == null) {
                    self.item_types[item_id.index()] = self.expr_types[expr_id.index()];
                } else {
                    const expected_type = self.item_types[item_id.index()];
                    const actual_type = self.expr_types[expr_id.index()];
                    if (try self.emitIntegerOverflowIfNeeded(field.range, expr_id, expected_type)) {
                        // Keep lowering/recovery moving after reporting the overflow.
                    } else if (!typesAssignable(expected_type, actual_type) and actual_type.kind() != .unknown) {
                        try self.emitRangeError(field.range, "field '{s}' expects type '{s}', found '{s}'", .{
                            field.name,
                            typeDisplayName(expected_type),
                            typeDisplayName(actual_type),
                        });
                    }
                }
            },
            .Constant => |constant| {
                try self.visitExpr(constant.value);
                if (constant.type_expr == null) {
                    self.item_types[item_id.index()] = self.expr_types[constant.value.index()];
                } else {
                    const expected_type = self.item_types[item_id.index()];
                    const actual_type = self.expr_types[constant.value.index()];
                    if (try self.emitIntegerOverflowIfNeeded(constant.range, constant.value, expected_type)) {
                        // Keep lowering/recovery moving after reporting the overflow.
                    } else if (!typesAssignable(expected_type, actual_type) and actual_type.kind() != .unknown) {
                        try self.emitRangeError(constant.range, "constant '{s}' expects type '{s}', found '{s}'", .{
                            constant.name,
                            typeDisplayName(expected_type),
                            typeDisplayName(actual_type),
                        });
                    }
                }
            },
            .GhostBlock => |ghost_block| try self.visitBody(ghost_block.body),
            else => {},
        }
    }

    fn visitBody(self: *TypeChecker, body_id: ast.BodyId) anyerror!void {
        const body = self.file.body(body_id).*;
        for (body.statements) |statement_id| {
            try self.visitStmt(statement_id);
        }
    }

    fn visitStmt(self: *TypeChecker, statement_id: ast.StmtId) anyerror!void {
        switch (self.file.statement(statement_id).*) {
            .VariableDecl => |decl| {
                if (decl.value) |expr_id| {
                    try self.visitExpr(expr_id);
                    const actual_type = self.expr_types[expr_id.index()];
                    if (decl.type_expr == null) {
                        self.pattern_types[decl.pattern.index()] = LocatedType.unlocated(actual_type);
                    } else {
                        const expected_type = self.pattern_types[decl.pattern.index()].type;
                        if (try self.emitIntegerOverflowIfNeeded(decl.range, expr_id, expected_type)) {
                            // Keep lowering/recovery moving after reporting the overflow.
                        } else if (!region_rules.isAssignable(LocatedType.unlocated(actual_type), self.pattern_types[decl.pattern.index()]) and actual_type.kind() != .unknown) {
                            try self.emitRangeError(decl.range, "declaration expects type '{s}', found '{s}'", .{
                                typeDisplayName(expected_type),
                                typeDisplayName(actual_type),
                            });
                        }
                    }
                }
            },
            .Return => |ret| if (ret.value) |expr_id| {
                try self.visitExpr(expr_id);
                const actual_type = self.expr_types[expr_id.index()];
                if (self.current_return_type) |expected_type| {
                    if (try self.emitIntegerOverflowIfNeeded(ret.range, expr_id, expected_type)) {
                        // Keep lowering/recovery moving after reporting the overflow.
                    } else if (!typesAssignable(expected_type, actual_type) and actual_type.kind() != .unknown) {
                        try self.emitRangeError(ret.range, "return expects type '{s}', found '{s}'", .{
                            typeDisplayName(expected_type),
                            typeDisplayName(actual_type),
                        });
                    }
                }
            },
            .If => |if_stmt| {
                try self.visitExpr(if_stmt.condition);
                try self.checkBoolCondition(if_stmt.condition, "if condition");
                try self.visitBody(if_stmt.then_body);
                if (if_stmt.else_body) |else_body| try self.visitBody(else_body);
            },
            .While => |while_stmt| {
                try self.visitExpr(while_stmt.condition);
                try self.checkBoolCondition(while_stmt.condition, "while condition");
                for (while_stmt.invariants) |expr_id| try self.visitExpr(expr_id);
                try self.visitBody(while_stmt.body);
            },
            .For => |for_stmt| {
                try self.visitExpr(for_stmt.iterable);
                for (for_stmt.invariants) |expr_id| try self.visitExpr(expr_id);
                try self.visitBody(for_stmt.body);
            },
            .Switch => |switch_stmt| {
                try self.visitExpr(switch_stmt.condition);
                for (switch_stmt.arms) |arm| {
                    switch (arm.pattern) {
                        .Expr => |expr_id| try self.visitExpr(expr_id),
                        .Range => |range_pattern| {
                            try self.visitExpr(range_pattern.start);
                            try self.visitExpr(range_pattern.end);
                        },
                        .Else => {},
                    }
                    try self.visitBody(arm.body);
                }
                if (switch_stmt.else_body) |else_body| try self.visitBody(else_body);
            },
            .Try => |try_stmt| {
                try self.visitBody(try_stmt.try_body);
                if (try_stmt.catch_clause) |catch_clause| try self.visitBody(catch_clause.body);
            },
            .Log => |log_stmt| {
                for (log_stmt.args) |arg| try self.visitExpr(arg);
            },
            .Lock => |lock_stmt| try self.visitExpr(lock_stmt.path),
            .Unlock => |unlock_stmt| try self.visitExpr(unlock_stmt.path),
            .Assert => |assert_stmt| {
                try self.visitExpr(assert_stmt.condition);
                try self.checkBoolCondition(assert_stmt.condition, "assert condition");
            },
            .Assume => |assume_stmt| {
                try self.visitExpr(assume_stmt.condition);
                try self.checkBoolCondition(assume_stmt.condition, "assume condition");
            },
            .Havoc => {},
            .Assign => |assign| {
                try self.visitExpr(assign.value);
                const expected = self.patternLocatedType(assign.target);
                const expected_type = expected.type;
                const actual_type = self.expr_types[assign.value.index()];
                if (try self.emitIntegerOverflowIfNeeded(assign.range, assign.value, expected_type)) {
                    // Keep lowering/recovery moving after reporting the overflow.
                } else if (actual_type.kind() != .unknown and expected_type.kind() != .unknown) {
                    const actual = self.exprLocatedType(assign.value);
                    if (!region_rules.isAssignable(actual, expected)) {
                        if (typesAssignable(expected_type, actual_type)) {
                            try self.emitRangeError(assign.range, "assignment expects region '{s}', found '{s}'", .{
                                region_rules.regionDisplayName(expected.region),
                                region_rules.regionDisplayName(actual.region),
                            });
                        } else {
                            try self.emitRangeError(assign.range, "assignment expects type '{s}', found '{s}'", .{
                                typeDisplayName(expected_type),
                                typeDisplayName(actual_type),
                            });
                        }
                    }
                }
            },
            .Expr => |expr_stmt| try self.visitExpr(expr_stmt.expr),
            .Block => |block| try self.visitBody(block.body),
            .LabeledBlock => |block| try self.visitBody(block.body),
            else => {},
        }
    }

    fn visitExpr(self: *TypeChecker, expr_id: ast.ExprId) anyerror!void {
        switch (self.file.expression(expr_id).*) {
            .IntegerLiteral => |literal| self.expr_types[expr_id.index()] = integerLiteralType(literal.text),
            .StringLiteral => self.expr_types[expr_id.index()] = .{ .string = {} },
            .BoolLiteral => self.expr_types[expr_id.index()] = .{ .bool = {} },
            .AddressLiteral => self.expr_types[expr_id.index()] = .{ .address = {} },
            .BytesLiteral => self.expr_types[expr_id.index()] = .{ .bytes = {} },
            .Tuple => |tuple| {
                for (tuple.elements) |element| try self.visitExpr(element);
                const elements = try self.arena.alloc(Type, tuple.elements.len);
                for (tuple.elements, 0..) |element, index| {
                    elements[index] = self.expr_types[element.index()];
                }
                self.expr_types[expr_id.index()] = .{ .tuple = elements };
            },
            .ArrayLiteral => |array| {
                var element_type: Type = .{ .unknown = {} };
                var saw_mismatch = false;
                for (array.elements) |element| try self.visitExpr(element);
                for (array.elements) |element| {
                    const next_type = self.expr_types[element.index()];
                    if (!saw_mismatch and element_type.kind() != .unknown and next_type.kind() != .unknown and !typesAssignable(element_type, next_type) and !typesAssignable(next_type, element_type)) {
                        saw_mismatch = true;
                        try self.emitExprError(expr_id, "array literal elements have incompatible types '{s}' and '{s}'", .{
                            typeDisplayName(element_type),
                            typeDisplayName(next_type),
                        });
                    }
                    if (!saw_mismatch) {
                        element_type = mergeExprType(element_type, next_type);
                    }
                }
                if (saw_mismatch) element_type = .{ .unknown = {} };
                self.expr_types[expr_id.index()] = .{ .array = .{
                    .element_type = try self.storeType(element_type),
                    .len = @intCast(array.elements.len),
                } };
            },
            .StructLiteral => |struct_literal| {
                for (struct_literal.fields) |field| try self.visitExpr(field.value);
                self.expr_types[expr_id.index()] = self.structLiteralType(struct_literal.type_name);
            },
            .Switch => |switch_expr| {
                try self.visitExpr(switch_expr.condition);
                var result_type: Type = .{ .unknown = {} };
                var saw_mismatch = false;
                for (switch_expr.arms) |arm| {
                    switch (arm.pattern) {
                        .Expr => |pattern_expr| try self.visitExpr(pattern_expr),
                        .Range => |range_pattern| {
                            try self.visitExpr(range_pattern.start);
                            try self.visitExpr(range_pattern.end);
                        },
                        .Else => {},
                    }
                    try self.visitExpr(arm.value);
                    const arm_type = self.expr_types[arm.value.index()];
                    if (!saw_mismatch and result_type.kind() != .unknown and arm_type.kind() != .unknown and !typesAssignable(result_type, arm_type) and !typesAssignable(arm_type, result_type)) {
                        saw_mismatch = true;
                        try self.emitExprError(expr_id, "switch expression branches have incompatible types '{s}' and '{s}'", .{
                            typeDisplayName(result_type),
                            typeDisplayName(arm_type),
                        });
                    }
                    if (!saw_mismatch) {
                        result_type = mergeExprType(result_type, self.expr_types[arm.value.index()]);
                    }
                }
                if (switch_expr.else_expr) |else_expr| {
                    try self.visitExpr(else_expr);
                    const else_type = self.expr_types[else_expr.index()];
                    if (!saw_mismatch and result_type.kind() != .unknown and else_type.kind() != .unknown and !typesAssignable(result_type, else_type) and !typesAssignable(else_type, result_type)) {
                        saw_mismatch = true;
                        try self.emitExprError(expr_id, "switch expression branches have incompatible types '{s}' and '{s}'", .{
                            typeDisplayName(result_type),
                            typeDisplayName(else_type),
                        });
                    }
                    if (!saw_mismatch) {
                        result_type = mergeExprType(result_type, self.expr_types[else_expr.index()]);
                    }
                }
                if (saw_mismatch) result_type = .{ .unknown = {} };
                self.expr_types[expr_id.index()] = result_type;
            },
            .Comptime => |comptime_expr| {
                try self.visitBody(comptime_expr.body);
                const body = self.file.body(comptime_expr.body).*;
                if (body.statements.len == 0) {
                    self.expr_types[expr_id.index()] = .{ .unknown = {} };
                } else switch (self.file.statement(body.statements[body.statements.len - 1]).*) {
                    .Expr => |expr_stmt| self.expr_types[expr_id.index()] = self.expr_types[expr_stmt.expr.index()],
                    .Return => |ret| self.expr_types[expr_id.index()] = if (ret.value) |value| self.expr_types[value.index()] else .{ .void = {} },
                    else => self.expr_types[expr_id.index()] = .{ .unknown = {} },
                }
            },
            .ErrorReturn => |error_return| {
                for (error_return.args) |arg| try self.visitExpr(arg);
                self.expr_types[expr_id.index()] = .{ .named = .{ .name = error_return.name } };
            },
            .Name => {
                self.expr_types[expr_id.index()] = self.typeForBinding(self.resolution.expr_bindings[expr_id.index()]);
            },
            .Result => {
                self.expr_types[expr_id.index()] = self.current_return_type orelse .{ .unknown = {} };
            },
            .Unary => |unary| {
                try self.visitExpr(unary.operand);
                const operand_type = self.expr_types[unary.operand.index()];
                const result_type = self.unaryResultType(unary.op, operand_type);
                self.expr_types[expr_id.index()] = result_type;
                if (result_type.kind() == .unknown and operand_type.kind() != .unknown) {
                    try self.emitExprError(expr_id, "invalid unary operator '{s}' for type '{s}'", .{
                        unaryOpName(unary.op),
                        typeDisplayName(operand_type),
                    });
                }
            },
            .Binary => |binary| {
                try self.visitExpr(binary.lhs);
                try self.visitExpr(binary.rhs);
                const lhs_type = self.expr_types[binary.lhs.index()];
                const rhs_type = self.expr_types[binary.rhs.index()];
                const result_type = self.binaryResultType(
                    binary.op,
                    lhs_type,
                    rhs_type,
                );
                var final_type = result_type;
                if (result_type.kind() != .unknown and try self.hasInvalidConstantShiftAmount(expr_id, binary.op, lhs_type, binary.rhs)) {
                    final_type = .{ .unknown = {} };
                }
                self.expr_types[expr_id.index()] = final_type;
                if (final_type.kind() == .unknown and result_type.kind() == .unknown and lhs_type.kind() != .unknown and rhs_type.kind() != .unknown) {
                    try self.emitExprError(expr_id, "invalid binary operator '{s}' for types '{s}' and '{s}'", .{
                        binaryOpName(binary.op),
                        typeDisplayName(lhs_type),
                        typeDisplayName(rhs_type),
                    });
                }
            },
            .Call => |call| {
                try self.visitExpr(call.callee);
                for (call.args) |arg| try self.visitExpr(arg);
                const callee_type = self.callableType(call.callee);
                const callee_expr_type = self.expr_types[call.callee.index()];
                const result_type = self.callReturnType(call);
                self.expr_types[expr_id.index()] = result_type;
                if (callee_type.kind() != .function) {
                    const bad_type = if (callee_expr_type.kind() != .unknown) callee_expr_type else callee_type;
                    if (bad_type.kind() != .unknown) {
                        try self.emitExprError(expr_id, "type '{s}' is not callable", .{typeDisplayName(bad_type)});
                    }
                } else if (result_type.kind() == .unknown and callee_type.paramTypes().len != call.args.len) {
                    try self.emitExprError(expr_id, "expected {d} arguments, found {d}", .{ callee_type.paramTypes().len, call.args.len });
                }
            },
            .Builtin => |builtin| {
                for (builtin.args) |arg| try self.visitExpr(arg);
                const result_type = self.builtinReturnType(builtin);
                self.expr_types[expr_id.index()] = result_type;
                if (try self.emitBuiltinIntegerOverflowIfNeeded(expr_id, builtin, result_type)) {
                    self.expr_types[expr_id.index()] = .{ .unknown = {} };
                }
            },
            .Field => |field| {
                try self.visitExpr(field.base);
                const base_type = self.expr_types[field.base.index()];
                const result_type = try self.fieldAccessType(base_type, field.name);
                self.expr_types[expr_id.index()] = result_type;
                if (result_type.kind() == .unknown and base_type.kind() != .unknown) {
                    try self.emitExprError(expr_id, "type '{s}' has no field '{s}'", .{
                        typeDisplayName(base_type),
                        field.name,
                    });
                }
            },
            .Index => |index| {
                try self.visitExpr(index.base);
                try self.visitExpr(index.index);
                const base_type = self.expr_types[index.base.index()];
                const result_type = self.indexAccessType(base_type, index.index);
                self.expr_types[expr_id.index()] = result_type;
                if (result_type.kind() == .unknown and base_type.kind() != .unknown) {
                    try self.emitExprError(expr_id, "type '{s}' is not indexable", .{typeDisplayName(base_type)});
                }
            },
            .Group => |group| {
                try self.visitExpr(group.expr);
                self.expr_types[expr_id.index()] = self.expr_types[group.expr.index()];
            },
            .Old => |old| {
                try self.visitExpr(old.expr);
                self.expr_types[expr_id.index()] = self.expr_types[old.expr.index()];
            },
            .Quantified => |quantified| {
                self.pattern_types[quantified.pattern.index()] = LocatedType.unlocated(
                    try descriptorFromTypeExpr(self.arena, self.file, self.item_index, quantified.type_expr),
                );
                if (quantified.condition) |condition| try self.visitExpr(condition);
                try self.visitExpr(quantified.body);
                self.expr_types[expr_id.index()] = .{ .bool = {} };
            },
            .Error => self.expr_types[expr_id.index()] = .{ .unknown = {} },
        }
    }

    fn typeForBinding(self: *const TypeChecker, binding: ?ResolvedBinding) Type {
        if (binding) |resolved| {
            return switch (resolved) {
                .item => |item_id| self.item_types[item_id.index()],
                .pattern => |pattern_id| self.pattern_types[pattern_id.index()].type,
            };
        }
        return .{ .unknown = {} };
    }

    fn locatedTypeForBinding(self: *const TypeChecker, binding: ?ResolvedBinding) LocatedType {
        if (binding) |resolved| {
            return switch (resolved) {
                .item => |item_id| self.itemLocatedType(item_id),
                .pattern => |pattern_id| self.pattern_types[pattern_id.index()],
            };
        }
        return LocatedType.unlocated(.{ .unknown = {} });
    }

    fn callReturnType(self: *const TypeChecker, call: ast.CallExpr) Type {
        const callee_type = self.callableType(call.callee);
        if (callee_type.kind() != .function) return .{ .unknown = {} };

        const param_types = callee_type.paramTypes();
        if (param_types.len != call.args.len) return .{ .unknown = {} };

        const return_types = callee_type.returnTypes();
        if (return_types.len > 0) return return_types[0];
        return .{ .void = {} };
    }

    fn callableType(self: *const TypeChecker, expr_id: ast.ExprId) Type {
        switch (self.file.expression(expr_id).*) {
            .Name => {
                if (self.resolution.expr_bindings[expr_id.index()]) |binding| {
                    if (binding == .item) {
                        if (self.file.item(binding.item).* == .Function) {
                            return self.item_types[binding.item.index()];
                        }
                    }
                }
            },
            .Field => {
                const field_type = self.expr_types[expr_id.index()];
                if (field_type.kind() == .function) return field_type;
            },
            .Group => |group| return self.callableType(group.expr),
            else => {},
        }
        return .{ .unknown = {} };
    }

    fn structLiteralType(self: *const TypeChecker, name: []const u8) Type {
        if (self.item_index.lookup(name)) |item_id| {
            return self.item_types[item_id.index()];
        }
        return .{ .named = .{ .name = name } };
    }

    fn builtinReturnType(self: *const TypeChecker, builtin: ast.BuiltinExpr) Type {
        if (std.mem.eql(u8, builtin.name, "cast") or
            std.mem.eql(u8, builtin.name, "bitCast") or
            std.mem.eql(u8, builtin.name, "truncate"))
        {
            if (builtin.type_arg) |type_expr| return descriptorFromTypeExpr(self.arena, self.file, self.item_index, type_expr) catch .{ .unknown = {} };
            if (std.mem.eql(u8, builtin.name, "truncate") and builtin.args.len > 0) {
                return self.expr_types[builtin.args[0].index()];
            }
            return .{ .unknown = {} };
        }

        if (builtin.args.len > 0 and (std.mem.eql(u8, builtin.name, "divTrunc") or
            std.mem.eql(u8, builtin.name, "divFloor") or
            std.mem.eql(u8, builtin.name, "divCeil") or
            std.mem.eql(u8, builtin.name, "divExact")))
        {
            return self.expr_types[builtin.args[0].index()];
        }

        if (std.mem.eql(u8, builtin.name, "divmod") or
            std.mem.eql(u8, builtin.name, "addWithOverflow") or
            std.mem.eql(u8, builtin.name, "subWithOverflow") or
            std.mem.eql(u8, builtin.name, "mulWithOverflow") or
            std.mem.eql(u8, builtin.name, "divWithOverflow") or
            std.mem.eql(u8, builtin.name, "modWithOverflow") or
            std.mem.eql(u8, builtin.name, "negWithOverflow") or
            std.mem.eql(u8, builtin.name, "shlWithOverflow") or
            std.mem.eql(u8, builtin.name, "shrWithOverflow") or
            std.mem.eql(u8, builtin.name, "powerWithOverflow"))
        {
            if (builtin.args.len > 0) {
                const value_type = self.expr_types[builtin.args[0].index()];
                const tuple_types = self.arena.alloc(Type, 2) catch return .{ .unknown = {} };
                tuple_types[0] = value_type;
                tuple_types[1] = .{ .bool = {} };
                return .{ .tuple = tuple_types };
            }
            return .{ .unknown = {} };
        }

        return .{ .unknown = {} };
    }

    fn summarizeFunctionEffects(self: *TypeChecker, function: ast.FunctionItem) !Effect {
        var reads: std.ArrayList([]const u8) = .{};
        var writes: std.ArrayList([]const u8) = .{};
        try self.collectBodyEffects(function.body, &reads, &writes);
        return buildEffect(reads.items, writes.items);
    }

    fn buildEffect(reads: []const []const u8, writes: []const []const u8) Effect {
        if (reads.len == 0 and writes.len == 0) return .pure;
        if (reads.len == 0) return .{ .writes = .{ .slots = writes } };
        if (writes.len == 0) return .{ .reads = .{ .slots = reads } };
        return .{ .reads_writes = .{ .reads = reads, .writes = writes } };
    }

    fn collectBodyEffects(self: *TypeChecker, body_id: ast.BodyId, reads: *std.ArrayList([]const u8), writes: *std.ArrayList([]const u8)) anyerror!void {
        const body = self.file.body(body_id).*;
        for (body.statements) |statement_id| {
            try self.collectStmtEffects(statement_id, reads, writes);
        }
    }

    fn collectStmtEffects(self: *TypeChecker, statement_id: ast.StmtId, reads: *std.ArrayList([]const u8), writes: *std.ArrayList([]const u8)) anyerror!void {
        switch (self.file.statement(statement_id).*) {
            .VariableDecl => |decl| if (decl.value) |expr_id| {
                try self.collectExprEffects(expr_id, reads, writes);
            },
            .Return => |ret| if (ret.value) |expr_id| {
                try self.collectExprEffects(expr_id, reads, writes);
            },
            .If => |if_stmt| {
                try self.collectExprEffects(if_stmt.condition, reads, writes);
                try self.collectBodyEffects(if_stmt.then_body, reads, writes);
                if (if_stmt.else_body) |else_body| try self.collectBodyEffects(else_body, reads, writes);
            },
            .While => |while_stmt| {
                try self.collectExprEffects(while_stmt.condition, reads, writes);
                for (while_stmt.invariants) |expr_id| try self.collectExprEffects(expr_id, reads, writes);
                try self.collectBodyEffects(while_stmt.body, reads, writes);
            },
            .For => |for_stmt| {
                try self.collectExprEffects(for_stmt.iterable, reads, writes);
                for (for_stmt.invariants) |expr_id| try self.collectExprEffects(expr_id, reads, writes);
                try self.collectBodyEffects(for_stmt.body, reads, writes);
            },
            .Switch => |switch_stmt| {
                try self.collectExprEffects(switch_stmt.condition, reads, writes);
                for (switch_stmt.arms) |arm| {
                    switch (arm.pattern) {
                        .Expr => |expr_id| try self.collectExprEffects(expr_id, reads, writes),
                        .Range => |range_pattern| {
                            try self.collectExprEffects(range_pattern.start, reads, writes);
                            try self.collectExprEffects(range_pattern.end, reads, writes);
                        },
                        .Else => {},
                    }
                    try self.collectBodyEffects(arm.body, reads, writes);
                }
                if (switch_stmt.else_body) |else_body| try self.collectBodyEffects(else_body, reads, writes);
            },
            .Try => |try_stmt| {
                try self.collectBodyEffects(try_stmt.try_body, reads, writes);
                if (try_stmt.catch_clause) |catch_clause| try self.collectBodyEffects(catch_clause.body, reads, writes);
            },
            .Log => |log_stmt| {
                for (log_stmt.args) |arg| try self.collectExprEffects(arg, reads, writes);
            },
            .Lock, .Unlock, .Havoc, .Break, .Continue => {},
            .Assert => |assert_stmt| try self.collectExprEffects(assert_stmt.condition, reads, writes),
            .Assume => |assume_stmt| try self.collectExprEffects(assume_stmt.condition, reads, writes),
            .Assign => |assign| {
                try self.collectExprEffects(assign.value, reads, writes);
                try self.collectPatternTargetEffects(assign.target, assign.op, reads, writes);
            },
            .Expr => |expr_stmt| try self.collectExprEffects(expr_stmt.expr, reads, writes),
            .Block => |block| try self.collectBodyEffects(block.body, reads, writes),
            .LabeledBlock => |block| try self.collectBodyEffects(block.body, reads, writes),
            .Error => {},
        }
    }

    fn collectExprEffects(self: *TypeChecker, expr_id: ast.ExprId, reads: *std.ArrayList([]const u8), writes: *std.ArrayList([]const u8)) anyerror!void {
        switch (self.file.expression(expr_id).*) {
            .IntegerLiteral, .StringLiteral, .BoolLiteral, .AddressLiteral, .BytesLiteral, .Result, .Error => {},
            .Tuple => |tuple| for (tuple.elements) |element| try self.collectExprEffects(element, reads, writes),
            .ArrayLiteral => |array| for (array.elements) |element| try self.collectExprEffects(element, reads, writes),
            .StructLiteral => |struct_literal| for (struct_literal.fields) |field| try self.collectExprEffects(field.value, reads, writes),
            .Switch => |switch_expr| {
                try self.collectExprEffects(switch_expr.condition, reads, writes);
                for (switch_expr.arms) |arm| {
                    switch (arm.pattern) {
                        .Expr => |pattern_expr| try self.collectExprEffects(pattern_expr, reads, writes),
                        .Range => |range_pattern| {
                            try self.collectExprEffects(range_pattern.start, reads, writes);
                            try self.collectExprEffects(range_pattern.end, reads, writes);
                        },
                        .Else => {},
                    }
                    try self.collectExprEffects(arm.value, reads, writes);
                }
                if (switch_expr.else_expr) |else_expr| try self.collectExprEffects(else_expr, reads, writes);
            },
            .Comptime => {},
            .ErrorReturn => |error_return| for (error_return.args) |arg| try self.collectExprEffects(arg, reads, writes),
            .Name => {
                if (self.fieldSlotForBinding(self.resolution.expr_bindings[expr_id.index()])) |slot| {
                    try self.appendUniqueSlot(reads, slot);
                }
            },
            .Unary => |unary| try self.collectExprEffects(unary.operand, reads, writes),
            .Binary => |binary| {
                try self.collectExprEffects(binary.lhs, reads, writes);
                try self.collectExprEffects(binary.rhs, reads, writes);
            },
            .Call => |call| {
                try self.collectExprEffects(call.callee, reads, writes);
                for (call.args) |arg| try self.collectExprEffects(arg, reads, writes);
            },
            .Builtin => |builtin| for (builtin.args) |arg| try self.collectExprEffects(arg, reads, writes),
            .Field => |field| try self.collectExprEffects(field.base, reads, writes),
            .Index => |index| {
                try self.collectExprEffects(index.base, reads, writes);
                try self.collectExprEffects(index.index, reads, writes);
            },
            .Group => |group| try self.collectExprEffects(group.expr, reads, writes),
            .Old, .Quantified => {},
        }
    }

    fn collectPatternTargetEffects(self: *TypeChecker, pattern_id: ast.PatternId, op: ast.AssignmentOp, reads: *std.ArrayList([]const u8), writes: *std.ArrayList([]const u8)) anyerror!void {
        switch (self.file.pattern(pattern_id).*) {
            .Name => {
                if (self.patternRootFieldSlot(pattern_id)) |slot_name| {
                    if (op != .assign) try self.appendUniqueSlot(reads, slot_name);
                    try self.appendUniqueSlot(writes, slot_name);
                }
            },
            .Field => |field| {
                try self.collectPatternTargetEffects(field.base, .add_assign, reads, writes);
            },
            .Index => |index| {
                try self.collectPatternExprReads(index.base, reads, writes);
                try self.collectExprEffects(index.index, reads, writes);
                if (self.patternRootFieldSlot(pattern_id)) |slot| {
                    try self.appendUniqueSlot(reads, slot);
                    try self.appendUniqueSlot(writes, slot);
                }
            },
            .StructDestructure => {},
            .Error => {},
        }
    }

    fn collectPatternExprReads(self: *TypeChecker, pattern_id: ast.PatternId, reads: *std.ArrayList([]const u8), writes: *std.ArrayList([]const u8)) anyerror!void {
        switch (self.file.pattern(pattern_id).*) {
            .Name => {
                if (self.patternRootFieldSlot(pattern_id)) |slot| {
                    try self.appendUniqueSlot(reads, slot);
                }
            },
            .Field => |field| try self.collectPatternExprReads(field.base, reads, writes),
            .Index => |index| {
                try self.collectPatternExprReads(index.base, reads, writes);
                try self.collectExprEffects(index.index, reads, writes);
            },
            .StructDestructure => {},
            .Error => {},
        }
    }

    fn patternRootFieldSlot(self: *TypeChecker, pattern_id: ast.PatternId) ?[]const u8 {
        return switch (self.file.pattern(pattern_id).*) {
            .Name => |name| self.lookupNamedFieldSlot(name.name),
            .Field => |field| self.patternRootFieldSlot(field.base),
            .Index => |index| self.patternRootFieldSlot(index.base),
            .StructDestructure => null,
            .Error => null,
        };
    }

    fn fieldSlotForBinding(self: *TypeChecker, binding: ?ResolvedBinding) ?[]const u8 {
        if (binding) |resolved| {
            return switch (resolved) {
                .item => |item_id| switch (self.file.item(item_id).*) {
                    .Field => |field| if (field.storage_class != .none) field.name else null,
                    else => null,
                },
                .pattern => null,
            };
        }
        return null;
    }

    fn lookupNamedFieldSlot(self: *TypeChecker, name: []const u8) ?[]const u8 {
        if (self.current_contract) |contract_id| {
            const contract = self.file.item(contract_id).Contract;
            for (contract.members) |member_id| {
                switch (self.file.item(member_id).*) {
                    .Field => |field| {
                        if (field.storage_class == .none) continue;
                        if (std.mem.eql(u8, field.name, name)) return field.name;
                    },
                    else => {},
                }
            }
        }
        return self.fieldSlotForBinding(if (self.item_index.lookup(name)) |item_id| .{ .item = item_id } else null);
    }

    fn appendUniqueSlot(self: *TypeChecker, slots: *std.ArrayList([]const u8), slot: []const u8) !void {
        for (slots.items) |existing| {
            if (std.mem.eql(u8, existing, slot)) return;
        }
        try slots.append(self.arena, slot);
    }

    fn storeType(self: *TypeChecker, ty: Type) !*const Type {
        const stored = try self.arena.create(Type);
        stored.* = ty;
        return stored;
    }

    fn unaryResultType(self: *const TypeChecker, op: ast.UnaryOp, operand_type: Type) Type {
        _ = self;
        return switch (op) {
            .neg => if (isIntegerType(operand_type)) operand_type else .{ .unknown = {} },
            .not_ => if (operand_type.kind() == .bool) .{ .bool = {} } else .{ .unknown = {} },
            .bit_not => if (isIntegerType(operand_type)) operand_type else .{ .unknown = {} },
            .try_ => if (operand_type.payloadType()) |payload| payload.* else .{ .unknown = {} },
        };
    }

    fn binaryResultType(self: *const TypeChecker, op: ast.BinaryOp, lhs_type: Type, rhs_type: Type) Type {
        _ = self;
        return switch (op) {
            .add, .wrapping_add, .sub, .wrapping_sub, .mul, .wrapping_mul, .div, .mod, .pow, .wrapping_pow => arithmeticResultType(lhs_type, rhs_type),
            .bit_and, .bit_or, .bit_xor, .shl, .wrapping_shl, .shr, .wrapping_shr => bitwiseResultType(lhs_type, rhs_type),
            .and_and, .or_or => if (lhs_type.kind() == .bool and rhs_type.kind() == .bool) .{ .bool = {} } else .{ .unknown = {} },
            .eq, .ne => if (typesComparable(lhs_type, rhs_type)) .{ .bool = {} } else .{ .unknown = {} },
            .lt, .le, .gt, .ge => if (orderedTypesComparable(lhs_type, rhs_type)) .{ .bool = {} } else .{ .unknown = {} },
        };
    }

    fn fieldAccessType(self: *const TypeChecker, base_type: Type, field_name: []const u8) !Type {
        const item_id = self.itemIdForType(base_type) orelse return .{ .unknown = {} };
        return switch (self.file.item(item_id).*) {
            .Struct => |struct_item| blk: {
                for (struct_item.fields) |field| {
                    if (std.mem.eql(u8, field.name, field_name)) {
                        break :blk try descriptorFromTypeExpr(self.arena, self.file, self.item_index, field.type_expr);
                    }
                }
                break :blk .{ .unknown = {} };
            },
            .Bitfield => |bitfield_item| blk: {
                for (bitfield_item.fields) |field| {
                    if (std.mem.eql(u8, field.name, field_name)) {
                        break :blk try descriptorFromTypeExpr(self.arena, self.file, self.item_index, field.type_expr);
                    }
                }
                break :blk .{ .unknown = {} };
            },
            .Contract => |contract| blk: {
                for (contract.members) |member_id| {
                    const member = self.file.item(member_id).*;
                    switch (member) {
                        .Field => |field| if (std.mem.eql(u8, field.name, field_name)) break :blk self.item_types[member_id.index()],
                        .Constant => |constant| if (std.mem.eql(u8, constant.name, field_name)) break :blk self.item_types[member_id.index()],
                        .Function => |function| if (std.mem.eql(u8, function.name, field_name)) break :blk self.item_types[member_id.index()],
                        .Struct => |struct_item| if (std.mem.eql(u8, struct_item.name, field_name)) break :blk self.item_types[member_id.index()],
                        .Bitfield => |bitfield_item| if (std.mem.eql(u8, bitfield_item.name, field_name)) break :blk self.item_types[member_id.index()],
                        .Enum => |enum_item| if (std.mem.eql(u8, enum_item.name, field_name)) break :blk self.item_types[member_id.index()],
                        else => {},
                    }
                }
                break :blk .{ .unknown = {} };
            },
            .Enum => |enum_item| blk: {
                for (enum_item.variants) |variant| {
                    if (std.mem.eql(u8, variant.name, field_name)) break :blk base_type;
                }
                break :blk .{ .unknown = {} };
            },
            else => .{ .unknown = {} },
        };
    }

    fn indexAccessType(self: *const TypeChecker, base_type: Type, index_expr_id: ast.ExprId) Type {
        if (base_type.elementType()) |element| return element.*;
        return switch (base_type) {
            .map => |map| if (map.value_type) |value| value.* else .{ .unknown = {} },
            .tuple => |elements| blk: {
                const tuple_index = self.constTupleIndex(index_expr_id) orelse break :blk .{ .unknown = {} };
                if (tuple_index >= elements.len) break :blk .{ .unknown = {} };
                break :blk elements[tuple_index];
            },
            else => .{ .unknown = {} },
        };
    }

    fn constTupleIndex(self: *const TypeChecker, expr_id: ast.ExprId) ?usize {
        const value = self.const_eval.values[expr_id.index()] orelse return null;
        return switch (value) {
            .integer => |integer| const_bridge.positiveShiftAmount(integer),
            else => null,
        };
    }

    fn itemIdForType(self: *const TypeChecker, ty: Type) ?ast.ItemId {
        const name = ty.name() orelse return null;
        return self.item_index.lookup(name);
    }

    fn hasInvalidConstantShiftAmount(self: *TypeChecker, expr_id: ast.ExprId, op: ast.BinaryOp, lhs_type: Type, rhs_expr_id: ast.ExprId) !bool {
        switch (op) {
            .shl, .wrapping_shl, .shr, .wrapping_shr => {},
            else => return false,
        }
        if (lhs_type.kind() != .integer) return false;
        const bits = lhs_type.integer.bits orelse return false;
        const value = self.const_eval.values[rhs_expr_id.index()] orelse return false;
        const amount = switch (value) {
            .integer => |integer| integer,
            else => return false,
        };
        const amount_usize = const_bridge.positiveShiftAmount(amount) orelse {
            const amount_text = try self.integerValueText(amount);
            try self.emitExprError(expr_id, "shift amount {s} out of range for type '{s}'", .{
                amount_text,
                typeDisplayName(lhs_type),
            });
            return true;
        };
        if (amount_usize >= bits) {
            const amount_text = try self.integerValueText(amount);
            try self.emitExprError(expr_id, "shift amount {s} out of range for type '{s}'", .{
                amount_text,
                typeDisplayName(lhs_type),
            });
            return true;
        }
        return false;
    }

    fn emitIntegerOverflowIfNeeded(self: *TypeChecker, range: source.TextRange, expr_id: ast.ExprId, expected_type: Type) !bool {
        const value = self.const_eval.values[expr_id.index()] orelse return false;
        if (value != .integer or expected_type.kind() != .integer) return false;
        if (integerValueFitsType(value.integer, expected_type.integer)) return false;
        const value_text = try self.integerValueText(value.integer);
        try self.emitRangeError(range, "constant value {s} does not fit in type '{s}'", .{
            value_text,
            typeDisplayName(expected_type),
        });
        return true;
    }

    fn emitBuiltinIntegerOverflowIfNeeded(self: *TypeChecker, expr_id: ast.ExprId, builtin: ast.BuiltinExpr, result_type: Type) !bool {
        if (!std.mem.eql(u8, builtin.name, "cast")) return false;
        if (builtin.args.len == 0) return false;
        const value = self.const_eval.values[builtin.args[0].index()] orelse return false;
        if (value != .integer or result_type.kind() != .integer) return false;
        if (integerValueFitsType(value.integer, result_type.integer)) return false;
        const value_text = try self.integerValueText(value.integer);
        try self.emitExprError(expr_id, "constant value {s} does not fit in cast target type '{s}'", .{
            value_text,
            typeDisplayName(result_type),
        });
        return true;
    }

    fn integerValueText(self: *TypeChecker, value: BigInt) ![]const u8 {
        return try value.toString(self.arena, 10, .lower);
    }

    fn emitExprError(self: *TypeChecker, expr_id: ast.ExprId, comptime fmt: []const u8, args: anytype) !void {
        var buffer: [256]u8 = undefined;
        const message = try std.fmt.bufPrint(&buffer, fmt, args);
        try self.diagnostics.appendError(message, .{
            .file_id = self.file_id,
            .range = source.rangeOf(self.file.expression(expr_id).*),
        });
    }

    fn emitRangeError(self: *TypeChecker, range: source.TextRange, comptime fmt: []const u8, args: anytype) !void {
        var buffer: [256]u8 = undefined;
        const message = try std.fmt.bufPrint(&buffer, fmt, args);
        try self.diagnostics.appendError(message, .{
            .file_id = self.file_id,
            .range = range,
        });
    }

    fn patternType(self: *const TypeChecker, pattern_id: ast.PatternId) Type {
        return switch (self.file.pattern(pattern_id).*) {
            .Name => |name| blk: {
                const direct = self.pattern_types[pattern_id.index()].type;
                if (direct.kind() != .unknown) break :blk direct;
                break :blk self.lookupNamedPatternType(name.name);
            },
            .Field => |field| self.fieldPatternType(field),
            .Index => |index| self.indexPatternType(index),
            .StructDestructure => self.pattern_types[pattern_id.index()].type,
            .Error => .{ .unknown = {} },
        };
    }

    fn patternLocatedType(self: *const TypeChecker, pattern_id: ast.PatternId) LocatedType {
        return switch (self.file.pattern(pattern_id).*) {
            .Name => |name| blk: {
                const direct = self.pattern_types[pattern_id.index()];
                if (direct.kind() != .unknown) break :blk direct;
                break :blk self.lookupNamedPatternLocatedType(name.name);
            },
            .StructDestructure => self.pattern_types[pattern_id.index()],
            .Field => |field| blk: {
                const base = self.patternLocatedType(field.base);
                break :blk .{ .type = self.fieldPatternType(field), .region = base.region };
            },
            .Index => |index| blk: {
                const base = self.patternLocatedType(index.base);
                break :blk .{ .type = self.indexPatternType(index), .region = base.region };
            },
            .Error => LocatedType.unlocated(.{ .unknown = {} }),
        };
    }

    fn exprLocatedType(self: *const TypeChecker, expr_id: ast.ExprId) LocatedType {
        return switch (self.file.expression(expr_id).*) {
            .Name => self.locatedTypeForBinding(self.resolution.expr_bindings[expr_id.index()]),
            .Group => |group| self.exprLocatedType(group.expr),
            .Old => |old| self.exprLocatedType(old.expr),
            .Field => |field| blk: {
                const base = self.exprLocatedType(field.base);
                break :blk .{ .type = self.expr_types[expr_id.index()], .region = base.region };
            },
            .Index => |index| blk: {
                const base = self.exprLocatedType(index.base);
                break :blk .{ .type = self.expr_types[expr_id.index()], .region = base.region };
            },
            else => LocatedType.unlocated(self.expr_types[expr_id.index()]),
        };
    }

    fn fieldPatternType(self: *const TypeChecker, field: ast.FieldPattern) Type {
        const base_type = self.patternType(field.base);
        return self.fieldAccessType(base_type, field.name) catch .{ .unknown = {} };
    }

    fn indexPatternType(self: *const TypeChecker, index: ast.IndexPattern) Type {
        const base_type = self.patternType(index.base);
        return self.indexAccessType(base_type, index.index);
    }

    fn checkBoolCondition(self: *TypeChecker, expr_id: ast.ExprId, label: []const u8) !void {
        const ty = self.expr_types[expr_id.index()];
        if (ty.kind() == .unknown or ty.kind() == .bool) return;
        try self.emitExprError(expr_id, "{s} must be 'bool', found '{s}'", .{
            label,
            typeDisplayName(ty),
        });
    }

    fn lookupNamedPatternType(self: *const TypeChecker, name: []const u8) Type {
        return self.lookupNamedPatternLocatedType(name).type;
    }

    fn lookupNamedPatternLocatedType(self: *const TypeChecker, name: []const u8) LocatedType {
        if (self.item_index.lookup(name)) |item_id| {
            return self.itemLocatedType(item_id);
        }
        for (self.file.patterns, 0..) |pattern, index| {
            if (pattern != .Name) continue;
            if (!std.mem.eql(u8, pattern.Name.name, name)) continue;
            const ty = self.pattern_types[index];
            if (ty.kind() != .unknown) return ty;
        }
        return LocatedType.unlocated(.{ .unknown = {} });
    }

    fn itemLocatedType(self: *const TypeChecker, item_id: ast.ItemId) LocatedType {
        return .{
            .type = self.item_types[item_id.index()],
            .region = self.item_regions[item_id.index()],
        };
    }
};

fn arithmeticResultType(lhs_type: Type, rhs_type: Type) Type {
    if (!isIntegerType(lhs_type) or !isIntegerType(rhs_type)) return .{ .unknown = {} };
    if (sameConcreteType(lhs_type, rhs_type)) return lhs_type;
    return lhs_type;
}

fn bitwiseResultType(lhs_type: Type, rhs_type: Type) Type {
    if (!isIntegerType(lhs_type) or !isIntegerType(rhs_type)) return .{ .unknown = {} };
    return lhs_type;
}

fn integerLiteralType(text: []const u8) Type {
    if (integerTypeSuffix(text)) |integer| {
        return .{ .integer = integer };
    }
    return .{ .integer = .{} };
}

fn integerTypeSuffix(text: []const u8) ?model.IntegerType {
    const unsigned_index = std.mem.lastIndexOfScalar(u8, text, 'u');
    const signed_index = std.mem.lastIndexOfScalar(u8, text, 'i');
    const suffix_index = switch (unsigned_index != null and signed_index != null) {
        true => @max(unsigned_index.?, signed_index.?),
        false => unsigned_index orelse signed_index,
    };
    const start = suffix_index orelse return null;
    if (start == 0 or start + 1 >= text.len) return null;
    const suffix = text[start..];
    const signed = switch (suffix[0]) {
        'u' => false,
        'i' => true,
        else => return null,
    };
    const bits = std.fmt.parseInt(u16, suffix[1..], 10) catch return null;
    return .{
        .bits = bits,
        .signed = signed,
        .spelling = suffix,
    };
}

fn typesComparable(lhs_type: Type, rhs_type: Type) bool {
    if (sameConcreteType(lhs_type, rhs_type)) return true;
    if (isIntegerType(lhs_type) and isIntegerType(rhs_type)) return true;
    return false;
}

fn orderedTypesComparable(lhs_type: Type, rhs_type: Type) bool {
    if (lhs_type.kind() == .bool or rhs_type.kind() == .bool) return false;
    return typesComparable(lhs_type, rhs_type);
}

fn isIntegerType(ty: Type) bool {
    return ty.kind() == .integer;
}

fn integerValueFitsType(value: BigInt, integer: model.IntegerType) bool {
    const bits = integer.bits orelse return true;
    const signed = integer.signed orelse return true;
    if (bits == 0) return value.eqlZero();
    return value.fitsInTwosComp(if (signed) .signed else .unsigned, bits);
}

fn typesAssignable(expected_type: Type, actual_type: Type) bool {
    if (expected_type.kind() == .unknown or actual_type.kind() == .unknown) return true;
    if (isIntegerType(expected_type) and isIntegerType(actual_type)) return true;
    return typeEql(expected_type, actual_type);
}

fn sameConcreteType(lhs_type: Type, rhs_type: Type) bool {
    if (lhs_type.kind() != rhs_type.kind()) return false;
    return switch (lhs_type) {
        .unknown, .void, .bool, .string, .address, .bytes => true,
        .integer => |left| blk: {
            const right = rhs_type.integer;
            break :blk left.bits == right.bits and left.signed == right.signed and std.meta.eql(left.spelling, right.spelling);
        },
        .named => |left| std.mem.eql(u8, left.name, rhs_type.named.name),
        .contract => |left| std.mem.eql(u8, left.name, rhs_type.contract.name),
        .struct_ => |left| std.mem.eql(u8, left.name, rhs_type.struct_.name),
        .bitfield => |left| std.mem.eql(u8, left.name, rhs_type.bitfield.name),
        .enum_ => |left| std.mem.eql(u8, left.name, rhs_type.enum_.name),
        .refinement => |left| blk: {
            const right = rhs_type.refinement;
            break :blk std.mem.eql(u8, left.name, right.name) and sameConcreteType(left.base_type.*, right.base_type.*);
        },
        else => false,
    };
}

fn unaryOpName(op: ast.UnaryOp) []const u8 {
    return switch (op) {
        .neg => "-",
        .not_ => "!",
        .bit_not => "~",
        .try_ => "try",
    };
}

fn binaryOpName(op: ast.BinaryOp) []const u8 {
    return switch (op) {
        .add => "+",
        .wrapping_add => "+%",
        .sub => "-",
        .wrapping_sub => "-%",
        .mul => "*",
        .wrapping_mul => "*%",
        .div => "/",
        .mod => "%",
        .pow => "**",
        .wrapping_pow => "**%",
        .eq => "==",
        .ne => "!=",
        .lt => "<",
        .le => "<=",
        .gt => ">",
        .ge => ">=",
        .and_and => "&&",
        .or_or => "||",
        .bit_and => "&",
        .bit_or => "|",
        .bit_xor => "^",
        .shl => "<<",
        .wrapping_shl => "<<%",
        .shr => ">>",
        .wrapping_shr => ">>%",
    };
}

fn typeDisplayName(ty: Type) []const u8 {
    return switch (ty) {
        .unknown => "unknown",
        .void => "void",
        .bool => "bool",
        .integer => |integer| integer.spelling orelse "integer",
        .string => "string",
        .address => "address",
        .bytes => "bytes",
        .named => |named| named.name,
        .function => |function| function.name orelse "function",
        .contract => |named| named.name,
        .struct_ => |named| named.name,
        .bitfield => |named| named.name,
        .enum_ => |named| named.name,
        .tuple => "tuple",
        .array => "array",
        .slice => "slice",
        .map => "map",
        .error_union => "error union",
        .refinement => |refinement| refinement.name,
    };
}

test "typesAssignable accepts structurally equal compound types" {
    const testing = std.testing;

    const tuple_type: Type = .{ .tuple = &.{
        .{ .integer = .{ .bits = 8, .signed = false, .spelling = "u8" } },
        .{ .bool = {} },
    } };
    const same_tuple: Type = .{ .tuple = &.{
        .{ .integer = .{ .bits = 8, .signed = false, .spelling = "u8" } },
        .{ .bool = {} },
    } };

    const array_element = try testing.allocator.create(Type);
    defer testing.allocator.destroy(array_element);
    array_element.* = .{ .integer = .{ .bits = 8, .signed = false, .spelling = "u8" } };

    const array_element_copy = try testing.allocator.create(Type);
    defer testing.allocator.destroy(array_element_copy);
    array_element_copy.* = .{ .integer = .{ .bits = 8, .signed = false, .spelling = "u8" } };

    const array_type: Type = .{ .array = .{ .element_type = array_element, .len = 2 } };
    const same_array: Type = .{ .array = .{ .element_type = array_element_copy, .len = 2 } };

    try testing.expect(typesAssignable(tuple_type, same_tuple));
    try testing.expect(typesAssignable(array_type, same_array));
}
