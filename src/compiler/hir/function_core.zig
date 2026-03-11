const std = @import("std");
const mlir = @import("mlir_c_api").c;
const ast = @import("../ast/mod.zig");
const sema = @import("../sema/mod.zig");
const source = @import("../source/mod.zig");
const hir_locals = @import("locals.zig");
const support = @import("support.zig");

const appendEmptyScfYield = support.appendEmptyScfYield;
const appendEmptyYield = support.appendEmptyYield;
const appendOp = support.appendOp;
const appendOraYieldValues = support.appendOraYieldValues;
const appendScfYieldValues = support.appendScfYieldValues;
const appendValueOp = support.appendValueOp;
const boolType = support.boolType;
const createIntegerConstant = support.createIntegerConstant;
const defaultIntegerType = support.defaultIntegerType;
const namedBoolAttr = support.namedBoolAttr;
const nullStringRef = support.nullStringRef;
const strRef = support.strRef;
const LocalEnv = hir_locals.LocalEnv;
const LocalId = hir_locals.LocalId;

pub fn mixin(FunctionLowerer: type, Lowerer: type) type {
    return struct {
        pub fn init(parent: *Lowerer, item_id: ast.ItemId, function: ast.FunctionItem, op: mlir.MlirOperation, return_type: ?mlir.MlirType) FunctionLowerer {
            const block = mlir.oraFuncOpGetBodyBlock(op);
            var self = FunctionLowerer{
                .parent = parent,
                .item_id = item_id,
                .function = function,
                .op = op,
                .block = block,
                .locals = LocalEnv.init(parent.allocator),
                .return_type = return_type,
                .in_try_block = false,
            };

            for (function.parameters, 0..) |parameter, index| {
                self.locals.bindPattern(parent.file, parameter.pattern, mlir.oraBlockGetArgument(block, index)) catch {};
            }
            return self;
        }

        pub fn initContractContext(parent: *Lowerer, block: mlir.MlirBlock) FunctionLowerer {
            return .{
                .parent = parent,
                .item_id = null,
                .function = null,
                .op = std.mem.zeroes(mlir.MlirOperation),
                .block = block,
                .locals = LocalEnv.init(parent.allocator),
                .return_type = null,
                .in_try_block = false,
            };
        }

        pub fn lower(self: *FunctionLowerer) anyerror!void {
            if (self.function) |function| {
                for (function.clauses) |clause| {
                    if (clause.kind != .requires) continue;
                    const condition = try self.lowerExpr(clause.expr, &self.locals);
                    const op = mlir.oraRequiresOpCreate(self.parent.context, self.parent.location(clause.range), condition);
                    if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                    appendOp(self.block, op);
                }

                var locals = try self.cloneLocals(&self.locals);
                const terminated = try self.lowerBody(function.body, &locals);
                if (!terminated) {
                    if (self.return_type) |return_type| {
                        const value = try self.defaultValue(return_type, function.range);
                        const ret = mlir.oraReturnOpCreate(self.parent.context, self.parent.location(function.range), &[_]mlir.MlirValue{value}, 1);
                        if (mlir.oraOperationIsNull(ret)) return error.MlirOperationCreationFailed;
                        appendOp(self.block, ret);
                    } else {
                        const ret = mlir.oraReturnOpCreate(self.parent.context, self.parent.location(function.range), null, 0);
                        if (mlir.oraOperationIsNull(ret)) return error.MlirOperationCreationFailed;
                        appendOp(self.block, ret);
                    }
                }
            }
        }

        pub fn cloneLocals(self: *FunctionLowerer, locals: *const LocalEnv) anyerror!LocalEnv {
            _ = self;
            return locals.clone();
        }

        fn wrapValueForReturn(self: *FunctionLowerer, value: mlir.MlirValue, range: source.TextRange) anyerror!mlir.MlirValue {
            const return_type = self.return_type orelse return value;
            const success_type = mlir.oraErrorUnionTypeGetSuccessType(return_type);
            if (mlir.oraTypeIsNull(success_type)) return value;
            if (mlir.oraTypeEqual(mlir.oraValueGetType(value), return_type)) return value;

            const op = mlir.oraErrorOkOpCreate(self.parent.context, self.parent.location(range), value, return_type);
            if (mlir.oraOperationIsNull(op)) return value;
            return appendValueOp(self.block, op);
        }

        pub fn lowerBody(self: *FunctionLowerer, body_id: ast.BodyId, locals: *LocalEnv) anyerror!bool {
            const body = self.parent.file.body(body_id).*;
            for (body.statements) |statement_id| {
                if (try self.lowerStmt(statement_id, locals)) return true;
            }
            return false;
        }

        pub fn lowerStmt(self: *FunctionLowerer, statement_id: ast.StmtId, locals: *LocalEnv) anyerror!bool {
            switch (self.parent.file.statement(statement_id).*) {
                .VariableDecl => |decl| {
                    const value = if (decl.value) |expr_id|
                        try self.lowerExpr(expr_id, locals)
                    else if (decl.type_expr) |type_expr|
                        try self.defaultValue(self.parent.lowerTypeExpr(type_expr), decl.range)
                    else
                        try self.defaultValue(defaultIntegerType(self.parent.context), decl.range);
                    try self.bindPatternValue(decl.pattern, value, locals);
                    return false;
                },
                .Return => |ret| {
                    const loc = self.parent.location(ret.range);
                    if (ret.value) |expr_id| {
                        const raw_value = try self.lowerExpr(expr_id, locals);
                        const value = try @This().wrapValueForReturn(self, raw_value, ret.range);
                        self.current_return_value = value;
                        defer self.current_return_value = null;
                        if (self.function) |function| {
                            for (function.clauses) |clause| {
                                if (clause.kind != .ensures) continue;
                                const condition = try self.lowerExpr(clause.expr, locals);
                                const ensure = mlir.oraEnsuresOpCreate(self.parent.context, self.parent.location(clause.range), condition);
                                if (mlir.oraOperationIsNull(ensure)) return error.MlirOperationCreationFailed;
                                appendOp(self.block, ensure);
                            }
                        }
                        const op = mlir.oraReturnOpCreate(self.parent.context, loc, &[_]mlir.MlirValue{value}, 1);
                        if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                        appendOp(self.block, op);
                    } else {
                        const op = mlir.oraReturnOpCreate(self.parent.context, loc, null, 0);
                        if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                        appendOp(self.block, op);
                    }
                    return true;
                },
                .Assign => |assign| {
                    const value = try self.lowerExpr(assign.value, locals);
                    try self.storePattern(assign.target, value, locals);
                    return false;
                },
                .Expr => |expr_stmt| {
                    _ = try self.lowerExpr(expr_stmt.expr, locals);
                    return false;
                },
                .Log => |log_stmt| {
                    var args: std.ArrayList(mlir.MlirValue) = .{};
                    for (log_stmt.args) |arg| {
                        try args.append(self.parent.allocator, try self.lowerExpr(arg, locals));
                    }
                    const op = mlir.oraLogOpCreate(
                        self.parent.context,
                        self.parent.location(log_stmt.range),
                        strRef(log_stmt.name),
                        if (args.items.len == 0) null else args.items.ptr,
                        args.items.len,
                    );
                    if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                    appendOp(self.block, op);
                    return false;
                },
                .Lock => |lock_stmt| {
                    try @This().lowerLockLikeStmt(self, lock_stmt.range, lock_stmt.path, locals, true);
                    return false;
                },
                .Unlock => |unlock_stmt| {
                    try @This().lowerLockLikeStmt(self, unlock_stmt.range, unlock_stmt.path, locals, false);
                    return false;
                },
                .Assert => |assert_stmt| {
                    const condition = try self.lowerExpr(assert_stmt.condition, locals);
                    const message = if (assert_stmt.message) |msg| strRef(msg) else nullStringRef();
                    const op = mlir.oraAssertOpCreate(self.parent.context, self.parent.location(assert_stmt.range), condition, message);
                    if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                    appendOp(self.block, op);
                    return false;
                },
                .Assume => |assume_stmt| {
                    const condition = try self.lowerExpr(assume_stmt.condition, locals);
                    const op = mlir.oraAssumeOpCreate(self.parent.context, self.parent.location(assume_stmt.range), condition);
                    if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                    appendOp(self.block, op);
                    return false;
                },
                .Havoc => |havoc_stmt| {
                    const op = mlir.oraHavocOpCreate(self.parent.context, self.parent.location(havoc_stmt.range), strRef(havoc_stmt.name));
                    if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                    appendOp(self.block, op);
                    return false;
                },
                .Block => |block_stmt| {
                    var child_locals = try self.cloneLocals(locals);
                    return self.lowerBody(block_stmt.body, &child_locals);
                },
                .LabeledBlock => |block_stmt| {
                    var child_locals = try self.cloneLocals(locals);
                    return self.lowerBody(block_stmt.body, &child_locals);
                },
                .If => |if_stmt| return self.lowerIfStmt(if_stmt, locals),
                .While => |while_stmt| return self.lowerWhileStmt(while_stmt, locals),
                .For => |for_stmt| {
                    _ = try self.lowerExpr(for_stmt.iterable, locals);
                    const op = try self.parent.createPlaceholderOp(
                        "ora.for_placeholder",
                        self.parent.location(for_stmt.range),
                        &.{namedBoolAttr(self.parent.context, "ora.unsupported", true)},
                    );
                    appendOp(self.block, op);
                    return false;
                },
                .Switch => |switch_stmt| return self.lowerSwitchStmt(switch_stmt, locals),
                .Try => |try_stmt| return self.lowerTryStmt(try_stmt, locals),
                .Break => |jump| {
                    if (self.switch_context != null) {
                        try appendEmptyYield(self.parent.context, self.block, self.parent.location(jump.range));
                        return true;
                    }
                    if (self.loop_context) |loop_context| {
                        const loc = self.parent.location(jump.range);
                        const true_value = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, boolType(self.parent.context), 1));
                        const set_break = mlir.oraMemrefStoreOpCreate(self.parent.context, loc, true_value, loop_context.break_flag, null, 0);
                        if (mlir.oraOperationIsNull(set_break)) return error.MlirOperationCreationFailed;
                        appendOp(self.block, set_break);
                        try self.appendScfYieldFromLocals(self.block, jump.range, locals, loop_context.carried_locals);
                        return true;
                    }
                    const op = try self.parent.createPlaceholderOp("ora.break", self.parent.location(jump.range), &.{});
                    appendOp(self.block, op);
                    return false;
                },
                .Continue => |jump| {
                    if (self.switch_context != null) {
                        const op = try self.parent.createPlaceholderOp("ora.continue", self.parent.location(jump.range), &.{});
                        appendOp(self.block, op);
                        return false;
                    }
                    if (self.loop_context != null) {
                        try self.appendScfYieldFromLocals(self.block, jump.range, locals, self.loop_context.?.carried_locals);
                        return true;
                    }
                    const op = try self.parent.createPlaceholderOp("ora.continue", self.parent.location(jump.range), &.{});
                    appendOp(self.block, op);
                    return false;
                },
                .Error => return false,
            }
        }

        pub fn bindPatternValue(self: *FunctionLowerer, pattern_id: ast.PatternId, value: mlir.MlirValue, locals: *LocalEnv) anyerror!void {
            try locals.bindPattern(self.parent.file, pattern_id, value);
        }

        pub fn storePattern(self: *FunctionLowerer, pattern_id: ast.PatternId, value: mlir.MlirValue, locals: *LocalEnv) anyerror!void {
            switch (self.parent.file.pattern(pattern_id).*) {
                .Name => |name| {
                    if (locals.lookupName(name.name)) |local_id| {
                        try locals.setValue(local_id, value);
                        return;
                    }
                    if (self.parent.item_index.lookup(name.name)) |item_id| {
                        const item = self.parent.file.item(item_id).*;
                        if (item == .Field) {
                            const field = item.Field;
                            const loc = self.parent.location(name.range);
                            const op = switch (field.storage_class) {
                                .storage => blk: {
                                    try @This().maybeEmitGuardedStorageWrite(self, field.name, name.range);
                                    break :blk mlir.oraSStoreOpCreate(self.parent.context, loc, value, strRef(field.name));
                                },
                                .memory => mlir.oraMStoreOpCreate(self.parent.context, loc, value, strRef(field.name)),
                                .tstore => mlir.oraTStoreOpCreate(self.parent.context, loc, value, strRef(field.name)),
                                .none => std.mem.zeroes(mlir.MlirOperation),
                            };
                            if (!mlir.oraOperationIsNull(op)) appendOp(self.block, op);
                            return;
                        }
                    }
                    return error.UnknownAssignmentTarget;
                },
                .StructDestructure => |destructure| {
                    for (destructure.fields) |field| {
                        try self.storePattern(field.binding, value, locals);
                    }
                    return;
                },
                .Field => |field| {
                    const base_value = try @This().lowerPatternValue(self, field.base, locals);
                    const op = mlir.oraStructFieldUpdateOpCreate(
                        self.parent.context,
                        self.parent.location(field.range),
                        base_value,
                        strRef(field.name),
                        value,
                    );
                    if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                    const updated_struct = appendValueOp(self.block, op);
                    try self.storePattern(field.base, updated_struct, locals);
                    return;
                },
                .Index => |index| {
                    const map_value = try @This().lowerPatternValue(self, index.base, locals);
                    const map_type = mlir.oraValueGetType(map_value);
                    const map_value_type = mlir.oraMapTypeGetValueType(map_type);
                    if (map_value_type.ptr != null) {
                        const key_value = try self.lowerExpr(index.index, locals);
                        try @This().appendMapStore(self, index.range, map_value, key_value, value);
                        if (self.parent.file.pattern(index.base).* == .Index) {
                            try self.storePattern(index.base, map_value, locals);
                        }
                        return;
                    }
                },
                else => {},
            }
        }

        fn appendMapStore(
            self: *FunctionLowerer,
            range: source.TextRange,
            map_value: mlir.MlirValue,
            key_value: mlir.MlirValue,
            value: mlir.MlirValue,
        ) anyerror!void {
            const op = mlir.oraMapStoreOpCreate(self.parent.context, self.parent.location(range), map_value, key_value, value);
            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
            appendOp(self.block, op);
        }

        fn maybeEmitGuardedStorageWrite(self: *FunctionLowerer, field_name: []const u8, range: source.TextRange) anyerror!void {
            const guarded_roots = self.parent.guarded_storage_roots orelse return;
            if (!guarded_roots.contains(field_name)) return;

            const guard = mlir.oraTStoreGuardOpCreate(self.parent.context, self.parent.location(range), strRef(field_name));
            if (mlir.oraOperationIsNull(guard)) return error.MlirOperationCreationFailed;
            appendOp(self.block, guard);
        }

        fn lowerPatternValue(self: *FunctionLowerer, pattern_id: ast.PatternId, locals: *LocalEnv) anyerror!mlir.MlirValue {
            return switch (self.parent.file.pattern(pattern_id).*) {
                .Name => |name| blk: {
                    if (locals.lookupName(name.name)) |local_id| {
                        if (locals.getValue(local_id)) |value| break :blk value;
                    }
                    if (self.parent.item_index.lookup(name.name)) |item_id| {
                        const item = self.parent.file.item(item_id).*;
                        if (item == .Field) {
                            const field = item.Field;
                            const result_type = if (field.type_expr) |type_expr| self.parent.lowerTypeExpr(type_expr) else self.parent.lowerSemaType(self.parent.typecheck.pattern_types[pattern_id.index()], name.range);
                            const op = switch (field.storage_class) {
                                .storage => mlir.oraSLoadOpCreate(self.parent.context, self.parent.location(name.range), strRef(field.name), result_type),
                                .memory => mlir.oraMLoadOpCreate(self.parent.context, self.parent.location(name.range), strRef(field.name), result_type),
                                .tstore => mlir.oraTLoadOpCreate(self.parent.context, self.parent.location(name.range), strRef(field.name), result_type),
                                .none => std.mem.zeroes(mlir.MlirOperation),
                            };
                            if (!mlir.oraOperationIsNull(op)) break :blk appendValueOp(self.block, op);
                        }
                    }
                    return error.UnknownAssignmentTarget;
                },
                .Index => |index| blk: {
                    const base_value = try @This().lowerPatternValue(self, index.base, locals);
                    const key_value = try self.lowerExpr(index.index, locals);
                    const result_type = blk2: {
                        const map_value_type = mlir.oraMapTypeGetValueType(mlir.oraValueGetType(base_value));
                        if (map_value_type.ptr != null) break :blk2 map_value_type;
                        break :blk2 self.parent.lowerSemaType(self.parent.typecheck.pattern_types[pattern_id.index()], index.range);
                    };
                    const op = mlir.oraMapGetOpCreate(self.parent.context, self.parent.location(index.range), base_value, key_value, result_type);
                    if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                    break :blk appendValueOp(self.block, op);
                },
                .Field => |field| blk: {
                    const base_value = try @This().lowerPatternValue(self, field.base, locals);
                    const result_type = self.parent.lowerSemaType(self.parent.typecheck.pattern_types[pattern_id.index()], field.range);
                    const op = mlir.oraStructFieldExtractOpCreate(
                        self.parent.context,
                        self.parent.location(field.range),
                        base_value,
                        strRef(field.name),
                        result_type,
                    );
                    if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                    break :blk appendValueOp(self.block, op);
                },
                else => return error.UnknownAssignmentTarget,
            };
        }

        pub fn appendUnsupportedControlPlaceholder(self: *FunctionLowerer, op_name: []const u8, range: source.TextRange) anyerror!void {
            const op = try self.parent.createPlaceholderOp(
                op_name,
                self.parent.location(range),
                &.{namedBoolAttr(self.parent.context, "ora.unsupported", true)},
            );
            appendOp(self.block, op);
        }

        fn lowerLockLikeStmt(
            self: *FunctionLowerer,
            range: source.TextRange,
            path_expr: ast.ExprId,
            locals: *LocalEnv,
            is_lock: bool,
        ) anyerror!void {
            const loc = self.parent.location(range);
            const resource_expr = lockResourceExpr(self.parent.file, path_expr);
            const resource = try self.lowerExpr(resource_expr, locals);
            const owned_key = buildRuntimeLockKey(self.parent.allocator, self.parent.file, self.parent.resolution, path_expr);
            const key = owned_key orelse {
                const op = try self.parent.createPlaceholderOp(
                    if (is_lock) "ora.lock_placeholder" else "ora.unlock_placeholder",
                    loc,
                    &.{namedBoolAttr(self.parent.context, "ora.unsupported", true)},
                );
                appendOp(self.block, op);
                return;
            };
            defer self.parent.allocator.free(key);
            const op = if (is_lock)
                mlir.oraLockOpCreateWithKey(self.parent.context, loc, resource, strRef(key))
            else
                mlir.oraUnlockOpCreateWithKey(self.parent.context, loc, resource, strRef(key));
            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
            appendOp(self.block, op);
        }

        fn lockResourceExpr(file: *const ast.AstFile, expr_id: ast.ExprId) ast.ExprId {
            return switch (file.expression(expr_id).*) {
                .Index => |index| switch (file.expression(index.base).*) {
                    .Name => index.index,
                    else => expr_id,
                },
                else => expr_id,
            };
        }

        fn buildRuntimeLockKey(
            allocator: std.mem.Allocator,
            file: *const ast.AstFile,
            resolution: *const sema.NameResolutionResult,
            expr_id: ast.ExprId,
        ) ?[]const u8 {
            return switch (file.expression(expr_id).*) {
                .Name => |name| if (isStorageFieldName(file, resolution, expr_id))
                    std.fmt.allocPrint(allocator, "{s}", .{name.name}) catch null
                else
                    null,
                .Index => |index| switch (file.expression(index.base).*) {
                    .Name => |name| if (isStorageFieldName(file, resolution, index.base))
                        std.fmt.allocPrint(allocator, "{s}[]", .{name.name}) catch null
                    else
                        null,
                    else => null,
                },
                else => null,
            };
        }

        fn isStorageFieldName(file: *const ast.AstFile, resolution: *const sema.NameResolutionResult, expr_id: ast.ExprId) bool {
            const binding = resolution.expr_bindings[expr_id.index()] orelse return false;
            return switch (binding) {
                .item => |item_id| switch (file.item(item_id).*) {
                    .Field => |field| field.storage_class == .storage,
                    else => false,
                },
                else => false,
            };
        }

        pub fn buildCarriedResultTypes(
            self: *FunctionLowerer,
            locals: *const LocalEnv,
            carried_locals: []const LocalId,
        ) anyerror!?std.ArrayList(mlir.MlirType) {
            var result_types: std.ArrayList(mlir.MlirType) = .{};
            for (carried_locals) |local_id| {
                const value = locals.getValue(local_id) orelse return null;
                try result_types.append(self.parent.allocator, mlir.oraValueGetType(value));
            }
            return result_types;
        }

        pub fn appendOraYieldFromLocals(
            self: *FunctionLowerer,
            block: mlir.MlirBlock,
            range: source.TextRange,
            locals: *const LocalEnv,
            carried_locals: []const LocalId,
        ) anyerror!void {
            if (carried_locals.len == 0) {
                try appendEmptyYield(self.parent.context, block, self.parent.location(range));
                return;
            }

            var yielded_values: std.ArrayList(mlir.MlirValue) = .{};
            for (carried_locals) |local_id| {
                const value = locals.getValue(local_id) orelse return error.MlirOperationCreationFailed;
                try yielded_values.append(self.parent.allocator, value);
            }
            try appendOraYieldValues(self.parent.context, block, self.parent.location(range), yielded_values.items);
        }

        pub fn appendScfYieldFromLocals(
            self: *FunctionLowerer,
            block: mlir.MlirBlock,
            range: source.TextRange,
            locals: *const LocalEnv,
            carried_locals: []const LocalId,
        ) anyerror!void {
            if (carried_locals.len == 0) {
                try appendEmptyScfYield(self.parent.context, block, self.parent.location(range));
                return;
            }

            var yielded_values: std.ArrayList(mlir.MlirValue) = .{};
            for (carried_locals) |local_id| {
                const value = locals.getValue(local_id) orelse return error.MlirOperationCreationFailed;
                try yielded_values.append(self.parent.allocator, value);
            }
            try appendScfYieldValues(self.parent.context, block, self.parent.location(range), yielded_values.items);
        }

        pub fn writeBackCarriedLocals(locals: *LocalEnv, carried_locals: []const LocalId, op: mlir.MlirOperation) anyerror!void {
            for (carried_locals, 0..) |local_id, index| {
                try locals.setValue(local_id, mlir.oraOperationGetResult(op, index));
            }
        }
    };
}
