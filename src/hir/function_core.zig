const std = @import("std");
const mlir = @import("mlir_c_api").c;
const ast = @import("../ast/mod.zig");
const sema = @import("../sema/mod.zig");
const source = @import("../source/mod.zig");
const hir_locals = @import("locals.zig");
const support = @import("support.zig");
const analysis = @import("analysis.zig");

const appendEmptyScfYield = support.appendEmptyScfYield;
const appendEmptyYield = support.appendEmptyYield;
const appendOp = support.appendOp;
const appendOraYieldValues = support.appendOraYieldValues;
const appendScfYieldValues = support.appendScfYieldValues;
const appendValueOp = support.appendValueOp;
const boolType = support.boolType;
const clearKnownTerminator = support.clearKnownTerminator;
const createIntegerConstant = support.createIntegerConstant;
const defaultIntegerType = support.defaultIntegerType;
const cmpPredicate = support.cmpPredicate;
const namedStringAttr = support.namedStringAttr;
const namedBoolAttr = support.namedBoolAttr;
const nullStringRef = support.nullStringRef;
const strRef = support.strRef;
const LocalEnv = hir_locals.LocalEnv;
const LocalId = hir_locals.LocalId;
const LocalIdList = hir_locals.LocalIdList;
const LocalIdSet = hir_locals.LocalIdSet;
const LoopContext = support.LoopContext;
const SwitchContext = support.SwitchContext;
const bodyContainsLoopControl = analysis.bodyContainsLoopControl;
const bodyMayReturn = analysis.bodyMayReturn;
const collectLoopCarriedLocals = analysis.collectLoopCarriedLocals;

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
                .deferred_return_flag = null,
                .deferred_return_value_slot = null,
                .deferred_return_kind = .none,
                .deferred_return_carried_locals = &.{},
                .in_try_block = false,
                .in_ghost_context = function.is_ghost,
            };

            var runtime_index: usize = 0;
            for (function.parameters) |parameter| {
                if (parameter.is_comptime) {
                    if (parent.patternName(parameter.pattern)) |name| {
                        if (parent.substitutedInteger(name)) |integer_text| {
                            const param_type = parent.lowerSemaType(parent.typecheck.pattern_types[parameter.pattern.index()].type, parameter.range);
                            const parsed = support.parseIntLiteral(integer_text) orelse 0;
                            const value = appendValueOp(block, createIntegerConstant(parent.context, parent.location(parameter.range), param_type, parsed));
                            self.locals.bindPattern(parent.file, parameter.pattern, value) catch {};
                        }
                    }
                    continue;
                }
                self.locals.bindPattern(parent.file, parameter.pattern, mlir.oraBlockGetArgument(block, runtime_index)) catch {};
                runtime_index += 1;
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
                .deferred_return_flag = null,
                .deferred_return_value_slot = null,
                .deferred_return_kind = .none,
                .deferred_return_carried_locals = &.{},
                .in_try_block = false,
                .in_ghost_context = false,
            };
        }

        pub fn lower(self: *FunctionLowerer) anyerror!void {
            if (self.function) |function| {
                try @This().insertParameterRefinementGuards(self, function);
                for (function.clauses) |clause| {
                    if (clause.kind != .requires) continue;
                    const condition = try self.lowerExpr(clause.expr, &self.locals);
                    const op = mlir.oraRequiresOpCreate(self.parent.context, self.parent.location(clause.range), condition);
                    if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                    appendOp(self.block, op);
                }
                for (self.extra_verification_clauses) |clause| {
                    if (clause.kind != .requires) continue;
                    const condition = try self.lowerExpr(clause.expr, &self.locals);
                    const op = mlir.oraRequiresOpCreate(self.parent.context, self.parent.location(clause.range), condition);
                    if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                    appendOp(self.block, op);
                }

                var locals = try self.cloneLocals(&self.locals);
                const terminated = try self.lowerBody(function.body, &locals);
                if (!terminated) {
                    try @This().emitEnsuresClauses(self, &locals);
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

        fn emitEnsuresClauses(self: *FunctionLowerer, locals: *LocalEnv) anyerror!void {
            if (self.function) |function| {
                for (function.clauses) |clause| {
                    if (clause.kind != .ensures) continue;
                    const condition = try self.lowerExpr(clause.expr, locals);
                    const ensure = mlir.oraEnsuresOpCreate(self.parent.context, self.parent.location(clause.range), condition);
                    if (mlir.oraOperationIsNull(ensure)) return error.MlirOperationCreationFailed;
                    appendOp(self.block, ensure);
                }
            }
            for (self.extra_verification_clauses) |clause| {
                if (clause.kind != .ensures) continue;
                const condition = try self.lowerExpr(clause.expr, locals);
                const ensure = mlir.oraEnsuresOpCreate(self.parent.context, self.parent.location(clause.range), condition);
                if (mlir.oraOperationIsNull(ensure)) return error.MlirOperationCreationFailed;
                appendOp(self.block, ensure);
            }
        }

        fn insertParameterRefinementGuards(self: *FunctionLowerer, function: ast.FunctionItem) anyerror!void {
            for (function.parameters) |parameter| {
                const param_type = self.parent.typecheck.pattern_types[parameter.pattern.index()].type;
                if (param_type.kind() != .refinement) continue;
                const local_id = self.locals.resolvePatternTarget(self.parent.file, parameter.pattern) orelse continue;
                const param_value = self.locals.getValue(local_id) orelse continue;
                try @This().insertRefinementGuard(self, param_value, param_type.refinement, parameter.range, patternName(self.parent.file, parameter.pattern));
            }
        }

        fn insertRefinementGuard(self: *FunctionLowerer, value: mlir.MlirValue, refinement: sema.RefinementType, range: source.TextRange, var_name: ?[]const u8) anyerror!void {
            if (std.mem.eql(u8, refinement.name, "Exact") or std.mem.eql(u8, refinement.name, "Scaled")) return;

            const loc = self.parent.location(range);
            const base_value = try @This().unwrapRefinementValue(self, value, loc);
            const condition = if (std.mem.eql(u8, refinement.name, "MinValue"))
                try @This().buildMinValueCheck(self, base_value, refinement)
            else if (std.mem.eql(u8, refinement.name, "MaxValue"))
                try @This().buildMaxValueCheck(self, base_value, refinement)
            else if (std.mem.eql(u8, refinement.name, "InRange"))
                try @This().buildInRangeCheck(self, base_value, refinement)
            else if (std.mem.eql(u8, refinement.name, "NonZeroAddress"))
                try @This().buildNonZeroAddressCheck(self, base_value, range)
            else
                return;

            const message = try @This().refinementMessage(self, refinement);
            const op = mlir.oraRefinementGuardOpCreate(self.parent.context, loc, condition, strRef(message));
            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
            defer self.parent.allocator.free(message);

            mlir.oraOperationSetAttributeByName(op, strRef("ora.verification"), namedBoolAttr(self.parent.context, "ora.verification", true).attribute);
            mlir.oraOperationSetAttributeByName(op, strRef("ora.formal"), namedBoolAttr(self.parent.context, "ora.formal", true).attribute);
            mlir.oraOperationSetAttributeByName(op, strRef("ora.verification_type"), namedStringAttr(self.parent.context, "ora.verification_type", "refinement_guard").attribute);
            mlir.oraOperationSetAttributeByName(op, strRef("ora.verification_context"), namedStringAttr(self.parent.context, "ora.verification_context", "parameter_refinement").attribute);
            mlir.oraOperationSetAttributeByName(op, strRef("ora.refinement_kind"), namedStringAttr(self.parent.context, "ora.refinement_kind", refinement.name).attribute);
            if (var_name) |name| {
                const guard_id = try std.fmt.allocPrint(self.parent.allocator, "guard:{s}:{d}:{d}:{d}:{s}:{s}", .{
                    self.parent.sources.file(self.parent.file.file_id).path,
                    self.parent.sources.lineColumn(.{ .file_id = self.parent.file.file_id, .range = range }).line,
                    self.parent.sources.lineColumn(.{ .file_id = self.parent.file.file_id, .range = range }).column,
                    range.len(),
                    refinement.name,
                    name,
                });
                defer self.parent.allocator.free(guard_id);
                mlir.oraOperationSetAttributeByName(op, strRef("ora.guard_id"), namedStringAttr(self.parent.context, "ora.guard_id", guard_id).attribute);
            }

            appendOp(self.block, op);
        }

        fn unwrapRefinementValue(self: *FunctionLowerer, value: mlir.MlirValue, loc: mlir.MlirLocation) anyerror!mlir.MlirValue {
            const value_type = mlir.oraValueGetType(value);
            const base_type = mlir.oraRefinementTypeGetBaseType(value_type);
            if (mlir.oraTypeIsNull(base_type)) return value;
            const op = mlir.oraRefinementToBaseOpCreate(self.parent.context, loc, value, self.block);
            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
            return mlir.oraOperationGetResult(op, 0);
        }

        fn buildMinValueCheck(self: *FunctionLowerer, value: mlir.MlirValue, refinement: sema.RefinementType) anyerror!mlir.MlirValue {
            const min_value = refinementIntArg(refinement.args, 1) orelse return error.MlirOperationCreationFailed;
            const constant = try @This().createTypedIntegerConstant(self, mlir.oraValueGetType(value), min_value, self.parent.location(.{ .start = 0, .end = 0 }));
            const predicate = comparePredicateForBase(refinement.base_type.*, true);
            const op = mlir.oraArithCmpIOpCreate(self.parent.context, self.parent.location(.{ .start = 0, .end = 0 }), predicate, value, constant);
            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
            return appendValueOp(self.block, op);
        }

        fn buildMaxValueCheck(self: *FunctionLowerer, value: mlir.MlirValue, refinement: sema.RefinementType) anyerror!mlir.MlirValue {
            const max_value = refinementIntArg(refinement.args, 1) orelse return error.MlirOperationCreationFailed;
            const constant = try @This().createTypedIntegerConstant(self, mlir.oraValueGetType(value), max_value, self.parent.location(.{ .start = 0, .end = 0 }));
            const predicate = comparePredicateForBase(refinement.base_type.*, false);
            const op = mlir.oraArithCmpIOpCreate(self.parent.context, self.parent.location(.{ .start = 0, .end = 0 }), predicate, value, constant);
            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
            return appendValueOp(self.block, op);
        }

        fn buildInRangeCheck(self: *FunctionLowerer, value: mlir.MlirValue, refinement: sema.RefinementType) anyerror!mlir.MlirValue {
            const min_value = refinementIntArg(refinement.args, 1) orelse return error.MlirOperationCreationFailed;
            const max_value = refinementIntArg(refinement.args, 2) orelse return error.MlirOperationCreationFailed;
            const ty = mlir.oraValueGetType(value);
            const min_constant = try @This().createTypedIntegerConstant(self, ty, min_value, self.parent.location(.{ .start = 0, .end = 0 }));
            const max_constant = try @This().createTypedIntegerConstant(self, ty, max_value, self.parent.location(.{ .start = 0, .end = 0 }));
            const ge_predicate = comparePredicateForBase(refinement.base_type.*, true);
            const le_predicate = comparePredicateForBase(refinement.base_type.*, false);
            const ge_op = mlir.oraArithCmpIOpCreate(self.parent.context, self.parent.location(.{ .start = 0, .end = 0 }), ge_predicate, value, min_constant);
            if (mlir.oraOperationIsNull(ge_op)) return error.MlirOperationCreationFailed;
            const le_op = mlir.oraArithCmpIOpCreate(self.parent.context, self.parent.location(.{ .start = 0, .end = 0 }), le_predicate, value, max_constant);
            if (mlir.oraOperationIsNull(le_op)) return error.MlirOperationCreationFailed;
            const ge_value = appendValueOp(self.block, ge_op);
            const le_value = appendValueOp(self.block, le_op);
            const and_op = mlir.oraArithAndIOpCreate(self.parent.context, self.parent.location(.{ .start = 0, .end = 0 }), ge_value, le_value);
            if (mlir.oraOperationIsNull(and_op)) return error.MlirOperationCreationFailed;
            return appendValueOp(self.block, and_op);
        }

        fn buildNonZeroAddressCheck(self: *FunctionLowerer, value: mlir.MlirValue, range: source.TextRange) anyerror!mlir.MlirValue {
            const loc = self.parent.location(range);
            const addr_to_i160 = mlir.oraAddrToI160OpCreate(self.parent.context, loc, value);
            if (mlir.oraOperationIsNull(addr_to_i160)) return error.MlirOperationCreationFailed;
            const i160_value = appendValueOp(self.block, addr_to_i160);
            const i160_type = mlir.oraIntegerTypeCreate(self.parent.context, 160);
            const zero = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, i160_type, 0));
            const cmp_op = mlir.oraArithCmpIOpCreate(self.parent.context, loc, cmpPredicate("ne"), i160_value, zero);
            if (mlir.oraOperationIsNull(cmp_op)) return error.MlirOperationCreationFailed;
            return appendValueOp(self.block, cmp_op);
        }

        fn createTypedIntegerConstant(self: *FunctionLowerer, ty: mlir.MlirType, value: u256, loc: mlir.MlirLocation) anyerror!mlir.MlirValue {
            const attr = if (value <= std.math.maxInt(i64))
                mlir.oraIntegerAttrCreateI64FromType(ty, @intCast(value))
            else blk: {
                var decimal_buf: [80]u8 = undefined;
                const decimal_text = std.fmt.bufPrint(&decimal_buf, "{}", .{value}) catch return error.MlirOperationCreationFailed;
                break :blk mlir.oraIntegerAttrGetFromString(ty, strRef(decimal_text));
            };
            const op = mlir.oraArithConstantOpCreate(self.parent.context, loc, ty, attr);
            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
            return appendValueOp(self.block, op);
        }

        fn refinementMessage(self: *FunctionLowerer, refinement: sema.RefinementType) anyerror![]const u8 {
            if (std.mem.eql(u8, refinement.name, "MinValue")) {
                const value = refinementIntArg(refinement.args, 1) orelse return error.MlirOperationCreationFailed;
                return std.fmt.allocPrint(self.parent.allocator, "Refinement violation: expected MinValue<u256, {d}>", .{value});
            }
            if (std.mem.eql(u8, refinement.name, "MaxValue")) {
                const value = refinementIntArg(refinement.args, 1) orelse return error.MlirOperationCreationFailed;
                return std.fmt.allocPrint(self.parent.allocator, "Refinement violation: expected MaxValue<u256, {d}>", .{value});
            }
            if (std.mem.eql(u8, refinement.name, "InRange")) {
                const min_value = refinementIntArg(refinement.args, 1) orelse return error.MlirOperationCreationFailed;
                const max_value = refinementIntArg(refinement.args, 2) orelse return error.MlirOperationCreationFailed;
                return std.fmt.allocPrint(self.parent.allocator, "Refinement violation: expected InRange<u256, {d}, {d}>", .{ min_value, max_value });
            }
            if (std.mem.eql(u8, refinement.name, "NonZeroAddress")) {
                return self.parent.allocator.dupe(u8, "Refinement violation: expected NonZeroAddress");
            }
            return error.MlirOperationCreationFailed;
        }

        fn refinementIntArg(args: []const ast.TypeArg, index: usize) ?u256 {
            if (index >= args.len) return null;
            return switch (args[index]) {
                .Integer => |literal| parseU256Literal(literal.text),
                else => null,
            };
        }

        fn parseU256Literal(text: []const u8) ?u256 {
            const base: u8 = if (std.mem.startsWith(u8, text, "0x")) 16 else if (std.mem.startsWith(u8, text, "0b")) 2 else 10;
            const digits = if (base == 10) text else text[2..];
            return std.fmt.parseInt(u256, digits, base) catch null;
        }

        fn comparePredicateForBase(base_type: sema.Type, lower_bound: bool) i64 {
            const is_signed = switch (base_type) {
                .integer => |integer| integer.signed orelse false,
                else => false,
            };
            if (lower_bound) return if (is_signed) cmpPredicate("sge") else 9;
            return if (is_signed) cmpPredicate("sle") else 7;
        }

        fn patternName(file: *const ast.AstFile, pattern_id: ast.PatternId) ?[]const u8 {
            return switch (file.pattern(pattern_id).*) {
                .Name => |name| name.name,
                else => null,
            };
        }

        fn patternRange(file: *const ast.AstFile, pattern_id: ast.PatternId) source.TextRange {
            return switch (file.pattern(pattern_id).*) {
                .Name => |name| name.range,
                .Field => |field| field.range,
                .Index => |index| index.range,
                .StructDestructure => |destructure| destructure.range,
                .Error => |err| err.range,
            };
        }

        fn wrapValueForReturn(self: *FunctionLowerer, value: mlir.MlirValue, range: source.TextRange) anyerror!mlir.MlirValue {
            const return_type = self.return_type orelse return value;
            const success_type = mlir.oraErrorUnionTypeGetSuccessType(return_type);
            if (mlir.oraTypeIsNull(success_type)) {
                return @This().convertValueForFlow(self, value, return_type, range);
            }
            if (mlir.oraTypeEqual(mlir.oraValueGetType(value), return_type)) return value;

            const payload = try @This().convertValueForFlow(self, value, success_type, range);
            const op = mlir.oraErrorOkOpCreate(self.parent.context, self.parent.location(range), payload, return_type);
            if (mlir.oraOperationIsNull(op)) return payload;
            return appendValueOp(self.block, op);
        }

        pub fn ensureDeferredReturnSlots(self: *FunctionLowerer, range: source.TextRange) anyerror!void {
            if (self.deferred_return_flag != null) return;

            const loc = self.parent.location(range);
            const flag_alloc = mlir.oraMemrefAllocaOpCreate(self.parent.context, loc, support.memRefType(self.parent.context, boolType(self.parent.context)));
            if (mlir.oraOperationIsNull(flag_alloc)) return error.MlirOperationCreationFailed;
            self.deferred_return_flag = appendValueOp(self.block, flag_alloc);

            const false_value = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, boolType(self.parent.context), 0));
            const clear_flag = mlir.oraMemrefStoreOpCreate(self.parent.context, loc, false_value, self.deferred_return_flag.?, null, 0);
            if (mlir.oraOperationIsNull(clear_flag)) return error.MlirOperationCreationFailed;
            appendOp(self.block, clear_flag);

            if (self.return_type) |return_type| {
                const value_alloc = mlir.oraMemrefAllocaOpCreate(self.parent.context, loc, support.memRefType(self.parent.context, return_type));
                if (mlir.oraOperationIsNull(value_alloc)) return error.MlirOperationCreationFailed;
                self.deferred_return_value_slot = appendValueOp(self.block, value_alloc);
            }
        }

        pub fn appendDeferredReturnTerminator(self: *FunctionLowerer, range: source.TextRange, locals: *LocalEnv) anyerror!void {
            switch (self.deferred_return_kind) {
                .none => return,
                .ora_yield => try self.appendOraYieldFromLocals(self.block, range, locals, self.deferred_return_carried_locals),
                .scf_yield => try self.appendScfYieldFromLocals(self.block, range, locals, self.deferred_return_carried_locals),
            }
        }

        pub fn appendDeferredReturnCheck(self: *FunctionLowerer, range: source.TextRange) anyerror!void {
            const flag = self.deferred_return_flag orelse return;
            const loc = self.parent.location(range);
            const flag_load = mlir.oraMemrefLoadOpCreate(self.parent.context, loc, flag, null, 0, boolType(self.parent.context));
            if (mlir.oraOperationIsNull(flag_load)) return error.MlirOperationCreationFailed;
            const should_return = appendValueOp(self.block, flag_load);

            const branch = mlir.oraConditionalReturnOpCreate(self.parent.context, loc, should_return);
            if (mlir.oraOperationIsNull(branch)) return error.MlirOperationCreationFailed;
            appendOp(self.block, branch);

            const then_block = mlir.oraConditionalReturnOpGetThenBlock(branch);
            const else_block = mlir.oraConditionalReturnOpGetElseBlock(branch);
            if (mlir.oraBlockIsNull(then_block) or mlir.oraBlockIsNull(else_block)) {
                return error.MlirOperationCreationFailed;
            }

            if (self.deferred_return_value_slot) |slot| {
                const return_type = self.return_type orelse return error.MlirOperationCreationFailed;
                const value_load = mlir.oraMemrefLoadOpCreate(self.parent.context, loc, slot, null, 0, return_type);
                if (mlir.oraOperationIsNull(value_load)) return error.MlirOperationCreationFailed;
                const value = appendValueOp(then_block, value_load);
                const ret = mlir.oraReturnOpCreate(self.parent.context, loc, &[_]mlir.MlirValue{value}, 1);
                if (mlir.oraOperationIsNull(ret)) return error.MlirOperationCreationFailed;
                appendOp(then_block, ret);
            } else {
                const ret = mlir.oraReturnOpCreate(self.parent.context, loc, null, 0);
                if (mlir.oraOperationIsNull(ret)) return error.MlirOperationCreationFailed;
                appendOp(then_block, ret);
            }

            try appendEmptyYield(self.parent.context, else_block, loc);
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
                    if (self.deferred_return_flag) |return_flag| {
                        if (self.deferred_return_kind == .none) {
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
                                for (self.extra_verification_clauses) |clause| {
                                    if (clause.kind != .ensures) continue;
                                    const condition = try self.lowerExpr(clause.expr, locals);
                                    const ensure = mlir.oraEnsuresOpCreate(self.parent.context, self.parent.location(clause.range), condition);
                                    if (mlir.oraOperationIsNull(ensure)) return error.MlirOperationCreationFailed;
                                    appendOp(self.block, ensure);
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
                        }
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
                            for (self.extra_verification_clauses) |clause| {
                                if (clause.kind != .ensures) continue;
                                const condition = try self.lowerExpr(clause.expr, locals);
                                const ensure = mlir.oraEnsuresOpCreate(self.parent.context, self.parent.location(clause.range), condition);
                                if (mlir.oraOperationIsNull(ensure)) return error.MlirOperationCreationFailed;
                                appendOp(self.block, ensure);
                            }
                            if (self.deferred_return_value_slot) |slot| {
                                const store_value = mlir.oraMemrefStoreOpCreate(self.parent.context, loc, value, slot, null, 0);
                                if (mlir.oraOperationIsNull(store_value)) return error.MlirOperationCreationFailed;
                                appendOp(self.block, store_value);
                            }
                        }

                        const true_value = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, boolType(self.parent.context), 1));
                        const set_flag = mlir.oraMemrefStoreOpCreate(self.parent.context, loc, true_value, return_flag, null, 0);
                        if (mlir.oraOperationIsNull(set_flag)) return error.MlirOperationCreationFailed;
                        appendOp(self.block, set_flag);
                        try self.appendDeferredReturnTerminator(ret.range, locals);
                        return true;
                    }

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
                        for (self.extra_verification_clauses) |clause| {
                            if (clause.kind != .ensures) continue;
                            const condition = try self.lowerExpr(clause.expr, locals);
                            const ensure = mlir.oraEnsuresOpCreate(self.parent.context, self.parent.location(clause.range), condition);
                            if (mlir.oraOperationIsNull(ensure)) return error.MlirOperationCreationFailed;
                            appendOp(self.block, ensure);
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
                    const value = switch (assign.op) {
                        .assign => try self.lowerExpr(assign.value, locals),
                        .add_assign => blk: {
                            const lhs = try @This().lowerPatternValue(self, assign.target, locals);
                            const rhs = try self.lowerExpr(assign.value, locals);
                            const op = mlir.oraArithAddIOpCreate(self.parent.context, self.parent.location(assign.range), lhs, rhs);
                            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                            break :blk appendValueOp(self.block, op);
                        },
                        .wrapping_add_assign => blk: {
                            const lhs = try @This().lowerPatternValue(self, assign.target, locals);
                            const rhs = try self.lowerExpr(assign.value, locals);
                            const op = mlir.oraAddWrappingOpCreate(self.parent.context, self.parent.location(assign.range), lhs, rhs, mlir.oraValueGetType(lhs));
                            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                            break :blk appendValueOp(self.block, op);
                        },
                        .bit_and_assign => blk: {
                            const lhs = try @This().lowerPatternValue(self, assign.target, locals);
                            const rhs = try self.lowerExpr(assign.value, locals);
                            const op = mlir.oraArithAndIOpCreate(self.parent.context, self.parent.location(assign.range), lhs, rhs);
                            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                            break :blk appendValueOp(self.block, op);
                        },
                        .sub_assign => blk: {
                            const lhs = try @This().lowerPatternValue(self, assign.target, locals);
                            const rhs = try self.lowerExpr(assign.value, locals);
                            const op = mlir.oraArithSubIOpCreate(self.parent.context, self.parent.location(assign.range), lhs, rhs);
                            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                            break :blk appendValueOp(self.block, op);
                        },
                        .wrapping_sub_assign => blk: {
                            const lhs = try @This().lowerPatternValue(self, assign.target, locals);
                            const rhs = try self.lowerExpr(assign.value, locals);
                            const op = mlir.oraSubWrappingOpCreate(self.parent.context, self.parent.location(assign.range), lhs, rhs, mlir.oraValueGetType(lhs));
                            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                            break :blk appendValueOp(self.block, op);
                        },
                        .bit_or_assign => blk: {
                            const lhs = try @This().lowerPatternValue(self, assign.target, locals);
                            const rhs = try self.lowerExpr(assign.value, locals);
                            const op = mlir.oraArithOrIOpCreate(self.parent.context, self.parent.location(assign.range), lhs, rhs);
                            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                            break :blk appendValueOp(self.block, op);
                        },
                        .mul_assign => blk: {
                            const lhs = try @This().lowerPatternValue(self, assign.target, locals);
                            const rhs = try self.lowerExpr(assign.value, locals);
                            const op = mlir.oraArithMulIOpCreate(self.parent.context, self.parent.location(assign.range), lhs, rhs);
                            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                            break :blk appendValueOp(self.block, op);
                        },
                        .wrapping_mul_assign => blk: {
                            const lhs = try @This().lowerPatternValue(self, assign.target, locals);
                            const rhs = try self.lowerExpr(assign.value, locals);
                            const op = mlir.oraMulWrappingOpCreate(self.parent.context, self.parent.location(assign.range), lhs, rhs, mlir.oraValueGetType(lhs));
                            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                            break :blk appendValueOp(self.block, op);
                        },
                        .bit_xor_assign => blk: {
                            const lhs = try @This().lowerPatternValue(self, assign.target, locals);
                            const rhs = try self.lowerExpr(assign.value, locals);
                            const op = mlir.oraArithXorIOpCreate(self.parent.context, self.parent.location(assign.range), lhs, rhs);
                            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                            break :blk appendValueOp(self.block, op);
                        },
                        .shl_assign => blk: {
                            const lhs = try @This().lowerPatternValue(self, assign.target, locals);
                            const rhs = try self.lowerExpr(assign.value, locals);
                            const op = mlir.oraArithShlIOpCreate(self.parent.context, self.parent.location(assign.range), lhs, rhs);
                            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                            break :blk appendValueOp(self.block, op);
                        },
                        .shr_assign => blk: {
                            const lhs = try @This().lowerPatternValue(self, assign.target, locals);
                            const rhs = try self.lowerExpr(assign.value, locals);
                            const op = mlir.oraArithShrSIOpCreate(self.parent.context, self.parent.location(assign.range), lhs, rhs);
                            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                            break :blk appendValueOp(self.block, op);
                        },
                        .pow_assign => blk: {
                            const lhs = try @This().lowerPatternValue(self, assign.target, locals);
                            const rhs = try self.lowerExpr(assign.value, locals);
                            break :blk try @This().lowerCheckedPower(self, lhs, rhs, mlir.oraValueGetType(lhs), assign.range);
                        },
                        .div_assign => blk: {
                            const lhs = try @This().lowerPatternValue(self, assign.target, locals);
                            const rhs = try self.lowerExpr(assign.value, locals);
                            const op = mlir.oraArithDivSIOpCreate(self.parent.context, self.parent.location(assign.range), lhs, rhs);
                            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                            break :blk appendValueOp(self.block, op);
                        },
                        .mod_assign => blk: {
                            const lhs = try @This().lowerPatternValue(self, assign.target, locals);
                            const rhs = try self.lowerExpr(assign.value, locals);
                            const op = mlir.oraArithRemSIOpCreate(self.parent.context, self.parent.location(assign.range), lhs, rhs);
                            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                            break :blk appendValueOp(self.block, op);
                        },
                    };
                    try self.storePattern(assign.target, value, locals);
                    return false;
                },
                .Expr => |expr_stmt| {
                    if (self.parent.file.expression(expr_stmt.expr).* == .TypeValue) {
                        return false;
                    }
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
                    if (self.in_ghost_context) {
                        mlir.oraOperationSetAttributeByName(op, strRef("ora.ghost"), namedBoolAttr(self.parent.context, "ora.ghost", true).attribute);
                        mlir.oraOperationSetAttributeByName(op, strRef("ora.verification"), namedBoolAttr(self.parent.context, "ora.verification", true).attribute);
                        mlir.oraOperationSetAttributeByName(op, strRef("ora.formal"), namedBoolAttr(self.parent.context, "ora.formal", true).attribute);
                        mlir.oraOperationSetAttributeByName(op, strRef("ora.verification_type"), namedStringAttr(self.parent.context, "ora.verification_type", "assert").attribute);
                        mlir.oraOperationSetAttributeByName(op, strRef("ora.verification_context"), namedStringAttr(self.parent.context, "ora.verification_context", "ghost_assertion").attribute);
                    }
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
                .For => |for_stmt| return @This().lowerForStmt(self, for_stmt, locals),
                .Switch => |switch_stmt| return self.lowerSwitchStmt(switch_stmt, locals),
                .Try => |try_stmt| return self.lowerTryStmt(try_stmt, locals),
                .Break => |jump| {
                    if (@This().findTargetSwitchContext(self, jump.label)) |switch_context| {
                        if (switch_context.carried_locals.len == 0) {
                            try appendEmptyYield(self.parent.context, self.block, self.parent.location(jump.range));
                        } else {
                            try self.appendOraYieldFromLocals(self.block, jump.range, locals, switch_context.carried_locals);
                        }
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
                    const op = mlir.oraBreakOpCreate(self.parent.context, self.parent.location(jump.range), nullStringRef(), null, 0);
                    if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                    appendOp(self.block, op);
                    return false;
                },
                .Continue => |jump| {
                    if (@This().findContinuableSwitchContext(self, jump.label)) |switch_context| {
                        const loc = self.parent.location(jump.range);
                        if (jump.value) |expr_id| {
                            const raw_value = try self.lowerExpr(expr_id, locals);
                            const target_type = switch_context.value_type orelse mlir.oraValueGetType(raw_value);
                            const value = try @This().convertValueForFlow(self, raw_value, target_type, jump.range);
                            const store = mlir.oraMemrefStoreOpCreate(self.parent.context, loc, value, switch_context.value_slot.?, null, 0);
                            if (mlir.oraOperationIsNull(store)) return error.MlirOperationCreationFailed;
                            appendOp(self.block, store);
                        }
                        const true_value = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, boolType(self.parent.context), 1));
                        const set_continue = mlir.oraMemrefStoreOpCreate(self.parent.context, loc, true_value, switch_context.continue_flag.?, null, 0);
                        if (mlir.oraOperationIsNull(set_continue)) return error.MlirOperationCreationFailed;
                        appendOp(self.block, set_continue);
                        if (switch_context.carried_locals.len == 0) {
                            try appendEmptyYield(self.parent.context, self.block, loc);
                        } else {
                            try self.appendOraYieldFromLocals(self.block, jump.range, locals, switch_context.carried_locals);
                        }
                        return true;
                    }
                    if (self.switch_context != null and jump.label == null) {
                        const op = mlir.oraContinueOpCreate(self.parent.context, self.parent.location(jump.range), nullStringRef());
                        if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                        appendOp(self.block, op);
                        return false;
                    }
                    if (self.loop_context != null) {
                        try self.appendScfYieldFromLocals(self.block, jump.range, locals, self.loop_context.?.carried_locals);
                        return true;
                    }
                    const op = mlir.oraContinueOpCreate(self.parent.context, self.parent.location(jump.range), nullStringRef());
                    if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                    appendOp(self.block, op);
                    return false;
                },
                .Error => return false,
            }
        }

        fn findTargetSwitchContext(self: *FunctionLowerer, label: ?[]const u8) ?*const SwitchContext {
            var current = self.switch_context;
            while (current) |switch_context| : (current = switch_context.parent) {
                if (label) |target| {
                    if (switch_context.label) |current_label| {
                        if (std.mem.eql(u8, current_label, target)) return switch_context;
                    }
                } else {
                    return switch_context;
                }
            }
            return null;
        }

        fn findContinuableSwitchContext(self: *FunctionLowerer, label: ?[]const u8) ?*const SwitchContext {
            var current = self.switch_context;
            while (current) |switch_context| : (current = switch_context.parent) {
                if (switch_context.continue_flag == null or switch_context.value_slot == null) continue;
                if (label) |target| {
                    if (switch_context.label) |current_label| {
                        if (std.mem.eql(u8, current_label, target)) return switch_context;
                    }
                } else {
                    return switch_context;
                }
            }
            return null;
        }

        pub fn bindPatternValue(self: *FunctionLowerer, pattern_id: ast.PatternId, value: mlir.MlirValue, locals: *LocalEnv) anyerror!void {
            switch (self.parent.file.pattern(pattern_id).*) {
                .StructDestructure => |destructure| {
                    for (destructure.fields) |field| {
                        const result_type = self.parent.lowerSemaType(self.parent.typecheck.pattern_types[field.binding.index()].type, field.range);
                        const op = mlir.oraStructFieldExtractOpCreate(
                            self.parent.context,
                            self.parent.location(field.range),
                            value,
                            strRef(field.name),
                            result_type,
                        );
                        if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                        const field_value = appendValueOp(self.block, op);
                        try self.bindPatternValue(field.binding, field_value, locals);
                    }
                },
                else => {
                    const range = patternRange(self.parent.file, pattern_id);
                    const target_type = self.parent.lowerSemaType(self.parent.typecheck.pattern_types[pattern_id.index()].type, range);
                    const converted = try @This().convertValueForFlow(self, value, target_type, range);
                    try locals.bindPattern(self.parent.file, pattern_id, converted);
                },
            }
        }

        pub fn convertValueForFlow(
            self: *FunctionLowerer,
            value: mlir.MlirValue,
            target_type: mlir.MlirType,
            range: source.TextRange,
        ) anyerror!mlir.MlirValue {
            const value_type = mlir.oraValueGetType(value);
            if (mlir.oraTypeEqual(value_type, target_type)) return value;

            const loc = self.parent.location(range);
            const value_ref_base = mlir.oraRefinementTypeGetBaseType(value_type);
            const target_ref_base = mlir.oraRefinementTypeGetBaseType(target_type);

            if (!mlir.oraTypeIsNull(value_ref_base) and mlir.oraTypeEqual(value_ref_base, target_type)) {
                const op = mlir.oraRefinementToBaseOpCreate(self.parent.context, loc, value, self.block);
                if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                return mlir.oraOperationGetResult(op, 0);
            }

            if (!mlir.oraTypeIsNull(target_ref_base) and mlir.oraTypeEqual(value_type, target_ref_base)) {
                const op = mlir.oraBaseToRefinementOpCreate(self.parent.context, loc, value, target_type, self.block);
                if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                return mlir.oraOperationGetResult(op, 0);
            }

            if (!mlir.oraTypeIsNull(value_ref_base) and !mlir.oraTypeIsNull(target_ref_base) and mlir.oraTypeEqual(value_ref_base, target_ref_base)) {
                const base_op = mlir.oraRefinementToBaseOpCreate(self.parent.context, loc, value, self.block);
                if (mlir.oraOperationIsNull(base_op)) return error.MlirOperationCreationFailed;
                const base_value = mlir.oraOperationGetResult(base_op, 0);
                const refine_op = mlir.oraBaseToRefinementOpCreate(self.parent.context, loc, base_value, target_type, self.block);
                if (mlir.oraOperationIsNull(refine_op)) return error.MlirOperationCreationFailed;
                return mlir.oraOperationGetResult(refine_op, 0);
            }

            const value_is_int = mlir.oraTypeIsAInteger(value_type);
            const target_is_int = mlir.oraTypeIsAInteger(target_type);

            if (mlir.oraTypeIsAddressType(value_type) and target_is_int and mlir.oraIntegerTypeGetWidth(target_type) == 160) {
                const op = mlir.oraAddrToI160OpCreate(self.parent.context, loc, value);
                if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                return appendValueOp(self.block, op);
            }

            if (value_is_int and mlir.oraTypeIsAddressType(target_type) and mlir.oraIntegerTypeGetWidth(value_type) == 160) {
                const op = mlir.oraI160ToAddrOpCreate(self.parent.context, loc, value);
                if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                return appendValueOp(self.block, op);
            }

            if (mlir.oraTypeEqual(target_type, boolType(self.parent.context)) and value_is_int) {
                const zero = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, value_type, 0));
                return appendValueOp(self.block, self.createCompareOp(loc, "ne", value, zero));
            }

            if (!(value_is_int and target_is_int)) return value;

            const value_width = mlir.oraIntegerTypeGetWidth(value_type);
            const target_width = mlir.oraIntegerTypeGetWidth(target_type);
            if (value_width == target_width) {
                const op = mlir.oraArithBitcastOpCreate(self.parent.context, loc, value, target_type);
                if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                return appendValueOp(self.block, op);
            }

            if (value_width > target_width) {
                const op = mlir.oraArithTruncIOpCreate(self.parent.context, loc, value, target_type);
                if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                return appendValueOp(self.block, op);
            }

            const op = if (mlir.oraIntegerTypeIsSigned(value_type))
                mlir.oraArithExtSIOpCreate(self.parent.context, loc, value, target_type)
            else
                mlir.oraArithExtUIOpCreate(self.parent.context, loc, value, target_type);
            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
            return appendValueOp(self.block, op);
        }

        pub fn lowerPowerWithOverflow(
            self: *FunctionLowerer,
            lhs: mlir.MlirValue,
            rhs: mlir.MlirValue,
            result_type: mlir.MlirType,
            range: source.TextRange,
        ) anyerror!struct { value: mlir.MlirValue, overflow: mlir.MlirValue } {
            const loc = self.parent.location(range);
            const is_signed = mlir.oraTypeIsAInteger(result_type) and mlir.oraIntegerTypeIsSigned(result_type);
            const value_ty = result_type;
            const one = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, value_ty, 1));
            const zero = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, value_ty, 0));
            var result = one;
            var factor = lhs;
            var overflow = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, boolType(self.parent.context), 0));

            const bit_width: u32 = if (mlir.oraTypeIsAInteger(value_ty))
                @intCast(mlir.oraIntegerTypeGetWidth(value_ty))
            else
                256;

            var bit_index: u32 = 0;
            while (bit_index < bit_width) : (bit_index += 1) {
                const shift_amount = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, value_ty, @intCast(bit_index)));
                const shifted = try @This().createArithShiftRightUnsigned(self, rhs, shift_amount, loc);
                const current_bit = appendValueOp(self.block, mlir.oraArithAndIOpCreate(self.parent.context, loc, shifted, one));
                const bit_is_set = appendValueOp(self.block, self.createCompareOp(loc, "ne", current_bit, zero));

                const multiplied = appendValueOp(self.block, mlir.oraArithMulIOpCreate(self.parent.context, loc, result, factor));
                const mul_overflow = try @This().computeMulOverflowFlag(self, multiplied, result, factor, value_ty, is_signed, range);
                const overflow_with_mul = appendValueOp(self.block, mlir.oraArithOrIOpCreate(self.parent.context, loc, overflow, mul_overflow));
                const after_mul = try @This().selectPowerState(self, bit_is_set, multiplied, overflow_with_mul, result, overflow, value_ty, boolType(self.parent.context), loc);
                result = after_mul.value;
                overflow = after_mul.overflow;

                if (bit_index + 1 < bit_width) {
                    const next_shift_amount = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, value_ty, @intCast(bit_index + 1)));
                    const remaining = try @This().createArithShiftRightUnsigned(self, rhs, next_shift_amount, loc);
                    const has_more_bits = appendValueOp(self.block, self.createCompareOp(loc, "ne", remaining, zero));

                    const squared = appendValueOp(self.block, mlir.oraArithMulIOpCreate(self.parent.context, loc, factor, factor));
                    const sq_overflow = try @This().computeMulOverflowFlag(self, squared, factor, factor, value_ty, is_signed, range);
                    const overflow_with_square = appendValueOp(self.block, mlir.oraArithOrIOpCreate(self.parent.context, loc, overflow, sq_overflow));
                    const after_square = try @This().selectPowerState(self, has_more_bits, squared, overflow_with_square, factor, overflow, value_ty, boolType(self.parent.context), loc);
                    factor = after_square.value;
                    overflow = after_square.overflow;
                }
            }

            const power_op = mlir.oraPowerOpCreate(self.parent.context, loc, lhs, rhs, value_ty);
            if (mlir.oraOperationIsNull(power_op)) return error.MlirOperationCreationFailed;
            const power_value = appendValueOp(self.block, power_op);
            return .{ .value = power_value, .overflow = overflow };
        }

        pub fn lowerCheckedPower(
            self: *FunctionLowerer,
            lhs: mlir.MlirValue,
            rhs: mlir.MlirValue,
            result_type: mlir.MlirType,
            range: source.TextRange,
        ) anyerror!mlir.MlirValue {
            const power = try @This().lowerPowerWithOverflow(self, lhs, rhs, result_type, range);
            try @This().emitOverflowAssert(self, power.overflow, "checked power overflow", range);
            return power.value;
        }

        fn createArithShiftRightUnsigned(self: *FunctionLowerer, lhs: mlir.MlirValue, rhs: mlir.MlirValue, loc: mlir.MlirLocation) anyerror!mlir.MlirValue {
            const op = mlir.oraArithShrUIOpCreate(self.parent.context, loc, lhs, rhs);
            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
            return appendValueOp(self.block, op);
        }

        fn selectPowerState(
            self: *FunctionLowerer,
            condition: mlir.MlirValue,
            then_value: mlir.MlirValue,
            then_overflow: mlir.MlirValue,
            else_value: mlir.MlirValue,
            else_overflow: mlir.MlirValue,
            value_ty: mlir.MlirType,
            flag_ty: mlir.MlirType,
            loc: mlir.MlirLocation,
        ) anyerror!struct { value: mlir.MlirValue, overflow: mlir.MlirValue } {
            const result_types = [_]mlir.MlirType{ value_ty, flag_ty };
            const if_op = mlir.oraScfIfOpCreate(self.parent.context, loc, condition, &result_types, result_types.len, true);
            if (mlir.oraOperationIsNull(if_op)) return error.MlirOperationCreationFailed;
            appendOp(self.block, if_op);

            const then_block = mlir.oraScfIfOpGetThenBlock(if_op);
            const else_block = mlir.oraScfIfOpGetElseBlock(if_op);
            if (mlir.oraBlockIsNull(then_block) or mlir.oraBlockIsNull(else_block)) return error.MlirOperationCreationFailed;

            try appendScfYieldValues(self.parent.context, then_block, loc, &[_]mlir.MlirValue{ then_value, then_overflow });
            try appendScfYieldValues(self.parent.context, else_block, loc, &[_]mlir.MlirValue{ else_value, else_overflow });

            return .{
                .value = mlir.oraOperationGetResult(if_op, 0),
                .overflow = mlir.oraOperationGetResult(if_op, 1),
            };
        }

        fn computeMulOverflowFlag(
            self: *FunctionLowerer,
            product: mlir.MlirValue,
            lhs: mlir.MlirValue,
            rhs: mlir.MlirValue,
            value_ty: mlir.MlirType,
            is_signed: bool,
            range: source.TextRange,
        ) anyerror!mlir.MlirValue {
            const loc = self.parent.location(range);
            const zero = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, value_ty, 0));
            const rhs_non_zero = appendValueOp(self.block, self.createCompareOp(loc, "ne", rhs, zero));

            if (!is_signed) {
                const quot_op = mlir.oraArithDivUIOpCreate(self.parent.context, loc, product, rhs);
                if (mlir.oraOperationIsNull(quot_op)) return error.MlirOperationCreationFailed;
                mlir.oraOperationSetAttributeByName(quot_op, strRef("ora.guard_internal"), mlir.oraStringAttrCreate(self.parent.context, strRef("true")));
                const quotient = appendValueOp(self.block, quot_op);
                const mismatch = appendValueOp(self.block, self.createCompareOp(loc, "ne", quotient, lhs));
                const and_op = mlir.oraArithAndIOpCreate(self.parent.context, loc, mismatch, rhs_non_zero);
                if (mlir.oraOperationIsNull(and_op)) return error.MlirOperationCreationFailed;
                return appendValueOp(self.block, and_op);
            }

            const bit_width: u32 = if (mlir.oraTypeIsAInteger(value_ty))
                @intCast(mlir.oraIntegerTypeGetWidth(value_ty))
            else
                256;
            const one = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, value_ty, 1));
            const shift_amt = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, value_ty, @intCast(bit_width - 1)));
            const min_int_op = mlir.oraArithShlIOpCreate(self.parent.context, loc, one, shift_amt);
            if (mlir.oraOperationIsNull(min_int_op)) return error.MlirOperationCreationFailed;
            const min_int = appendValueOp(self.block, min_int_op);
            const neg1 = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, value_ty, -1));

            const lhs_is_min = appendValueOp(self.block, self.createCompareOp(loc, "eq", lhs, min_int));
            const rhs_is_neg1 = appendValueOp(self.block, self.createCompareOp(loc, "eq", rhs, neg1));
            const special_case = appendValueOp(self.block, mlir.oraArithAndIOpCreate(self.parent.context, loc, lhs_is_min, rhs_is_neg1));

            const quot_op = mlir.oraArithDivSIOpCreate(self.parent.context, loc, product, rhs);
            if (mlir.oraOperationIsNull(quot_op)) return error.MlirOperationCreationFailed;
            mlir.oraOperationSetAttributeByName(quot_op, strRef("ora.guard_internal"), mlir.oraStringAttrCreate(self.parent.context, strRef("true")));
            const quotient = appendValueOp(self.block, quot_op);
            const mismatch = appendValueOp(self.block, self.createCompareOp(loc, "ne", quotient, lhs));
            const general_case = appendValueOp(self.block, mlir.oraArithAndIOpCreate(self.parent.context, loc, mismatch, rhs_non_zero));
            const overflow_op = mlir.oraArithOrIOpCreate(self.parent.context, loc, special_case, general_case);
            if (mlir.oraOperationIsNull(overflow_op)) return error.MlirOperationCreationFailed;
            return appendValueOp(self.block, overflow_op);
        }

        fn emitOverflowAssert(self: *FunctionLowerer, overflow_flag: mlir.MlirValue, message: []const u8, range: source.TextRange) anyerror!void {
            const loc = self.parent.location(range);
            const true_val = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, boolType(self.parent.context), 1));
            const not_op = mlir.oraArithXorIOpCreate(self.parent.context, loc, overflow_flag, true_val);
            if (mlir.oraOperationIsNull(not_op)) return error.MlirOperationCreationFailed;
            const no_overflow = appendValueOp(self.block, not_op);
            const assert_op = mlir.oraAssertOpCreate(self.parent.context, loc, no_overflow, strRef(message));
            if (mlir.oraOperationIsNull(assert_op)) return error.MlirOperationCreationFailed;
            appendOp(self.block, assert_op);
        }

        pub fn storePattern(self: *FunctionLowerer, pattern_id: ast.PatternId, value: mlir.MlirValue, locals: *LocalEnv) anyerror!void {
            switch (self.parent.file.pattern(pattern_id).*) {
                .Name => |name| {
                    if (locals.lookupName(name.name)) |local_id| {
                        const target_type = self.parent.lowerSemaType(self.parent.typecheck.pattern_types[local_id.index()].type, name.range);
                        const converted = try @This().convertValueForFlow(self, value, target_type, name.range);
                        try locals.setValue(local_id, converted);
                        return;
                    }
                    if (self.parent.item_index.lookup(name.name)) |item_id| {
                        const item = self.parent.file.item(item_id).*;
                        if (item == .Field) {
                            const field = item.Field;
                            const target_type = if (field.type_expr) |type_expr|
                                self.parent.lowerTypeExpr(type_expr)
                            else
                                self.parent.lowerSemaType(self.parent.typecheck.pattern_types[pattern_id.index()].type, name.range);
                            const converted = try @This().convertValueForFlow(self, value, target_type, name.range);
                            const loc = self.parent.location(name.range);
                            const op = switch (field.storage_class) {
                                .storage => blk: {
                                    try @This().maybeEmitGuardedStorageWrite(self, field.name, name.range);
                                    break :blk mlir.oraSStoreOpCreate(self.parent.context, loc, converted, strRef(field.name));
                                },
                                .memory => mlir.oraMStoreOpCreate(self.parent.context, loc, converted, strRef(field.name)),
                                .tstore => mlir.oraTStoreOpCreate(self.parent.context, loc, converted, strRef(field.name)),
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
                        const result_type = self.parent.lowerSemaType(self.parent.typecheck.pattern_types[field.binding.index()].type, field.range);
                        const op = mlir.oraStructFieldExtractOpCreate(
                            self.parent.context,
                            self.parent.location(field.range),
                            value,
                            strRef(field.name),
                            result_type,
                        );
                        if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                        const field_value = appendValueOp(self.block, op);
                        try self.storePattern(field.binding, field_value, locals);
                    }
                    return;
                },
                .Field => |field| {
                    const base_type = @This().patternType(self, field.base, locals);
                    if (@This().isBitfieldLikeType(self, base_type)) {
                        const updated_bitfield = try @This().createBitfieldFieldUpdate(self, try @This().lowerPatternValue(self, field.base, locals), base_type, field.name, value, field.range);
                        try self.storePattern(field.base, updated_bitfield, locals);
                        return;
                    }
                    const base_value = try @This().lowerPatternValue(self, field.base, locals);
                    const target_type = self.parent.lowerSemaType(self.parent.typecheck.pattern_types[pattern_id.index()].type, field.range);
                    const converted = try @This().convertValueForFlow(self, value, target_type, field.range);
                    const op = mlir.oraStructFieldUpdateOpCreate(
                        self.parent.context,
                        self.parent.location(field.range),
                        base_value,
                        strRef(field.name),
                        converted,
                    );
                    if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                    const updated_struct = appendValueOp(self.block, op);
                    try self.storePattern(field.base, updated_struct, locals);
                    return;
                },
                .Index => |index| {
                    const base_value = try @This().lowerPatternValue(self, index.base, locals);
                    const base_type = mlir.oraValueGetType(base_value);
                    const target_type = self.parent.lowerSemaType(self.parent.typecheck.pattern_types[pattern_id.index()].type, index.range);
                    const converted = try @This().convertValueForFlow(self, value, target_type, index.range);
                    if (mlir.oraTypeIsAMemRef(base_type)) {
                        const key_value = try self.lowerExpr(index.index, locals);
                        const index_value = try @This().convertIndexToIndexType(self, key_value, index.range);
                        const op = mlir.oraMemrefStoreOpCreate(
                            self.parent.context,
                            self.parent.location(index.range),
                            converted,
                            base_value,
                            &[_]mlir.MlirValue{index_value},
                            1,
                        );
                        if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                        appendOp(self.block, op);
                        return;
                    }
                    const map_value_type = mlir.oraMapTypeGetValueType(base_type);
                    if (map_value_type.ptr != null) {
                        const key_value = try self.lowerExpr(index.index, locals);
                        try @This().appendMapStore(self, index.range, base_value, key_value, converted);
                        if (self.parent.file.pattern(index.base).* == .Index) {
                            try self.storePattern(index.base, base_value, locals);
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
            const map_type = mlir.oraValueGetType(map_value);
            const key_type = mlir.oraMapTypeGetKeyType(map_type);
            const converted_key = if (!mlir.oraTypeIsNull(key_type))
                try @This().convertValueForFlow(self, key_value, key_type, range)
            else
                key_value;
            const op = mlir.oraMapStoreOpCreate(self.parent.context, self.parent.location(range), map_value, converted_key, value);
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

        pub fn createBitfieldFieldExtract(
            self: *FunctionLowerer,
            base_value: mlir.MlirValue,
            base_type: sema.Type,
            field_name: []const u8,
            result_type: mlir.MlirType,
            range: source.TextRange,
        ) anyerror!mlir.MlirValue {
            const resolved = @This().bitfieldField(self, base_type, field_name) orelse return error.UnknownAssignmentTarget;
            const loc = self.parent.location(range);
            const word_type = mlir.oraValueGetType(base_value);
            const offset = resolved.offset;
            const width = resolved.width;

            var shifted = base_value;
            if (offset != 0) {
                const offset_value = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, word_type, @intCast(offset)));
                const shr_op = mlir.oraArithShrUIOpCreate(self.parent.context, loc, base_value, offset_value);
                if (mlir.oraOperationIsNull(shr_op)) return error.MlirOperationCreationFailed;
                shifted = appendValueOp(self.block, shr_op);
            }

            var raw = shifted;
            if (width < 256) {
                const mask = try @This().createWideIntegerConstant(self, word_type, @This().bitfieldMask(width), false, range);
                const and_op = mlir.oraArithAndIOpCreate(self.parent.context, loc, shifted, mask);
                if (mlir.oraOperationIsNull(and_op)) return error.MlirOperationCreationFailed;
                raw = appendValueOp(self.block, and_op);
            }

            const is_signed = resolved.sign == 's';
            var value_word = raw;
            if (is_signed and width < 256) {
                const shift_amount: i64 = @intCast(256 - width);
                const shift_value = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, word_type, shift_amount));
                const shl_op = mlir.oraArithShlIOpCreate(self.parent.context, loc, raw, shift_value);
                if (mlir.oraOperationIsNull(shl_op)) return error.MlirOperationCreationFailed;
                const shl_result = appendValueOp(self.block, shl_op);
                const sar_op = mlir.oraArithShrSIOpCreate(self.parent.context, loc, shl_result, shift_value);
                if (mlir.oraOperationIsNull(sar_op)) return error.MlirOperationCreationFailed;
                value_word = appendValueOp(self.block, sar_op);
            }

            return try @This().convertBitfieldWordToResult(self, value_word, result_type, range);
        }

        pub fn createBitfieldFieldUpdate(
            self: *FunctionLowerer,
            base_value: mlir.MlirValue,
            base_type: sema.Type,
            field_name: []const u8,
            new_value: mlir.MlirValue,
            range: source.TextRange,
        ) anyerror!mlir.MlirValue {
            const resolved = @This().bitfieldField(self, base_type, field_name) orelse return error.UnknownAssignmentTarget;
            const loc = self.parent.location(range);
            const word_type = mlir.oraValueGetType(base_value);
            const offset = resolved.offset;
            const width = resolved.width;
            const field_type = if (resolved.field_type) |field_type|
                self.parent.lowerSemaType(field_type, range)
            else
                self.parent.lowerTypeExpr(resolved.field.type_expr);
            const is_signed = resolved.sign == 's';

            var field_value = try @This().convertValueForFlow(self, new_value, field_type, range);
            field_value = try @This().convertBitfieldValueToWord(self, field_value, word_type, is_signed, range);

            const field_mask = @This().bitfieldMask(width);
            const mask_value = try @This().createWideIntegerConstant(self, word_type, field_mask, false, range);
            const masked_field_op = mlir.oraArithAndIOpCreate(self.parent.context, loc, field_value, mask_value);
            if (mlir.oraOperationIsNull(masked_field_op)) return error.MlirOperationCreationFailed;
            var prepared_field = appendValueOp(self.block, masked_field_op);

            if (offset != 0) {
                const offset_value = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, word_type, @intCast(offset)));
                const shl_op = mlir.oraArithShlIOpCreate(self.parent.context, loc, prepared_field, offset_value);
                if (mlir.oraOperationIsNull(shl_op)) return error.MlirOperationCreationFailed;
                prepared_field = appendValueOp(self.block, shl_op);
            }

            const clear_mask = ~(@as(u256, field_mask) << @intCast(offset));
            const clear_value = try @This().createWideIntegerConstant(self, word_type, clear_mask, false, range);
            const cleared_op = mlir.oraArithAndIOpCreate(self.parent.context, loc, base_value, clear_value);
            if (mlir.oraOperationIsNull(cleared_op)) return error.MlirOperationCreationFailed;
            const cleared = appendValueOp(self.block, cleared_op);

            const or_op = mlir.oraArithOrIOpCreate(self.parent.context, loc, cleared, prepared_field);
            if (mlir.oraOperationIsNull(or_op)) return error.MlirOperationCreationFailed;
            return appendValueOp(self.block, or_op);
        }

        fn convertBitfieldValueToWord(
            self: *FunctionLowerer,
            value: mlir.MlirValue,
            word_type: mlir.MlirType,
            is_signed: bool,
            range: source.TextRange,
        ) anyerror!mlir.MlirValue {
            const value_type = mlir.oraValueGetType(value);
            if (mlir.oraTypeEqual(value_type, word_type)) return value;

            const loc = self.parent.location(range);
            if (mlir.oraTypeEqual(value_type, boolType(self.parent.context))) {
                const ext_op = mlir.oraArithExtUIOpCreate(self.parent.context, loc, value, word_type);
                if (mlir.oraOperationIsNull(ext_op)) return error.MlirOperationCreationFailed;
                return appendValueOp(self.block, ext_op);
            }
            if (!mlir.oraTypeIsAInteger(value_type)) return value;

            const value_width = mlir.oraIntegerTypeGetWidth(value_type);
            const word_width = mlir.oraIntegerTypeGetWidth(word_type);
            if (value_width == word_width) {
                const op = mlir.oraArithBitcastOpCreate(self.parent.context, loc, value, word_type);
                if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                return appendValueOp(self.block, op);
            }
            if (value_width < word_width) {
                const op = if (is_signed)
                    mlir.oraArithExtSIOpCreate(self.parent.context, loc, value, word_type)
                else
                    mlir.oraArithExtUIOpCreate(self.parent.context, loc, value, word_type);
                if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                return appendValueOp(self.block, op);
            }
            const op = mlir.oraArithTruncIOpCreate(self.parent.context, loc, value, word_type);
            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
            return appendValueOp(self.block, op);
        }

        fn createWideIntegerConstant(
            self: *FunctionLowerer,
            target_type: mlir.MlirType,
            value: u256,
            negative: bool,
            range: source.TextRange,
        ) anyerror!mlir.MlirValue {
            const attr = if (!negative and value <= std.math.maxInt(i64))
                mlir.oraIntegerAttrCreateI64FromType(target_type, @intCast(value))
            else blk: {
                var buf: [80]u8 = undefined;
                const text = if (negative)
                    std.fmt.bufPrint(&buf, "-{}", .{value}) catch return error.MlirOperationCreationFailed
                else
                    std.fmt.bufPrint(&buf, "{}", .{value}) catch return error.MlirOperationCreationFailed;
                break :blk mlir.oraIntegerAttrGetFromString(target_type, strRef(text));
            };
            const op = mlir.oraArithConstantOpCreate(self.parent.context, self.parent.location(range), target_type, attr);
            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
            return appendValueOp(self.block, op);
        }

        fn convertBitfieldWordToResult(
            self: *FunctionLowerer,
            value: mlir.MlirValue,
            result_type: mlir.MlirType,
            range: source.TextRange,
        ) anyerror!mlir.MlirValue {
            const value_type = mlir.oraValueGetType(value);
            if (mlir.oraTypeEqual(value_type, result_type)) return value;

            const loc = self.parent.location(range);
            if (mlir.oraTypeEqual(result_type, boolType(self.parent.context))) {
                const zero = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, value_type, 0));
                return appendValueOp(self.block, self.createCompareOp(loc, "ne", value, zero));
            }
            if (!mlir.oraTypeIsAInteger(result_type)) return value;

            const value_width = mlir.oraIntegerTypeGetWidth(value_type);
            const result_width = mlir.oraIntegerTypeGetWidth(result_type);
            if (value_width == result_width) {
                const op = mlir.oraArithBitcastOpCreate(self.parent.context, loc, value, result_type);
                if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                return appendValueOp(self.block, op);
            }
            if (value_width > result_width) {
                const op = mlir.oraArithTruncIOpCreate(self.parent.context, loc, value, result_type);
                if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                return appendValueOp(self.block, op);
            }
            const op = if (mlir.oraIntegerTypeIsSigned(value_type))
                mlir.oraArithExtSIOpCreate(self.parent.context, loc, value, result_type)
            else
                mlir.oraArithExtUIOpCreate(self.parent.context, loc, value, result_type);
            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
            return appendValueOp(self.block, op);
        }

        fn bitfieldField(self: *FunctionLowerer, base_type: sema.Type, field_name: []const u8) ?Lowerer.ResolvedBitfieldField {
            if (base_type.kind() != .bitfield) return null;
            return self.parent.resolveBitfieldField(base_type.name() orelse return null, field_name);
        }

        fn bitfieldMask(width: u32) u256 {
            if (width >= 256) return std.math.maxInt(u256);
            if (width == 0) return 0;
            return (@as(u256, 1) << @intCast(width)) - 1;
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
                            const result_type = if (field.type_expr) |type_expr| self.parent.lowerTypeExpr(type_expr) else self.parent.lowerSemaType(self.parent.typecheck.pattern_types[pattern_id.index()].type, name.range);
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
                    if (mlir.oraTypeIsAMemRef(mlir.oraValueGetType(base_value))) {
                        const index_value = try @This().convertIndexToIndexType(self, key_value, index.range);
                        const result_type = self.parent.lowerSemaType(self.parent.typecheck.pattern_types[pattern_id.index()].type, index.range);
                        const op = mlir.oraMemrefLoadOpCreate(
                            self.parent.context,
                            self.parent.location(index.range),
                            base_value,
                            &[_]mlir.MlirValue{index_value},
                            1,
                            result_type,
                        );
                        if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                        break :blk appendValueOp(self.block, op);
                    }
                    const result_type = blk2: {
                        const map_value_type = mlir.oraMapTypeGetValueType(mlir.oraValueGetType(base_value));
                        if (map_value_type.ptr != null) break :blk2 map_value_type;
                        break :blk2 self.parent.lowerSemaType(self.parent.typecheck.pattern_types[pattern_id.index()].type, index.range);
                    };
                    const map_key_type = mlir.oraMapTypeGetKeyType(mlir.oraValueGetType(base_value));
                    const converted_key = if (!mlir.oraTypeIsNull(map_key_type))
                        try @This().convertValueForFlow(self, key_value, map_key_type, index.range)
                    else
                        key_value;
                    const op = mlir.oraMapGetOpCreate(self.parent.context, self.parent.location(index.range), base_value, converted_key, result_type);
                    if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                    break :blk appendValueOp(self.block, op);
                },
                .Field => |field| blk: {
                    const base_type = @This().patternType(self, field.base, locals);
                    if (@This().isBitfieldLikeType(self, base_type)) {
                        const result_type = self.parent.lowerSemaType(self.parent.typecheck.pattern_types[pattern_id.index()].type, field.range);
                        break :blk try @This().createBitfieldFieldExtract(self, try @This().lowerPatternValue(self, field.base, locals), base_type, field.name, result_type, field.range);
                    }
                    const base_value = try @This().lowerPatternValue(self, field.base, locals);
                    const result_type = self.parent.lowerSemaType(self.parent.typecheck.pattern_types[pattern_id.index()].type, field.range);
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

        fn patternType(self: *FunctionLowerer, pattern_id: ast.PatternId, locals: *LocalEnv) sema.Type {
            return switch (self.parent.file.pattern(pattern_id).*) {
                .Name => |name| blk: {
                    if (locals.lookupName(name.name)) |local_id| {
                        break :blk self.parent.typecheck.pattern_types[local_id.index()].type;
                    }
                    if (self.parent.item_index.lookup(name.name)) |item_id| {
                        break :blk self.parent.typecheck.item_types[item_id.index()];
                    }
                    break :blk self.parent.typecheck.pattern_types[pattern_id.index()].type;
                },
                else => self.parent.typecheck.pattern_types[pattern_id.index()].type,
            };
        }

        fn isBitfieldLikeType(self: *FunctionLowerer, ty: sema.Type) bool {
            if (ty.kind() == .bitfield) return true;
            const name = ty.name() orelse return false;
            if (self.parent.typecheck.instantiatedBitfieldByName(name) != null) return true;
            const item_id = self.parent.item_index.lookup(name) orelse return false;
            return self.parent.file.item(item_id).* == .Bitfield;
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

        fn lowerForStmt(self: *FunctionLowerer, for_stmt: ast.ForStmt, locals: *LocalEnv) anyerror!bool {
            const loc = self.parent.location(for_stmt.range);
            const iterable = try self.lowerExpr(for_stmt.iterable, locals);
            const iterable_type = mlir.oraValueGetType(iterable);
            const has_return = bodyMayReturn(self.parent.file, for_stmt.body);

            if (!mlir.oraTypeIsAMemRef(iterable_type) or
                mlir.oraShapedTypeGetRank(iterable_type) != 1)
            {
                try self.appendUnsupportedControlPlaceholder("ora.for_placeholder", for_stmt.range);
                return false;
            }
            const created_deferred_return = has_return and self.deferred_return_flag == null;
            if (created_deferred_return) {
                try self.ensureDeferredReturnSlots(for_stmt.range);
            }

            const has_loop_control = bodyContainsLoopControl(self.parent.file, for_stmt.body);
            const break_flag = if (has_loop_control) blk: {
                const break_flag_alloc = mlir.oraMemrefAllocaOpCreate(
                    self.parent.context,
                    loc,
                    support.memRefType(self.parent.context, boolType(self.parent.context)),
                );
                if (mlir.oraOperationIsNull(break_flag_alloc)) return error.MlirOperationCreationFailed;
                const flag = appendValueOp(self.block, break_flag_alloc);

                const break_flag_zero = appendValueOp(
                    self.block,
                    createIntegerConstant(self.parent.context, loc, boolType(self.parent.context), 0),
                );
                const clear_break = mlir.oraMemrefStoreOpCreate(self.parent.context, loc, break_flag_zero, flag, null, 0);
                if (mlir.oraOperationIsNull(clear_break)) return error.MlirOperationCreationFailed;
                appendOp(self.block, clear_break);
                break :blk flag;
            } else std.mem.zeroes(mlir.MlirValue);

            var carried_locals: LocalIdList = .{};
            var carried_seen = LocalIdSet.init(self.parent.allocator);
            const carried_supported = try collectLoopCarriedLocals(self.parent.allocator, self.parent.file, for_stmt.body, locals, &carried_locals, &carried_seen);
            if (!carried_supported) {
                try self.appendUnsupportedControlPlaceholder("ora.for_placeholder", for_stmt.range);
                return false;
            }
            carried_locals = try self.filterCarriedLocals(locals, carried_locals.items);

            var init_operands: std.ArrayList(mlir.MlirValue) = .{};
            for (carried_locals.items) |local_id| {
                const value = try self.materializeCarriedLocalValue(locals, local_id);
                try init_operands.append(self.parent.allocator, value);
            }

            const index_type = mlir.oraIndexTypeCreate(self.parent.context);
            const lower_bound = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, index_type, 0));
            const upper_bound = if (mlir.oraShapedTypeHasStaticShape(iterable_type)) blk: {
                const dim_size = mlir.oraShapedTypeGetDimSize(iterable_type, 0);
                break :blk appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, index_type, @intCast(dim_size)));
            } else blk: {
                const dim_index = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, index_type, 0));
                const dim_op = mlir.oraMemrefDimOpCreate(self.parent.context, loc, iterable, dim_index);
                if (mlir.oraOperationIsNull(dim_op)) {
                    try self.appendUnsupportedControlPlaceholder("ora.for_placeholder", for_stmt.range);
                    return false;
                }
                break :blk appendValueOp(self.block, dim_op);
            };
            const step = appendValueOp(self.block, createIntegerConstant(self.parent.context, loc, index_type, 1));

            const for_op = mlir.oraScfForOpCreate(
                self.parent.context,
                loc,
                lower_bound,
                upper_bound,
                step,
                if (init_operands.items.len == 0) null else init_operands.items.ptr,
                init_operands.items.len,
                false,
            );
            if (mlir.oraOperationIsNull(for_op)) return error.MlirOperationCreationFailed;
            appendOp(self.block, for_op);

            const body_block = mlir.oraScfForOpGetBodyBlock(for_op);
            if (mlir.oraBlockIsNull(body_block)) return error.MlirOperationCreationFailed;
            clearKnownTerminator(body_block);

            const induction_var = mlir.oraBlockGetArgument(body_block, 0);
            const index_value_op = mlir.oraArithIndexCastUIOpCreate(
                self.parent.context,
                loc,
                induction_var,
                defaultIntegerType(self.parent.context),
            );
            if (mlir.oraOperationIsNull(index_value_op)) return error.MlirOperationCreationFailed;
            appendOp(body_block, index_value_op);
            const index_value = mlir.oraOperationGetResult(index_value_op, 0);

            const item_type = self.parent.lowerSemaType(self.parent.typecheck.pattern_types[for_stmt.item_pattern.index()].type, for_stmt.range);
            const item_load = mlir.oraMemrefLoadOpCreate(
                self.parent.context,
                loc,
                iterable,
                &[_]mlir.MlirValue{induction_var},
                1,
                item_type,
            );
            if (mlir.oraOperationIsNull(item_load)) {
                try self.appendUnsupportedControlPlaceholder("ora.for_placeholder", for_stmt.range);
                return false;
            }
            appendOp(body_block, item_load);
            const item_value = mlir.oraOperationGetResult(item_load, 0);

            var body_lowerer = self.*;
            body_lowerer.block = body_block;
            var loop_context = LoopContext{
                .parent = self.loop_context,
                .break_flag = break_flag,
                .carried_locals = carried_locals.items,
            };
            if (has_loop_control) {
                body_lowerer.loop_context = &loop_context;
            }
            var body_locals = try self.cloneLocals(locals);
            for (carried_locals.items, 0..) |local_id, index| {
                try body_locals.setValue(local_id, mlir.oraBlockGetArgument(body_block, index + 1));
            }
            try body_lowerer.bindPatternValue(for_stmt.item_pattern, item_value, &body_locals);
            if (for_stmt.index_pattern) |index_pattern| {
                try body_lowerer.bindPatternValue(index_pattern, index_value, &body_locals);
            }

            const has_execution_guard = has_loop_control or has_return;
            if (has_execution_guard) {
                var loop_condition = if (has_loop_control) guard_blk: {
                    const break_flag_value = appendValueOp(body_block, load_blk: {
                        const load = mlir.oraMemrefLoadOpCreate(self.parent.context, loc, break_flag, null, 0, boolType(self.parent.context));
                        if (mlir.oraOperationIsNull(load)) return error.MlirOperationCreationFailed;
                        break :load_blk load;
                    });
                    const break_flag_clear = appendValueOp(
                        body_block,
                        createIntegerConstant(self.parent.context, loc, boolType(self.parent.context), 0),
                    );
                    const loop_enabled = body_lowerer.createCompareOp(loc, "eq", break_flag_value, break_flag_clear);
                    if (mlir.oraOperationIsNull(loop_enabled)) return error.MlirOperationCreationFailed;
                    break :guard_blk appendValueOp(body_block, loop_enabled);
                } else appendValueOp(
                    body_block,
                    createIntegerConstant(self.parent.context, loc, boolType(self.parent.context), 1),
                );

                if (has_return) {
                    const return_flag_value = appendValueOp(body_block, blk: {
                        const load = mlir.oraMemrefLoadOpCreate(self.parent.context, loc, self.deferred_return_flag.?, null, 0, boolType(self.parent.context));
                        if (mlir.oraOperationIsNull(load)) return error.MlirOperationCreationFailed;
                        break :blk load;
                    });
                    const return_flag_clear = appendValueOp(
                        body_block,
                        createIntegerConstant(self.parent.context, loc, boolType(self.parent.context), 0),
                    );
                    const return_enabled = body_lowerer.createCompareOp(loc, "eq", return_flag_value, return_flag_clear);
                    if (mlir.oraOperationIsNull(return_enabled)) return error.MlirOperationCreationFailed;
                    loop_condition = appendValueOp(
                        body_block,
                        mlir.oraArithAndIOpCreate(self.parent.context, loc, loop_condition, appendValueOp(body_block, return_enabled)),
                    );
                }

                const result_types = if (carried_locals.items.len == 0)
                    null
                else
                    (try self.buildCarriedResultTypes(locals, carried_locals.items)) orelse {
                        try self.appendUnsupportedControlPlaceholder("ora.for_placeholder", for_stmt.range);
                        return false;
                    };

                const if_op = mlir.oraScfIfOpCreate(
                    self.parent.context,
                    loc,
                    loop_condition,
                    if (result_types) |types| types.items.ptr else null,
                    if (result_types) |types| types.items.len else 0,
                    true,
                );
                if (mlir.oraOperationIsNull(if_op)) return error.MlirOperationCreationFailed;
                appendOp(body_block, if_op);

                const then_block = mlir.oraScfIfOpGetThenBlock(if_op);
                const else_block = mlir.oraScfIfOpGetElseBlock(if_op);
                if (mlir.oraBlockIsNull(then_block) or mlir.oraBlockIsNull(else_block)) {
                    return error.MlirOperationCreationFailed;
                }

                var then_lowerer = body_lowerer;
                then_lowerer.block = then_block;
                if (has_return) {
                    then_lowerer.deferred_return_kind = .scf_yield;
                    then_lowerer.deferred_return_carried_locals = carried_locals.items;
                }
                var then_locals = try self.cloneLocals(&body_locals);
                try @This().lowerLoopInvariants(&then_lowerer, for_stmt.invariants, &then_locals);
                _ = try then_lowerer.lowerBody(for_stmt.body, &then_locals);
                if (!support.blockEndsWithTerminator(then_block)) {
                    try then_lowerer.appendScfYieldFromLocals(then_block, for_stmt.range, &then_locals, carried_locals.items);
                }

                var else_locals = try self.cloneLocals(&body_locals);
                if (!support.blockEndsWithTerminator(else_block)) {
                    try body_lowerer.appendScfYieldFromLocals(else_block, for_stmt.range, &else_locals, carried_locals.items);
                }

                if (carried_locals.items.len > 0) {
                    try FunctionLowerer.writeBackCarriedLocals(&body_locals, carried_locals.items, if_op);
                }
            } else {
                try @This().lowerLoopInvariants(&body_lowerer, for_stmt.invariants, &body_locals);
                _ = try body_lowerer.lowerBody(for_stmt.body, &body_locals);
            }
            if (!support.blockEndsWithTerminator(body_block)) {
                try body_lowerer.appendScfYieldFromLocals(body_block, for_stmt.range, &body_locals, carried_locals.items);
            }
            try FunctionLowerer.writeBackCarriedLocals(locals, carried_locals.items, for_op);
            if (created_deferred_return) {
                try self.appendDeferredReturnCheck(for_stmt.range);
            }
            return false;
        }

        fn lowerLoopInvariants(self: *FunctionLowerer, invariants: []const ast.ExprId, locals: *LocalEnv) anyerror!void {
            for (invariants) |expr_id| {
                const value = try self.lowerExpr(expr_id, locals);
                const op = mlir.oraInvariantOpCreate(
                    self.parent.context,
                    self.parent.location(support.exprRange(self.parent.file, expr_id)),
                    value,
                );
                if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
                appendOp(self.block, op);
            }
        }

        fn convertIndexToIndexType(self: *FunctionLowerer, index: mlir.MlirValue, range: source.TextRange) anyerror!mlir.MlirValue {
            const index_type = mlir.oraIndexTypeCreate(self.parent.context);
            if (mlir.oraTypeEqual(mlir.oraValueGetType(index), index_type)) {
                return index;
            }

            const op = mlir.oraArithIndexCastUIOpCreate(
                self.parent.context,
                self.parent.location(range),
                index,
                index_type,
            );
            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
            return appendValueOp(self.block, op);
        }

        fn lockResourceExpr(file: *const ast.AstFile, expr_id: ast.ExprId) ast.ExprId {
            return switch (file.expression(expr_id).*) {
                .Group => |group| lockResourceExpr(file, group.expr),
                .Index => |index| switch (file.expression(index.base).*) {
                    .Name => index.index,
                    .Group => lockResourceExpr(file, index.base),
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
                .Group => |group| buildRuntimeLockKey(allocator, file, resolution, group.expr),
                .Name => |name| if (isStorageFieldName(file, resolution, expr_id))
                    std.fmt.allocPrint(allocator, "{s}", .{name.name}) catch null
                else
                    null,
                .Index => |index| blk: {
                    const base_expr = switch (file.expression(index.base).*) {
                        .Group => |group| group.expr,
                        else => index.base,
                    };
                    switch (file.expression(base_expr).*) {
                        .Name => |name| if (isStorageFieldName(file, resolution, base_expr))
                            break :blk std.fmt.allocPrint(allocator, "{s}[]", .{name.name}) catch null
                        else
                            break :blk null,
                        else => break :blk null,
                    }
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
            _ = locals;
            var result_types: std.ArrayList(mlir.MlirType) = .{};
            for (carried_locals) |local_id| {
                const sema_type = self.parent.typecheck.pattern_types[local_id.index()].type;
                if (sema_type.kind() == .unknown) return null;
                try result_types.append(self.parent.allocator, self.parent.lowerSemaType(sema_type, patternRange(self.parent.file, local_id)));
            }
            return result_types;
        }

        pub fn filterCarriedLocals(
            self: *FunctionLowerer,
            locals: *const LocalEnv,
            carried_locals: []const LocalId,
        ) anyerror!LocalIdList {
            var filtered: LocalIdList = .{};
            for (carried_locals) |local_id| {
                if (!locals.hasLocal(local_id)) continue;
                if (self.parent.typecheck.pattern_types[local_id.index()].type.kind() == .unknown) continue;
                try filtered.append(self.parent.allocator, local_id);
            }
            return filtered;
        }

        pub fn materializeCarriedLocalValue(
            self: *FunctionLowerer,
            locals: *const LocalEnv,
            local_id: LocalId,
        ) anyerror!mlir.MlirValue {
            if (locals.getValue(local_id)) |value| return value;

            const sema_type = self.parent.typecheck.pattern_types[local_id.index()].type;
            if (sema_type.kind() == .unknown) return error.MlirOperationCreationFailed;
            return self.defaultValue(
                self.parent.lowerSemaType(sema_type, patternRange(self.parent.file, local_id)),
                patternRange(self.parent.file, local_id),
            );
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
                const value = try self.materializeCarriedLocalValue(locals, local_id);
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
                const value = try self.materializeCarriedLocalValue(locals, local_id);
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
