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
const EffectSlot = model.EffectSlot;
const KeySegment = model.KeySegment;
const InstantiatedStruct = model.InstantiatedStruct;
const InstantiatedStructField = model.InstantiatedStructField;
const InstantiatedEnum = model.InstantiatedEnum;
const InstantiatedBitfield = model.InstantiatedBitfield;
const InstantiatedBitfieldField = model.InstantiatedBitfieldField;
const EffectSummaryState = enum { unvisited, visiting, done };
const ConstEvalResult = model.ConstEvalResult;
const ConstValue = model.ConstValue;
const BigInt = std.math.big.int.Managed;
const descriptorFromTypeExpr = descriptors.descriptorFromTypeExpr;
const descriptorFromGenericType = descriptors.descriptorFromGenericType;
const descriptorFromPathName = descriptors.descriptorFromPathName;
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

fn keySegmentEql(lhs: KeySegment, rhs: KeySegment) bool {
    return switch (lhs) {
        .parameter => |index| rhs == .parameter and rhs.parameter == index,
        .constant => |value| rhs == .constant and std.mem.eql(u8, rhs.constant, value),
        .self_ref => rhs == .self_ref,
        .unknown => rhs == .unknown,
    };
}

fn keySegmentMayAlias(lhs: KeySegment, rhs: KeySegment) bool {
    return switch (lhs) {
        .unknown, .self_ref => true,
        .parameter => |lhs_index| switch (rhs) {
            .parameter => |rhs_index| lhs_index == rhs_index,
            .unknown, .self_ref, .constant => true,
        },
        .constant => |lhs_value| switch (rhs) {
            .constant => |rhs_value| std.mem.eql(u8, lhs_value, rhs_value),
            .unknown, .self_ref, .parameter => true,
        },
    };
}

fn keyPathsEql(lhs: ?[]const KeySegment, rhs: ?[]const KeySegment) bool {
    if (lhs == null and rhs == null) return true;
    const lhs_path = lhs orelse return false;
    const rhs_path = rhs orelse return false;
    if (lhs_path.len != rhs_path.len) return false;
    for (lhs_path, rhs_path) |lhs_segment, rhs_segment| {
        if (!keySegmentEql(lhs_segment, rhs_segment)) return false;
    }
    return true;
}

fn keyPathsMayAlias(lhs: ?[]const KeySegment, rhs: ?[]const KeySegment) bool {
    if (lhs == null or rhs == null) return true;
    const lhs_path = lhs.?;
    const rhs_path = rhs.?;
    if (lhs_path.len != rhs_path.len) return true;
    for (lhs_path, rhs_path) |lhs_segment, rhs_segment| {
        if (!keySegmentMayAlias(lhs_segment, rhs_segment)) return false;
    }
    return true;
}

fn effectSlotEql(lhs: EffectSlot, rhs: EffectSlot) bool {
    return lhs.region == rhs.region and
        std.mem.eql(u8, lhs.name, rhs.name) and
        keyPathsEql(lhs.key_path, rhs.key_path);
}

fn effectSlotsEql(lhs: []const EffectSlot, rhs: []const EffectSlot) bool {
    if (lhs.len != rhs.len) return false;
    for (lhs, rhs) |lhs_slot, rhs_slot| {
        if (!effectSlotEql(lhs_slot, rhs_slot)) return false;
    }
    return true;
}

fn effectEql(lhs: Effect, rhs: Effect) bool {
    return switch (lhs) {
        .pure => rhs == .pure,
        .external => rhs == .external,
        .side_effects => |lhs_effects| rhs == .side_effects and
            lhs_effects.has_external == rhs.side_effects.has_external and
            lhs_effects.has_log == rhs.side_effects.has_log and
            lhs_effects.has_havoc == rhs.side_effects.has_havoc and
            lhs_effects.has_lock == rhs.side_effects.has_lock and
            lhs_effects.has_unlock == rhs.side_effects.has_unlock,
        .reads => |lhs_effects| rhs == .reads and
            effectSlotsEql(lhs_effects.slots, rhs.reads.slots) and
            lhs_effects.has_external == rhs.reads.has_external and
            lhs_effects.has_log == rhs.reads.has_log and
            lhs_effects.has_havoc == rhs.reads.has_havoc and
            lhs_effects.has_lock == rhs.reads.has_lock and
            lhs_effects.has_unlock == rhs.reads.has_unlock,
        .writes => |lhs_effects| rhs == .writes and
            effectSlotsEql(lhs_effects.slots, rhs.writes.slots) and
            lhs_effects.has_external == rhs.writes.has_external and
            lhs_effects.has_log == rhs.writes.has_log and
            lhs_effects.has_havoc == rhs.writes.has_havoc and
            lhs_effects.has_lock == rhs.writes.has_lock and
            lhs_effects.has_unlock == rhs.writes.has_unlock,
        .reads_writes => |lhs_effects| rhs == .reads_writes and
            effectSlotsEql(lhs_effects.reads, rhs.reads_writes.reads) and
            effectSlotsEql(lhs_effects.writes, rhs.reads_writes.writes) and
            lhs_effects.has_external == rhs.reads_writes.has_external and
            lhs_effects.has_log == rhs.reads_writes.has_log and
            lhs_effects.has_havoc == rhs.reads_writes.has_havoc and
            lhs_effects.has_lock == rhs.reads_writes.has_lock and
            lhs_effects.has_unlock == rhs.reads_writes.has_unlock,
    };
}

test "effect slot aliasing distinguishes parameters and constants" {
    const same_param = EffectSlot{
        .name = "balances",
        .region = .storage,
        .key_path = &[_]KeySegment{.{ .parameter = 0 }},
    };
    const other_param = EffectSlot{
        .name = "balances",
        .region = .storage,
        .key_path = &[_]KeySegment{.{ .parameter = 1 }},
    };
    const const_one = EffectSlot{
        .name = "balances",
        .region = .storage,
        .key_path = &[_]KeySegment{.{ .constant = "1" }},
    };
    const const_two = EffectSlot{
        .name = "balances",
        .region = .storage,
        .key_path = &[_]KeySegment{.{ .constant = "2" }},
    };
    const unknown = EffectSlot{
        .name = "balances",
        .region = .storage,
        .key_path = &[_]KeySegment{.{ .unknown = {} }},
    };

    try std.testing.expect(keyPathsMayAlias(same_param.key_path, same_param.key_path));
    try std.testing.expect(!keyPathsMayAlias(same_param.key_path, other_param.key_path));
    try std.testing.expect(!keyPathsMayAlias(const_one.key_path, const_two.key_path));
    try std.testing.expect(keyPathsMayAlias(unknown.key_path, same_param.key_path));
}

test "buildEffect preserves external marker across slot summaries" {
    const slots = [_]EffectSlot{
        .{ .name = "total", .region = .storage },
    };

    try std.testing.expect(TypeChecker.buildEffect(&.{}, &.{}, true, false, false, false, false) == .external);
    try std.testing.expect(TypeChecker.buildEffect(&slots, &.{}, true, false, false, false, false).reads.has_external);
    try std.testing.expect(TypeChecker.buildEffect(&.{}, &slots, true, false, false, false, false).writes.has_external);
    try std.testing.expect(TypeChecker.buildEffect(&slots, &slots, true, false, false, false, false).reads_writes.has_external);
}

test "buildEffect preserves log and havoc markers across slot summaries" {
    const slots = [_]EffectSlot{
        .{ .name = "total", .region = .storage },
    };

    try std.testing.expect(TypeChecker.buildEffect(&slots, &.{}, false, true, false, false, false).reads.has_log);
    try std.testing.expect(TypeChecker.buildEffect(&.{}, &slots, false, false, true, false, false).writes.has_havoc);
    const mixed = TypeChecker.buildEffect(&slots, &slots, true, true, true, true, true).reads_writes;
    try std.testing.expect(mixed.has_external);
    try std.testing.expect(mixed.has_log);
    try std.testing.expect(mixed.has_havoc);
    try std.testing.expect(mixed.has_lock);
    try std.testing.expect(mixed.has_unlock);
    const effects_only = TypeChecker.buildEffect(&.{}, &.{}, false, true, true, true, true).side_effects;
    try std.testing.expect(effects_only.has_log);
    try std.testing.expect(effects_only.has_havoc);
    try std.testing.expect(effects_only.has_lock);
    try std.testing.expect(effects_only.has_unlock);
}

test "unknown locked call diagnostics mention each locked slot" {
    var diags = diagnostics.DiagnosticList.init(std.testing.allocator);
    defer diags.deinit();

    var checker = TypeChecker{
        .arena = std.testing.allocator,
        .file_id = source.FileId.fromIndex(0),
        .file = undefined,
        .item_index = undefined,
        .resolution = undefined,
        .const_eval = undefined,
        .item_types = &.{},
        .item_regions = &.{},
        .item_effects = &.{},
        .pattern_types = &.{},
        .expr_types = &.{},
        .expr_effects = &.{},
        .effect_states = &.{},
        .current_function_item = null,
        .diagnostics = &diags,
    };

    const locked = [_]EffectSlot{
        .{ .name = "total", .region = .storage },
        .{ .name = "pending", .region = .transient },
    };
    try checker.emitUnknownLockedCallDiagnostics(source.TextRange.init(3, 7), &locked);

    try std.testing.expectEqual(@as(usize, 2), diags.items.items.len);
    try std.testing.expect(std.mem.eql(u8, diags.items.items[0].message, "unresolved call may write locked storage slot 'total'"));
    try std.testing.expect(std.mem.eql(u8, diags.items.items[1].message, "unresolved call may write locked transient slot 'pending'"));
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
        .expr_effects = &.{},
        .body_types = &.{},
        .instantiated_structs = &.{},
        .instantiated_enums = &.{},
        .instantiated_bitfields = &.{},
        .diagnostics = diagnostics.DiagnosticList.init(allocator),
    };
    errdefer result.deinit();

    const arena = result.arena.allocator();
    var item_types = try arena.alloc(Type, file.items.len);
    var item_regions = try arena.alloc(Region, file.items.len);
    const item_effects = try arena.alloc(Effect, file.items.len);
    var pattern_types = try arena.alloc(LocatedType, file.patterns.len);
    const expr_types = try arena.alloc(Type, file.expressions.len);
    const expr_effects = try arena.alloc(Effect, file.expressions.len);
    var body_types = try arena.alloc(Type, file.bodies.len);
    const effect_states = try arena.alloc(EffectSummaryState, file.items.len);
    @memset(item_types, .{ .unknown = {} });
    @memset(item_regions, .none);
    @memset(item_effects, .pure);
    @memset(pattern_types, LocatedType.unlocated(.{ .unknown = {} }));
    @memset(expr_types, .{ .unknown = {} });
    @memset(expr_effects, .pure);
    @memset(body_types, .{ .void = {} });
    @memset(effect_states, .unvisited);

    for (file.items, 0..) |item, index| {
        item_types[index] = try inferItemType(arena, file, item_index, item);
        switch (item) {
            .Function => |function| {
                for (function.parameters) |parameter| {
                    pattern_types[parameter.pattern.index()] = LocatedType.withRegion(.{ .unknown = {} }, .calldata);
                }
                body_types[function.body.index()] = if (function.return_type) |_| .{ .unknown = {} } else .{ .void = {} };
            },
            .Field => |field| {
                item_regions[index] = declarationRegion(field.storage_class);
                if (field.type_expr) |_| item_types[index] = .{ .unknown = {} };
            },
            .Constant => |constant| {
                if (constant.type_expr) |_| item_types[index] = .{ .unknown = {} };
            },
            else => {},
        }
    }

    for (file.statements) |statement| {
        if (statement == .VariableDecl) {
            const decl = statement.VariableDecl;
            if (decl.type_expr) |type_expr| {
                _ = type_expr;
                pattern_types[decl.pattern.index()] = LocatedType.unlocated(.{ .unknown = {} });
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
        .expr_effects = expr_effects,
        .effect_states = effect_states,
        .instantiated_structs = .{},
        .instantiated_enums = .{},
        .instantiated_bitfields = .{},
        .diagnostics = &result.diagnostics,
    };

    try result.diagnostics.appendList(&const_eval.diagnostics);

    for (file.items, 0..) |item, index| {
        switch (item) {
            .Function => |function| {
                const resolved_params = try arena.alloc(Type, function.parameters.len);
                for (function.parameters, 0..) |parameter, param_index| {
                    const resolved_type = try typechecker.resolveTypeExpr(parameter.type_expr);
                    resolved_params[param_index] = resolved_type;
                    pattern_types[parameter.pattern.index()] = LocatedType.withRegion(resolved_type, .calldata);
                }
                const resolved_returns = if (function.return_type) |type_expr| blk_returns: {
                    const slice = try arena.alloc(Type, 1);
                    slice[0] = try typechecker.resolveTypeExpr(type_expr);
                    break :blk_returns slice;
                } else &.{};
                item_types[index] = .{ .function = .{
                    .name = function.name,
                    .param_types = resolved_params,
                    .return_types = resolved_returns,
                } };
                body_types[function.body.index()] = if (function.return_type) |type_expr|
                    try typechecker.resolveTypeExpr(type_expr)
                else
                    .{ .void = {} };
            },
            .Field => |field| {
                if (field.type_expr) |type_expr| item_types[index] = try typechecker.resolveTypeExpr(type_expr);
            },
            .Constant => |constant| {
                if (constant.type_expr) |type_expr| item_types[index] = try typechecker.resolveTypeExpr(type_expr);
            },
            .TypeAlias => |type_alias| {
                item_types[index] = if (type_alias.is_generic)
                    .{ .unknown = {} }
                else
                    try typechecker.resolveTypeExpr(type_alias.target_type);
            },
            else => {},
        }
    }

    for (file.statements) |statement| {
        if (statement == .VariableDecl) {
            const decl = statement.VariableDecl;
            if (decl.type_expr) |type_expr| {
                pattern_types[decl.pattern.index()] = LocatedType.unlocated(try typechecker.resolveTypeExpr(type_expr));
            }
        }
    }

    for (file.root_items) |item_id| {
        try typechecker.visitItem(item_id);
    }

    result.item_types = item_types;
    result.item_regions = item_regions;
    result.item_effects = item_effects;
    result.pattern_types = pattern_types;
    result.expr_types = expr_types;
    result.expr_effects = expr_effects;
    result.body_types = body_types;
    result.instantiated_structs = try typechecker.instantiated_structs.toOwnedSlice(arena);
    result.instantiated_enums = try typechecker.instantiated_enums.toOwnedSlice(arena);
    result.instantiated_bitfields = try typechecker.instantiated_bitfields.toOwnedSlice(arena);
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
    expr_effects: []Effect,
    effect_states: []EffectSummaryState,
    instantiated_structs: std.ArrayList(InstantiatedStruct),
    instantiated_enums: std.ArrayList(InstantiatedEnum),
    instantiated_bitfields: std.ArrayList(InstantiatedBitfield),
    active_aliases: std.ArrayList(ast.ItemId) = .{},
    current_return_type: ?Type = null,
    current_contract: ?ast.ItemId = null,
    current_function_item: ?ast.ItemId = null,
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
                const previous_function_item = self.current_function_item;
                self.current_return_type = if (function.return_type) |type_expr| try self.resolveTypeExpr(type_expr) else .{ .void = {} };
                self.current_function_item = item_id;
                defer self.current_function_item = previous_function_item;
                defer self.current_return_type = previous_return_type;
                for (function.clauses) |clause| try self.visitExpr(clause.expr);
                try self.visitBody(function.body);
                try self.ensureFunctionEffectSummary(item_id, function);
                var locked_slots: std.ArrayList(EffectSlot) = .{};
                try self.validateBodyLocks(function.body, &locked_slots);
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
            .TypeAlias => {},
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
                } else if (result_type.kind() == .unknown) {
                    const expected_args = self.expectedCallArgCount(call) orelse callee_type.paramTypes().len;
                    if (expected_args != call.args.len) {
                        try self.emitExprError(expr_id, "expected {d} arguments, found {d}", .{ expected_args, call.args.len });
                    }
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

    fn callReturnType(self: *TypeChecker, call: ast.CallExpr) Type {
        if (self.genericCallReturnType(call)) |result| return result;
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

    fn genericCallReturnType(self: *TypeChecker, call: ast.CallExpr) ?Type {
        const callee_id = self.calleeFunctionItem(call.callee) orelse return null;
        const function = switch (self.file.item(callee_id).*) {
            .Function => |function| function,
            else => return null,
        };
        if (!function.is_generic) return null;

        const bindings = self.genericTypeBindingsForCall(function, call) orelse return .{ .unknown = {} };
        const runtime_parameters = self.runtimeFunctionParameters(function) catch return .{ .unknown = {} };
        if (runtime_parameters.len != call.args.len - bindings.len) return .{ .unknown = {} };
        if (function.return_type) |type_expr| {
            return self.resolveTypeExprWithBindings(type_expr, bindings) catch .{ .unknown = {} };
        }
        return .{ .void = {} };
    }

    fn expectedCallArgCount(self: *const TypeChecker, call: ast.CallExpr) ?usize {
        const callee_id = self.calleeFunctionItem(call.callee) orelse return null;
        const function = switch (self.file.item(callee_id).*) {
            .Function => |function| function,
            else => return null,
        };
        if (!function.is_generic) return function.parameters.len;
        return function.parameters.len;
    }

    fn genericTypeBindingsForCall(self: *const TypeChecker, function: ast.FunctionItem, call: ast.CallExpr) ?[]const GenericTypeBinding {
        var generic_count: usize = 0;
        for (function.parameters) |parameter| {
            if (!parameter.is_comptime) break;
            generic_count += 1;
        }
        if (generic_count == 0) return &.{};
        if (call.args.len < generic_count) return null;

        const bindings = self.arena.alloc(GenericTypeBinding, generic_count) catch return null;
        for (function.parameters[0..generic_count], 0..) |parameter, index| {
            const name = self.patternName(parameter.pattern) orelse return null;
            const value = if (self.isGenericTypeParameter(parameter))
                blk: {
                    const arg_name = self.typeArgNameFromExpr(call.args[index]) orelse return null;
                    break :blk GenericBindingValue{ .ty = descriptorFromPathName(self.file, self.item_index, arg_name) };
                }
            else if (self.comptimeIntegerParameter(parameter))
                blk: {
                    const integer = self.exprIntegerText(call.args[index]) orelse return null;
                    break :blk GenericBindingValue{ .integer = integer };
                }
            else
                return null;
            bindings[index] = .{ .name = name, .value = value };
        }
        return bindings;
    }

    fn runtimeFunctionParameters(self: *const TypeChecker, function: ast.FunctionItem) ![]ast.Parameter {
        var parameters: std.ArrayList(ast.Parameter) = .{};
        for (function.parameters) |parameter| {
            if (parameter.is_comptime) continue;
            try parameters.append(self.arena, parameter);
        }
        return parameters.toOwnedSlice(self.arena);
    }

    fn isGenericTypeParameter(self: *const TypeChecker, parameter: ast.Parameter) bool {
        if (!parameter.is_comptime) return false;
        return switch (self.file.typeExpr(parameter.type_expr).*) {
            .Path => |path| std.mem.eql(u8, path.name, "type"),
            else => false,
        };
    }

    fn patternName(self: *const TypeChecker, pattern_id: ast.PatternId) ?[]const u8 {
        return switch (self.file.pattern(pattern_id).*) {
            .Name => |name| name.name,
            else => null,
        };
    }

    fn typeArgNameFromExpr(self: *const TypeChecker, expr_id: ast.ExprId) ?[]const u8 {
        return switch (self.file.expression(expr_id).*) {
            .Name => |name| name.name,
            .Group => |group| self.typeArgNameFromExpr(group.expr),
            else => null,
        };
    }

    fn exprIntegerText(self: *const TypeChecker, expr_id: ast.ExprId) ?[]const u8 {
        return switch (self.file.expression(expr_id).*) {
            .IntegerLiteral => |literal| std.mem.trim(u8, literal.text, " \t\n\r"),
            .Group => |group| self.exprIntegerText(group.expr),
            else => null,
        };
    }

    const GenericBindingValue = union(enum) {
        ty: Type,
        integer: []const u8,
    };

    const GenericTypeBinding = struct {
        name: []const u8,
        value: GenericBindingValue,
    };

    fn genericBindingType(binding: GenericTypeBinding) ?Type {
        return switch (binding.value) {
            .ty => |ty| ty,
            .integer => null,
        };
    }

    fn genericBindingInteger(binding: GenericTypeBinding) ?[]const u8 {
        return switch (binding.value) {
            .integer => |text| text,
            .ty => null,
        };
    }

    fn comptimeIntegerParameter(self: *const TypeChecker, parameter: ast.Parameter) bool {
        if (!parameter.is_comptime) return false;
        const ty = descriptorFromTypeExpr(self.arena, self.file, self.item_index, parameter.type_expr) catch Type{ .unknown = {} };
        return ty.kind() == .integer;
    }

    fn typeExprIntegerBinding(self: *const TypeChecker, type_expr: ast.TypeExprId, bindings: []const GenericTypeBinding) ?[]const u8 {
        return switch (self.file.typeExpr(type_expr).*) {
            .Path => |path| blk: {
                const trimmed = std.mem.trim(u8, path.name, " \t\n\r");
                for (bindings) |binding| {
                    if (!std.mem.eql(u8, trimmed, binding.name)) continue;
                    break :blk genericBindingInteger(binding);
                }
                break :blk null;
            },
            else => null,
        };
    }

    fn resolveIntegerGenericArg(
        self: *const TypeChecker,
        arg: ast.TypeArg,
        bindings: []const GenericTypeBinding,
    ) ?[]const u8 {
        return switch (arg) {
            .Integer => |literal| std.mem.trim(u8, literal.text, " \t\n\r"),
            .Type => |type_expr| self.typeExprIntegerBinding(type_expr, bindings),
        };
    }

    fn sanitizeGenericMangleSegment(self: *TypeChecker, text: []const u8) ![]const u8 {
        const trimmed = std.mem.trim(u8, text, " \t\n\r");
        var result = std.ArrayList(u8){};
        for (trimmed) |ch| {
            try result.append(self.arena, if (std.ascii.isAlphanumeric(ch)) ch else '_');
        }
        return result.toOwnedSlice(self.arena);
    }

    fn substituteGenericType(self: *TypeChecker, ty: Type, bindings: []const GenericTypeBinding) !Type {
        return switch (ty) {
            .named => |named| blk: {
                for (bindings) |binding| {
                    if (!std.mem.eql(u8, named.name, binding.name)) continue;
                    if (genericBindingType(binding)) |bound_type| break :blk bound_type;
                }
                break :blk ty;
            },
            .refinement => |refinement| blk: {
                const base_copy = try self.substituteGenericType(refinement.base_type.*, bindings);
                var refinement_copy = refinement;
                refinement_copy.base_type = try self.storeType(base_copy);
                break :blk .{ .refinement = refinement_copy };
            },
            .array => |array| blk: {
                const element = try self.substituteGenericType(array.element_type.*, bindings);
                break :blk .{ .array = .{
                    .element_type = try self.storeType(element),
                    .len = array.len,
                } };
            },
            .slice => |slice| blk: {
                const element = try self.substituteGenericType(slice.element_type.*, bindings);
                break :blk .{ .slice = .{
                    .element_type = try self.storeType(element),
                } };
            },
            .map => |map| blk: {
                const key_ptr = if (map.key_type) |key_type| blk_key: {
                    const key_copy = try self.substituteGenericType(key_type.*, bindings);
                    break :blk_key try self.storeType(key_copy);
                } else null;
                const value_ptr = if (map.value_type) |value_type| blk_value: {
                    const value_copy = try self.substituteGenericType(value_type.*, bindings);
                    break :blk_value try self.storeType(value_copy);
                } else null;
                break :blk .{ .map = .{
                    .key_type = key_ptr,
                    .value_type = value_ptr,
                } };
            },
            .tuple => |elements| blk: {
                const substituted = try self.arena.alloc(Type, elements.len);
                for (elements, 0..) |element, index| {
                    substituted[index] = try self.substituteGenericType(element, bindings);
                }
                break :blk .{ .tuple = substituted };
            },
            .error_union => |error_union| blk: {
                const payload = try self.substituteGenericType(error_union.payload_type.*, bindings);
                const errors = try self.arena.alloc(Type, error_union.error_types.len);
                for (error_union.error_types, 0..) |error_type, index| {
                    errors[index] = try self.substituteGenericType(error_type, bindings);
                }
                break :blk .{ .error_union = .{
                    .payload_type = try self.storeType(payload),
                    .error_types = errors,
                } };
            },
            else => ty,
        };
    }

    fn resolveTypeExpr(self: *TypeChecker, type_expr: ast.TypeExprId) anyerror!Type {
        return self.resolveTypeExprWithBindings(type_expr, &.{});
    }

    fn typeExprRange(self: *const TypeChecker, type_expr: ast.TypeExprId) source.TextRange {
        return switch (self.file.typeExpr(type_expr).*) {
            .Path => |node| node.range,
            .Generic => |node| node.range,
            .Tuple => |node| node.range,
            .Array => |node| node.range,
            .Slice => |node| node.range,
            .ErrorUnion => |node| node.range,
            .Error => |node| node.range,
        };
    }

    fn substituteGenericArgs(self: *TypeChecker, args: []const ast.TypeArg, bindings: []const GenericTypeBinding) ![]const ast.TypeArg {
        const resolved = try self.arena.alloc(ast.TypeArg, args.len);
        for (args, 0..) |arg, index| {
            resolved[index] = switch (arg) {
                .Integer => arg,
                .Type => |type_expr| if (self.typeExprIntegerBinding(type_expr, bindings)) |integer|
                    .{ .Integer = .{
                        .range = self.typeExprRange(type_expr),
                        .text = integer,
                    } }
                else
                    arg,
            };
        }
        return resolved;
    }

    fn resolveArraySize(self: *const TypeChecker, size: ast.TypeArraySize, bindings: []const GenericTypeBinding) ?u32 {
        _ = self;
        return switch (size) {
            .Integer => |literal| std.fmt.parseInt(u32, literal.text, 10) catch null,
            .Name => |name| blk: {
                const trimmed = std.mem.trim(u8, name.name, " \t\n\r");
                var integer_text: ?[]const u8 = null;
                for (bindings) |binding| {
                    if (!std.mem.eql(u8, trimmed, binding.name)) continue;
                    integer_text = genericBindingInteger(binding);
                    break;
                }
                const text = integer_text orelse break :blk null;
                break :blk std.fmt.parseInt(u32, text, 10) catch null;
            },
        };
    }

    fn resolveTypeExprWithBindings(self: *TypeChecker, type_expr: ast.TypeExprId, bindings: []const GenericTypeBinding) anyerror!Type {
        return switch (self.file.typeExpr(type_expr).*) {
            .Path => |path| blk: {
                const trimmed = std.mem.trim(u8, path.name, " \t\n\r");
                for (bindings) |binding| {
                    if (!std.mem.eql(u8, trimmed, binding.name)) continue;
                    if (genericBindingType(binding)) |bound_type| break :blk bound_type;
                }
                if (self.item_index.lookup(trimmed)) |item_id| {
                    switch (self.file.item(item_id).*) {
                        .TypeAlias => |type_alias| break :blk try self.resolveTypeAliasTarget(item_id, type_alias, bindings),
                        else => {},
                    }
                }
                break :blk descriptorFromPathName(self.file, self.item_index, trimmed);
            },
            .Generic => |generic| self.resolveGenericTypeWithBindings(generic, bindings),
            .Tuple => |tuple| blk: {
                const elements = try self.arena.alloc(Type, tuple.elements.len);
                for (tuple.elements, 0..) |element, index| {
                    elements[index] = try self.resolveTypeExprWithBindings(element, bindings);
                }
                break :blk .{ .tuple = elements };
            },
            .Array => |array| .{ .array = .{
                .element_type = try self.storeType(try self.resolveTypeExprWithBindings(array.element, bindings)),
                .len = self.resolveArraySize(array.size, bindings),
            } },
            .Slice => |slice| .{ .slice = .{
                .element_type = try self.storeType(try self.resolveTypeExprWithBindings(slice.element, bindings)),
            } },
            .ErrorUnion => |error_union| blk: {
                const errors = try self.arena.alloc(Type, error_union.errors.len);
                for (error_union.errors, 0..) |error_type, index| {
                    errors[index] = try self.resolveTypeExprWithBindings(error_type, bindings);
                }
                break :blk .{ .error_union = .{
                    .payload_type = try self.storeType(try self.resolveTypeExprWithBindings(error_union.payload, bindings)),
                    .error_types = errors,
                } };
            },
            .Error => .{ .unknown = {} },
        };
    }

    fn resolveGenericType(self: *TypeChecker, generic: ast.GenericTypeExpr) anyerror!Type {
        return self.resolveGenericTypeWithBindings(generic, &.{});
    }

    fn resolveGenericTypeWithBindings(self: *TypeChecker, generic: ast.GenericTypeExpr, bindings: []const GenericTypeBinding) anyerror!Type {
        if (self.item_index.lookup(generic.name)) |item_id| {
            switch (self.file.item(item_id).*) {
                .Struct => |struct_item| if (struct_item.is_generic) {
                    return try self.instantiateGenericStruct(item_id, struct_item, generic, bindings);
                },
                .Enum => |enum_item| if (enum_item.is_generic) {
                    return try self.instantiateGenericEnum(item_id, enum_item, generic, bindings);
                },
                .Bitfield => |bitfield_item| if (bitfield_item.is_generic) {
                    return try self.instantiateGenericBitfield(item_id, bitfield_item, generic, bindings);
                },
                .TypeAlias => |type_alias| if (type_alias.is_generic) {
                    return try self.instantiateGenericTypeAlias(item_id, type_alias, generic, bindings);
                },
                else => {},
            }
        }
        if (std.mem.eql(u8, generic.name, "map")) {
            return .{ .map = .{
                .key_type = if (generic.args.len > 0 and generic.args[0] == .Type)
                    try self.storeType(try self.resolveTypeExprWithBindings(generic.args[0].Type, bindings))
                else
                    null,
                .value_type = if (generic.args.len > 1 and generic.args[1] == .Type)
                    try self.storeType(try self.resolveTypeExprWithBindings(generic.args[1].Type, bindings))
                else
                    null,
            } };
        }

        if (std.mem.eql(u8, generic.name, "MinValue") or
            std.mem.eql(u8, generic.name, "MaxValue") or
            std.mem.eql(u8, generic.name, "InRange") or
            std.mem.eql(u8, generic.name, "Scaled") or
            std.mem.eql(u8, generic.name, "Exact") or
            std.mem.eql(u8, generic.name, "NonZero") or
            std.mem.eql(u8, generic.name, "NonZeroAddress") or
            std.mem.eql(u8, generic.name, "BasisPoints"))
        {
            if (generic.args.len > 0 and generic.args[0] == .Type) {
                const resolved_args = try self.substituteGenericArgs(generic.args, bindings);
                return .{ .refinement = .{
                    .name = generic.name,
                    .base_type = try self.storeType(try self.resolveTypeExprWithBindings(generic.args[0].Type, bindings)),
                    .args = resolved_args,
                } };
            }
        }

        if (bindings.len == 0) {
            return descriptorFromGenericType(self.arena, self.file, self.item_index, generic);
        }

        return descriptorFromPathName(self.file, self.item_index, generic.name);
    }

    fn structLiteralType(self: *const TypeChecker, name: []const u8) Type {
        if (self.item_index.lookup(name)) |item_id| {
            return self.item_types[item_id.index()];
        }
        if (self.instantiatedStructByName(name) != null) {
            return .{ .struct_ = .{ .name = name } };
        }
        if (self.instantiatedEnumByName(name) != null) {
            return .{ .enum_ = .{ .name = name } };
        }
        if (self.instantiatedBitfieldByName(name) != null) {
            return .{ .bitfield = .{ .name = name } };
        }
        return .{ .named = .{ .name = name } };
    }

    fn instantiateGenericStruct(
        self: *TypeChecker,
        item_id: ast.ItemId,
        struct_item: ast.StructItem,
        generic: ast.GenericTypeExpr,
        outer_bindings: []const GenericTypeBinding,
    ) anyerror!Type {
        const bindings = try self.genericTypeBindingsForStruct(struct_item, generic, outer_bindings);
        const mangled_name = try self.mangleGenericStructName(struct_item.name, bindings);

        if (self.instantiatedStructByName(mangled_name) == null) {
            const fields = try self.arena.alloc(InstantiatedStructField, struct_item.fields.len);
            for (struct_item.fields, 0..) |field, index| {
                fields[index] = .{
                    .name = field.name,
                    .ty = try self.resolveTypeExprWithBindings(field.type_expr, bindings),
                };
            }
            try self.instantiated_structs.append(self.arena, .{
                .template_item_id = item_id,
                .mangled_name = mangled_name,
                .fields = fields,
            });
        }

        return .{ .struct_ = .{ .name = mangled_name } };
    }

    fn genericTypeBindingsForStruct(
        self: *TypeChecker,
        struct_item: ast.StructItem,
        generic: ast.GenericTypeExpr,
        outer_bindings: []const GenericTypeBinding,
    ) anyerror![]const GenericTypeBinding {
        if (struct_item.template_parameters.len != generic.args.len) return error.InvalidGenericStructInstantiation;

        const bindings = try self.arena.alloc(GenericTypeBinding, struct_item.template_parameters.len);
        for (struct_item.template_parameters, generic.args, 0..) |parameter, arg, index| {
            const name = self.patternName(parameter.pattern) orelse return error.InvalidGenericStructInstantiation;
            const value = if (self.isGenericTypeParameter(parameter))
                blk: {
                    const ty = switch (arg) {
                        .Type => |type_expr| try self.resolveTypeExprWithBindings(type_expr, outer_bindings),
                        else => return error.InvalidGenericStructInstantiation,
                    };
                    break :blk GenericBindingValue{ .ty = ty };
                }
            else if (self.comptimeIntegerParameter(parameter))
                blk: {
                    const integer = self.resolveIntegerGenericArg(arg, outer_bindings) orelse return error.InvalidGenericStructInstantiation;
                    break :blk GenericBindingValue{ .integer = integer };
                }
            else
                return error.InvalidGenericStructInstantiation;
            bindings[index] = .{
                .name = name,
                .value = value,
            };
        }
        return bindings;
    }

    fn mangleGenericStructName(self: *TypeChecker, base_name: []const u8, bindings: []const GenericTypeBinding) anyerror![]const u8 {
        var name = std.ArrayList(u8){};
        try name.appendSlice(self.arena, base_name);
        for (bindings) |binding| {
            try name.appendSlice(self.arena, "__");
            switch (binding.value) {
                .ty => |ty| try self.appendTypeMangleName(&name, ty),
                .integer => |integer| try name.appendSlice(self.arena, try self.sanitizeGenericMangleSegment(integer)),
            }
        }
        return name.toOwnedSlice(self.arena);
    }

    fn appendTypeMangleName(self: *TypeChecker, buffer: *std.ArrayList(u8), ty: Type) anyerror!void {
        switch (ty) {
            .bool => try buffer.appendSlice(self.arena, "bool"),
            .address => try buffer.appendSlice(self.arena, "address"),
            .string => try buffer.appendSlice(self.arena, "string"),
            .bytes => try buffer.appendSlice(self.arena, "bytes"),
            .void => try buffer.appendSlice(self.arena, "void"),
            .integer => |integer| try buffer.appendSlice(self.arena, integer.spelling orelse "int"),
            .named => |named| try buffer.appendSlice(self.arena, named.name),
            .struct_ => |named| try buffer.appendSlice(self.arena, named.name),
            .contract => |named| try buffer.appendSlice(self.arena, named.name),
            .bitfield => |named| try buffer.appendSlice(self.arena, named.name),
            .enum_ => |named| try buffer.appendSlice(self.arena, named.name),
            .slice => |slice| {
                try buffer.appendSlice(self.arena, "slice_");
                try self.appendTypeMangleName(buffer, slice.element_type.*);
            },
            .array => |array| {
                try buffer.appendSlice(self.arena, "array_");
                try self.appendTypeMangleName(buffer, array.element_type.*);
                if (array.len) |len| {
                    try buffer.append(self.arena, '_');
                    try buffer.writer(self.arena).print("{d}", .{len});
                }
            },
            .map => |map| {
                try buffer.appendSlice(self.arena, "map");
                if (map.key_type) |key| {
                    try buffer.append(self.arena, '_');
                    try self.appendTypeMangleName(buffer, key.*);
                }
                if (map.value_type) |value| {
                    try buffer.append(self.arena, '_');
                    try self.appendTypeMangleName(buffer, value.*);
                }
            },
            else => try buffer.appendSlice(self.arena, "type"),
        }
    }

    fn instantiatedStructByName(self: *const TypeChecker, name: []const u8) ?InstantiatedStruct {
        for (self.instantiated_structs.items) |instantiated| {
            if (std.mem.eql(u8, instantiated.mangled_name, name)) return instantiated;
        }
        return null;
    }

    fn instantiateGenericEnum(
        self: *TypeChecker,
        item_id: ast.ItemId,
        enum_item: ast.EnumItem,
        generic: ast.GenericTypeExpr,
        outer_bindings: []const GenericTypeBinding,
    ) anyerror!Type {
        const bindings = try self.genericTypeBindingsForEnum(enum_item, generic, outer_bindings);
        const mangled_name = try self.mangleGenericStructName(enum_item.name, bindings);

        if (self.instantiatedEnumByName(mangled_name) == null) {
            try self.instantiated_enums.append(self.arena, .{
                .template_item_id = item_id,
                .mangled_name = mangled_name,
            });
        }

        return .{ .enum_ = .{ .name = mangled_name } };
    }

    fn genericTypeBindingsForEnum(
        self: *TypeChecker,
        enum_item: ast.EnumItem,
        generic: ast.GenericTypeExpr,
        outer_bindings: []const GenericTypeBinding,
    ) anyerror![]const GenericTypeBinding {
        if (enum_item.template_parameters.len != generic.args.len) return error.InvalidGenericEnumInstantiation;

        const bindings = try self.arena.alloc(GenericTypeBinding, enum_item.template_parameters.len);
        for (enum_item.template_parameters, generic.args, 0..) |parameter, arg, index| {
            const name = self.patternName(parameter.pattern) orelse return error.InvalidGenericEnumInstantiation;
            const value = if (self.isGenericTypeParameter(parameter))
                blk: {
                    const ty = switch (arg) {
                        .Type => |type_expr| try self.resolveTypeExprWithBindings(type_expr, outer_bindings),
                        else => return error.InvalidGenericEnumInstantiation,
                    };
                    break :blk GenericBindingValue{ .ty = ty };
                }
            else if (self.comptimeIntegerParameter(parameter))
                blk: {
                    const integer = self.resolveIntegerGenericArg(arg, outer_bindings) orelse return error.InvalidGenericEnumInstantiation;
                    break :blk GenericBindingValue{ .integer = integer };
                }
            else
                return error.InvalidGenericEnumInstantiation;
            bindings[index] = .{
                .name = name,
                .value = value,
            };
        }
        return bindings;
    }

    fn instantiatedEnumByName(self: *const TypeChecker, name: []const u8) ?InstantiatedEnum {
        for (self.instantiated_enums.items) |instantiated| {
            if (std.mem.eql(u8, instantiated.mangled_name, name)) return instantiated;
        }
        return null;
    }

    fn instantiateGenericBitfield(
        self: *TypeChecker,
        item_id: ast.ItemId,
        bitfield_item: ast.BitfieldItem,
        generic: ast.GenericTypeExpr,
        outer_bindings: []const GenericTypeBinding,
    ) anyerror!Type {
        const bindings = try self.genericTypeBindingsForBitfield(bitfield_item, generic, outer_bindings);
        const mangled_name = try self.mangleGenericStructName(bitfield_item.name, bindings);

        if (self.instantiatedBitfieldByName(mangled_name) == null) {
            const fields = try self.arena.alloc(InstantiatedBitfieldField, bitfield_item.fields.len);
            for (bitfield_item.fields, 0..) |field, index| {
                fields[index] = .{
                    .name = field.name,
                    .ty = try self.resolveTypeExprWithBindings(field.type_expr, bindings),
                    .offset = field.offset,
                    .width = field.width,
                };
            }
            try self.instantiated_bitfields.append(self.arena, .{
                .template_item_id = item_id,
                .mangled_name = mangled_name,
                .base_type = if (bitfield_item.base_type) |type_expr|
                    try self.resolveTypeExprWithBindings(type_expr, bindings)
                else
                    null,
                .fields = fields,
            });
        }

        return .{ .bitfield = .{ .name = mangled_name } };
    }

    fn instantiateGenericTypeAlias(
        self: *TypeChecker,
        item_id: ast.ItemId,
        type_alias: ast.TypeAliasItem,
        generic: ast.GenericTypeExpr,
        outer_bindings: []const GenericTypeBinding,
    ) anyerror!Type {
        const bindings = try self.genericTypeBindingsForAlias(type_alias, generic, outer_bindings);
        return self.resolveTypeAliasTarget(item_id, type_alias, bindings);
    }

    fn genericTypeBindingsForAlias(
        self: *TypeChecker,
        type_alias: ast.TypeAliasItem,
        generic: ast.GenericTypeExpr,
        outer_bindings: []const GenericTypeBinding,
    ) anyerror![]const GenericTypeBinding {
        if (type_alias.template_parameters.len != generic.args.len) return error.InvalidGenericTypeAliasInstantiation;

        const bindings = try self.arena.alloc(GenericTypeBinding, type_alias.template_parameters.len);
        for (type_alias.template_parameters, generic.args, 0..) |parameter, arg, index| {
            const name = self.patternName(parameter.pattern) orelse return error.InvalidGenericTypeAliasInstantiation;
            const value = if (self.isGenericTypeParameter(parameter))
                blk: {
                    const ty = switch (arg) {
                        .Type => |type_expr| try self.resolveTypeExprWithBindings(type_expr, outer_bindings),
                        else => return error.InvalidGenericTypeAliasInstantiation,
                    };
                    break :blk GenericBindingValue{ .ty = ty };
                }
            else if (self.comptimeIntegerParameter(parameter))
                blk: {
                    const integer = self.resolveIntegerGenericArg(arg, outer_bindings) orelse return error.InvalidGenericTypeAliasInstantiation;
                    break :blk GenericBindingValue{ .integer = integer };
                }
            else
                return error.InvalidGenericTypeAliasInstantiation;
            bindings[index] = .{
                .name = name,
                .value = value,
            };
        }
        return bindings;
    }

    fn resolveTypeAliasTarget(
        self: *TypeChecker,
        item_id: ast.ItemId,
        type_alias: ast.TypeAliasItem,
        bindings: []const GenericTypeBinding,
    ) anyerror!Type {
        for (self.active_aliases.items) |active_id| {
            if (active_id == item_id) {
                try self.emitRangeError(type_alias.range, "recursive type alias '{s}' is not supported", .{type_alias.name});
                return .{ .unknown = {} };
            }
        }
        try self.active_aliases.append(self.arena, item_id);
        defer _ = self.active_aliases.pop();
        return self.resolveTypeExprWithBindings(type_alias.target_type, bindings);
    }

    fn genericTypeBindingsForBitfield(
        self: *TypeChecker,
        bitfield_item: ast.BitfieldItem,
        generic: ast.GenericTypeExpr,
        outer_bindings: []const GenericTypeBinding,
    ) anyerror![]const GenericTypeBinding {
        if (bitfield_item.template_parameters.len != generic.args.len) return error.InvalidGenericBitfieldInstantiation;

        const bindings = try self.arena.alloc(GenericTypeBinding, bitfield_item.template_parameters.len);
        for (bitfield_item.template_parameters, generic.args, 0..) |parameter, arg, index| {
            const name = self.patternName(parameter.pattern) orelse return error.InvalidGenericBitfieldInstantiation;
            const value = if (self.isGenericTypeParameter(parameter))
                blk: {
                    const ty = switch (arg) {
                        .Type => |type_expr| try self.resolveTypeExprWithBindings(type_expr, outer_bindings),
                        else => return error.InvalidGenericBitfieldInstantiation,
                    };
                    break :blk GenericBindingValue{ .ty = ty };
                }
            else if (self.comptimeIntegerParameter(parameter))
                blk: {
                    const integer = self.resolveIntegerGenericArg(arg, outer_bindings) orelse return error.InvalidGenericBitfieldInstantiation;
                    break :blk GenericBindingValue{ .integer = integer };
                }
            else
                return error.InvalidGenericBitfieldInstantiation;
            bindings[index] = .{
                .name = name,
                .value = value,
            };
        }
        return bindings;
    }

    fn instantiatedBitfieldByName(self: *const TypeChecker, name: []const u8) ?InstantiatedBitfield {
        for (self.instantiated_bitfields.items) |instantiated| {
            if (std.mem.eql(u8, instantiated.mangled_name, name)) return instantiated;
        }
        return null;
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

    fn summarizeFunctionEffects(self: *TypeChecker, item_id: ast.ItemId, function: ast.FunctionItem) !Effect {
        const previous_function_item = self.current_function_item;
        self.current_function_item = item_id;
        defer self.current_function_item = previous_function_item;
        var state = EffectCollectorState.init();
        try self.collectBodyEffects(function.body, &state);
        return self.effectFromState(state);
    }

    fn ensureFunctionEffectSummary(self: *TypeChecker, item_id: ast.ItemId, function: ast.FunctionItem) !void {
        switch (self.effect_states[item_id.index()]) {
            .done => return,
            .visiting => return,
            .unvisited => {},
        }
        _ = function;

        const node_count = self.file.items.len;
        const indexes = try self.arena.alloc(?u32, node_count);
        const lowlinks = try self.arena.alloc(u32, node_count);
        const on_stack = try self.arena.alloc(bool, node_count);
        @memset(indexes, null);
        @memset(lowlinks, 0);
        @memset(on_stack, false);
        var stack: std.ArrayList(ast.ItemId) = .{};
        var next_index: u32 = 0;

        try self.strongConnectEffectFunction(item_id, &next_index, indexes, lowlinks, on_stack, &stack);
    }

    fn strongConnectEffectFunction(
        self: *TypeChecker,
        item_id: ast.ItemId,
        next_index: *u32,
        indexes: []?u32,
        lowlinks: []u32,
        on_stack: []bool,
        stack: *std.ArrayList(ast.ItemId),
    ) !void {
        if (self.effect_states[item_id.index()] == .done) return;
        if (indexes[item_id.index()] != null) return;

        const function = switch (self.file.item(item_id).*) {
            .Function => |function| function,
            else => return,
        };

        indexes[item_id.index()] = next_index.*;
        lowlinks[item_id.index()] = next_index.*;
        next_index.* += 1;
        self.effect_states[item_id.index()] = .visiting;
        self.item_effects[item_id.index()] = .pure;
        try stack.append(self.arena, item_id);
        on_stack[item_id.index()] = true;

        var callees: std.ArrayList(ast.ItemId) = .{};
        try self.collectFunctionDirectCallees(item_id, function, &callees);
        for (callees.items) |callee_id| {
            if (self.file.item(callee_id).* != .Function) continue;
            if (self.effect_states[callee_id.index()] == .done) continue;
            if (indexes[callee_id.index()] == null) {
                try self.strongConnectEffectFunction(callee_id, next_index, indexes, lowlinks, on_stack, stack);
                lowlinks[item_id.index()] = @min(lowlinks[item_id.index()], lowlinks[callee_id.index()]);
            } else if (on_stack[callee_id.index()]) {
                lowlinks[item_id.index()] = @min(lowlinks[item_id.index()], indexes[callee_id.index()].?);
            }
        }

        if (lowlinks[item_id.index()] != indexes[item_id.index()].?) return;

        var component: std.ArrayList(ast.ItemId) = .{};
        while (stack.items.len > 0) {
            const member = stack.pop().?;
            on_stack[member.index()] = false;
            try component.append(self.arena, member);
            if (member.index() == item_id.index()) break;
        }

        var changed = true;
        while (changed) {
            changed = false;
            for (component.items) |member_id| {
                const member_function = switch (self.file.item(member_id).*) {
                    .Function => |member_function| member_function,
                    else => continue,
                };
                const new_effect = try self.summarizeFunctionEffects(member_id, member_function);
                if (!effectEql(new_effect, self.item_effects[member_id.index()])) {
                    self.item_effects[member_id.index()] = new_effect;
                    changed = true;
                }
            }
        }

        for (component.items) |member_id| {
            self.effect_states[member_id.index()] = .done;
        }
    }

    fn validateBodyLocks(self: *TypeChecker, body_id: ast.BodyId, locked_slots: *std.ArrayList(EffectSlot)) anyerror!void {
        const body = self.file.body(body_id).*;
        for (body.statements) |statement_id| {
            try self.validateStmtLocks(statement_id, locked_slots);
        }
    }

    fn validateStmtLocks(self: *TypeChecker, statement_id: ast.StmtId, locked_slots: *std.ArrayList(EffectSlot)) anyerror!void {
        switch (self.file.statement(statement_id).*) {
            .VariableDecl, .Return, .Assert, .Assume, .Expr, .Havoc, .Break, .Continue, .Error => {
                if (self.statementExpr(statement_id)) |expr_id| try self.validateExprLocks(expr_id, locked_slots);
            },
            .If => |if_stmt| {
                try self.validateExprLocks(if_stmt.condition, locked_slots);
                var then_locked = try self.cloneEffectSlots(locked_slots.items);
                var else_locked = try self.cloneEffectSlots(locked_slots.items);
                defer then_locked.deinit(self.arena);
                defer else_locked.deinit(self.arena);
                try self.validateBodyLocks(if_stmt.then_body, &then_locked);
                if (if_stmt.else_body) |else_body| try self.validateBodyLocks(else_body, &else_locked);
                locked_slots.* = try self.intersectLockedSlots(then_locked.items, else_locked.items);
            },
            .While => |while_stmt| {
                try self.validateExprLocks(while_stmt.condition, locked_slots);
                for (while_stmt.invariants) |expr_id| try self.validateExprLocks(expr_id, locked_slots);
                var loop_locked = try self.cloneEffectSlots(locked_slots.items);
                defer loop_locked.deinit(self.arena);
                try self.validateBodyLocks(while_stmt.body, &loop_locked);
                locked_slots.* = try self.intersectLockedSlots(locked_slots.items, loop_locked.items);
            },
            .For => |for_stmt| {
                try self.validateExprLocks(for_stmt.iterable, locked_slots);
                for (for_stmt.invariants) |expr_id| try self.validateExprLocks(expr_id, locked_slots);
                var loop_locked = try self.cloneEffectSlots(locked_slots.items);
                defer loop_locked.deinit(self.arena);
                try self.validateBodyLocks(for_stmt.body, &loop_locked);
                locked_slots.* = try self.intersectLockedSlots(locked_slots.items, loop_locked.items);
            },
            .Switch => |switch_stmt| {
                try self.validateExprLocks(switch_stmt.condition, locked_slots);
                var merged_locked = try self.cloneEffectSlots(locked_slots.items);
                for (switch_stmt.arms) |arm| {
                    switch (arm.pattern) {
                        .Expr => |expr_id| try self.validateExprLocks(expr_id, locked_slots),
                        .Range => |range_pattern| {
                            try self.validateExprLocks(range_pattern.start, locked_slots);
                            try self.validateExprLocks(range_pattern.end, locked_slots);
                        },
                        .Else => {},
                    }
                    var case_locked = try self.cloneEffectSlots(locked_slots.items);
                    defer case_locked.deinit(self.arena);
                    try self.validateBodyLocks(arm.body, &case_locked);
                    const next = try self.intersectLockedSlots(merged_locked.items, case_locked.items);
                    merged_locked.deinit(self.arena);
                    merged_locked = next;
                }
                if (switch_stmt.else_body) |else_body| {
                    var else_locked = try self.cloneEffectSlots(locked_slots.items);
                    defer else_locked.deinit(self.arena);
                    try self.validateBodyLocks(else_body, &else_locked);
                    const next = try self.intersectLockedSlots(merged_locked.items, else_locked.items);
                    merged_locked.deinit(self.arena);
                    merged_locked = next;
                }
                locked_slots.* = merged_locked;
            },
            .Try => |try_stmt| {
                var try_locked = try self.cloneEffectSlots(locked_slots.items);
                defer try_locked.deinit(self.arena);
                try self.validateBodyLocks(try_stmt.try_body, &try_locked);
                if (try_stmt.catch_clause) |catch_clause| {
                    var catch_locked = try self.cloneEffectSlots(locked_slots.items);
                    defer catch_locked.deinit(self.arena);
                    try self.validateBodyLocks(catch_clause.body, &catch_locked);
                    locked_slots.* = try self.intersectLockedSlots(try_locked.items, catch_locked.items);
                } else {
                    locked_slots.* = try self.cloneEffectSlots(try_locked.items);
                }
            },
            .Log => |log_stmt| for (log_stmt.args) |arg| try self.validateExprLocks(arg, locked_slots),
            .Block => |block| {
                var nested_locked = try self.cloneEffectSlots(locked_slots.items);
                defer nested_locked.deinit(self.arena);
                try self.validateBodyLocks(block.body, &nested_locked);
                locked_slots.* = try self.cloneEffectSlots(nested_locked.items);
            },
            .LabeledBlock => |block| {
                var nested_locked = try self.cloneEffectSlots(locked_slots.items);
                defer nested_locked.deinit(self.arena);
                try self.validateBodyLocks(block.body, &nested_locked);
                locked_slots.* = try self.cloneEffectSlots(nested_locked.items);
            },
            .Lock => |lock_stmt| {
                try self.validateExprLocks(lock_stmt.path, locked_slots);
                if (self.lockSlotForExpr(lock_stmt.path)) |slot| try self.appendUniqueSlot(locked_slots, slot);
            },
            .Unlock => |unlock_stmt| {
                try self.validateExprLocks(unlock_stmt.path, locked_slots);
                if (self.lockSlotForExpr(unlock_stmt.path)) |slot| self.removeLockedSlot(locked_slots, slot);
            },
            .Assign => |assign| {
                try self.validateExprLocks(assign.value, locked_slots);
                var state = EffectCollectorState.init();
                try self.collectPatternTargetEffects(assign.target, assign.op, &state);
                try self.emitLockedWriteDiagnostics(assign.range, state.writes.items, locked_slots.items);
            },
        }
    }

    fn validateExprLocks(self: *TypeChecker, expr_id: ast.ExprId, locked_slots: *std.ArrayList(EffectSlot)) anyerror!void {
        switch (self.file.expression(expr_id).*) {
            .Call => |call| {
                try self.validateExprLocks(call.callee, locked_slots);
                for (call.args) |arg| try self.validateExprLocks(arg, locked_slots);
                if (self.calleeFunctionItem(call.callee)) |callee_id| {
                    switch (self.file.item(callee_id).*) {
                        .Function => |function| {
                            try self.ensureFunctionEffectSummary(callee_id, function);
                            try self.emitLockedWriteDiagnostics(call.range, self.effectWrites(self.item_effects[callee_id.index()]), locked_slots.items);
                        },
                        else => {},
                    }
                } else if (self.callableType(call.callee).kind() == .function) {
                    try self.emitUnknownLockedCallDiagnostics(call.range, locked_slots.items);
                }
            },
            .Unary => |unary| try self.validateExprLocks(unary.operand, locked_slots),
            .Binary => |binary| {
                try self.validateExprLocks(binary.lhs, locked_slots);
                try self.validateExprLocks(binary.rhs, locked_slots);
            },
            .Tuple => |tuple| for (tuple.elements) |element| try self.validateExprLocks(element, locked_slots),
            .ArrayLiteral => |array| for (array.elements) |element| try self.validateExprLocks(element, locked_slots),
            .StructLiteral => |struct_literal| for (struct_literal.fields) |field| try self.validateExprLocks(field.value, locked_slots),
            .Switch => |switch_expr| {
                try self.validateExprLocks(switch_expr.condition, locked_slots);
                for (switch_expr.arms) |arm| {
                    switch (arm.pattern) {
                        .Expr => |pattern_expr| try self.validateExprLocks(pattern_expr, locked_slots),
                        .Range => |range_pattern| {
                            try self.validateExprLocks(range_pattern.start, locked_slots);
                            try self.validateExprLocks(range_pattern.end, locked_slots);
                        },
                        .Else => {},
                    }
                    try self.validateExprLocks(arm.value, locked_slots);
                }
                if (switch_expr.else_expr) |else_expr| try self.validateExprLocks(else_expr, locked_slots);
            },
            .Builtin => |builtin| for (builtin.args) |arg| try self.validateExprLocks(arg, locked_slots),
            .Field => |field| try self.validateExprLocks(field.base, locked_slots),
            .Index => |index| {
                try self.validateExprLocks(index.base, locked_slots);
                try self.validateExprLocks(index.index, locked_slots);
            },
            .Group => |group| try self.validateExprLocks(group.expr, locked_slots),
            .Comptime, .Old, .Quantified, .ErrorReturn, .Name, .IntegerLiteral, .StringLiteral, .BoolLiteral, .AddressLiteral, .BytesLiteral, .Result, .Error => {},
        }
    }

    fn statementExpr(self: *TypeChecker, statement_id: ast.StmtId) ?ast.ExprId {
        return switch (self.file.statement(statement_id).*) {
            .VariableDecl => |decl| decl.value,
            .Return => |ret| ret.value,
            .Assert => |assert_stmt| assert_stmt.condition,
            .Assume => |assume_stmt| assume_stmt.condition,
            .Expr => |expr_stmt| expr_stmt.expr,
            else => null,
        };
    }

    fn lockSlotForExpr(self: *TypeChecker, expr_id: ast.ExprId) ?EffectSlot {
        return switch (self.file.expression(expr_id).*) {
            .Name => |name| self.lookupNamedFieldSlot(name.name),
            .Group => |group| self.lockSlotForExpr(group.expr),
            .Field => |field| self.lockSlotForExpr(field.base),
            .Index => |index| blk: {
                const base = self.lockSlotForExpr(index.base) orelse break :blk null;
                break :blk self.slotWithIndexKey(base, index.index);
            },
            else => null,
        };
    }

    fn emitLockedWriteDiagnostics(self: *TypeChecker, range: source.TextRange, writes: []const EffectSlot, locked_slots: []const EffectSlot) !void {
        for (writes) |write_slot| {
            for (locked_slots) |locked_slot| {
                if (!self.effectSlotsMayAlias(write_slot, locked_slot)) continue;
                try self.emitRangeError(range, "cannot write locked {s} slot '{s}'", .{
                    region_rules.regionDisplayName(write_slot.region),
                    write_slot.name,
                });
                break;
            }
        }
    }

    fn emitUnknownLockedCallDiagnostics(self: *TypeChecker, range: source.TextRange, locked_slots: []const EffectSlot) !void {
        for (locked_slots) |locked_slot| {
            try self.emitRangeError(range, "unresolved call may write locked {s} slot '{s}'", .{
                region_rules.regionDisplayName(locked_slot.region),
                locked_slot.name,
            });
        }
    }

    fn effectWrites(self: *TypeChecker, effect: Effect) []const EffectSlot {
        _ = self;
        return switch (effect) {
            .pure, .external, .side_effects, .reads => &.{},
            .writes => |write_effect| write_effect.slots,
            .reads_writes => |read_write| read_write.writes,
        };
    }

    fn effectHasExternal(self: *TypeChecker, effect: Effect) bool {
        _ = self;
        return switch (effect) {
            .pure => false,
            .external => true,
            .side_effects => |side_effects| side_effects.has_external,
            .reads => |read_effect| read_effect.has_external,
            .writes => |write_effect| write_effect.has_external,
            .reads_writes => |read_write| read_write.has_external,
        };
    }

    fn effectHasLog(self: *TypeChecker, effect: Effect) bool {
        _ = self;
        return switch (effect) {
            .pure, .external => false,
            .side_effects => |side_effects| side_effects.has_log,
            .reads => |read_effect| read_effect.has_log,
            .writes => |write_effect| write_effect.has_log,
            .reads_writes => |read_write| read_write.has_log,
        };
    }

    fn effectHasHavoc(self: *TypeChecker, effect: Effect) bool {
        _ = self;
        return switch (effect) {
            .pure, .external => false,
            .side_effects => |side_effects| side_effects.has_havoc,
            .reads => |read_effect| read_effect.has_havoc,
            .writes => |write_effect| write_effect.has_havoc,
            .reads_writes => |read_write| read_write.has_havoc,
        };
    }

    fn effectHasLock(self: *TypeChecker, effect: Effect) bool {
        _ = self;
        return switch (effect) {
            .pure, .external => false,
            .side_effects => |side_effects| side_effects.has_lock,
            .reads => |read_effect| read_effect.has_lock,
            .writes => |write_effect| write_effect.has_lock,
            .reads_writes => |read_write| read_write.has_lock,
        };
    }

    fn effectHasUnlock(self: *TypeChecker, effect: Effect) bool {
        _ = self;
        return switch (effect) {
            .pure, .external => false,
            .side_effects => |side_effects| side_effects.has_unlock,
            .reads => |read_effect| read_effect.has_unlock,
            .writes => |write_effect| write_effect.has_unlock,
            .reads_writes => |read_write| read_write.has_unlock,
        };
    }

    const EffectCollectorState = struct {
        reads: std.ArrayList(EffectSlot) = .{},
        writes: std.ArrayList(EffectSlot) = .{},
        has_external: bool = false,
        has_log: bool = false,
        has_havoc: bool = false,
        has_lock: bool = false,
        has_unlock: bool = false,

        fn init() EffectCollectorState {
            return .{};
        }
    };

    fn cloneEffectSlots(self: *TypeChecker, items: []const EffectSlot) !std.ArrayList(EffectSlot) {
        var clone: std.ArrayList(EffectSlot) = .{};
        for (items) |item| try clone.append(self.arena, item);
        return clone;
    }

    fn mergeLockedSlots(self: *TypeChecker, dst: *std.ArrayList(EffectSlot), src: []const EffectSlot) !void {
        for (src) |slot| try self.appendUniqueSlot(dst, slot);
    }

    fn intersectLockedSlots(self: *TypeChecker, lhs: []const EffectSlot, rhs: []const EffectSlot) !std.ArrayList(EffectSlot) {
        var result: std.ArrayList(EffectSlot) = .{};
        for (lhs) |lhs_slot| {
            for (rhs) |rhs_slot| {
                if (!effectSlotEql(lhs_slot, rhs_slot)) continue;
                try result.append(self.arena, lhs_slot);
                break;
            }
        }
        return result;
    }

    fn removeLockedSlot(self: *TypeChecker, slots: *std.ArrayList(EffectSlot), slot: EffectSlot) void {
        _ = self;
        var index: usize = 0;
        while (index < slots.items.len) : (index += 1) {
            const existing = slots.items[index];
            if (!effectSlotEql(existing, slot)) continue;
            _ = slots.swapRemove(index);
            return;
        }
    }

    fn buildEffect(reads: []const EffectSlot, writes: []const EffectSlot, has_external: bool, has_log: bool, has_havoc: bool, has_lock: bool, has_unlock: bool) Effect {
        if (reads.len == 0 and writes.len == 0) {
            if (has_external and !has_log and !has_havoc) return .external;
            if (has_external or has_log or has_havoc or has_lock or has_unlock) return .{ .side_effects = .{
                .has_external = has_external,
                .has_log = has_log,
                .has_havoc = has_havoc,
                .has_lock = has_lock,
                .has_unlock = has_unlock,
            } };
            return .pure;
        }
        if (reads.len == 0) return .{ .writes = .{
            .slots = writes,
            .has_external = has_external,
            .has_log = has_log,
            .has_havoc = has_havoc,
            .has_lock = has_lock,
            .has_unlock = has_unlock,
        } };
        if (writes.len == 0) return .{ .reads = .{
            .slots = reads,
            .has_external = has_external,
            .has_log = has_log,
            .has_havoc = has_havoc,
            .has_lock = has_lock,
            .has_unlock = has_unlock,
        } };
        return .{ .reads_writes = .{
            .reads = reads,
            .writes = writes,
            .has_external = has_external,
            .has_log = has_log,
            .has_havoc = has_havoc,
            .has_lock = has_lock,
            .has_unlock = has_unlock,
        } };
    }

    fn effectFromState(self: *TypeChecker, state: EffectCollectorState) Effect {
        _ = self;
        return buildEffect(state.reads.items, state.writes.items, state.has_external, state.has_log, state.has_havoc, state.has_lock, state.has_unlock);
    }

    fn collectBodyEffects(self: *TypeChecker, body_id: ast.BodyId, state: *EffectCollectorState) anyerror!void {
        const body = self.file.body(body_id).*;
        for (body.statements) |statement_id| {
            try self.collectStmtEffects(statement_id, state);
        }
    }

    fn collectStmtEffects(self: *TypeChecker, statement_id: ast.StmtId, state: *EffectCollectorState) anyerror!void {
        switch (self.file.statement(statement_id).*) {
            .VariableDecl => |decl| if (decl.value) |expr_id| {
                try self.collectExprEffects(expr_id, state);
            },
            .Return => |ret| if (ret.value) |expr_id| {
                try self.collectExprEffects(expr_id, state);
            },
            .If => |if_stmt| {
                try self.collectExprEffects(if_stmt.condition, state);
                try self.collectBodyEffects(if_stmt.then_body, state);
                if (if_stmt.else_body) |else_body| try self.collectBodyEffects(else_body, state);
            },
            .While => |while_stmt| {
                try self.collectExprEffects(while_stmt.condition, state);
                for (while_stmt.invariants) |expr_id| try self.collectExprEffects(expr_id, state);
                try self.collectBodyEffects(while_stmt.body, state);
            },
            .For => |for_stmt| {
                try self.collectExprEffects(for_stmt.iterable, state);
                for (for_stmt.invariants) |expr_id| try self.collectExprEffects(expr_id, state);
                try self.collectBodyEffects(for_stmt.body, state);
            },
            .Switch => |switch_stmt| {
                try self.collectExprEffects(switch_stmt.condition, state);
                for (switch_stmt.arms) |arm| {
                    switch (arm.pattern) {
                        .Expr => |expr_id| try self.collectExprEffects(expr_id, state),
                        .Range => |range_pattern| {
                            try self.collectExprEffects(range_pattern.start, state);
                            try self.collectExprEffects(range_pattern.end, state);
                        },
                        .Else => {},
                    }
                    try self.collectBodyEffects(arm.body, state);
                }
                if (switch_stmt.else_body) |else_body| try self.collectBodyEffects(else_body, state);
            },
            .Try => |try_stmt| {
                try self.collectBodyEffects(try_stmt.try_body, state);
                if (try_stmt.catch_clause) |catch_clause| try self.collectBodyEffects(catch_clause.body, state);
            },
            .Log => |log_stmt| {
                state.has_log = true;
                for (log_stmt.args) |arg| try self.collectExprEffects(arg, state);
            },
            .Havoc => state.has_havoc = true,
            .Lock => state.has_lock = true,
            .Unlock => state.has_unlock = true,
            .Break, .Continue => {},
            .Assert => |assert_stmt| try self.collectExprEffects(assert_stmt.condition, state),
            .Assume => |assume_stmt| try self.collectExprEffects(assume_stmt.condition, state),
            .Assign => |assign| {
                try self.collectExprEffects(assign.value, state);
                try self.collectPatternTargetEffects(assign.target, assign.op, state);
            },
            .Expr => |expr_stmt| try self.collectExprEffects(expr_stmt.expr, state),
            .Block => |block| try self.collectBodyEffects(block.body, state),
            .LabeledBlock => |block| try self.collectBodyEffects(block.body, state),
            .Error => {},
        }
    }

    fn collectExprEffects(self: *TypeChecker, expr_id: ast.ExprId, state: *EffectCollectorState) anyerror!void {
        var expr_state = EffectCollectorState.init();
        switch (self.file.expression(expr_id).*) {
            .IntegerLiteral, .StringLiteral, .BoolLiteral, .AddressLiteral, .BytesLiteral, .Result, .Error => {},
            .Tuple => |tuple| for (tuple.elements) |element| try self.collectExprEffects(element, &expr_state),
            .ArrayLiteral => |array| for (array.elements) |element| try self.collectExprEffects(element, &expr_state),
            .StructLiteral => |struct_literal| for (struct_literal.fields) |field| try self.collectExprEffects(field.value, &expr_state),
            .Switch => |switch_expr| {
                try self.collectExprEffects(switch_expr.condition, &expr_state);
                for (switch_expr.arms) |arm| {
                    switch (arm.pattern) {
                        .Expr => |pattern_expr| try self.collectExprEffects(pattern_expr, &expr_state),
                        .Range => |range_pattern| {
                            try self.collectExprEffects(range_pattern.start, &expr_state);
                            try self.collectExprEffects(range_pattern.end, &expr_state);
                        },
                        .Else => {},
                    }
                    try self.collectExprEffects(arm.value, &expr_state);
                }
                if (switch_expr.else_expr) |else_expr| try self.collectExprEffects(else_expr, &expr_state);
            },
            .Comptime => {},
            .ErrorReturn => |error_return| for (error_return.args) |arg| try self.collectExprEffects(arg, &expr_state),
            .Name => {
                if (self.fieldSlotForBinding(self.resolution.expr_bindings[expr_id.index()])) |slot| {
                    try self.appendUniqueSlot(&expr_state.reads, slot);
                }
            },
            .Unary => |unary| try self.collectExprEffects(unary.operand, &expr_state),
            .Binary => |binary| {
                try self.collectExprEffects(binary.lhs, &expr_state);
                try self.collectExprEffects(binary.rhs, &expr_state);
            },
            .Call => |call| {
                try self.collectExprEffects(call.callee, &expr_state);
                for (call.args) |arg| try self.collectExprEffects(arg, &expr_state);
                if (self.calleeFunctionItem(call.callee)) |callee_id| {
                    switch (self.file.item(callee_id).*) {
                        .Function => |function| {
                            try self.ensureFunctionEffectSummary(callee_id, function);
                            try self.mergeEffect(&expr_state, self.item_effects[callee_id.index()]);
                        },
                        else => {},
                    }
                } else if (self.callableType(call.callee).kind() == .function) {
                    expr_state.has_external = true;
                }
            },
            .Builtin => |builtin| for (builtin.args) |arg| try self.collectExprEffects(arg, &expr_state),
            .Field => |field| try self.collectExprEffects(field.base, &expr_state),
            .Index => |index| {
                try self.collectExprEffects(index.index, &expr_state);
                if (self.lockSlotForExpr(expr_id)) |slot| {
                    try self.appendUniqueSlot(&expr_state.reads, slot);
                } else {
                    try self.collectExprEffects(index.base, &expr_state);
                }
            },
            .Group => |group| try self.collectExprEffects(group.expr, &expr_state),
            .Old, .Quantified => {},
        }
        self.expr_effects[expr_id.index()] = self.effectFromState(expr_state);
        try self.mergeEffect(state, self.expr_effects[expr_id.index()]);
    }

    fn collectPatternTargetEffects(self: *TypeChecker, pattern_id: ast.PatternId, op: ast.AssignmentOp, state: *EffectCollectorState) anyerror!void {
        switch (self.file.pattern(pattern_id).*) {
            .Name => {
                if (self.patternFieldSlot(pattern_id)) |slot| {
                    if (op != .assign) try self.appendUniqueSlot(&state.reads, slot);
                    try self.appendUniqueSlot(&state.writes, slot);
                }
            },
            .Field => |field| {
                try self.collectPatternTargetEffects(field.base, .add_assign, state);
            },
            .Index => |index| {
                try self.collectExprEffects(index.index, state);
                if (self.patternFieldSlot(pattern_id)) |slot| {
                    try self.appendUniqueSlot(&state.reads, slot);
                    try self.appendUniqueSlot(&state.writes, slot);
                } else {
                    try self.collectPatternExprReads(index.base, state);
                }
            },
            .StructDestructure => {},
            .Error => {},
        }
    }

    fn collectPatternExprReads(self: *TypeChecker, pattern_id: ast.PatternId, state: *EffectCollectorState) anyerror!void {
        switch (self.file.pattern(pattern_id).*) {
            .Name => {
                if (self.patternFieldSlot(pattern_id)) |slot| {
                    try self.appendUniqueSlot(&state.reads, slot);
                }
            },
            .Field => |field| try self.collectPatternExprReads(field.base, state),
            .Index => |index| {
                try self.collectExprEffects(index.index, state);
                if (self.patternFieldSlot(pattern_id)) |slot| {
                    try self.appendUniqueSlot(&state.reads, slot);
                } else {
                    try self.collectPatternExprReads(index.base, state);
                }
            },
            .StructDestructure => {},
            .Error => {},
        }
    }

    fn patternFieldSlot(self: *TypeChecker, pattern_id: ast.PatternId) ?EffectSlot {
        return switch (self.file.pattern(pattern_id).*) {
            .Name => |name| self.lookupNamedFieldSlot(name.name),
            .Field => |field| self.patternFieldSlot(field.base),
            .Index => |index| blk: {
                const base = self.patternFieldSlot(index.base) orelse break :blk null;
                break :blk self.slotWithIndexKey(base, index.index);
            },
            .StructDestructure => null,
            .Error => null,
        };
    }

    fn fieldSlotForBinding(self: *TypeChecker, binding: ?ResolvedBinding) ?EffectSlot {
        if (binding) |resolved| {
            return switch (resolved) {
                .item => |item_id| switch (self.file.item(item_id).*) {
                    .Field => |field| if (field.storage_class != .none) .{
                        .name = field.name,
                        .region = declarationRegion(field.storage_class),
                    } else null,
                    else => null,
                },
                .pattern => null,
            };
        }
        return null;
    }

    fn lookupNamedFieldSlot(self: *TypeChecker, name: []const u8) ?EffectSlot {
        if (self.current_contract) |contract_id| {
            const contract = self.file.item(contract_id).Contract;
            for (contract.members) |member_id| {
                switch (self.file.item(member_id).*) {
                    .Field => |field| {
                        if (field.storage_class == .none) continue;
                        if (std.mem.eql(u8, field.name, name)) return .{
                            .name = field.name,
                            .region = declarationRegion(field.storage_class),
                        };
                    },
                    else => {},
                }
            }
        }
        return self.fieldSlotForBinding(if (self.item_index.lookup(name)) |item_id| .{ .item = item_id } else null);
    }

    fn appendUniqueSlot(self: *TypeChecker, slots: *std.ArrayList(EffectSlot), slot: EffectSlot) !void {
        for (slots.items) |existing| {
            if (effectSlotEql(existing, slot)) return;
        }
        try slots.append(self.arena, slot);
    }

    fn slotWithIndexKey(self: *TypeChecker, base: EffectSlot, index_expr: ast.ExprId) ?EffectSlot {
        const segment = self.keySegmentForExpr(index_expr);
        const base_len: usize = if (base.key_path) |path| path.len else 0;
        const path = self.arena.alloc(KeySegment, base_len + 1) catch return null;
        if (base.key_path) |existing| @memcpy(path[0..existing.len], existing);
        path[base_len] = segment;
        var slot = base;
        slot.key_path = path;
        return slot;
    }

    fn keySegmentForExpr(self: *TypeChecker, expr_id: ast.ExprId) KeySegment {
        return switch (self.file.expression(expr_id).*) {
            .Group => |group| self.keySegmentForExpr(group.expr),
            .Name => blk: {
                if (self.resolution.expr_bindings[expr_id.index()]) |binding| {
                    switch (binding) {
                        .pattern => |pattern_id| if (self.parameterIndexForPattern(pattern_id)) |index| {
                            break :blk .{ .parameter = index };
                        },
                        .item => {},
                    }
                }
                break :blk .unknown;
            },
            .IntegerLiteral => |literal| .{ .constant = literal.text },
            .StringLiteral => |literal| .{ .constant = literal.text },
            .AddressLiteral => |literal| .{ .constant = literal.text },
            .BytesLiteral => |literal| .{ .constant = literal.text },
            .BoolLiteral => |literal| .{ .constant = if (literal.value) "true" else "false" },
            .Field => |field| blk: {
                const base = self.file.expression(field.base).*;
                if (base == .Name and std.mem.eql(u8, base.Name.name, "msg") and std.mem.eql(u8, field.name, "sender")) {
                    break :blk .self_ref;
                }
                if (base == .Name and std.mem.eql(u8, base.Name.name, "tx") and std.mem.eql(u8, field.name, "origin")) {
                    break :blk .self_ref;
                }
                break :blk .unknown;
            },
            else => .unknown,
        };
    }

    fn parameterIndexForPattern(self: *const TypeChecker, pattern_id: ast.PatternId) ?u32 {
        const function_item = self.current_function_item orelse return null;
        const function = switch (self.file.item(function_item).*) {
            .Function => |function| function,
            else => return null,
        };
        for (function.parameters, 0..) |parameter, index| {
            if (parameter.pattern.index() == pattern_id.index()) return @intCast(index);
        }
        return null;
    }

    fn effectSlotsMayAlias(self: *const TypeChecker, lhs: EffectSlot, rhs: EffectSlot) bool {
        _ = self;
        if (lhs.region != rhs.region) return false;
        if (!std.mem.eql(u8, lhs.name, rhs.name)) return false;
        return keyPathsMayAlias(lhs.key_path, rhs.key_path);
    }

    fn mergeEffect(self: *TypeChecker, state: *EffectCollectorState, effect: Effect) !void {
        switch (effect) {
            .pure => {},
            .external => state.has_external = true,
            .side_effects => |side_effects| {
                state.has_external = state.has_external or side_effects.has_external;
                state.has_log = state.has_log or side_effects.has_log;
                state.has_havoc = state.has_havoc or side_effects.has_havoc;
                state.has_lock = state.has_lock or side_effects.has_lock;
                state.has_unlock = state.has_unlock or side_effects.has_unlock;
            },
            .reads => |read_effect| {
                state.has_external = state.has_external or read_effect.has_external;
                state.has_log = state.has_log or read_effect.has_log;
                state.has_havoc = state.has_havoc or read_effect.has_havoc;
                state.has_lock = state.has_lock or read_effect.has_lock;
                state.has_unlock = state.has_unlock or read_effect.has_unlock;
                for (read_effect.slots) |slot| try self.appendUniqueSlot(&state.reads, slot);
            },
            .writes => |write_effect| {
                state.has_external = state.has_external or write_effect.has_external;
                state.has_log = state.has_log or write_effect.has_log;
                state.has_havoc = state.has_havoc or write_effect.has_havoc;
                state.has_lock = state.has_lock or write_effect.has_lock;
                state.has_unlock = state.has_unlock or write_effect.has_unlock;
                for (write_effect.slots) |slot| try self.appendUniqueSlot(&state.writes, slot);
            },
            .reads_writes => |read_write| {
                state.has_external = state.has_external or read_write.has_external;
                state.has_log = state.has_log or read_write.has_log;
                state.has_havoc = state.has_havoc or read_write.has_havoc;
                state.has_lock = state.has_lock or read_write.has_lock;
                state.has_unlock = state.has_unlock or read_write.has_unlock;
                for (read_write.reads) |slot| try self.appendUniqueSlot(&state.reads, slot);
                for (read_write.writes) |slot| try self.appendUniqueSlot(&state.writes, slot);
            },
        }
    }

    fn calleeFunctionItem(self: *const TypeChecker, expr_id: ast.ExprId) ?ast.ItemId {
        return switch (self.file.expression(expr_id).*) {
            .Name => if (self.resolution.expr_bindings[expr_id.index()]) |binding| switch (binding) {
                .item => |item_id| switch (self.file.item(item_id).*) {
                    .Function => item_id,
                    else => null,
                },
                .pattern => |pattern_id| if (self.initializerExprForPattern(pattern_id)) |init_expr|
                    self.calleeFunctionItem(init_expr)
                else
                    null,
            } else null,
            .Field => |field| blk: {
                const base_type = self.expr_types[field.base.index()];
                const contract_item_id = self.itemIdForType(base_type) orelse break :blk null;
                switch (self.file.item(contract_item_id).*) {
                    .Contract => |contract| {
                        for (contract.members) |member_id| {
                            switch (self.file.item(member_id).*) {
                                .Function => |function| if (std.mem.eql(u8, function.name, field.name)) break :blk member_id,
                                else => {},
                            }
                        }
                    },
                    else => {},
                }
                break :blk null;
            },
            .Group => |group| self.calleeFunctionItem(group.expr),
            else => null,
        };
    }

    fn collectFunctionDirectCallees(self: *TypeChecker, item_id: ast.ItemId, function: ast.FunctionItem, callees: *std.ArrayList(ast.ItemId)) anyerror!void {
        const previous_function_item = self.current_function_item;
        self.current_function_item = item_id;
        defer self.current_function_item = previous_function_item;
        try self.collectBodyDirectCallees(function.body, callees);
    }

    fn collectBodyDirectCallees(self: *TypeChecker, body_id: ast.BodyId, callees: *std.ArrayList(ast.ItemId)) anyerror!void {
        const body = self.file.body(body_id).*;
        for (body.statements) |statement_id| try self.collectStmtDirectCallees(statement_id, callees);
    }

    fn collectStmtDirectCallees(self: *TypeChecker, statement_id: ast.StmtId, callees: *std.ArrayList(ast.ItemId)) anyerror!void {
        switch (self.file.statement(statement_id).*) {
            .VariableDecl => |decl| if (decl.value) |expr_id| try self.collectExprDirectCallees(expr_id, callees),
            .Return => |ret| if (ret.value) |expr_id| try self.collectExprDirectCallees(expr_id, callees),
            .If => |if_stmt| {
                try self.collectExprDirectCallees(if_stmt.condition, callees);
                try self.collectBodyDirectCallees(if_stmt.then_body, callees);
                if (if_stmt.else_body) |else_body| try self.collectBodyDirectCallees(else_body, callees);
            },
            .While => |while_stmt| {
                try self.collectExprDirectCallees(while_stmt.condition, callees);
                for (while_stmt.invariants) |expr_id| try self.collectExprDirectCallees(expr_id, callees);
                try self.collectBodyDirectCallees(while_stmt.body, callees);
            },
            .For => |for_stmt| {
                try self.collectExprDirectCallees(for_stmt.iterable, callees);
                for (for_stmt.invariants) |expr_id| try self.collectExprDirectCallees(expr_id, callees);
                try self.collectBodyDirectCallees(for_stmt.body, callees);
            },
            .Switch => |switch_stmt| {
                try self.collectExprDirectCallees(switch_stmt.condition, callees);
                for (switch_stmt.arms) |arm| {
                    switch (arm.pattern) {
                        .Expr => |expr_id| try self.collectExprDirectCallees(expr_id, callees),
                        .Range => |range_pattern| {
                            try self.collectExprDirectCallees(range_pattern.start, callees);
                            try self.collectExprDirectCallees(range_pattern.end, callees);
                        },
                        .Else => {},
                    }
                    try self.collectBodyDirectCallees(arm.body, callees);
                }
                if (switch_stmt.else_body) |else_body| try self.collectBodyDirectCallees(else_body, callees);
            },
            .Try => |try_stmt| {
                try self.collectBodyDirectCallees(try_stmt.try_body, callees);
                if (try_stmt.catch_clause) |catch_clause| try self.collectBodyDirectCallees(catch_clause.body, callees);
            },
            .Log => |log_stmt| for (log_stmt.args) |arg| try self.collectExprDirectCallees(arg, callees),
            .Lock => |lock_stmt| try self.collectExprDirectCallees(lock_stmt.path, callees),
            .Unlock => |unlock_stmt| try self.collectExprDirectCallees(unlock_stmt.path, callees),
            .Assert => |assert_stmt| try self.collectExprDirectCallees(assert_stmt.condition, callees),
            .Assume => |assume_stmt| try self.collectExprDirectCallees(assume_stmt.condition, callees),
            .Assign => |assign| {
                try self.collectExprDirectCallees(assign.value, callees);
                try self.collectPatternDirectCallees(assign.target, callees);
            },
            .Expr => |expr_stmt| try self.collectExprDirectCallees(expr_stmt.expr, callees),
            .Block => |block| try self.collectBodyDirectCallees(block.body, callees),
            .LabeledBlock => |block| try self.collectBodyDirectCallees(block.body, callees),
            .Havoc, .Break, .Continue, .Error => {},
        }
    }

    fn collectExprDirectCallees(self: *TypeChecker, expr_id: ast.ExprId, callees: *std.ArrayList(ast.ItemId)) anyerror!void {
        switch (self.file.expression(expr_id).*) {
            .Tuple => |tuple| for (tuple.elements) |element| try self.collectExprDirectCallees(element, callees),
            .ArrayLiteral => |array| for (array.elements) |element| try self.collectExprDirectCallees(element, callees),
            .StructLiteral => |struct_literal| for (struct_literal.fields) |field| try self.collectExprDirectCallees(field.value, callees),
            .Switch => |switch_expr| {
                try self.collectExprDirectCallees(switch_expr.condition, callees);
                for (switch_expr.arms) |arm| {
                    switch (arm.pattern) {
                        .Expr => |pattern_expr| try self.collectExprDirectCallees(pattern_expr, callees),
                        .Range => |range_pattern| {
                            try self.collectExprDirectCallees(range_pattern.start, callees);
                            try self.collectExprDirectCallees(range_pattern.end, callees);
                        },
                        .Else => {},
                    }
                    try self.collectExprDirectCallees(arm.value, callees);
                }
                if (switch_expr.else_expr) |else_expr| try self.collectExprDirectCallees(else_expr, callees);
            },
            .ErrorReturn => |error_return| for (error_return.args) |arg| try self.collectExprDirectCallees(arg, callees),
            .Unary => |unary| try self.collectExprDirectCallees(unary.operand, callees),
            .Binary => |binary| {
                try self.collectExprDirectCallees(binary.lhs, callees);
                try self.collectExprDirectCallees(binary.rhs, callees);
            },
            .Call => |call| {
                try self.collectExprDirectCallees(call.callee, callees);
                for (call.args) |arg| try self.collectExprDirectCallees(arg, callees);
                if (self.calleeFunctionItem(call.callee)) |callee_id| try self.appendUniqueItemId(callees, callee_id);
            },
            .Builtin => |builtin| for (builtin.args) |arg| try self.collectExprDirectCallees(arg, callees),
            .Field => |field| try self.collectExprDirectCallees(field.base, callees),
            .Index => |index| {
                try self.collectExprDirectCallees(index.base, callees);
                try self.collectExprDirectCallees(index.index, callees);
            },
            .Group => |group| try self.collectExprDirectCallees(group.expr, callees),
            .Comptime, .Old, .Quantified, .Name, .IntegerLiteral, .StringLiteral, .BoolLiteral, .AddressLiteral, .BytesLiteral, .Result, .Error => {},
        }
    }

    fn collectPatternDirectCallees(self: *TypeChecker, pattern_id: ast.PatternId, callees: *std.ArrayList(ast.ItemId)) anyerror!void {
        switch (self.file.pattern(pattern_id).*) {
            .Field => |field| try self.collectPatternDirectCallees(field.base, callees),
            .Index => |index| {
                try self.collectPatternDirectCallees(index.base, callees);
                try self.collectExprDirectCallees(index.index, callees);
            },
            .Name, .StructDestructure, .Error => {},
        }
    }

    fn appendUniqueItemId(self: *TypeChecker, items: *std.ArrayList(ast.ItemId), item_id: ast.ItemId) !void {
        for (items.items) |existing| {
            if (existing.index() == item_id.index()) return;
        }
        try items.append(self.arena, item_id);
    }

    fn initializerExprForPattern(self: *const TypeChecker, pattern_id: ast.PatternId) ?ast.ExprId {
        for (self.file.statements) |statement| {
            switch (statement) {
                .VariableDecl => |decl| {
                    if (decl.pattern != pattern_id) continue;
                    return decl.value;
                },
                else => {},
            }
        }
        return null;
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
        if (base_type.kind() == .struct_) {
            if (self.instantiatedStructByName(base_type.struct_.name)) |instantiated| {
                for (instantiated.fields) |field| {
                    if (std.mem.eql(u8, field.name, field_name)) return field.ty;
                }
                return .{ .unknown = {} };
            }
        }
        if (base_type.kind() == .enum_) {
            if (self.instantiatedEnumByName(base_type.enum_.name)) |instantiated| {
                const enum_item = self.file.item(instantiated.template_item_id).Enum;
                for (enum_item.variants) |variant| {
                    if (std.mem.eql(u8, variant.name, field_name)) return base_type;
                }
                return .{ .unknown = {} };
            }
        }
        if (base_type.kind() == .bitfield) {
            if (self.instantiatedBitfieldByName(base_type.bitfield.name)) |instantiated| {
                for (instantiated.fields) |field| {
                    if (std.mem.eql(u8, field.name, field_name)) return field.ty;
                }
                return .{ .unknown = {} };
            }
        }
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
                if (self.instantiatedEnumByName(enum_item.name)) |_| break :blk base_type;
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
        if (self.instantiatedStructByName(name) != null) return null;
        if (self.instantiatedEnumByName(name) != null) return null;
        if (self.instantiatedBitfieldByName(name) != null) return null;
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
