const std = @import("std");
const ast = @import("../ast/mod.zig");
const const_bridge = @import("../comptime/compiler_const_bridge.zig");
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
const Provenance = model.Provenance;
const Effect = model.Effect;
const EffectSlot = model.EffectSlot;
const KeySegment = model.KeySegment;
const appendModelTypeMangleName = model.appendTypeMangleName;
const InstantiatedStruct = model.InstantiatedStruct;
const InstantiatedStructField = model.InstantiatedStructField;
const InstantiatedEnum = model.InstantiatedEnum;
const InstantiatedBitfield = model.InstantiatedBitfield;
const InstantiatedBitfieldField = model.InstantiatedBitfieldField;
const TraitMethodSignature = model.TraitMethodSignature;
const TraitInterface = model.TraitInterface;
const ImplInterface = model.ImplInterface;
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
const typesAssignable = descriptors.typesAssignable;

fn declarationRegion(storage_class: ast.StorageClass) Region {
    return switch (storage_class) {
        .none => .memory,
        .storage => .storage,
        .memory => .memory,
        .tstore => .transient,
    };
}

fn locatedValue(expr_type: Type, expr_region: Region, provenance: Provenance) LocatedType {
    return LocatedType.withRegionAndProvenance(expr_type, expr_region, provenance);
}

fn typesFlowCompatible(expected_type: Type, actual_type: Type) bool {
    if (typesAssignable(expected_type, actual_type)) return true;

    if (expected_type.kind() == .refinement and actual_type.kind() == .refinement) {
        return typesAssignable(expected_type.refinement.base_type.*, actual_type.refinement.base_type.*) and
            typesAssignable(actual_type.refinement.base_type.*, expected_type.refinement.base_type.*);
    }

    return false;
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
        .trait_interfaces = &.{},
        .impl_interfaces = &.{},
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
    const catch_error_tag_patterns = try arena.alloc(bool, file.patterns.len);
    @memset(item_types, .{ .unknown = {} });
    @memset(item_regions, .none);
    @memset(item_effects, .pure);
    @memset(pattern_types, LocatedType.unlocated(.{ .unknown = {} }));
    @memset(expr_types, .{ .unknown = {} });
    @memset(expr_effects, .pure);
    @memset(body_types, .{ .void = {} });
    @memset(effect_states, .unvisited);
    @memset(catch_error_tag_patterns, false);

    for (file.items, 0..) |item, index| {
        item_types[index] = try inferItemType(arena, file, item_index, item);
        switch (item) {
            .Function => |function| {
                for (function.parameters) |parameter| {
                    pattern_types[parameter.pattern.index()] = LocatedType.withRegionAndProvenance(.{ .unknown = {} }, .memory, .calldata);
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
                pattern_types[decl.pattern.index()] = LocatedType.withRegion(.{ .unknown = {} }, declarationRegion(decl.storage_class));
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
        .trait_interfaces = .{},
        .impl_interfaces = .{},
        .catch_error_tag_patterns = catch_error_tag_patterns,
        .diagnostics = &result.diagnostics,
    };

    for (file.items, 0..) |item, index| {
        switch (item) {
            .Function => |function| {
                for (function.parameters) |parameter| {
                    const resolved_type = if (typechecker.parameterTypeForFunctionItem(ast.ItemId.fromIndex(index), parameter)) |ty|
                        ty
                    else
                        Type{ .unknown = {} };
                    pattern_types[parameter.pattern.index()] = LocatedType.withRegionAndProvenance(resolved_type, .memory, .calldata);
                }
            },
            else => {},
        }
    }

    try result.diagnostics.appendList(&const_eval.diagnostics);

    for (file.items, 0..) |item, index| {
        switch (item) {
            .Function => |function| {
                const function_bindings = try typechecker.genericBindingsForFunctionTemplate(function);
                const resolved_params = try arena.alloc(Type, function.parameters.len);
                for (function.parameters, 0..) |parameter, param_index| {
                    const resolved_type = if (typechecker.parameterTypeForFunctionItem(ast.ItemId.fromIndex(index), parameter)) |ty|
                        ty
                    else
                        try typechecker.resolveTypeExprWithBindings(parameter.type_expr, function_bindings);
                    resolved_params[param_index] = resolved_type;
                    pattern_types[parameter.pattern.index()] = LocatedType.withRegionAndProvenance(resolved_type, .memory, .calldata);
                }
                const resolved_returns = if (function.return_type) |type_expr| blk_returns: {
                    const slice = try arena.alloc(Type, 1);
                    slice[0] = try typechecker.resolveTypeExprWithBindings(type_expr, function_bindings);
                    break :blk_returns slice;
                } else &.{};
                item_types[index] = .{ .function = .{
                    .name = function.name,
                    .param_types = resolved_params,
                    .return_types = resolved_returns,
                } };
                body_types[function.body.index()] = if (function.return_type) |type_expr|
                    try typechecker.resolveTypeExprWithBindings(type_expr, function_bindings)
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
                pattern_types[decl.pattern.index()] = LocatedType.withRegion(try typechecker.resolveTypeExpr(type_expr), declarationRegion(decl.storage_class));
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
    result.trait_interfaces = try typechecker.trait_interfaces.toOwnedSlice(arena);
    result.impl_interfaces = try typechecker.impl_interfaces.toOwnedSlice(arena);
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
    trait_interfaces: std.ArrayList(TraitInterface),
    impl_interfaces: std.ArrayList(ImplInterface),
    catch_error_tag_patterns: []bool = &.{},
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
            .LogDecl => |log_decl| {
                var indexed_count: usize = 0;
                for (log_decl.fields) |field| {
                    const field_type = try self.resolveTypeExpr(field.type_expr);
                    if (!field.indexed) continue;
                    indexed_count += 1;
                    if (indexed_count > 3) {
                        try self.emitRangeError(field.range, "log declarations support at most 3 indexed fields", .{});
                    }
                    if (!self.logIndexedFieldTypeSupported(field_type)) {
                        try self.emitRangeError(field.range, "indexed log field '{s}' has unsupported type '{s}'", .{
                            field.name,
                            typeDisplayName(field_type),
                        });
                    }
                }
            },
            .Function => |function| {
                try self.validateConstructorFunction(function);
                const previous_return_type = self.current_return_type;
                const previous_function_item = self.current_function_item;
                const function_bindings = try self.genericBindingsForFunctionTemplate(function);
                self.current_return_type = if (function.return_type) |type_expr|
                    try self.resolveTypeExprWithBindings(type_expr, function_bindings)
                else
                    .{ .void = {} };
                self.current_function_item = item_id;
                defer self.current_function_item = previous_function_item;
                defer self.current_return_type = previous_return_type;
                for (function.clauses) |clause| try self.visitExpr(clause.expr);
                try self.visitBody(function.body);
                try self.ensureFunctionEffectSummary(item_id, function);
                var locked_slots: std.ArrayList(EffectSlot) = .{};
                try self.validateBodyLocks(function.body, &locked_slots);
                var external_call_state = try self.initExternalCallValidationState();
                defer external_call_state.deinit(self.arena);
                try self.validateBodyExternalCalls(function.body, &external_call_state);
            },
            .Trait => |trait_item| {
                if (trait_item.is_extern and trait_item.ghost_block != null) {
                    try self.emitRangeError(trait_item.range, "extern trait '{s}' cannot declare a ghost block", .{
                        trait_item.name,
                    });
                }
                for (trait_item.methods) |method| {
                    if (trait_item.is_extern and method.extern_call_kind == .none) {
                        try self.emitRangeError(method.range, "extern trait method '{s}' must use 'call fn' or 'staticcall fn'", .{
                            method.name,
                        });
                    }
                    for (method.errors) |error_name| {
                        const error_item_id = self.item_index.lookup(error_name);
                        if (error_item_id == null or self.file.item(error_item_id.?).* != .ErrorDecl) {
                            try self.emitRangeError(method.range, "extern trait method '{s}' declares unknown error '{s}'", .{
                                method.name,
                                error_name,
                            });
                        }
                    }
                    for (method.parameters) |parameter| {
                        _ = try self.resolveTypeExpr(parameter.type_expr);
                    }
                    if (method.return_type) |type_expr| {
                        _ = try self.resolveTypeExpr(type_expr);
                    }
                    for (method.clauses) |clause| try self.visitExpr(clause.expr);
                }
                try self.recordTraitInterface(item_id, trait_item);
            },
            .Impl => |impl_item| {
                try self.checkImplConformance(impl_item);

                const previous_contract = self.current_contract;
                if (self.item_index.lookup(impl_item.target_name)) |target_item_id| {
                    if (self.file.item(target_item_id).* == .Contract) {
                        self.current_contract = target_item_id;
                    }
                }
                defer self.current_contract = previous_contract;

                if (self.item_index.lookup(impl_item.trait_name)) |trait_item_id| {
                    if (self.file.item(trait_item_id).* == .Trait) {
                        const trait_item = self.file.item(trait_item_id).Trait;
                        if (trait_item.ghost_block) |ghost_id| {
                            try self.visitItem(ghost_id);
                        }
                    }
                }

                for (impl_item.methods) |method_id| {
                    try self.visitItem(method_id);
                }
                try self.recordImplInterface(item_id, impl_item);
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
                    } else if (actual_type.kind() != .unknown and expected_type.kind() != .unknown) {
                        const expected = LocatedType.withRegion(expected_type, self.item_regions[item_id.index()]);
                        const actual_located = self.exprLocatedType(expr_id);
                        const actual = locatedValue(actual_type, actual_located.region, actual_located.provenance);
                        if (!typesFlowCompatible(expected_type, actual_type)) {
                            try self.emitRangeError(field.range, "field '{s}' expects type '{s}', found '{s}'", .{
                                field.name,
                                typeDisplayName(expected_type),
                                typeDisplayName(actual_type),
                            });
                        } else if (!region_rules.regionAssignable(actual.region, expected.region)) {
                            try self.emitRangeError(field.range, "field '{s}' expects region '{s}', found '{s}'", .{
                                field.name,
                                region_rules.regionDisplayName(expected.region),
                                region_rules.regionDisplayName(actual.region),
                            });
                        }
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
            .ErrorDecl => |error_decl| {
                for (error_decl.parameters) |parameter| {
                    self.pattern_types[parameter.pattern.index()] = LocatedType.unlocated(try self.resolveTypeExpr(parameter.type_expr));
                }
            },
            .GhostBlock => |ghost_block| try self.visitBody(ghost_block.body),
            .TypeAlias => {},
            else => {},
        }
    }

    fn validateConstructorFunction(self: *TypeChecker, function: ast.FunctionItem) anyerror!void {
        if (!std.mem.eql(u8, function.name, "init")) return;

        if (self.current_contract == null) {
            try self.emitRangeError(function.range, "init() is only supported as a contract constructor", .{});
        }
        if (function.visibility != .public) {
            try self.emitRangeError(function.range, "constructor init() must be public", .{});
        }
        if (self.functionHasBareSelf(function)) {
            try self.emitRangeError(function.range, "constructor init() must not declare a self parameter", .{});
        }
        if (function.return_type != null) {
            try self.emitRangeError(function.range, "constructor init() must not return values", .{});
        }
    }

    fn checkImplConformance(self: *TypeChecker, impl_item: anytype) anyerror!void {
        const trait_item_id = self.item_index.lookup(impl_item.trait_name) orelse {
            try self.emitRangeError(impl_item.range, "impl references unknown trait '{s}'", .{
                impl_item.trait_name,
            });
            return;
        };
        const trait_item = switch (self.file.item(trait_item_id).*) {
            .Trait => |trait_item| trait_item,
            else => {
                try self.emitRangeError(impl_item.range, "'{s}' is not a trait", .{
                    impl_item.trait_name,
                });
                return;
            },
        };
        if (trait_item.is_extern) {
            try self.emitRangeError(impl_item.range, "extern trait '{s}' cannot be implemented with an impl block", .{
                trait_item.name,
            });
            return;
        }

        const target_item_id = self.item_index.lookup(impl_item.target_name) orelse {
            try self.emitRangeError(impl_item.range, "impl references unknown target '{s}'", .{
                impl_item.target_name,
            });
            return;
        };
        switch (self.file.item(target_item_id).*) {
            .Contract, .Struct, .Bitfield, .Enum => {},
            else => {
                try self.emitRangeError(impl_item.range, "impl target '{s}' must name a contract, struct, bitfield, or enum", .{
                    impl_item.target_name,
                });
                return;
            },
        }

        var impl_count: usize = 0;
        for (self.item_index.impl_entries) |entry| {
            if (std.mem.eql(u8, entry.trait_name, impl_item.trait_name) and
                std.mem.eql(u8, entry.target_name, impl_item.target_name))
            {
                impl_count += 1;
            }
        }
        if (impl_count > 1) {
            try self.emitRangeError(impl_item.range, "duplicate impl for trait '{s}' and type '{s}'", .{
                impl_item.trait_name,
                impl_item.target_name,
            });
        }

        for (trait_item.methods) |trait_method| {
            const impl_method = self.findImplMethodByName(impl_item, trait_method.name);
            if (impl_method == null) {
                try self.emitRangeError(impl_item.range, "impl missing method '{s}' required by trait '{s}'", .{
                    trait_method.name,
                    trait_item.name,
                });
                continue;
            }
            try self.checkImplMethodSignature(trait_item.name, trait_method, impl_method.?);
        }

        for (impl_item.methods) |method_id| {
            const method = self.file.item(method_id).Function;
            if (!self.traitHasMethodNamed(trait_item, method.name)) {
                try self.emitRangeError(method.range, "impl contains method '{s}' which is not part of trait '{s}'", .{
                    method.name,
                    trait_item.name,
                });
            }
        }
    }

    fn traitHasMethodNamed(self: *TypeChecker, trait_item: anytype, name: []const u8) bool {
        _ = self;
        for (trait_item.methods) |method| {
            if (std.mem.eql(u8, method.name, name)) return true;
        }
        return false;
    }

    fn findImplMethodByName(self: *TypeChecker, impl_item: anytype, name: []const u8) ?ast.FunctionItem {
        for (impl_item.methods) |method_id| {
            const item = self.file.item(method_id).*;
            if (item != .Function) continue;
            if (std.mem.eql(u8, item.Function.name, name)) return item.Function;
        }
        return null;
    }

    fn checkImplMethodSignature(self: *TypeChecker, trait_name: []const u8, trait_method: anytype, impl_method: ast.FunctionItem) anyerror!void {
        if (impl_method.is_comptime != trait_method.is_comptime) {
            try self.emitRangeError(impl_method.range, "method '{s}' has wrong signature for trait '{s}': expected {s}comptime fn", .{
                impl_method.name,
                trait_name,
                if (trait_method.is_comptime) "" else "non-",
            });
            return;
        }

        const impl_receiver_kind: ast.ReceiverKind = if (self.functionHasBareSelf(impl_method)) .value_self else .none;
        const expected_impl_receiver_kind: ast.ReceiverKind = switch (trait_method.receiver_kind) {
            .extern_self => .value_self,
            else => trait_method.receiver_kind,
        };
        if (impl_receiver_kind != expected_impl_receiver_kind) {
            try self.emitRangeError(impl_method.range, "method '{s}' has wrong signature: expected {s}self parameter", .{
                impl_method.name,
                if (expected_impl_receiver_kind != .none) "" else "no ",
            });
            return;
        }

        const impl_offset: usize = if (impl_receiver_kind != .none) 1 else 0;
        const impl_runtime_params = impl_method.parameters.len -| impl_offset;
        if (impl_runtime_params != trait_method.parameters.len) {
            try self.emitRangeError(impl_method.range, "method '{s}' has wrong signature for trait '{s}': expected {d} parameters, found {d}", .{
                impl_method.name,
                trait_name,
                trait_method.parameters.len,
                impl_runtime_params,
            });
            return;
        }

        for (trait_method.parameters, 0..) |trait_param, index| {
            const impl_param = impl_method.parameters[index + impl_offset];
            const trait_type = try self.resolveTypeExpr(trait_param.type_expr);
            const impl_type = try self.resolveTypeExpr(impl_param.type_expr);
            if (trait_type.kind() == .unknown or impl_type.kind() == .unknown) continue;
            if (!typesAssignable(trait_type, impl_type) or !typesAssignable(impl_type, trait_type)) {
                try self.emitRangeError(impl_method.range, "method '{s}' has wrong signature for trait '{s}': parameter {d} expects '{s}', found '{s}'", .{
                    impl_method.name,
                    trait_name,
                    index,
                    typeDisplayName(trait_type),
                    typeDisplayName(impl_type),
                });
                return;
            }
        }

        const trait_return = if (trait_method.return_type) |type_expr|
            try self.resolveTypeExpr(type_expr)
        else
            Type{ .void = {} };
        const impl_return = if (impl_method.return_type) |type_expr|
            try self.resolveTypeExpr(type_expr)
        else
            Type{ .void = {} };
        if (trait_return.kind() != .unknown and impl_return.kind() != .unknown and
            (!typesAssignable(trait_return, impl_return) or !typesAssignable(impl_return, trait_return)))
        {
            try self.emitRangeError(impl_method.range, "method '{s}' has wrong signature for trait '{s}': expected return '{s}', found '{s}'", .{
                impl_method.name,
                trait_name,
                typeDisplayName(trait_return),
                typeDisplayName(impl_return),
            });
        }
    }

    fn functionHasBareSelf(self: *const TypeChecker, function: ast.FunctionItem) bool {
        for (function.parameters) |parameter| {
            if (parameter.is_comptime) continue;
            const pattern = self.file.pattern(parameter.pattern).*;
            if (pattern != .Name) return false;
            return std.mem.eql(u8, pattern.Name.name, "self");
        }
        return false;
    }

    fn parameterTypeForFunctionItem(self: *TypeChecker, function_item_id: ast.ItemId, parameter: ast.Parameter) ?Type {
        if (!self.parameterIsBareSelf(parameter)) return null;
        const impl_item = self.enclosingImplForMethod(function_item_id) orelse return null;
        const target_item_id = self.item_index.lookup(impl_item.target_name) orelse return null;
        return self.nominalItemSelfType(target_item_id);
    }

    fn genericBindingsForFunctionTemplate(self: *TypeChecker, function: ast.FunctionItem) ![]const GenericTypeBinding {
        if (!function.is_generic) return &.{};

        var count: usize = 0;
        for (function.parameters) |parameter| {
            if (!parameter.is_comptime) continue;
            if (!self.isGenericTypeParameter(parameter) and !self.comptimeIntegerParameter(parameter)) continue;
            count += 1;
        }

        const bindings = try self.arena.alloc(GenericTypeBinding, count);
        var index: usize = 0;
        for (function.parameters) |parameter| {
            if (!parameter.is_comptime) continue;
            const name = self.patternName(parameter.pattern) orelse continue;
            if (self.isGenericTypeParameter(parameter)) {
                bindings[index] = .{
                    .name = name,
                    .value = .{ .ty = .{ .named = .{ .name = name } } },
                };
                index += 1;
                continue;
            }
            if (self.comptimeIntegerParameter(parameter)) {
                bindings[index] = .{
                    .name = name,
                    .value = .{ .integer = name },
                };
                index += 1;
            }
        }
        return bindings[0..index];
    }

    fn currentFunctionGenericBindings(self: *TypeChecker) ![]const GenericTypeBinding {
        const function_item_id = self.current_function_item orelse return &.{};
        const item = self.file.item(function_item_id).*;
        if (item != .Function) return &.{};
        return self.genericBindingsForFunctionTemplate(item.Function);
    }

    fn resolveTypeExprInCurrentContext(self: *TypeChecker, type_expr: ast.TypeExprId) !Type {
        return self.resolveTypeExprWithBindings(type_expr, try self.currentFunctionGenericBindings());
    }

    fn parameterIsBareSelf(self: *TypeChecker, parameter: ast.Parameter) bool {
        const pattern = self.file.pattern(parameter.pattern).*;
        if (pattern != .Name) return false;
        return std.mem.eql(u8, pattern.Name.name, "self");
    }

    fn enclosingImplForMethod(self: *const TypeChecker, method_item_id: ast.ItemId) ?ast.ImplItem {
        for (self.file.items) |item| {
            if (item != .Impl) continue;
            for (item.Impl.methods) |candidate_id| {
                if (candidate_id.index() == method_item_id.index()) return item.Impl;
            }
        }
        return null;
    }

    fn nominalItemSelfType(self: *const TypeChecker, item_id: ast.ItemId) ?Type {
        return switch (self.file.item(item_id).*) {
            .Contract => |contract| Type{ .contract = .{ .name = contract.name } },
            .Struct => |struct_item| Type{ .struct_ = .{ .name = struct_item.name } },
            .Enum => |enum_item| Type{ .enum_ = .{ .name = enum_item.name } },
            .Bitfield => |bitfield| Type{ .bitfield = .{ .name = bitfield.name } },
            else => null,
        };
    }

    fn buildTraitMethodSignature(self: *TypeChecker, trait_method: anytype) anyerror!TraitMethodSignature {
        const param_types = try self.arena.alloc(Type, trait_method.parameters.len);
        for (trait_method.parameters, 0..) |parameter, index| {
            param_types[index] = try self.resolveTypeExpr(parameter.type_expr);
        }
        const return_type = if (trait_method.return_type) |type_expr|
            try self.resolveTypeExpr(type_expr)
        else
            Type{ .void = {} };
        return .{
            .name = trait_method.name,
            .receiver_kind = trait_method.receiver_kind,
            .is_comptime = trait_method.is_comptime,
            .extern_call_kind = trait_method.extern_call_kind,
            .errors = trait_method.errors,
            .param_types = param_types,
            .return_type = return_type,
        };
    }

    fn buildFunctionMethodSignature(self: *TypeChecker, function: ast.FunctionItem) anyerror!TraitMethodSignature {
        const has_self = self.functionHasBareSelf(function);
        const offset: usize = if (has_self) 1 else 0;
        const runtime_param_count = function.parameters.len -| offset;
        const param_types = try self.arena.alloc(Type, runtime_param_count);
        for (function.parameters[offset..], 0..) |parameter, index| {
            param_types[index] = try self.resolveTypeExpr(parameter.type_expr);
        }
        const return_type = if (function.return_type) |type_expr|
            try self.resolveTypeExpr(type_expr)
        else
            Type{ .void = {} };
        return .{
            .name = function.name,
            .receiver_kind = if (has_self) .value_self else .none,
            .is_comptime = function.is_comptime,
            .extern_call_kind = .none,
            .errors = &.{},
            .param_types = param_types,
            .return_type = return_type,
        };
    }

    fn recordTraitInterface(self: *TypeChecker, trait_item_id: ast.ItemId, trait_item: anytype) anyerror!void {
        if (self.findRecordedTraitInterfaceIndex(trait_item.name) != null) return;

        const methods = try self.arena.alloc(TraitMethodSignature, trait_item.methods.len);
        for (trait_item.methods, 0..) |method, index| {
            methods[index] = try self.buildTraitMethodSignature(method);
        }
        try self.trait_interfaces.append(self.arena, .{
            .trait_item_id = trait_item_id,
            .name = trait_item.name,
            .is_extern = trait_item.is_extern,
            .methods = methods,
        });
    }

    fn recordImplInterface(self: *TypeChecker, impl_item_id: ast.ItemId, impl_item: anytype) anyerror!void {
        if (self.findRecordedImplInterfaceIndex(impl_item.trait_name, impl_item.target_name) != null) return;

        const trait_item_id = self.item_index.lookup(impl_item.trait_name) orelse return;
        const target_item_id = self.item_index.lookup(impl_item.target_name) orelse return;
        const methods = try self.arena.alloc(TraitMethodSignature, impl_item.methods.len);
        for (impl_item.methods, 0..) |method_id, index| {
            const method = self.file.item(method_id).Function;
            methods[index] = try self.buildFunctionMethodSignature(method);
        }
        try self.impl_interfaces.append(self.arena, .{
            .impl_item_id = impl_item_id,
            .trait_item_id = trait_item_id,
            .target_item_id = target_item_id,
            .trait_name = impl_item.trait_name,
            .target_name = impl_item.target_name,
            .methods = methods,
        });
    }

    fn findRecordedTraitInterfaceIndex(self: *TypeChecker, name: []const u8) ?usize {
        for (self.trait_interfaces.items, 0..) |trait_interface, index| {
            if (std.mem.eql(u8, trait_interface.name, name)) return index;
        }
        return null;
    }

    fn findRecordedImplInterfaceIndex(self: *TypeChecker, trait_name: []const u8, target_name: []const u8) ?usize {
        for (self.impl_interfaces.items, 0..) |impl_interface, index| {
            if (std.mem.eql(u8, impl_interface.trait_name, trait_name) and
                std.mem.eql(u8, impl_interface.target_name, target_name))
            {
                return index;
            }
        }
        return null;
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
                if (decl.storage_class == .storage) {
                    try self.emitRangeError(decl.range, "storage declarations are only allowed at contract scope", .{});
                }
                if (decl.type_expr) |type_expr| {
                    if (self.pattern_types[decl.pattern.index()].type.kind() == .unknown) {
                        self.pattern_types[decl.pattern.index()] = LocatedType.withRegionAndProvenance(
                            try self.resolveTypeExpr(type_expr),
                            declarationRegion(decl.storage_class),
                            .local,
                        );
                    }
                }
                if (decl.value) |expr_id| {
                    try self.visitExpr(expr_id);
                    if (decl.type_expr != null) {
                        try self.contextualizeLiteral(expr_id, self.pattern_types[decl.pattern.index()].type);
                    }
                    const actual_type = self.expr_types[expr_id.index()];
                    const actual_located = self.exprLocatedType(expr_id);
                    if (decl.type_expr == null) {
                        self.pattern_types[decl.pattern.index()] = LocatedType.withRegionAndProvenance(
                            actual_type,
                            declarationRegion(decl.storage_class),
                            actual_located.provenance,
                        );
                    } else {
                        const expected = self.pattern_types[decl.pattern.index()];
                        const expected_type = expected.type;
                        if (try self.emitIntegerOverflowIfNeeded(decl.range, expr_id, expected_type)) {
                            // Keep lowering/recovery moving after reporting the overflow.
                        } else if (actual_type.kind() != .unknown and expected_type.kind() != .unknown) {
                            const actual = locatedValue(actual_type, actual_located.region, actual_located.provenance);
                            if (!typesFlowCompatible(expected_type, actual_type)) {
                                try self.emitRangeError(decl.range, "declaration expects type '{s}', found '{s}'", .{
                                    typeDisplayName(expected_type),
                                    typeDisplayName(actual_type),
                                });
                            } else if (!region_rules.regionAssignable(actual.region, expected.region)) {
                                try self.emitRangeError(decl.range, "declaration expects region '{s}', found '{s}'", .{
                                    region_rules.regionDisplayName(expected.region),
                                    region_rules.regionDisplayName(actual.region),
                                });
                            }
                        }
                    }
                }
            },
            .Return => |ret| if (ret.value) |expr_id| {
                try self.visitExpr(expr_id);
                if (self.current_return_type) |expected_type| {
                    try self.contextualizeLiteral(expr_id, expected_type);
                    const actual_type = self.expr_types[expr_id.index()];
                    if (try self.emitIntegerOverflowIfNeeded(ret.range, expr_id, expected_type)) {
                        // Keep lowering/recovery moving after reporting the overflow.
                    } else if (actual_type.kind() != .unknown and expected_type.kind() != .unknown) {
                        if (!typesFlowCompatible(expected_type, actual_type)) {
                            try self.emitRangeError(ret.range, "return expects type '{s}', found '{s}'", .{
                                typeDisplayName(expected_type),
                                typeDisplayName(actual_type),
                            });
                        }
                        // No region check — return values are loaded to stack,
                        // so all source regions are valid regardless of return type region.
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
                if (for_stmt.range_end) |end_expr| try self.visitExpr(end_expr);
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
            .Break => |jump| if (jump.value) |expr_id| try self.visitExpr(expr_id),
            .Continue => |jump| if (jump.value) |expr_id| try self.visitExpr(expr_id),
            .Try => |try_stmt| {
                try self.visitBody(try_stmt.try_body);
                if (try_stmt.catch_clause) |catch_clause| {
                    if (catch_clause.error_pattern) |pattern_id| {
                        const catch_type = try self.inferCatchPatternType(try_stmt.try_body);
                        self.pattern_types[pattern_id.index()] = LocatedType.unlocated(catch_type);
                        self.catch_error_tag_patterns[pattern_id.index()] = catch_type.kind() == .integer;
                    }
                    try self.visitBody(catch_clause.body);
                }
            },
            .Log => |log_stmt| {
                try self.checkLogStatement(log_stmt);
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
                    const actual_located = self.exprLocatedType(assign.value);
                    const actual = locatedValue(actual_type, actual_located.region, actual_located.provenance);
                    if (!typesFlowCompatible(expected_type, actual_type)) {
                        try self.emitRangeError(assign.range, "assignment expects type '{s}', found '{s}'", .{
                            typeDisplayName(expected_type),
                            typeDisplayName(actual_type),
                        });
                    } else if (!region_rules.regionAssignable(actual.region, expected.region)) {
                        try self.emitRangeError(assign.range, "assignment expects region '{s}', found '{s}'", .{
                            region_rules.regionDisplayName(expected.region),
                            region_rules.regionDisplayName(actual.region),
                        });
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
            .TypeValue => |type_value| {
                self.expr_types[expr_id.index()] = try self.resolveTypeExprInCurrentContext(type_value.type_expr);
            },
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
                self.expr_types[expr_id.index()] = if (struct_literal.type_expr) |type_expr|
                    try self.resolveTypeExprInCurrentContext(type_expr)
                else if (struct_literal.type_name.len != 0)
                    self.structLiteralType(struct_literal.type_name)
                else
                    .{ .unknown = {} };
            },
            .ExternalProxy => |proxy| {
                try self.visitExpr(proxy.address_expr);
                try self.visitExpr(proxy.gas_expr);
                const address_type = self.expr_types[proxy.address_expr.index()];
                const gas_type = self.expr_types[proxy.gas_expr.index()];
                if (address_type.kind() != .unknown and address_type.kind() != .address) {
                    try self.emitExprError(expr_id, "external proxy address must be 'address', found '{s}'", .{
                        typeDisplayName(address_type),
                    });
                }
                if (gas_type.kind() != .unknown and gas_type.kind() != .integer) {
                    try self.emitExprError(expr_id, "external proxy gas must be integer, found '{s}'", .{
                        typeDisplayName(gas_type),
                    });
                }
                const trait_interface = self.traitInterfaceByName(proxy.trait_name);
                if (trait_interface == null) {
                    try self.emitExprError(expr_id, "unknown extern trait '{s}'", .{proxy.trait_name});
                    self.expr_types[expr_id.index()] = .{ .unknown = {} };
                } else if (!trait_interface.?.is_extern) {
                    try self.emitExprError(expr_id, "trait '{s}' is not extern", .{proxy.trait_name});
                    self.expr_types[expr_id.index()] = .{ .unknown = {} };
                } else {
                    self.expr_types[expr_id.index()] = .{ .external_proxy = .{ .trait_name = proxy.trait_name } };
                }
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
                self.expr_types[expr_id.index()] = try self.checkErrorReturn(expr_id, error_return);
            },
            .Name => {
                const binding_type = self.typeForBinding(self.resolution.expr_bindings[expr_id.index()]);
                self.expr_types[expr_id.index()] = if (binding_type.kind() != .unknown)
                    binding_type
                else switch (self.file.expression(expr_id).*) {
                    .Name => |name| if (std.mem.eql(u8, name.name, "result"))
                        (self.current_return_type orelse .{ .unknown = {} })
                    else
                        self.typeValueNameType(name.name),
                    else => .{ .unknown = {} },
                };
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
                const result_type = if ((binary.op == .eq or binary.op == .ne) and self.exprIsTypeValue(binary.lhs) and self.exprIsTypeValue(binary.rhs))
                    Type{ .bool = {} }
                else
                    self.binaryResultType(
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
                const result_type = self.callReturnType(call);
                self.expr_types[expr_id.index()] = result_type;
                if (self.calleeErrorDeclItem(call.callee)) |item_id| {
                    const error_decl = self.file.item(item_id).ErrorDecl;
                    if (error_decl.parameters.len != call.args.len) {
                        try self.emitExprError(expr_id, "expected {d} arguments, found {d}", .{ error_decl.parameters.len, call.args.len });
                    } else {
                        try self.checkErrorDeclCallArguments(call, error_decl);
                    }
                } else {
                    const callee_type = self.callableType(call.callee);
                    const callee_expr_type = self.expr_types[call.callee.index()];
                    if (callee_type.kind() != .function) {
                        const bad_type = if (callee_expr_type.kind() != .unknown) callee_expr_type else callee_type;
                        if (bad_type.kind() != .unknown) {
                            try self.emitExprError(expr_id, "type '{s}' is not callable", .{typeDisplayName(bad_type)});
                        }
                    } else if (result_type.kind() == .unknown) {
                        if (self.calleeFunctionItem(call.callee)) |item_id| {
                            const function = switch (self.file.item(item_id).*) {
                                .Function => |function| function,
                                else => null,
                            };
                            if (function) |resolved_function| {
                                if (resolved_function.is_generic and self.genericTypeBindingsForCall(resolved_function, call) == null) {
                                    try self.emitExprError(expr_id, "could not infer generic type arguments", .{});
                                } else if (self.expectedCallArgCount(call)) |expected_args| {
                                    if (expected_args != call.args.len) {
                                        try self.emitExprError(expr_id, "expected {d} arguments, found {d}", .{ expected_args, call.args.len });
                                    }
                                }
                            }
                        } else if (self.expectedCallArgCount(call)) |expected_args| {
                            if (expected_args != call.args.len) {
                                try self.emitExprError(expr_id, "expected {d} arguments, found {d}", .{ expected_args, call.args.len });
                            }
                        }
                    } else {
                        try self.checkCallArguments(call, callee_type);
                    }
                }
            },
            .Builtin => |builtin| {
                for (builtin.args) |arg| try self.visitExpr(arg);
                if (std.mem.eql(u8, builtin.name, "lock") or std.mem.eql(u8, builtin.name, "unlock")) {
                    try self.emitExprError(expr_id, "@{s} is statement-only and cannot be used in expression position", .{
                        builtin.name,
                    });
                    self.expr_types[expr_id.index()] = .{ .unknown = {} };
                    return;
                }
                const result_type = self.builtinReturnType(builtin);
                self.expr_types[expr_id.index()] = result_type;
                if (try self.emitBuiltinIntegerOverflowIfNeeded(expr_id, builtin, result_type)) {
                    self.expr_types[expr_id.index()] = .{ .unknown = {} };
                }
            },
            .Field => |field| {
                try self.visitExpr(field.base);
                const base_type = self.expr_types[field.base.index()];
                const result_type = try self.fieldAccessTypeForExpr(field.base, field.name);
                self.expr_types[expr_id.index()] = result_type;
                if (result_type.kind() == .unknown and base_type.kind() != .unknown) {
                    if (!try self.emitTraitMethodFieldError(expr_id, field, base_type)) {
                        try self.emitExprError(expr_id, "type '{s}' has no field '{s}'", .{
                            typeDisplayName(base_type),
                            field.name,
                        });
                    }
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

    fn checkLogStatement(self: *TypeChecker, log_stmt: ast.LogStmt) !void {
        for (log_stmt.args) |arg| try self.visitExpr(arg);

        const log_item_id = self.item_index.lookup(log_stmt.name) orelse {
            try self.emitRangeError(log_stmt.range, "undefined log '{s}'", .{log_stmt.name});
            return;
        };
        const log_decl = switch (self.file.item(log_item_id).*) {
            .LogDecl => |log_decl| log_decl,
            else => {
                try self.emitRangeError(log_stmt.range, "'{s}' is not a log declaration", .{log_stmt.name});
                return;
            },
        };

        if (log_stmt.args.len != log_decl.fields.len) {
            try self.emitRangeError(log_stmt.range, "log '{s}' expects {d} arguments, found {d}", .{
                log_stmt.name,
                log_decl.fields.len,
                log_stmt.args.len,
            });
            return;
        }

        for (log_stmt.args, log_decl.fields) |arg, field| {
            const expected_type = try self.resolveTypeExpr(field.type_expr);
            try self.contextualizeLiteral(arg, expected_type);
            const actual_type = self.expr_types[arg.index()];
            if (try self.emitIntegerOverflowIfNeeded(self.exprRange(arg), arg, expected_type)) {
                continue;
            }
            if (actual_type.kind() != .unknown and expected_type.kind() != .unknown and !typesFlowCompatible(expected_type, actual_type)) {
                try self.emitRangeError(self.exprRange(arg), "log field '{s}' expects type '{s}', found '{s}'", .{
                    field.name,
                    typeDisplayName(expected_type),
                    typeDisplayName(actual_type),
                });
            }
        }
    }

    fn logIndexedFieldTypeSupported(self: *const TypeChecker, ty: Type) bool {
        _ = self;
        return switch (unwrapRefinement(ty).kind()) {
            .struct_, .tuple, .array, .slice, .map => false,
            else => true,
        };
    }

    fn callReturnType(self: *TypeChecker, call: ast.CallExpr) Type {
        if (self.genericCallReturnType(call)) |result| return result;
        if (self.externProxyCallReturnType(call)) |result| return result;
        if (self.calleeErrorDeclItem(call.callee)) |item_id| {
            return self.item_types[item_id.index()];
        }
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

    fn calleeErrorDeclItem(self: *const TypeChecker, expr_id: ast.ExprId) ?ast.ItemId {
        return switch (self.file.expression(expr_id).*) {
            .Group => |group| self.calleeErrorDeclItem(group.expr),
            else => blk: {
                const binding = self.resolution.expr_bindings[expr_id.index()] orelse break :blk null;
                break :blk switch (binding) {
                    .item => |item_id| switch (self.file.item(item_id).*) {
                        .ErrorDecl => item_id,
                        else => null,
                    },
                    else => null,
                };
            },
        };
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
        const comptime_count = self.leadingComptimeParameterCount(function);
        const method_receiver_supplied = self.callSuppliesMethodReceiver(call.callee) and self.functionHasRuntimeSelf(function);
        const effective_runtime_count = runtime_parameters.len - @as(usize, if (method_receiver_supplied) 1 else 0);
        const explicit_generics = call.args.len >= comptime_count + effective_runtime_count;
        const runtime_arg_count = if (explicit_generics)
            call.args.len - comptime_count
        else
            call.args.len;
        if (effective_runtime_count != runtime_arg_count) return .{ .unknown = {} };
        if (function.return_type) |type_expr| {
            return self.resolveTypeExprWithBindings(type_expr, bindings) catch .{ .unknown = {} };
        }
        return .{ .void = {} };
    }

    fn externProxyCallReturnType(self: *TypeChecker, call: ast.CallExpr) ?Type {
        const method = self.externProxyMethodSignature(call.callee) orelse return null;
        const error_types = self.arena.alloc(Type, method.errors.len + 1) catch return .{ .unknown = {} };
        error_types[0] = .{ .named = .{ .name = "ExternalCallFailed" } };
        for (method.errors, 0..) |error_name, index| {
            error_types[index + 1] = .{ .named = .{ .name = error_name } };
        }
        return .{ .error_union = .{
            .payload_type = self.storeType(method.return_type) catch return .{ .unknown = {} },
            .error_types = error_types,
        } };
    }

    fn expectedCallArgCount(self: *const TypeChecker, call: ast.CallExpr) ?usize {
        const callee_id = self.calleeFunctionItem(call.callee) orelse return null;
        const function = switch (self.file.item(callee_id).*) {
            .Function => |function| function,
            else => return null,
        };
        if (!function.is_generic) return function.parameters.len;
        return null;
    }

    fn genericTypeBindingsForCall(self: *TypeChecker, function: ast.FunctionItem, call: ast.CallExpr) ?[]const GenericTypeBinding {
        const comptime_count = self.leadingComptimeParameterCount(function);
        const inferable_type_count = self.leadingGenericTypeParameterCount(function);
        if (comptime_count == 0) return &.{};
        const runtime_params = self.runtimeFunctionParameters(function) catch return null;
        const method_receiver_supplied = self.callSuppliesMethodReceiver(call.callee) and self.functionHasRuntimeSelf(function);
        const effective_runtime_count = runtime_params.len - @as(usize, if (method_receiver_supplied) 1 else 0);

        if (call.args.len >= comptime_count + effective_runtime_count) {
            const bindings = self.arena.alloc(GenericTypeBinding, comptime_count) catch return null;
            for (function.parameters[0..comptime_count], 0..) |parameter, index| {
                const name = self.patternName(parameter.pattern) orelse return null;
                const value = self.genericBindingValueForCallArg(parameter, call.args[index]) orelse return null;
                bindings[index] = .{ .name = name, .value = value };
            }
            self.validateGenericTraitBounds(function, call.range, bindings) catch return null;
            return bindings;
        }

        if (call.args.len != effective_runtime_count) return null;
        if (comptime_count != inferable_type_count) return null;

        const bindings = self.arena.alloc(GenericTypeBinding, comptime_count) catch return null;
        for (bindings) |*binding| {
            binding.* = .{ .name = "", .value = .{ .ty = .{ .unknown = {} } } };
        }

        const infer_runtime_params = if (method_receiver_supplied) runtime_params[1..] else runtime_params;
        for (call.args, infer_runtime_params) |arg, param| {
            const arg_type = self.expr_types[arg.index()];
            if (arg_type.kind() == .unknown) continue;
            self.inferBindingFromArgType(function, inferable_type_count, param.type_expr, arg_type, bindings);
        }

        for (bindings) |binding| {
            if (binding.name.len == 0) return null;
        }
        self.validateGenericTraitBounds(function, call.range, bindings) catch return null;
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

    fn leadingComptimeParameterCount(self: *const TypeChecker, function: ast.FunctionItem) usize {
        _ = self;
        var count: usize = 0;
        for (function.parameters) |parameter| {
            if (!parameter.is_comptime) break;
            count += 1;
        }
        return count;
    }

    fn leadingGenericTypeParameterCount(self: *const TypeChecker, function: ast.FunctionItem) usize {
        var count: usize = 0;
        for (function.parameters) |parameter| {
            if (!parameter.is_comptime) break;
            if (!self.isGenericTypeParameter(parameter)) break;
            count += 1;
        }
        return count;
    }

    fn functionHasRuntimeSelf(self: *const TypeChecker, function: ast.FunctionItem) bool {
        for (function.parameters) |parameter| {
            if (parameter.is_comptime) continue;
            return std.mem.eql(u8, self.patternName(parameter.pattern) orelse "", "self");
        }
        return false;
    }

    fn callSuppliesMethodReceiver(self: *const TypeChecker, expr_id: ast.ExprId) bool {
        return switch (self.file.expression(expr_id).*) {
            .Field => true,
            .Group => |group| self.callSuppliesMethodReceiver(group.expr),
            else => false,
        };
    }

    fn exprRange(self: *const TypeChecker, expr_id: ast.ExprId) source.TextRange {
        return switch (self.file.expression(expr_id).*) {
            inline else => |expr| expr.range,
        };
    }

    fn checkCallArguments(self: *TypeChecker, call: ast.CallExpr, callee_type: Type) !void {
        const callee_id = self.calleeFunctionItem(call.callee);
        if (callee_id) |item_id| {
            const function = switch (self.file.item(item_id).*) {
                .Function => |function| function,
                else => return,
            };
            const comptime_count = self.leadingComptimeParameterCount(function);
            const runtime_parameters = try self.runtimeFunctionParameters(function);
            const bindings = if (function.is_generic)
                self.genericTypeBindingsForCall(function, call) orelse return
            else
                &.{};
            const explicit_generics = call.args.len >= comptime_count + runtime_parameters.len;
            const runtime_args = if (explicit_generics)
                call.args[comptime_count..]
            else
                call.args;
            if (runtime_parameters.len != runtime_args.len) return;
            for (runtime_args, runtime_parameters) |arg, parameter| {
                const param_type = if (function.is_generic)
                    self.resolveTypeExprWithBindings(parameter.type_expr, bindings) catch Type{ .unknown = {} }
                else
                    self.pattern_types[parameter.pattern.index()].type;
                if (try self.emitIntegerOverflowIfNeeded(self.exprRange(arg), arg, param_type)) continue;
                const arg_type = self.expr_types[arg.index()];
                if (arg_type.kind() != .unknown and param_type.kind() != .unknown and
                    !typesFlowCompatible(param_type, arg_type))
                {
                    try self.emitExprError(arg, "expected argument type '{s}', found '{s}'", .{
                        typeDisplayName(param_type), typeDisplayName(arg_type),
                    });
                }
            }
            return;
        }

        // Fall back to the callee type when the call target is not a directly-resolved function item.
        // This cannot recover comptime-vs-runtime parameter metadata, so only use it for non-item callables.
        const param_types = callee_type.paramTypes();
        if (param_types.len != call.args.len) return;
        for (call.args, param_types) |arg, param_type| {
            if (try self.emitIntegerOverflowIfNeeded(self.exprRange(arg), arg, param_type)) continue;
            const arg_type = self.expr_types[arg.index()];
            if (arg_type.kind() != .unknown and param_type.kind() != .unknown and
                !typesFlowCompatible(param_type, arg_type))
            {
                try self.emitExprError(arg, "expected argument type '{s}', found '{s}'", .{
                    typeDisplayName(param_type), typeDisplayName(arg_type),
                });
            }
        }
    }

    fn checkErrorDeclCallArguments(self: *TypeChecker, call: ast.CallExpr, error_decl: ast.ErrorDeclItem) !void {
        if (error_decl.parameters.len != call.args.len) return;
        for (call.args, error_decl.parameters) |arg, parameter| {
            const param_type = self.pattern_types[parameter.pattern.index()].type;
            if (try self.emitIntegerOverflowIfNeeded(self.exprRange(arg), arg, param_type)) continue;
            const arg_type = self.expr_types[arg.index()];
            if (arg_type.kind() != .unknown and param_type.kind() != .unknown and
                !typesFlowCompatible(param_type, arg_type))
            {
                try self.emitExprError(arg, "expected argument type '{s}', found '{s}'", .{
                    typeDisplayName(param_type), typeDisplayName(arg_type),
                });
            }
        }
    }

    fn inferBindingFromArgType(
        self: *TypeChecker,
        function: ast.FunctionItem,
        generic_count: usize,
        type_expr: ast.TypeExprId,
        arg_type: Type,
        bindings: []GenericTypeBinding,
    ) void {
        const type_expr_node = self.file.typeExpr(type_expr).*;
        const name = switch (type_expr_node) {
            .Path => |path| path.name,
            else => return,
        };

        for (function.parameters[0..generic_count], 0..) |param, index| {
            const param_name = self.patternName(param.pattern) orelse continue;
            if (!std.mem.eql(u8, name, param_name)) continue;
            if (!self.isGenericTypeParameter(param)) continue;

            if (bindings[index].name.len > 0) {
                if (genericBindingType(bindings[index])) |existing| {
                    if (!sameConcreteType(existing, arg_type)) {
                        bindings[index].name = "";
                    }
                }
                return;
            }

            bindings[index] = .{ .name = param_name, .value = .{ .ty = arg_type } };
            return;
        }
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
            .TypeValue => |type_value| self.typeArgNameFromTypeExpr(type_value.type_expr),
            .Group => |group| self.typeArgNameFromExpr(group.expr),
            else => null,
        };
    }

    fn typeArgNameFromTypeExpr(self: *const TypeChecker, type_expr_id: ast.TypeExprId) ?[]const u8 {
        return switch (self.file.typeExpr(type_expr_id).*) {
            .Path => |path| path.name,
            else => null,
        };
    }

    fn typeArgTypeFromExpr(self: *const TypeChecker, expr_id: ast.ExprId) ?Type {
        return switch (self.file.expression(expr_id).*) {
            .TypeValue => self.expr_types[expr_id.index()],
            .Group => |group| self.typeArgTypeFromExpr(group.expr),
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

    fn genericBindingValueForCallArg(self: *const TypeChecker, parameter: ast.Parameter, arg_expr: ast.ExprId) ?GenericBindingValue {
        if (self.isGenericTypeParameter(parameter)) {
            if (self.typeArgTypeFromExpr(arg_expr)) |arg_type| {
                if (arg_type.kind() != .unknown) return .{ .ty = arg_type };
            }
            const arg_name = self.typeArgNameFromExpr(arg_expr) orelse return null;
            return .{ .ty = descriptorFromPathName(self.file, self.item_index, arg_name) };
        }
        if (parameter.is_comptime) {
            const integer = self.exprIntegerText(arg_expr) orelse return null;
            return .{ .integer = integer };
        }
        return null;
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

    fn validateGenericTraitBounds(
        self: *TypeChecker,
        function: ast.FunctionItem,
        call_range: source.TextRange,
        bindings: []const GenericTypeBinding,
    ) anyerror!void {
        for (function.trait_bounds) |bound| {
            const binding = self.genericBindingByName(bindings, bound.parameter_name) orelse continue;
            const bound_type = genericBindingType(binding) orelse {
                try self.emitRangeError(call_range, "trait bound '{s}: {s}' requires a type argument", .{
                    bound.parameter_name,
                    bound.trait_name,
                });
                continue;
            };

            const target_name = bound_type.name() orelse {
                try self.emitRangeError(call_range, "type '{s}' does not implement trait '{s}'", .{
                    typeDisplayName(bound_type),
                    bound.trait_name,
                });
                continue;
            };
            if (self.item_index.lookupImpl(bound.trait_name, target_name) == null) {
                try self.emitRangeError(call_range, "type '{s}' does not implement trait '{s}'", .{
                    typeDisplayName(bound_type),
                    bound.trait_name,
                });
            }
        }
    }

    fn genericBindingByName(self: *TypeChecker, bindings: []const GenericTypeBinding, name: []const u8) ?GenericTypeBinding {
        _ = self;
        for (bindings) |binding| {
            if (std.mem.eql(u8, binding.name, name)) return binding;
        }
        return null;
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
            .anonymous_struct => |struct_type| blk: {
                const fields = try self.arena.alloc(model.AnonymousStructField, struct_type.fields.len);
                for (struct_type.fields, 0..) |field, index| {
                    fields[index] = .{
                        .name = field.name,
                        .ty = try self.substituteGenericType(field.ty, bindings),
                    };
                }
                break :blk .{ .anonymous_struct = .{ .fields = fields } };
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
            .AnonymousStruct => |node| node.range,
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
                if (std.mem.eql(u8, trimmed, "NonZeroAddress")) {
                    break :blk .{ .refinement = .{
                        .name = "NonZeroAddress",
                        .base_type = try self.storeType(.{ .address = {} }),
                        .args = &.{},
                    } };
                }
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
            .AnonymousStruct => |struct_type| blk: {
                const fields = try self.arena.alloc(model.AnonymousStructField, struct_type.fields.len);
                for (struct_type.fields, 0..) |field, index| {
                    fields[index] = .{
                        .name = field.name,
                        .ty = try self.resolveTypeExprWithBindings(field.type_expr, bindings),
                    };
                }
                break :blk .{ .anonymous_struct = .{ .fields = fields } };
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
        if (self.lookupTypeItemInScope(generic.name)) |item_id| {
            switch (self.file.item(item_id).*) {
                .Struct => |struct_item| if (struct_item.is_generic) {
                    if (struct_item.template_parameters.len != generic.args.len) {
                        try self.emitGenericArityError(generic.range, "struct", struct_item.name, struct_item.template_parameters.len, generic.args.len);
                        return .{ .unknown = {} };
                    }
                    return try self.instantiateGenericStruct(item_id, struct_item, generic, bindings);
                },
                .Enum => |enum_item| if (enum_item.is_generic) {
                    if (enum_item.template_parameters.len != generic.args.len) {
                        try self.emitGenericArityError(generic.range, "enum", enum_item.name, enum_item.template_parameters.len, generic.args.len);
                        return .{ .unknown = {} };
                    }
                    return try self.instantiateGenericEnum(item_id, enum_item, generic, bindings);
                },
                .Bitfield => |bitfield_item| if (bitfield_item.is_generic) {
                    if (bitfield_item.template_parameters.len != generic.args.len) {
                        try self.emitGenericArityError(generic.range, "bitfield", bitfield_item.name, bitfield_item.template_parameters.len, generic.args.len);
                        return .{ .unknown = {} };
                    }
                    return try self.instantiateGenericBitfield(item_id, bitfield_item, generic, bindings);
                },
                .TypeAlias => |type_alias| if (type_alias.is_generic) {
                    if (type_alias.template_parameters.len != generic.args.len) {
                        try self.emitGenericArityError(generic.range, "type alias", type_alias.name, type_alias.template_parameters.len, generic.args.len);
                        return .{ .unknown = {} };
                    }
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

    fn lookupTypeItemInScope(self: *const TypeChecker, name: []const u8) ?ast.ItemId {
        const trimmed = std.mem.trim(u8, name, " \t\n\r");
        if (self.current_contract) |contract_id| {
            const contract = self.file.item(contract_id).Contract;
            for (contract.members) |member_id| {
                switch (self.file.item(member_id).*) {
                    .Struct => |item| if (std.mem.eql(u8, item.name, trimmed)) return member_id,
                    .Enum => |item| if (std.mem.eql(u8, item.name, trimmed)) return member_id,
                    .Bitfield => |item| if (std.mem.eql(u8, item.name, trimmed)) return member_id,
                    .TypeAlias => |item| if (std.mem.eql(u8, item.name, trimmed)) return member_id,
                    else => {},
                }
            }
        }
        if (self.item_index.lookup(trimmed)) |item_id| return item_id;
        for (self.file.items, 0..) |item, index| {
            switch (item) {
                .Struct => |struct_item| if (std.mem.eql(u8, struct_item.name, trimmed)) return ast.ItemId.fromIndex(index),
                .Enum => |enum_item| if (std.mem.eql(u8, enum_item.name, trimmed)) return ast.ItemId.fromIndex(index),
                .Bitfield => |bitfield_item| if (std.mem.eql(u8, bitfield_item.name, trimmed)) return ast.ItemId.fromIndex(index),
                .TypeAlias => |type_alias| if (std.mem.eql(u8, type_alias.name, trimmed)) return ast.ItemId.fromIndex(index),
                else => {},
            }
        }
        return null;
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

    fn contextualizeLiteral(self: *TypeChecker, expr_id: ast.ExprId, expected_type: Type) !void {
        const expr = self.file.expression(expr_id).*;
        switch (expr) {
            .Group => |group| return self.contextualizeLiteral(group.expr, expected_type),
            .Tuple => |tuple| {
                if (expected_type.kind() != .tuple) return;
                if (tuple.elements.len != expected_type.tuple.len) return;
                for (tuple.elements, 0..) |element, index| {
                    const expected_element_type = expected_type.tuple[index];
                    try self.contextualizeLiteral(element, expected_element_type);
                    const actual_type = self.expr_types[element.index()];
                    if (!typesAssignable(expected_element_type, actual_type)) return;
                }
                self.expr_types[expr_id.index()] = expected_type;
            },
            .ArrayLiteral => |array| {
                if (expected_type.kind() != .array) return;
                if (expected_type.array.len == null or expected_type.array.len.? != array.elements.len) return;
                for (array.elements) |element| {
                    try self.contextualizeLiteral(element, expected_type.array.element_type.*);
                    const actual_type = self.expr_types[element.index()];
                    if (!typesAssignable(expected_type.array.element_type.*, actual_type)) return;
                }
                self.expr_types[expr_id.index()] = expected_type;
            },
            .StructLiteral => |struct_literal| {
                if (struct_literal.type_expr != null or struct_literal.type_name.len != 0) return;
                self.expr_types[expr_id.index()] = expected_type;
            },
            else => {},
        }
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
            const value = try self.genericBindingValueForTypeArg(parameter, arg, outer_bindings, error.InvalidGenericStructInstantiation);
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
                .ty => |ty| try appendModelTypeMangleName(self.arena, &name, ty),
                .integer => |integer| try name.appendSlice(self.arena, try self.sanitizeGenericMangleSegment(integer)),
            }
        }
        return name.toOwnedSlice(self.arena);
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
            const value = try self.genericBindingValueForTypeArg(parameter, arg, outer_bindings, error.InvalidGenericEnumInstantiation);
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
            const value = try self.genericBindingValueForTypeArg(parameter, arg, outer_bindings, error.InvalidGenericTypeAliasInstantiation);
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
            const value = try self.genericBindingValueForTypeArg(parameter, arg, outer_bindings, error.InvalidGenericBitfieldInstantiation);
            bindings[index] = .{
                .name = name,
                .value = value,
            };
        }
        return bindings;
    }

    fn genericBindingValueForTypeArg(
        self: *TypeChecker,
        parameter: ast.Parameter,
        arg: ast.TypeArg,
        outer_bindings: []const GenericTypeBinding,
        comptime invalid_instantiation: anytype,
    ) anyerror!GenericBindingValue {
        if (self.isGenericTypeParameter(parameter)) {
            const ty = switch (arg) {
                .Type => |type_expr| try self.resolveTypeExprWithBindings(type_expr, outer_bindings),
                else => return invalid_instantiation,
            };
            return .{ .ty = ty };
        }
        if (self.comptimeIntegerParameter(parameter)) {
            const integer = self.resolveIntegerGenericArg(arg, outer_bindings) orelse return invalid_instantiation;
            return .{ .integer = integer };
        }
        return invalid_instantiation;
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

        if (std.mem.eql(u8, builtin.name, "sizeOf") or std.mem.eql(u8, builtin.name, "keccak256")) {
            return descriptorFromPathName(self.file, self.item_index, "u256");
        }

        if (std.mem.eql(u8, builtin.name, "typeName")) {
            return .{ .string = {} };
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

    const ExternalCallValidationState = struct {
        current_writes: std.ArrayList(EffectSlot) = .{},
        frozen_pre_call_writes: std.ArrayList(EffectSlot) = .{},

        fn deinit(self: *ExternalCallValidationState, allocator: std.mem.Allocator) void {
            self.current_writes.deinit(allocator);
            self.frozen_pre_call_writes.deinit(allocator);
        }
    };

    fn initExternalCallValidationState(self: *TypeChecker) !ExternalCallValidationState {
        _ = self;
        return .{};
    }

    fn cloneExternalCallValidationState(self: *TypeChecker, src: ExternalCallValidationState) !ExternalCallValidationState {
        return .{
            .current_writes = try self.cloneEffectSlots(src.current_writes.items),
            .frozen_pre_call_writes = try self.cloneEffectSlots(src.frozen_pre_call_writes.items),
        };
    }

    fn mergeExternalCallValidationState(self: *TypeChecker, dst: *ExternalCallValidationState, src: ExternalCallValidationState) !void {
        const intersected_current = try self.intersectStorageSlots(dst.current_writes.items, src.current_writes.items);
        dst.current_writes.deinit(self.arena);
        dst.current_writes = intersected_current;
        try self.mergeStorageSlots(&dst.frozen_pre_call_writes, src.frozen_pre_call_writes.items);
    }

    fn validateBodyExternalCalls(self: *TypeChecker, body_id: ast.BodyId, state: *ExternalCallValidationState) anyerror!void {
        const body = self.file.body(body_id).*;
        for (body.statements) |statement_id| {
            try self.validateStmtExternalCalls(statement_id, state);
        }
    }

    fn overwriteExternalCallValidationState(self: *TypeChecker, dst: *ExternalCallValidationState, src: ExternalCallValidationState) !void {
        dst.deinit(self.arena);
        dst.* = try self.cloneExternalCallValidationState(src);
    }

    fn validateStmtExternalCalls(self: *TypeChecker, statement_id: ast.StmtId, state: *ExternalCallValidationState) anyerror!void {
        switch (self.file.statement(statement_id).*) {
            .VariableDecl, .Return, .Assert, .Assume, .Expr, .Havoc, .Break, .Continue, .Error => {
                if (self.statementExpr(statement_id)) |expr_id| try self.validateExprExternalCalls(expr_id, state);
            },
            .If => |if_stmt| {
                try self.validateExprExternalCalls(if_stmt.condition, state);
                var then_state = try self.cloneExternalCallValidationState(state.*);
                var else_state = try self.cloneExternalCallValidationState(state.*);
                defer then_state.deinit(self.arena);
                defer else_state.deinit(self.arena);
                try self.validateBodyExternalCalls(if_stmt.then_body, &then_state);
                if (if_stmt.else_body) |else_body| try self.validateBodyExternalCalls(else_body, &else_state);
                try self.mergeExternalCallValidationState(&then_state, else_state);
                try self.overwriteExternalCallValidationState(state, then_state);
            },
            .While => |while_stmt| {
                try self.validateExprExternalCalls(while_stmt.condition, state);
                for (while_stmt.invariants) |expr_id| try self.validateExprExternalCalls(expr_id, state);
                var loop_state = try self.cloneExternalCallValidationState(state.*);
                defer loop_state.deinit(self.arena);
                try self.validateBodyExternalCalls(while_stmt.body, &loop_state);
                try self.mergeExternalCallValidationState(state, loop_state);
            },
            .For => |for_stmt| {
                try self.validateExprExternalCalls(for_stmt.iterable, state);
                if (for_stmt.range_end) |end_expr| try self.validateExprExternalCalls(end_expr, state);
                for (for_stmt.invariants) |expr_id| try self.validateExprExternalCalls(expr_id, state);
                var loop_state = try self.cloneExternalCallValidationState(state.*);
                defer loop_state.deinit(self.arena);
                try self.validateBodyExternalCalls(for_stmt.body, &loop_state);
                try self.mergeExternalCallValidationState(state, loop_state);
            },
            .Switch => |switch_stmt| {
                try self.validateExprExternalCalls(switch_stmt.condition, state);
                var merged_state = try self.cloneExternalCallValidationState(state.*);
                for (switch_stmt.arms) |arm| {
                    switch (arm.pattern) {
                        .Expr => |expr_id| try self.validateExprExternalCalls(expr_id, state),
                        .Range => |range_pattern| {
                            try self.validateExprExternalCalls(range_pattern.start, state);
                            try self.validateExprExternalCalls(range_pattern.end, state);
                        },
                        .Else => {},
                    }
                    var case_state = try self.cloneExternalCallValidationState(state.*);
                    defer case_state.deinit(self.arena);
                    try self.validateBodyExternalCalls(arm.body, &case_state);
                    try self.mergeExternalCallValidationState(&merged_state, case_state);
                }
                if (switch_stmt.else_body) |else_body| {
                    var else_state = try self.cloneExternalCallValidationState(state.*);
                    defer else_state.deinit(self.arena);
                    try self.validateBodyExternalCalls(else_body, &else_state);
                    try self.mergeExternalCallValidationState(&merged_state, else_state);
                }
                defer merged_state.deinit(self.arena);
                try self.overwriteExternalCallValidationState(state, merged_state);
            },
            .Try => |try_stmt| {
                var try_state = try self.cloneExternalCallValidationState(state.*);
                defer try_state.deinit(self.arena);
                try self.validateBodyExternalCalls(try_stmt.try_body, &try_state);
                if (try_stmt.catch_clause) |catch_clause| {
                    var catch_state = try self.cloneExternalCallValidationState(state.*);
                    defer catch_state.deinit(self.arena);
                    try self.validateBodyExternalCalls(catch_clause.body, &catch_state);
                    try self.mergeExternalCallValidationState(&try_state, catch_state);
                }
                try self.overwriteExternalCallValidationState(state, try_state);
            },
            .Log => |log_stmt| for (log_stmt.args) |arg| try self.validateExprExternalCalls(arg, state),
            .Block => |block| try self.validateBodyExternalCalls(block.body, state),
            .LabeledBlock => |block| try self.validateBodyExternalCalls(block.body, state),
            .Lock => |lock_stmt| try self.validateExprExternalCalls(lock_stmt.path, state),
            .Unlock => |unlock_stmt| try self.validateExprExternalCalls(unlock_stmt.path, state),
            .Assign => |assign| {
                try self.validateExprExternalCalls(assign.value, state);
                var effects = EffectCollectorState.init();
                try self.collectPatternTargetEffects(assign.target, assign.op, &effects);
                try self.emitExternalCallWriteDiagnostics(assign.range, effects.writes.items, state.frozen_pre_call_writes.items);
                try self.mergeStorageSlots(&state.current_writes, effects.writes.items);
            },
        }
    }

    fn validateExprExternalCalls(self: *TypeChecker, expr_id: ast.ExprId, state: *ExternalCallValidationState) anyerror!void {
        switch (self.file.expression(expr_id).*) {
            .TypeValue => {},
            .Call => |call| {
                try self.validateExprExternalCalls(call.callee, state);
                for (call.args) |arg| try self.validateExprExternalCalls(arg, state);

                if (self.externProxyMethodSignature(call.callee)) |method| {
                    if (method.extern_call_kind == .call) {
                        try self.mergeStorageSlots(&state.frozen_pre_call_writes, state.current_writes.items);
                    }
                    return;
                }

                if (self.calleeFunctionItem(call.callee)) |callee_id| {
                    switch (self.file.item(callee_id).*) {
                        .Function => |function| {
                            try self.ensureFunctionEffectSummary(callee_id, function);
                            const writes = self.effectWrites(self.item_effects[callee_id.index()]);
                            try self.emitExternalCallWriteDiagnostics(call.range, writes, state.frozen_pre_call_writes.items);
                            try self.mergeStorageSlots(&state.current_writes, writes);
                        },
                        else => {},
                    }
                }
            },
            .Unary => |unary| try self.validateExprExternalCalls(unary.operand, state),
            .Binary => |binary| {
                try self.validateExprExternalCalls(binary.lhs, state);
                try self.validateExprExternalCalls(binary.rhs, state);
            },
            .Tuple => |tuple| for (tuple.elements) |element| try self.validateExprExternalCalls(element, state),
            .ArrayLiteral => |array| for (array.elements) |element| try self.validateExprExternalCalls(element, state),
            .StructLiteral => |struct_literal| for (struct_literal.fields) |field| try self.validateExprExternalCalls(field.value, state),
            .Switch => |switch_expr| {
                try self.validateExprExternalCalls(switch_expr.condition, state);
                for (switch_expr.arms) |arm| {
                    switch (arm.pattern) {
                        .Expr => |pattern_expr| try self.validateExprExternalCalls(pattern_expr, state),
                        .Range => |range_pattern| {
                            try self.validateExprExternalCalls(range_pattern.start, state);
                            try self.validateExprExternalCalls(range_pattern.end, state);
                        },
                        .Else => {},
                    }
                    try self.validateExprExternalCalls(arm.value, state);
                }
                if (switch_expr.else_expr) |else_expr| try self.validateExprExternalCalls(else_expr, state);
            },
            .Builtin => |builtin| for (builtin.args) |arg| try self.validateExprExternalCalls(arg, state),
            .Field => |field| try self.validateExprExternalCalls(field.base, state),
            .Index => |index| {
                try self.validateExprExternalCalls(index.base, state);
                try self.validateExprExternalCalls(index.index, state);
            },
            .Group => |group| try self.validateExprExternalCalls(group.expr, state),
            .Comptime, .ExternalProxy, .Old, .Quantified, .ErrorReturn, .Name, .IntegerLiteral, .StringLiteral, .BoolLiteral, .AddressLiteral, .BytesLiteral, .Result, .Error => {},
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
                if (for_stmt.range_end) |end_expr| try self.validateExprLocks(end_expr, locked_slots);
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
            .TypeValue => {},
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
            .Comptime, .ExternalProxy, .Old, .Quantified, .ErrorReturn, .Name, .IntegerLiteral, .StringLiteral, .BoolLiteral, .AddressLiteral, .BytesLiteral, .Result, .Error => {},
        }
    }

    fn statementExpr(self: *TypeChecker, statement_id: ast.StmtId) ?ast.ExprId {
        return switch (self.file.statement(statement_id).*) {
            .VariableDecl => |decl| decl.value,
            .Return => |ret| ret.value,
            .Assert => |assert_stmt| assert_stmt.condition,
            .Assume => |assume_stmt| assume_stmt.condition,
            .Expr => |expr_stmt| expr_stmt.expr,
            .Break => |jump| jump.value,
            .Continue => |jump| jump.value,
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

    fn emitExternalCallWriteDiagnostics(self: *TypeChecker, range: source.TextRange, writes: []const EffectSlot, frozen_pre_call_writes: []const EffectSlot) !void {
        for (writes) |write_slot| {
            if (write_slot.region != .storage) continue;
            for (frozen_pre_call_writes) |frozen_slot| {
                if (!self.effectSlotsMayAlias(write_slot, frozen_slot)) continue;
                try self.emitRangeError(range, "cannot write storage slot '{s}' after external call because it was written before the call", .{
                    write_slot.name,
                });
                break;
            }
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

    fn mergeStorageSlots(self: *TypeChecker, dst: *std.ArrayList(EffectSlot), src: []const EffectSlot) !void {
        for (src) |slot| {
            if (slot.region != .storage) continue;
            try self.appendUniqueSlot(dst, slot);
        }
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

    fn intersectStorageSlots(self: *TypeChecker, lhs: []const EffectSlot, rhs: []const EffectSlot) !std.ArrayList(EffectSlot) {
        var result: std.ArrayList(EffectSlot) = .{};
        for (lhs) |lhs_slot| {
            if (lhs_slot.region != .storage) continue;
            for (rhs) |rhs_slot| {
                if (rhs_slot.region != .storage) continue;
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
                if (for_stmt.range_end) |end_expr| try self.collectExprEffects(end_expr, state);
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
            .Break => |jump| if (jump.value) |expr_id| try self.collectExprEffects(expr_id, state),
            .Continue => |jump| if (jump.value) |expr_id| try self.collectExprEffects(expr_id, state),
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
            .IntegerLiteral, .StringLiteral, .BoolLiteral, .AddressLiteral, .BytesLiteral, .TypeValue, .Result, .Error => {},
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
            .Comptime, .ExternalProxy => {},
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
                if (for_stmt.range_end) |end_expr| try self.collectExprDirectCallees(end_expr, callees);
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
            .Havoc, .Error => {},
            .Break => |jump| if (jump.value) |expr_id| try self.collectExprDirectCallees(expr_id, callees),
            .Continue => |jump| if (jump.value) |expr_id| try self.collectExprDirectCallees(expr_id, callees),
        }
    }

    fn collectExprDirectCallees(self: *TypeChecker, expr_id: ast.ExprId, callees: *std.ArrayList(ast.ItemId)) anyerror!void {
        switch (self.file.expression(expr_id).*) {
            .TypeValue => {},
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
            .Comptime, .ExternalProxy, .Old, .Quantified, .Name, .IntegerLiteral, .StringLiteral, .BoolLiteral, .AddressLiteral, .BytesLiteral, .Result, .Error => {},
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
            .add => if (lhs_type.kind() == .string and rhs_type.kind() == .string)
                .{ .string = {} }
            else
                arithmeticResultType(lhs_type, rhs_type),
            .wrapping_add, .sub, .wrapping_sub, .mul, .wrapping_mul, .div, .mod, .pow, .wrapping_pow => arithmeticResultType(lhs_type, rhs_type),
            .bit_and, .bit_or, .bit_xor, .shl, .wrapping_shl, .shr, .wrapping_shr => bitwiseResultType(lhs_type, rhs_type),
            .and_and, .or_or => if (lhs_type.kind() == .bool and rhs_type.kind() == .bool) .{ .bool = {} } else .{ .unknown = {} },
            .eq, .ne => if (typesComparable(lhs_type, rhs_type)) .{ .bool = {} } else .{ .unknown = {} },
            .lt, .le, .gt, .ge => if (orderedTypesComparable(lhs_type, rhs_type)) .{ .bool = {} } else .{ .unknown = {} },
        };
    }

    fn exprIsTypeValue(self: *const TypeChecker, expr_id: ast.ExprId) bool {
        return switch (self.file.expression(expr_id).*) {
            .TypeValue => true,
            .Group => |group| self.exprIsTypeValue(group.expr),
            .Name => |name| blk: {
                const binding = self.resolution.expr_bindings[expr_id.index()];
                if (binding) |resolved| switch (resolved) {
                    .pattern => |pattern_id| {
                        const ty = self.pattern_types[pattern_id.index()].type;
                        break :blk ty.kind() == .named;
                    },
                    .item => |item_id| switch (self.file.item(item_id).*) {
                        .Struct, .Enum, .Bitfield, .Contract, .TypeAlias => break :blk true,
                        else => {},
                    },
                };
                break :blk self.typeValueNameType(name.name).kind() != .unknown;
            },
            else => false,
        };
    }

    fn typeValueNameType(self: *const TypeChecker, name: []const u8) Type {
        const trimmed = std.mem.trim(u8, name, " \t\n\r");
        if (std.mem.eql(u8, trimmed, "void") or
            std.mem.eql(u8, trimmed, "bool") or
            std.mem.eql(u8, trimmed, "string") or
            std.mem.eql(u8, trimmed, "address") or
            std.mem.eql(u8, trimmed, "bytes"))
        {
            return descriptorFromPathName(self.file, self.item_index, trimmed);
        }
        if (trimmed.len >= 2 and (trimmed[0] == 'u' or trimmed[0] == 'i')) {
            const digits = trimmed[1..];
            if (std.mem.eql(u8, digits, "8") or
                std.mem.eql(u8, digits, "16") or
                std.mem.eql(u8, digits, "32") or
                std.mem.eql(u8, digits, "64") or
                std.mem.eql(u8, digits, "128") or
                std.mem.eql(u8, digits, "256"))
            {
                return descriptorFromPathName(self.file, self.item_index, trimmed);
            }
        }
        if (self.item_index.lookup(trimmed)) |item_id| {
            return switch (self.file.item(item_id).*) {
                .Contract, .Struct, .Bitfield, .Enum, .TypeAlias => descriptorFromPathName(self.file, self.item_index, trimmed),
                else => .{ .unknown = {} },
            };
        }
        return .{ .unknown = {} };
    }

    fn fieldAccessType(self: *const TypeChecker, base_type: Type, field_name: []const u8) !Type {
        if (overflowTupleFieldType(base_type, field_name)) |field_type| return field_type;
        if (anonymousStructFieldType(base_type, field_name)) |field_type| return field_type;
        if (self.externalProxyMethodType(base_type, field_name)) |method_type| return method_type;
        if (self.traitBoundMethodType(base_type, field_name)) |method_type| return method_type;
        if (self.concreteImplMethodType(base_type, field_name)) |method_type| return method_type;
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
            .ErrorDecl => |error_decl| blk: {
                for (error_decl.parameters) |parameter| {
                    const pattern = self.file.pattern(parameter.pattern).*;
                    switch (pattern) {
                        .Name => |name| if (std.mem.eql(u8, name.name, field_name)) {
                            break :blk try descriptorFromTypeExpr(self.arena, self.file, self.item_index, parameter.type_expr);
                        },
                        else => {},
                    }
                }
                break :blk .{ .unknown = {} };
            },
            else => .{ .unknown = {} },
        };
    }

    fn fieldAccessTypeForExpr(self: *const TypeChecker, base_expr_id: ast.ExprId, field_name: []const u8) !Type {
        if (self.associatedMethodTypeForExpr(base_expr_id, field_name)) |method_type| return method_type;
        const base_type = self.expr_types[base_expr_id.index()];
        return self.fieldAccessType(base_type, field_name);
    }

    fn externalProxyMethodType(self: *const TypeChecker, base_type: Type, field_name: []const u8) ?Type {
        if (base_type.kind() != .external_proxy) return null;
        const trait_interface = self.traitInterfaceByName(base_type.external_proxy.trait_name) orelse return null;
        const method = self.findTraitMethodSignature(trait_interface, field_name) orelse return null;
        return self.functionTypeFromTraitSignature(method);
    }

    fn externProxyMethodSignature(self: *const TypeChecker, expr_id: ast.ExprId) ?TraitMethodSignature {
        const field = switch (self.file.expression(expr_id).*) {
            .Field => |field| field,
            .Group => |group| return self.externProxyMethodSignature(group.expr),
            else => return null,
        };
        const base_type = self.expr_types[field.base.index()];
        if (base_type.kind() != .external_proxy) return null;
        const trait_interface = self.traitInterfaceByName(base_type.external_proxy.trait_name) orelse return null;
        return self.findTraitMethodSignature(trait_interface, field.name);
    }

    fn emitTraitMethodFieldError(self: *TypeChecker, expr_id: ast.ExprId, field: ast.FieldExpr, base_type: Type) !bool {
        if (try self.emitCatchErrorTagFieldError(expr_id, field)) return true;
        if (try self.emitTraitBoundMethodFieldError(expr_id, field, base_type)) return true;
        if (try self.emitConcreteAssociatedMethodFieldError(expr_id, field)) return true;
        if (try self.emitConcreteImplMethodFieldError(expr_id, field)) return true;
        return false;
    }

    fn emitCatchErrorTagFieldError(self: *TypeChecker, expr_id: ast.ExprId, field: ast.FieldExpr) !bool {
        const binding = switch (self.file.expression(field.base).*) {
            .Name => self.resolution.expr_bindings[field.base.index()],
            .Group => |group| switch (self.file.expression(group.expr).*) {
                .Name => self.resolution.expr_bindings[group.expr.index()],
                else => null,
            },
            else => null,
        } orelse return false;
        if (binding != .pattern) return false;
        if (!self.catch_error_tag_patterns[binding.pattern.index()]) return false;
        try self.emitExprError(expr_id, "catch binding represents multiple possible error types; field access is not supported", .{});
        return true;
    }

    fn emitTraitBoundMethodFieldError(self: *TypeChecker, expr_id: ast.ExprId, field: ast.FieldExpr, base_type: Type) !bool {
        const function_item_id = self.current_function_item orelse return false;
        const function = switch (self.file.item(function_item_id).*) {
            .Function => |function| function,
            else => return false,
        };

        if (self.genericTypeParameterNameForExpr(field.base)) |parameter_name| {
            var matching_bounds: usize = 0;
            var bounds_with_method: usize = 0;
            var unique_trait: ?[]const u8 = null;
            for (function.trait_bounds) |bound| {
                if (!std.mem.eql(u8, bound.parameter_name, parameter_name)) continue;
                matching_bounds += 1;
                const trait_interface = self.traitInterfaceByName(bound.trait_name) orelse continue;
                const method = self.findTraitMethodSignature(trait_interface, field.name) orelse continue;
                _ = method;
                bounds_with_method += 1;
                if (unique_trait == null) unique_trait = bound.trait_name;
            }

            if (matching_bounds == 0) return false;
            if (bounds_with_method == 0) {
                try self.emitExprError(expr_id, "type parameter '{s}' has no trait bound providing method '{s}'", .{
                    parameter_name,
                    field.name,
                });
                return true;
            }
            if (bounds_with_method > 1) {
                try self.emitExprError(expr_id, "method '{s}' is ambiguous for type parameter '{s}' across multiple trait bounds", .{
                    field.name,
                    parameter_name,
                });
                return true;
            }

            return false;
        }

        if (base_type.name()) |receiver_name| {
            if (!self.currentFunctionHasGenericTypeParameterNamed(receiver_name)) return false;
            var bounds_with_method: usize = 0;
            var unique_trait: ?[]const u8 = null;
            for (function.trait_bounds) |bound| {
                if (!std.mem.eql(u8, bound.parameter_name, receiver_name)) continue;
                const trait_interface = self.traitInterfaceByName(bound.trait_name) orelse continue;
                const method = self.findTraitMethodSignature(trait_interface, field.name) orelse continue;
                if (method.receiver_kind != .value_self) continue;
                bounds_with_method += 1;
                if (unique_trait == null) unique_trait = bound.trait_name;
            }
            if (bounds_with_method > 1) {
                try self.emitExprError(expr_id, "method '{s}' is ambiguous for type parameter '{s}' across multiple trait bounds", .{
                    field.name,
                    receiver_name,
                });
                return true;
            }
            if (bounds_with_method == 0) {
                try self.emitExprError(expr_id, "type parameter '{s}' has no trait bound providing method '{s}'", .{
                    receiver_name,
                    field.name,
                });
                return true;
            }
        }

        return false;
    }

    fn emitConcreteImplMethodFieldError(self: *TypeChecker, expr_id: ast.ExprId, field: ast.FieldExpr) !bool {
        const target_name = self.concreteTypeNameForExpr(field.base) orelse return false;
        var matching_impls: usize = 0;
        for (self.impl_interfaces.items) |impl_interface| {
            if (!std.mem.eql(u8, impl_interface.target_name, target_name)) continue;
            for (impl_interface.methods) |method| {
                if (std.mem.eql(u8, method.name, field.name) and method.receiver_kind == .value_self) {
                    matching_impls += 1;
                    break;
                }
            }
        }
        if (matching_impls > 1) {
            try self.emitExprError(expr_id, "method '{s}' is ambiguous for type '{s}' across multiple impls", .{
                field.name,
                target_name,
            });
            return true;
        }
        if (matching_impls == 0 and self.anyTraitDefinesValueMethod(field.name)) {
            try self.emitExprError(expr_id, "type '{s}' has no impl providing method '{s}'", .{
                target_name,
                field.name,
            });
            return true;
        }
        return false;
    }

    fn emitConcreteAssociatedMethodFieldError(self: *TypeChecker, expr_id: ast.ExprId, field: ast.FieldExpr) !bool {
        if (!self.exprRepresentsType(field.base)) return false;
        const target_name = self.concreteTypeNameForExpr(field.base) orelse return false;
        var matching_impls: usize = 0;
        for (self.impl_interfaces.items) |impl_interface| {
            if (!std.mem.eql(u8, impl_interface.target_name, target_name)) continue;
            for (impl_interface.methods) |method| {
                if (std.mem.eql(u8, method.name, field.name) and method.receiver_kind == .none) {
                    matching_impls += 1;
                    break;
                }
            }
        }
        if (matching_impls > 1) {
            try self.emitExprError(expr_id, "method '{s}' is ambiguous for type '{s}' across multiple impls", .{
                field.name,
                target_name,
            });
            return true;
        }
        if (matching_impls == 0 and self.anyTraitDefinesAssociatedMethod(field.name)) {
            try self.emitExprError(expr_id, "type '{s}' has no impl providing method '{s}'", .{
                target_name,
                field.name,
            });
            return true;
        }
        return false;
    }

    fn anyTraitDefinesMethod(self: *const TypeChecker, method_name: []const u8) bool {
        for (self.trait_interfaces.items) |trait_interface| {
            if (self.findTraitMethodSignature(trait_interface, method_name) != null) return true;
        }
        return false;
    }

    fn anyTraitDefinesValueMethod(self: *const TypeChecker, method_name: []const u8) bool {
        for (self.trait_interfaces.items) |trait_interface| {
            const method = self.findTraitMethodSignature(trait_interface, method_name) orelse continue;
            if (method.receiver_kind == .value_self) return true;
        }
        return false;
    }

    fn anyTraitDefinesAssociatedMethod(self: *const TypeChecker, method_name: []const u8) bool {
        for (self.trait_interfaces.items) |trait_interface| {
            const method = self.findTraitMethodSignature(trait_interface, method_name) orelse continue;
            if (method.receiver_kind == .none) return true;
        }
        return false;
    }

    fn currentFunctionHasGenericTypeParameterNamed(self: *const TypeChecker, name: []const u8) bool {
        const function_item_id = self.current_function_item orelse return false;
        const function = switch (self.file.item(function_item_id).*) {
            .Function => |function| function,
            else => return false,
        };
        for (function.parameters) |parameter| {
            if (!self.isGenericTypeParameter(parameter)) continue;
            const pattern_name = self.patternName(parameter.pattern) orelse continue;
            if (std.mem.eql(u8, pattern_name, name)) return true;
        }
        return false;
    }

    fn associatedMethodTypeForExpr(self: *const TypeChecker, base_expr_id: ast.ExprId, field_name: []const u8) ?Type {
        if (!self.exprRepresentsType(base_expr_id)) return null;
        if (self.traitBoundAssociatedMethodType(base_expr_id, field_name)) |method_type| return method_type;
        return self.concreteAssociatedMethodType(base_expr_id, field_name);
    }

    fn exprRepresentsType(self: *const TypeChecker, expr_id: ast.ExprId) bool {
        return switch (self.file.expression(expr_id).*) {
            .TypeValue => true,
            .Group => |group| self.exprRepresentsType(group.expr),
            .Name => if (self.resolution.expr_bindings[expr_id.index()]) |binding| switch (binding) {
                .item => |item_id| switch (self.file.item(item_id).*) {
                    .Struct, .Enum, .Bitfield, .Contract => true,
                    else => false,
                },
                .pattern => |pattern_id| blk: {
                    const ty = self.pattern_types[pattern_id.index()].type;
                    break :blk if (ty.name()) |name| std.mem.eql(u8, name, "type") else false;
                },
            } else false,
            else => false,
        };
    }

    fn traitBoundAssociatedMethodType(self: *const TypeChecker, base_expr_id: ast.ExprId, field_name: []const u8) ?Type {
        const parameter_name = self.genericTypeParameterNameForExpr(base_expr_id) orelse return null;
        const function_item_id = self.current_function_item orelse return null;
        const function = switch (self.file.item(function_item_id).*) {
            .Function => |function| function,
            else => return null,
        };

        var matched: ?TraitMethodSignature = null;
        for (function.trait_bounds) |bound| {
            if (!std.mem.eql(u8, bound.parameter_name, parameter_name)) continue;
            const trait_interface = self.traitInterfaceByName(bound.trait_name) orelse continue;
            const method = self.findTraitMethodSignature(trait_interface, field_name) orelse continue;
            if (method.receiver_kind != .none) continue;
            if (matched != null) return null;
            matched = method;
        }
        return self.functionTypeFromTraitSignature(matched orelse return null);
    }

    fn concreteAssociatedMethodType(self: *const TypeChecker, base_expr_id: ast.ExprId, field_name: []const u8) ?Type {
        const target_name = self.concreteTypeNameForExpr(base_expr_id) orelse return null;
        var matched: ?TraitMethodSignature = null;
        for (self.impl_interfaces.items) |impl_interface| {
            if (!std.mem.eql(u8, impl_interface.target_name, target_name)) continue;
            for (impl_interface.methods) |method| {
                if (!std.mem.eql(u8, method.name, field_name) or method.receiver_kind != .none) continue;
                if (matched != null) return null;
                matched = method;
            }
        }
        return self.functionTypeFromTraitSignature(matched orelse return null);
    }

    fn genericTypeParameterNameForExpr(self: *const TypeChecker, expr_id: ast.ExprId) ?[]const u8 {
        return switch (self.file.expression(expr_id).*) {
            .Group => |group| self.genericTypeParameterNameForExpr(group.expr),
            .Name => if (self.resolution.expr_bindings[expr_id.index()]) |binding| switch (binding) {
                .pattern => |pattern_id| blk: {
                    const ty = self.pattern_types[pattern_id.index()].type;
                    const name = if (ty.name()) |n| n else break :blk null;
                    if (!std.mem.eql(u8, name, "type")) break :blk null;
                    const pattern = self.file.pattern(pattern_id).*;
                    if (pattern != .Name) break :blk null;
                    break :blk pattern.Name.name;
                },
                else => null,
            } else null,
            else => null,
        };
    }

    fn concreteTypeNameForExpr(self: *const TypeChecker, expr_id: ast.ExprId) ?[]const u8 {
        return switch (self.file.expression(expr_id).*) {
            .TypeValue => |type_value| switch (self.file.typeExpr(type_value.type_expr).*) {
                .Path => |path| std.mem.trim(u8, path.name, " \t\n\r"),
                else => null,
            },
            .Group => |group| self.concreteTypeNameForExpr(group.expr),
            .Name => if (self.resolution.expr_bindings[expr_id.index()]) |binding| switch (binding) {
                .item => |item_id| switch (self.file.item(item_id).*) {
                    .Struct => |item| item.name,
                    .Enum => |item| item.name,
                    .Bitfield => |item| item.name,
                    .Contract => |item| item.name,
                    else => null,
                },
                else => null,
            } else null,
            else => null,
        };
    }

    fn functionTypeFromTraitSignature(self: *const TypeChecker, method_signature: TraitMethodSignature) Type {
        const returns = self.arena.alloc(Type, 1) catch return .{ .unknown = {} };
        returns[0] = method_signature.return_type;
        return .{ .function = .{
            .name = method_signature.name,
            .param_types = method_signature.param_types,
            .return_types = returns,
        } };
    }

    fn traitBoundMethodType(self: *const TypeChecker, base_type: Type, field_name: []const u8) ?Type {
        const type_name = base_type.name() orelse return null;
        const function_item_id = self.current_function_item orelse return null;
        const function = switch (self.file.item(function_item_id).*) {
            .Function => |function| function,
            else => return null,
        };

        var matched: ?TraitMethodSignature = null;
        for (function.trait_bounds) |bound| {
            if (!std.mem.eql(u8, bound.parameter_name, type_name)) continue;
            const trait_interface = self.traitInterfaceByName(bound.trait_name) orelse continue;
            const method = self.findTraitMethodSignature(trait_interface, field_name) orelse continue;
            if (method.receiver_kind != .value_self) continue;
            if (matched != null) return null;
            matched = method;
        }

        const method_signature = matched orelse return null;
        const returns = self.arena.alloc(Type, 1) catch return .{ .unknown = {} };
        returns[0] = method_signature.return_type;
        return .{ .function = .{
            .name = method_signature.name,
            .param_types = method_signature.param_types,
            .return_types = returns,
        } };
    }

    fn concreteImplMethodType(self: *const TypeChecker, base_type: Type, field_name: []const u8) ?Type {
        const target_name = base_type.name() orelse return null;
        if (self.currentImplMethodType(target_name, field_name)) |method_type| return method_type;
        var matched: ?TraitMethodSignature = null;
        for (self.impl_interfaces.items) |impl_interface| {
            if (!std.mem.eql(u8, impl_interface.target_name, target_name)) continue;
            for (impl_interface.methods) |method| {
                if (!std.mem.eql(u8, method.name, field_name) or method.receiver_kind != .value_self) continue;
                if (matched != null) return null;
                matched = method;
            }
        }
        return self.functionTypeFromTraitSignature(matched orelse return null);
    }

    fn currentImplMethodType(self: *const TypeChecker, target_name: []const u8, field_name: []const u8) ?Type {
        const function_item_id = self.current_function_item orelse return null;
        const impl_item = self.enclosingImplForMethod(function_item_id) orelse return null;
        if (!std.mem.eql(u8, impl_item.target_name, target_name)) return null;

        var matched: ?Type = null;
        for (impl_item.methods) |method_item_id| {
            const item = self.file.item(method_item_id).*;
            if (item != .Function) continue;
            const function = item.Function;
            if (!std.mem.eql(u8, function.name, field_name) or !self.functionHasBareSelf(function)) continue;
            if (matched != null) return null;
            matched = self.item_types[method_item_id.index()];
        }
        return matched;
    }

    fn traitInterfaceByName(self: *const TypeChecker, name: []const u8) ?TraitInterface {
        for (self.trait_interfaces.items) |trait_interface| {
            if (std.mem.eql(u8, trait_interface.name, name)) return trait_interface;
        }
        return null;
    }

    fn findTraitMethodSignature(self: *const TypeChecker, trait_interface: TraitInterface, name: []const u8) ?TraitMethodSignature {
        _ = self;
        for (trait_interface.methods) |method| {
            if (std.mem.eql(u8, method.name, name)) return method;
        }
        return null;
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

    fn inferCatchPatternType(self: *TypeChecker, body_id: ast.BodyId) anyerror!Type {
        var error_types: std.ArrayList(Type) = .{};
        try self.collectBodyErrorTypes(body_id, &error_types);
        if (error_types.items.len == 1) return error_types.items[0];
        return .{ .integer = .{ .bits = 256, .signed = false, .spelling = "u256" } };
    }

    fn collectBodyErrorTypes(self: *TypeChecker, body_id: ast.BodyId, error_types: *std.ArrayList(Type)) anyerror!void {
        const body = self.file.body(body_id).*;
        for (body.statements) |statement_id| {
            try self.collectStmtErrorTypes(statement_id, error_types);
        }
    }

    fn collectStmtErrorTypes(self: *TypeChecker, statement_id: ast.StmtId, error_types: *std.ArrayList(Type)) anyerror!void {
        switch (self.file.statement(statement_id).*) {
            .VariableDecl => |decl| if (decl.value) |expr_id| try self.collectExprErrorTypes(expr_id, error_types),
            .Return => |ret| if (ret.value) |expr_id| try self.collectExprErrorTypes(expr_id, error_types),
            .If => |if_stmt| {
                try self.collectExprErrorTypes(if_stmt.condition, error_types);
                try self.collectBodyErrorTypes(if_stmt.then_body, error_types);
                if (if_stmt.else_body) |else_body| try self.collectBodyErrorTypes(else_body, error_types);
            },
            .While => |while_stmt| {
                try self.collectExprErrorTypes(while_stmt.condition, error_types);
                for (while_stmt.invariants) |expr_id| try self.collectExprErrorTypes(expr_id, error_types);
                try self.collectBodyErrorTypes(while_stmt.body, error_types);
            },
            .For => |for_stmt| {
                try self.collectExprErrorTypes(for_stmt.iterable, error_types);
                if (for_stmt.range_end) |end_expr| try self.collectExprErrorTypes(end_expr, error_types);
                for (for_stmt.invariants) |expr_id| try self.collectExprErrorTypes(expr_id, error_types);
                try self.collectBodyErrorTypes(for_stmt.body, error_types);
            },
            .Switch => |switch_stmt| {
                try self.collectExprErrorTypes(switch_stmt.condition, error_types);
                for (switch_stmt.arms) |arm| {
                    switch (arm.pattern) {
                        .Expr => |expr_id| try self.collectExprErrorTypes(expr_id, error_types),
                        .Range => |range_pattern| {
                            try self.collectExprErrorTypes(range_pattern.start, error_types);
                            try self.collectExprErrorTypes(range_pattern.end, error_types);
                        },
                        .Else => {},
                    }
                    try self.collectBodyErrorTypes(arm.body, error_types);
                }
                if (switch_stmt.else_body) |else_body| try self.collectBodyErrorTypes(else_body, error_types);
            },
            .Try => |try_stmt| {
                try self.collectBodyErrorTypes(try_stmt.try_body, error_types);
                if (try_stmt.catch_clause) |catch_clause| try self.collectBodyErrorTypes(catch_clause.body, error_types);
            },
            .Log => |log_stmt| for (log_stmt.args) |arg| try self.collectExprErrorTypes(arg, error_types),
            .Lock => |lock_stmt| try self.collectExprErrorTypes(lock_stmt.path, error_types),
            .Unlock => |unlock_stmt| try self.collectExprErrorTypes(unlock_stmt.path, error_types),
            .Assert => |assert_stmt| try self.collectExprErrorTypes(assert_stmt.condition, error_types),
            .Assume => |assume_stmt| try self.collectExprErrorTypes(assume_stmt.condition, error_types),
            .Assign => |assign| try self.collectExprErrorTypes(assign.value, error_types),
            .Expr => |expr_stmt| try self.collectExprErrorTypes(expr_stmt.expr, error_types),
            .Block => |block| try self.collectBodyErrorTypes(block.body, error_types),
            .LabeledBlock => |block| try self.collectBodyErrorTypes(block.body, error_types),
            else => {},
        }
    }

    fn collectExprErrorTypes(self: *TypeChecker, expr_id: ast.ExprId, error_types: *std.ArrayList(Type)) anyerror!void {
        const expr_type = self.expr_types[expr_id.index()];
        if (expr_type.kind() == .error_union) {
            for (expr_type.errorTypes()) |error_type| try self.appendUniqueErrorType(error_types, error_type);
        }

        switch (self.file.expression(expr_id).*) {
            .Unary => |unary| try self.collectExprErrorTypes(unary.operand, error_types),
            .Binary => |binary| {
                try self.collectExprErrorTypes(binary.lhs, error_types);
                try self.collectExprErrorTypes(binary.rhs, error_types);
            },
            .Call => |call| {
                try self.collectExprErrorTypes(call.callee, error_types);
                for (call.args) |arg| try self.collectExprErrorTypes(arg, error_types);
            },
            .Builtin => |builtin| for (builtin.args) |arg| try self.collectExprErrorTypes(arg, error_types),
            .Field => |field| try self.collectExprErrorTypes(field.base, error_types),
            .Index => |index| {
                try self.collectExprErrorTypes(index.base, error_types);
                try self.collectExprErrorTypes(index.index, error_types);
            },
            .Tuple => |tuple| for (tuple.elements) |element| try self.collectExprErrorTypes(element, error_types),
            .ArrayLiteral => |array| for (array.elements) |element| try self.collectExprErrorTypes(element, error_types),
            .StructLiteral => |struct_literal| for (struct_literal.fields) |field| try self.collectExprErrorTypes(field.value, error_types),
            .Switch => |switch_expr| {
                try self.collectExprErrorTypes(switch_expr.condition, error_types);
                for (switch_expr.arms) |arm| {
                    switch (arm.pattern) {
                        .Expr => |pattern_expr| try self.collectExprErrorTypes(pattern_expr, error_types),
                        .Range => |range_pattern| {
                            try self.collectExprErrorTypes(range_pattern.start, error_types);
                            try self.collectExprErrorTypes(range_pattern.end, error_types);
                        },
                        .Else => {},
                    }
                    try self.collectExprErrorTypes(arm.value, error_types);
                }
                if (switch_expr.else_expr) |else_expr| try self.collectExprErrorTypes(else_expr, error_types);
            },
            .Comptime => |comptime_expr| {
                try self.collectBodyErrorTypes(comptime_expr.body, error_types);
            },
            .Group => |group| try self.collectExprErrorTypes(group.expr, error_types),
            .Quantified => |quantified| {
                if (quantified.condition) |condition| try self.collectExprErrorTypes(condition, error_types);
                try self.collectExprErrorTypes(quantified.body, error_types);
            },
            .ErrorReturn => |error_return| for (error_return.args) |arg| try self.collectExprErrorTypes(arg, error_types),
            else => {},
        }
    }

    fn appendUniqueErrorType(self: *TypeChecker, error_types: *std.ArrayList(Type), error_type: Type) anyerror!void {
        for (error_types.items) |existing| {
            if (typeEql(existing, error_type)) return;
        }
        try error_types.append(self.arena, error_type);
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
        const checked_value = if (exprUsesWrappedIntegerValue(self, expr_id))
            try const_bridge.wrapIntegerToType(self.arena, value.integer, expected_type.integer)
        else
            value.integer;
        if (integerValueFitsType(checked_value, expected_type.integer)) return false;
        const value_text = try self.integerValueText(checked_value);
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

    fn checkErrorReturn(self: *TypeChecker, expr_id: ast.ExprId, error_return: ast.ErrorReturnExpr) !Type {
        const item_id = self.item_index.lookup(error_return.name) orelse {
            try self.emitExprError(expr_id, "unknown error '{s}'", .{error_return.name});
            return .{ .unknown = {} };
        };
        if (self.file.item(item_id).* != .ErrorDecl) {
            try self.emitExprError(expr_id, "'{s}' is not an error declaration", .{error_return.name});
            return .{ .unknown = {} };
        }

        const error_decl = self.file.item(item_id).ErrorDecl;
        if (error_decl.parameters.len != error_return.args.len) {
            try self.emitExprError(expr_id, "error '{s}' expects {d} arguments, found {d}", .{
                error_decl.name,
                error_decl.parameters.len,
                error_return.args.len,
            });
        }

        const arg_count = @min(error_decl.parameters.len, error_return.args.len);
        for (error_decl.parameters[0..arg_count], error_return.args[0..arg_count]) |parameter, arg_id| {
            const expected_type = try self.resolveTypeExpr(parameter.type_expr);
            const actual_type = self.expr_types[arg_id.index()];
            if (try self.emitIntegerOverflowIfNeeded(error_return.range, arg_id, expected_type)) continue;
            if (expected_type.kind() == .unknown or actual_type.kind() == .unknown) continue;
            if (!typesAssignable(expected_type, actual_type)) {
                const parameter_name = self.patternName(parameter.pattern) orelse "<error payload>";
                try self.emitExprError(expr_id, "error '{s}' argument '{s}' expects type '{s}', found '{s}'", .{
                    error_decl.name,
                    parameter_name,
                    typeDisplayName(expected_type),
                    typeDisplayName(actual_type),
                });
            }
        }

        if (self.current_return_type) |return_type| {
            if (return_type.kind() == .error_union) {
                const error_type: Type = .{ .named = .{ .name = error_decl.name } };
                var allowed = false;
                for (return_type.errorTypes()) |declared_error_type| {
                    if (typeEql(declared_error_type, error_type)) {
                        allowed = true;
                        break;
                    }
                }
                if (!allowed) {
                    try self.emitExprError(expr_id, "error '{s}' is not in function return error set", .{
                        error_decl.name,
                    });
                }
            }
        }

        return .{ .named = .{ .name = error_decl.name } };
    }

    fn integerValueText(self: *TypeChecker, value: BigInt) ![]const u8 {
        return try value.toString(self.arena, 10, .lower);
    }

    fn emitGenericArityError(
        self: *TypeChecker,
        range: source.TextRange,
        kind: []const u8,
        name: []const u8,
        expected: usize,
        found: usize,
    ) !void {
        try self.emitRangeError(range, "generic {s} '{s}' expects {d} arguments, found {d}", .{
            kind,
            name,
            expected,
            found,
        });
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
                const field_type = self.fieldPatternType(field);
                break :blk .{
                    .type = field_type,
                    .region = base.region,
                    .provenance = if (base.region == .storage) .storage else base.provenance,
                };
            },
            .Index => |index| blk: {
                const base = self.patternLocatedType(index.base);
                const index_type = self.indexPatternType(index);
                break :blk .{
                    .type = index_type,
                    .region = base.region,
                    .provenance = if (base.region == .storage) .storage else base.provenance,
                };
            },
            .Error => LocatedType.unlocated(.{ .unknown = {} }),
        };
    }

    fn exprLocatedType(self: *const TypeChecker, expr_id: ast.ExprId) LocatedType {
        return switch (self.file.expression(expr_id).*) {
            // Located: region inherited from binding or base expression.
            .Name => self.locatedTypeForBinding(self.resolution.expr_bindings[expr_id.index()]),
            .ExternalProxy => LocatedType.unlocated(self.expr_types[expr_id.index()]),
            .Field => |field| blk: {
                const base = self.exprLocatedType(field.base);
                break :blk .{
                    .type = self.expr_types[expr_id.index()],
                    .region = base.region,
                    .provenance = if (base.region == .storage) .storage else base.provenance,
                };
            },
            .Index => |index| blk: {
                const base = self.exprLocatedType(index.base);
                break :blk .{
                    .type = self.expr_types[expr_id.index()],
                    .region = base.region,
                    .provenance = if (base.region == .storage) .storage else base.provenance,
                };
            },
            // Transparent: region passes through wrapper.
            .Group => |group| self.exprLocatedType(group.expr),
            .Old => |old| self.exprLocatedType(old.expr),
            // Unlocated: stack-computed values, comptime constants, or verification constructs.
            // These have no storage region — the value exists on the EVM stack or in bytecode.
            .IntegerLiteral, .StringLiteral, .BoolLiteral, .AddressLiteral, .BytesLiteral, .TypeValue, .Comptime, .Unary, .Binary, .Call => if (self.externProxyMethodSignature(expr_id) != null)
                LocatedType.withRegionAndProvenance(self.expr_types[expr_id.index()], .none, .external)
            else
                LocatedType.unlocated(self.expr_types[expr_id.index()]),
            .Builtin,
            .Tuple,
            .ArrayLiteral,
            .StructLiteral,
            .Switch,
            .ErrorReturn,
            .Result,
            .Quantified,
            .Error,
            => LocatedType.unlocated(self.expr_types[expr_id.index()]),
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
            .provenance = if (self.item_regions[item_id.index()] == .storage) .storage else .local,
        };
    }
};

fn overflowTupleFieldType(base_type: Type, field_name: []const u8) ?Type {
    if (base_type.kind() != .tuple) return null;
    const elements = base_type.tuple;
    if (elements.len != 2) return null;
    if (elements[1].kind() != .bool) return null;
    if (std.mem.eql(u8, field_name, "value")) return elements[0];
    if (std.mem.eql(u8, field_name, "overflow")) return elements[1];
    return null;
}

fn anonymousStructFieldType(base_type: Type, field_name: []const u8) ?Type {
    if (base_type.kind() != .anonymous_struct) return null;
    for (base_type.anonymous_struct.fields) |field| {
        if (std.mem.eql(u8, field.name, field_name)) return field.ty;
    }
    return null;
}

fn arithmeticResultType(lhs_type: Type, rhs_type: Type) Type {
    const lhs = unwrapRefinement(lhs_type);
    const rhs = unwrapRefinement(rhs_type);
    if (isGenericTypeParam(lhs) and isGenericTypeParam(rhs) and sameConcreteType(lhs, rhs)) return lhs;
    if (!isIntegerType(lhs) or !isIntegerType(rhs)) return .{ .unknown = {} };
    if (sameConcreteType(lhs, rhs)) return lhs;
    return lhs;
}

fn bitwiseResultType(lhs_type: Type, rhs_type: Type) Type {
    const lhs = unwrapRefinement(lhs_type);
    const rhs = unwrapRefinement(rhs_type);
    if (isGenericTypeParam(lhs) and isGenericTypeParam(rhs) and sameConcreteType(lhs, rhs)) return lhs;
    if (!isIntegerType(lhs) or !isIntegerType(rhs)) return .{ .unknown = {} };
    return lhs;
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
    const lhs = unwrapRefinement(lhs_type);
    const rhs = unwrapRefinement(rhs_type);
    if (sameConcreteType(lhs, rhs)) return true;
    if (isIntegerType(lhs) and isIntegerType(rhs)) return true;
    return false;
}

fn orderedTypesComparable(lhs_type: Type, rhs_type: Type) bool {
    const lhs = unwrapRefinement(lhs_type);
    const rhs = unwrapRefinement(rhs_type);
    if (isGenericTypeParam(lhs) and isGenericTypeParam(rhs) and sameConcreteType(lhs, rhs)) return true;
    if (lhs.kind() == .bool or rhs.kind() == .bool) return false;
    return typesComparable(lhs_type, rhs_type);
}

fn isIntegerType(ty: Type) bool {
    return unwrapRefinement(ty).kind() == .integer;
}

fn unwrapRefinement(ty: Type) Type {
    return if (ty.refinementBaseType()) |base| base.* else ty;
}

fn isGenericTypeParam(ty: Type) bool {
    return unwrapRefinement(ty).kind() == .named;
}

fn exprIsWrappingOp(self: *const TypeChecker, expr_id: ast.ExprId) bool {
    return switch (self.file.expression(expr_id).*) {
        .Group => |group| exprIsWrappingOp(self, group.expr),
        .Binary => |binary| switch (binary.op) {
            .wrapping_add, .wrapping_sub, .wrapping_mul, .wrapping_pow, .wrapping_shl, .wrapping_shr => true,
            else => false,
        },
        else => false,
    };
}

fn exprUsesWrappedIntegerValue(self: *const TypeChecker, expr_id: ast.ExprId) bool {
    if (exprIsWrappingOp(self, expr_id)) return true;
    return switch (self.file.expression(expr_id).*) {
        .Group => |group| exprUsesWrappedIntegerValue(self, group.expr),
        .Name => if (self.resolution.expr_bindings[expr_id.index()]) |binding| switch (binding) {
            .pattern => |pattern_id| if (self.initializerExprForPattern(pattern_id)) |init_expr|
                exprUsesWrappedIntegerValue(self, init_expr)
            else
                false,
            else => false,
        } else false,
        else => false,
    };
}

fn integerValueFitsType(value: BigInt, integer: model.IntegerType) bool {
    const bits = integer.bits orelse return true;
    const signed = integer.signed orelse return true;
    if (bits == 0) return value.eqlZero();
    return value.fitsInTwosComp(if (signed) .signed else .unsigned, bits);
}

fn sameConcreteType(lhs_type: Type, rhs_type: Type) bool {
    if (lhs_type.kind() != rhs_type.kind()) return false;
    return switch (lhs_type) {
        .unknown, .void, .bool, .string, .address, .bytes => true,
        .external_proxy => |left| std.mem.eql(u8, left.trait_name, rhs_type.external_proxy.trait_name),
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
        .and_and => "and",
        .or_or => "or",
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
        .external_proxy => "external proxy",
        .named => |named| named.name,
        .function => |function| function.name orelse "function",
        .contract => |named| named.name,
        .struct_ => |named| named.name,
        .bitfield => |named| named.name,
        .enum_ => |named| named.name,
        .anonymous_struct => "anonymous struct",
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

test "typesAssignable widens error unions by error-set inclusion" {
    const testing = std.testing;

    const payload_a = try testing.allocator.create(Type);
    defer testing.allocator.destroy(payload_a);
    payload_a.* = .{ .integer = .{ .bits = 256, .signed = false, .spelling = "u256" } };

    const payload_b = try testing.allocator.create(Type);
    defer testing.allocator.destroy(payload_b);
    payload_b.* = .{ .integer = .{ .bits = 256, .signed = false, .spelling = "u256" } };

    const narrow_union: Type = .{ .error_union = .{
        .payload_type = payload_a,
        .error_types = &.{
            .{ .named = .{ .name = "ErrorA" } },
        },
    } };
    const wide_union: Type = .{ .error_union = .{
        .payload_type = payload_b,
        .error_types = &.{
            .{ .named = .{ .name = "ErrorA" } },
            .{ .named = .{ .name = "ErrorB" } },
        },
    } };

    try testing.expect(typesAssignable(wide_union, narrow_union));
    try testing.expect(!typesAssignable(narrow_union, wide_union));
    try testing.expect(typesAssignable(wide_union, .{ .named = .{ .name = "ErrorA" } }));
}

test "typesAssignable rejects integer narrowing and accepts widening" {
    const testing = std.testing;

    const u8_type: Type = .{ .integer = .{ .bits = 8, .signed = false, .spelling = "u8" } };
    const u16_type: Type = .{ .integer = .{ .bits = 16, .signed = false, .spelling = "u16" } };
    const u256_type: Type = .{ .integer = .{ .bits = 256, .signed = false, .spelling = "u256" } };

    try testing.expect(typesAssignable(u16_type, u8_type));
    try testing.expect(typesAssignable(u256_type, u16_type));
    try testing.expect(!typesAssignable(u8_type, u16_type));
    try testing.expect(!typesAssignable(u8_type, u256_type));
}

test "typesFlowCompatible accepts guardable refinement strengthening" {
    const testing = std.testing;

    const base = try testing.allocator.create(Type);
    defer testing.allocator.destroy(base);
    base.* = .{ .integer = .{ .bits = 256, .signed = false, .spelling = "u256" } };

    const min100: Type = .{ .refinement = .{
        .name = "MinValue",
        .base_type = base,
        .args = &.{
            ast.TypeArg{ .Type = ast.TypeExprId{ .value = 0 } },
            ast.TypeArg{ .Integer = .{ .range = undefined, .text = "100" } },
        },
    } };
    const min200: Type = .{ .refinement = .{
        .name = "MinValue",
        .base_type = base,
        .args = &.{
            ast.TypeArg{ .Type = ast.TypeExprId{ .value = 0 } },
            ast.TypeArg{ .Integer = .{ .range = undefined, .text = "200" } },
        },
    } };
    const range_wide: Type = .{ .refinement = .{
        .name = "InRange",
        .base_type = base,
        .args = &.{
            ast.TypeArg{ .Type = ast.TypeExprId{ .value = 0 } },
            ast.TypeArg{ .Integer = .{ .range = undefined, .text = "0" } },
            ast.TypeArg{ .Integer = .{ .range = undefined, .text = "10000" } },
        },
    } };
    const range_narrow: Type = .{ .refinement = .{
        .name = "InRange",
        .base_type = base,
        .args = &.{
            ast.TypeArg{ .Type = ast.TypeExprId{ .value = 0 } },
            ast.TypeArg{ .Integer = .{ .range = undefined, .text = "100" } },
            ast.TypeArg{ .Integer = .{ .range = undefined, .text = "5000" } },
        },
    } };

    try testing.expect(typesAssignable(min100, min200));
    try testing.expect(!typesAssignable(min200, min100));
    try testing.expect(typesAssignable(range_wide, range_narrow));
    try testing.expect(!typesAssignable(range_narrow, range_wide));

    try testing.expect(typesFlowCompatible(min100, min200));
    try testing.expect(typesFlowCompatible(min200, min100));
    try testing.expect(typesFlowCompatible(range_wide, range_narrow));
    try testing.expect(typesFlowCompatible(range_narrow, range_wide));
}

test "typesAssignable accepts semantically identical refinements from distinct sites" {
    const testing = std.testing;

    const base_a = try testing.allocator.create(Type);
    defer testing.allocator.destroy(base_a);
    base_a.* = .{ .integer = .{ .bits = 256, .signed = false, .spelling = "u256" } };

    const base_b = try testing.allocator.create(Type);
    defer testing.allocator.destroy(base_b);
    base_b.* = .{ .integer = .{ .bits = 256, .signed = false, .spelling = "u256" } };

    const lhs: Type = .{ .refinement = .{
        .name = "MinValue",
        .base_type = base_a,
        .args = &.{
            ast.TypeArg{ .Type = ast.TypeExprId{ .value = 1 } },
            ast.TypeArg{ .Integer = .{ .range = undefined, .text = "1" } },
        },
    } };
    const rhs: Type = .{ .refinement = .{
        .name = "MinValue",
        .base_type = base_b,
        .args = &.{
            ast.TypeArg{ .Type = ast.TypeExprId{ .value = 2 } },
            ast.TypeArg{ .Integer = .{ .range = undefined, .text = "1" } },
        },
    } };

    try testing.expect(typesAssignable(lhs, rhs));
    try testing.expect(typesAssignable(rhs, lhs));
}
