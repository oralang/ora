const std = @import("std");
const mlir = @import("mlir_c_api").c;
const ast = @import("../ast/mod.zig");
const sema = @import("../sema/mod.zig");
const source = @import("../source/mod.zig");
const support = @import("support.zig");

const appendOp = support.appendOp;
const createIntegerConstant = support.createIntegerConstant;
const defaultIntegerType = support.defaultIntegerType;
const identifier = support.identifier;
const namedBoolAttr = support.namedBoolAttr;
const namedStringAttr = support.namedStringAttr;
const namedTypeAttr = support.namedTypeAttr;
const strRef = support.strRef;
const zeroInitAttr = support.zeroInitAttr;

pub fn mixin(Lowerer: type, ContractLowerer: type, FunctionLowerer: type, HirSymbolKind: type) type {
    return struct {
        pub fn lowerItem(self: *Lowerer, item_id: ast.ItemId, parent_block: mlir.MlirBlock) anyerror!void {
            const item = self.file.item(item_id).*;
            switch (item) {
                .Import => |import_item| {
                    const op = try self.createPlaceholderOp(
                        "ora.import_decl",
                        self.location(import_item.range),
                        &.{
                            namedStringAttr(self.context, "ora.import_path", import_item.path),
                            namedBoolAttr(self.context, "ora.is_comptime", import_item.is_comptime),
                        },
                    );
                    appendOp(parent_block, op);
                },
                .Contract => |contract| try self.lowerContract(item_id, contract, parent_block),
                .Function => |function| try self.lowerFunction(item_id, function, parent_block),
                .Struct => |struct_item| try self.lowerStructDecl(item_id, struct_item, parent_block),
                .Bitfield => |bitfield| {
                    if (bitfield.is_generic) return;
                    try self.lowerDeclPlaceholder(item_id, .bitfield, bitfield.name, bitfield.range, "ora.bitfield_decl", parent_block);
                },
                .Enum => |enum_item| try self.lowerEnumDecl(item_id, enum_item, parent_block),
                .Trait => {},
                .Impl => |impl_item| try @This().lowerImpl(self, item_id, impl_item, parent_block),
                .TypeAlias => {},
                .LogDecl => |log_decl| try self.lowerLogDecl(item_id, log_decl, parent_block),
                .ErrorDecl => |error_decl| try self.lowerErrorDecl(item_id, error_decl, parent_block),
                .GhostBlock => |ghost_block| try @This().lowerGhostBlock(self, ghost_block, parent_block),
                .Field => |field| try self.lowerField(item_id, field, parent_block),
                .Constant => |constant| try self.lowerConstant(item_id, constant, parent_block),
                .Error => {},
            }
        }

        pub fn lowerContract(self: *Lowerer, item_id: ast.ItemId, contract: ast.ContractItem, parent_block: mlir.MlirBlock) anyerror!void {
            if (contract.is_generic) return;
            const op = mlir.oraContractOpCreate(self.context, self.location(contract.range), strRef(contract.name));
            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
            appendOp(parent_block, op);
            try self.appendItemHandle(item_id, .contract, contract.name, contract.range, op);

            const body = mlir.oraContractOpGetBodyBlock(op);
            if (mlir.oraBlockIsNull(body)) return error.MlirOperationCreationFailed;
            self.contract_body_blocks[item_id.index()] = body;

            var contract_lowerer = ContractLowerer{
                .parent = self,
                .block = body,
            };

            var guarded_storage_roots = try collectContractLockedStorageRoots(self, contract);
            defer guarded_storage_roots.deinit();
            const previous_guarded_roots = self.guarded_storage_roots;
            self.guarded_storage_roots = &guarded_storage_roots;
            defer self.guarded_storage_roots = previous_guarded_roots;

            for (contract.invariants) |expr_id| {
                try contract_lowerer.lowerInvariant(expr_id);
            }
            for (contract.members) |member_id| {
                try self.lowerItem(member_id, body);
            }
        }

        pub fn lowerFunction(self: *Lowerer, item_id: ast.ItemId, function: ast.FunctionItem, parent_block: mlir.MlirBlock) anyerror!void {
            if (function.is_generic) return;
            const parameters = try self.runtimeFunctionParameters(function);
            try @This().lowerConcreteFunction(self, item_id, function, function.name, parameters, parent_block, &.{});
        }

        pub fn lowerImpl(self: *Lowerer, impl_item_id: ast.ItemId, impl_item: ast.ImplItem, parent_block: mlir.MlirBlock) anyerror!void {
            _ = parent_block;
            const impl_parent_block = if (self.item_index.lookup(impl_item.target_name)) |target_item_id| blk: {
                if (self.file.item(target_item_id).* == .Contract) {
                    const block = self.contract_body_blocks[target_item_id.index()];
                    if (!mlir.oraBlockIsNull(block)) break :blk block;
                }
                break :blk self.module_body;
            } else self.module_body;

            for (impl_item.methods) |method_item_id| {
                const function = switch (self.file.item(method_item_id).*) {
                    .Function => |function| function,
                    else => continue,
                };
                if (function.is_generic) continue;
                const symbol_name = try @This().implMethodSymbolName(self, impl_item.target_name, function.name);
                if (self.monomorphized_function_names.contains(symbol_name)) continue;
                try @This().lowerConcreteFunction(self, method_item_id, function, symbol_name, function.parameters, impl_parent_block, &.{});
                try self.monomorphized_function_names.put(symbol_name, {});
                _ = impl_item_id;
            }
        }

        pub fn lowerInstantiatedStructDecl(self: *Lowerer, instantiated: sema.InstantiatedStruct, parent_block: mlir.MlirBlock) anyerror!void {
            const template_item = self.file.item(instantiated.template_item_id).Struct;
            const loc = self.location(template_item.range);
            const op = mlir.oraStructDeclOpCreate(self.context, loc, strRef(instantiated.mangled_name));
            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;

            if (instantiated.fields.len > 0) {
                const field_names = try self.allocator.alloc(mlir.MlirAttribute, instantiated.fields.len);
                const field_types = try self.allocator.alloc(mlir.MlirAttribute, instantiated.fields.len);
                for (instantiated.fields, 0..) |field, index| {
                    field_names[index] = mlir.oraStringAttrCreate(self.context, strRef(field.name));
                    field_types[index] = mlir.oraTypeAttrCreateFromType(self.lowerSemaType(field.ty, template_item.range));
                }
                mlir.oraOperationSetAttributeByName(op, strRef("ora.field_names"), mlir.oraArrayAttrCreate(self.context, @intCast(field_names.len), field_names.ptr));
                mlir.oraOperationSetAttributeByName(op, strRef("ora.field_types"), mlir.oraArrayAttrCreate(self.context, @intCast(field_types.len), field_types.ptr));
            }
            mlir.oraOperationSetAttributeByName(op, strRef("ora.struct_decl"), mlir.oraBoolAttrCreate(self.context, true));

            appendOp(parent_block, op);
        }

        pub fn lowerInstantiatedEnumDecl(self: *Lowerer, instantiated: sema.InstantiatedEnum, parent_block: mlir.MlirBlock) anyerror!void {
            const template_item = self.file.item(instantiated.template_item_id).Enum;
            const loc = self.location(template_item.range);
            const repr_type = defaultIntegerType(self.context);
            const op = mlir.oraEnumDeclOpCreate(self.context, loc, strRef(instantiated.mangled_name), repr_type);
            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;

            if (template_item.variants.len > 0) {
                const variant_names = try self.allocator.alloc(mlir.MlirAttribute, template_item.variants.len);
                const variant_values = try self.allocator.alloc(mlir.MlirAttribute, template_item.variants.len);
                for (template_item.variants, 0..) |variant, index| {
                    variant_names[index] = mlir.oraStringAttrCreate(self.context, strRef(variant.name));
                    variant_values[index] = mlir.oraIntegerAttrCreateI64FromType(repr_type, @intCast(index));
                }
                mlir.oraOperationSetAttributeByName(op, strRef("ora.variant_names"), mlir.oraArrayAttrCreate(self.context, @intCast(variant_names.len), variant_names.ptr));
                mlir.oraOperationSetAttributeByName(op, strRef("ora.variant_values"), mlir.oraArrayAttrCreate(self.context, @intCast(variant_values.len), variant_values.ptr));
            }
            mlir.oraOperationSetAttributeByName(op, strRef("ora.enum_decl"), mlir.oraBoolAttrCreate(self.context, true));
            mlir.oraOperationSetAttributeByName(op, strRef("ora.has_explicit_values"), mlir.oraBoolAttrCreate(self.context, false));

            appendOp(parent_block, op);
        }

        pub fn lowerInstantiatedBitfieldDecl(self: *Lowerer, instantiated: sema.InstantiatedBitfield, parent_block: mlir.MlirBlock) anyerror!void {
            const template_item = self.file.item(instantiated.template_item_id).Bitfield;
            const op = try self.createNamedPlaceholderOp("ora.bitfield_decl", instantiated.mangled_name, template_item.range, mlir.oraNoneTypeCreate(self.context));
            if (try self.bitfieldMetadataForType(.{ .bitfield = .{ .name = instantiated.mangled_name } })) |metadata| {
                mlir.oraOperationSetAttributeByName(op, strRef("ora.bitfield"), mlir.oraStringAttrCreate(self.context, strRef(metadata.name)));
                if (metadata.layout.len > 0) {
                    mlir.oraOperationSetAttributeByName(op, strRef("ora.bitfield_layout"), mlir.oraStringAttrCreate(self.context, strRef(metadata.layout)));
                }
            }
            appendOp(parent_block, op);
        }

        pub fn ensureMonomorphizedFunction(self: *Lowerer, item_id: ast.ItemId, function: ast.FunctionItem, call: ast.CallExpr, parameters: []const ast.Parameter) anyerror!?[]const u8 {
            if (!function.is_generic) return function.name;
            const bindings = (try self.genericTypeBindingsForCall(function, call)) orelse return null;
            const mangled_name = try self.mangleGenericFunctionName(function.name, bindings);
            if (!self.monomorphized_function_names.contains(mangled_name)) {
                const parent_block = if (function.parent_contract) |contract_id|
                    self.contract_body_blocks[contract_id.index()]
                else
                    self.module_body;
                try @This().lowerConcreteFunction(self, item_id, function, mangled_name, parameters, parent_block, bindings);
                try self.monomorphized_function_names.put(mangled_name, {});
            }
            return mangled_name;
        }

        pub fn ensureLoweredImplMethod(
            self: *Lowerer,
            impl_item_id: ast.ItemId,
            method_item_id: ast.ItemId,
            function: ast.FunctionItem,
            target_name: []const u8,
            call: ?ast.CallExpr,
        ) anyerror![]const u8 {
            const base_symbol_name = try @This().implMethodSymbolName(self, target_name, function.name);
            if (function.is_generic) {
                const method_call = call orelse return base_symbol_name;
                const bindings = (try self.genericTypeBindingsForCall(function, method_call)) orelse return base_symbol_name;
                const runtime_parameters = try self.runtimeFunctionParameters(function);
                const symbol_name = try self.mangleGenericFunctionName(base_symbol_name, bindings);
                if (!self.monomorphized_function_names.contains(symbol_name)) {
                    const parent_block = if (self.item_index.lookup(target_name)) |target_item_id| blk: {
                        if (self.file.item(target_item_id).* == .Contract) {
                            const block = self.contract_body_blocks[target_item_id.index()];
                            if (!mlir.oraBlockIsNull(block)) break :blk block;
                        }
                        break :blk self.module_body;
                    } else self.module_body;
                    try @This().lowerConcreteFunction(self, method_item_id, function, symbol_name, runtime_parameters, parent_block, bindings);
                    try self.monomorphized_function_names.put(symbol_name, {});
                    _ = impl_item_id;
                }
                return symbol_name;
            }

            const symbol_name = base_symbol_name;
            if (!self.monomorphized_function_names.contains(symbol_name)) {
                const parent_block = if (self.item_index.lookup(target_name)) |target_item_id| blk: {
                    if (self.file.item(target_item_id).* == .Contract) {
                        const block = self.contract_body_blocks[target_item_id.index()];
                        if (!mlir.oraBlockIsNull(block)) break :blk block;
                    }
                    break :blk self.module_body;
                } else self.module_body;
                try @This().lowerConcreteFunction(self, method_item_id, function, symbol_name, function.parameters, parent_block, &.{});
                try self.monomorphized_function_names.put(symbol_name, {});
                _ = impl_item_id;
            }
            return symbol_name;
        }

        fn implMethodSymbolName(self: *Lowerer, target_name: []const u8, method_name: []const u8) anyerror![]const u8 {
            var name = std.ArrayList(u8){};
            try name.appendSlice(self.allocator, target_name);
            try name.appendSlice(self.allocator, ".");
            try name.appendSlice(self.allocator, method_name);
            return name.toOwnedSlice(self.allocator);
        }

        fn lowerConcreteFunction(
            self: *Lowerer,
            item_id: ast.ItemId,
            function: ast.FunctionItem,
            symbol_name: []const u8,
            parameters: []const ast.Parameter,
            parent_block: mlir.MlirBlock,
            type_bindings: []const Lowerer.GenericTypeBinding,
        ) anyerror!void {
            const previous_type_bindings = self.active_type_bindings;
            self.active_type_bindings = type_bindings;
            defer self.active_type_bindings = previous_type_bindings;

            var attrs: std.ArrayList(mlir.MlirNamedAttribute) = .{};
            const return_type = if (function.return_type) |_| blk: {
                if (type_bindings.len == 0) {
                    break :blk self.lowerSemaType(self.typecheck.body_types[function.body.index()], function.range);
                }
                break :blk self.lowerSemaType(self.typecheck.body_types[function.body.index()], function.range);
            } else null;

            try attrs.append(self.allocator, namedStringAttr(self.context, "sym_name", symbol_name));
            try attrs.append(self.allocator, namedStringAttr(
                self.context,
                "ora.visibility",
                switch (function.visibility) {
                    .public => "pub",
                    .private => "private",
                },
            ));

            var param_types: std.ArrayList(mlir.MlirType) = .{};
            var param_locs: std.ArrayList(mlir.MlirLocation) = .{};
            for (parameters) |parameter| {
                const param_type = if (type_bindings.len == 0)
                    self.lowerSemaType(self.typecheck.pattern_types[parameter.pattern.index()].type, parameter.range)
                else
                    self.lowerTypeExpr(parameter.type_expr);
                try param_types.append(self.allocator, param_type);
                try param_locs.append(self.allocator, self.location(parameter.range));
            }

            var result_types: [1]mlir.MlirType = undefined;
            const fn_type = mlir.oraBuiltinFunctionTypeGet(
                self.context,
                param_types.items.len,
                if (param_types.items.len == 0) null else param_types.items.ptr,
                if (return_type == null) 0 else 1,
                if (return_type) |ty| blk: {
                    result_types[0] = ty;
                    break :blk &result_types;
                } else null,
            );
            try attrs.append(self.allocator, namedTypeAttr(self.context, "function_type", fn_type));

            const op = mlir.oraFuncFuncOpCreate(
                self.context,
                self.location(function.range),
                if (attrs.items.len == 0) null else attrs.items.ptr,
                attrs.items.len,
                if (param_types.items.len == 0) null else param_types.items.ptr,
                if (param_locs.items.len == 0) null else param_locs.items.ptr,
                param_types.items.len,
            );
            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
            if (function.is_ghost) @This().attachGhostAttrs(self, op, "ghost_function");

            for (parameters, 0..) |parameter, index| {
                try self.attachBitfieldParamMetadataForType(op, self.typecheck.pattern_types[parameter.pattern.index()].type, @intCast(index));
            }

            appendOp(parent_block, op);
            try self.appendItemHandle(item_id, .function, symbol_name, function.range, op);

            var specialized_function = function;
            specialized_function.name = symbol_name;
            specialized_function.is_generic = false;
            specialized_function.parameters = @constCast(parameters);
            var function_lowerer = FunctionLowerer.init(self, item_id, specialized_function, op, return_type);
            try function_lowerer.lower();
        }

        pub fn lowerField(self: *Lowerer, item_id: ast.ItemId, field: ast.FieldItem, parent_block: mlir.MlirBlock) anyerror!void {
            const loc = self.location(field.range);
            const ty = if (field.type_expr) |_|
                self.lowerSemaType(self.typecheck.item_types[item_id.index()], field.range)
            else
                defaultIntegerType(self.context);

            if (field.binding_kind == .immutable) {
                const op = if (field.value) |expr_id| blk: {
                    var function_lowerer = FunctionLowerer.initContractContext(self, parent_block);
                    const value = try function_lowerer.lowerExpr(expr_id, &function_lowerer.locals);
                    const created = mlir.oraImmutableOpCreate(self.context, loc, strRef(field.name), value, ty);
                    if (mlir.oraOperationIsNull(created)) {
                        break :blk try self.createNamedPlaceholderOp("ora.immutable_decl", field.name, field.range, ty);
                    }
                    break :blk created;
                } else try self.createNamedPlaceholderOp("ora.immutable_decl", field.name, field.range, ty);
                if (field.is_ghost) @This().attachGhostAttrs(self, op, "ghost_variable");

                if (field.type_expr) |_| {
                    try self.attachBitfieldOpMetadataForType(op, self.typecheck.item_types[item_id.index()]);
                }

                appendOp(parent_block, op);
                try self.appendItemHandle(item_id, .field, field.name, field.range, op);
                return;
            }

            const op = switch (field.storage_class) {
                .storage => blk: {
                    const init_attr = zeroInitAttr(ty);
                    const created = mlir.oraGlobalOpCreate(self.context, loc, strRef(field.name), ty, init_attr);
                    if (mlir.oraOperationIsNull(created)) {
                        break :blk try self.createNamedPlaceholderOp("ora.storage_field_decl", field.name, field.range, ty);
                    }
                    break :blk created;
                },
                .memory => blk: {
                    const created = mlir.oraMemoryGlobalOpCreate(self.context, loc, strRef(field.name), ty);
                    if (mlir.oraOperationIsNull(created)) {
                        break :blk try self.createNamedPlaceholderOp("ora.memory_field_decl", field.name, field.range, ty);
                    }
                    break :blk created;
                },
                .tstore => blk: {
                    const created = mlir.oraTStoreGlobalOpCreate(self.context, loc, strRef(field.name), ty);
                    if (mlir.oraOperationIsNull(created)) {
                        break :blk try self.createNamedPlaceholderOp("ora.tstore_field_decl", field.name, field.range, ty);
                    }
                    break :blk created;
                },
                .none => try self.createNamedPlaceholderOp("ora.field_decl", field.name, field.range, ty),
            };
            if (field.is_ghost) @This().attachGhostAttrs(self, op, "ghost_variable");

            if (field.type_expr) |_| {
                try self.attachBitfieldOpMetadataForType(op, self.typecheck.item_types[item_id.index()]);
            }

            appendOp(parent_block, op);
            try self.appendItemHandle(item_id, .field, field.name, field.range, op);
        }

        pub fn lowerConstant(self: *Lowerer, item_id: ast.ItemId, constant: ast.ConstantItem, parent_block: mlir.MlirBlock) anyerror!void {
            const expr = self.file.expression(constant.value).*;
            const declared_type = if (constant.type_expr) |_|
                self.lowerSemaType(self.typecheck.item_types[item_id.index()], constant.range)
            else
                self.lowerExprType(constant.value);
            const result_type = if (mlir.oraTypeIsAddressType(declared_type))
                mlir.oraIntegerTypeCreate(self.context, 160)
            else
                declared_type;
            if (self.const_eval.values[constant.value.index()]) |value| {
                if (try @This().constValueAttr(self, value, result_type)) |value_attr| {
                    const created = mlir.oraConstOpCreate(self.context, self.location(constant.range), strRef(constant.name), value_attr, result_type);
                    if (!mlir.oraOperationIsNull(created)) {
                        if (constant.is_ghost) @This().attachGhostAttrs(self, created, "ghost_constant");
                        appendOp(parent_block, created);
                        try self.appendItemHandle(item_id, .constant, constant.name, constant.range, created);
                        return;
                    }
                }
            }
            const op = switch (expr) {
                .IntegerLiteral => |literal| blk: {
                    const parsed = support.parseIntLiteral(literal.text) orelse 0;
                    const value_attr = mlir.oraIntegerAttrCreateI64FromType(result_type, parsed);
                    const created = mlir.oraConstOpCreate(self.context, self.location(constant.range), strRef(constant.name), value_attr, result_type);
                    if (mlir.oraOperationIsNull(created)) {
                        break :blk try self.createNamedPlaceholderOp("ora.constant_decl", constant.name, constant.range, result_type);
                    }
                    break :blk created;
                },
                .BoolLiteral => |literal| blk: {
                    const value_attr = mlir.oraBoolAttrCreate(self.context, literal.value);
                    const created = mlir.oraConstOpCreate(self.context, self.location(constant.range), strRef(constant.name), value_attr, result_type);
                    if (mlir.oraOperationIsNull(created)) {
                        break :blk try self.createNamedPlaceholderOp("ora.constant_decl", constant.name, constant.range, result_type);
                    }
                    break :blk created;
                },
                .StringLiteral => |literal| blk: {
                    const value_attr = mlir.oraStringAttrCreate(self.context, strRef(literal.text));
                    const created = mlir.oraConstOpCreate(self.context, self.location(constant.range), strRef(constant.name), value_attr, result_type);
                    if (mlir.oraOperationIsNull(created)) {
                        break :blk try self.createNamedPlaceholderOp("ora.constant_decl", constant.name, constant.range, result_type);
                    }
                    break :blk created;
                },
                .BytesLiteral => |literal| blk: {
                    const value_attr = mlir.oraStringAttrCreate(self.context, strRef(literal.text));
                    const created = mlir.oraConstOpCreate(self.context, self.location(constant.range), strRef(constant.name), value_attr, result_type);
                    if (mlir.oraOperationIsNull(created)) {
                        break :blk try self.createNamedPlaceholderOp("ora.constant_decl", constant.name, constant.range, result_type);
                    }
                    break :blk created;
                },
                .AddressLiteral => |literal| blk: {
                    const trimmed = if (std.mem.startsWith(u8, literal.text, "0x")) literal.text[2..] else literal.text;
                    const parsed = std.fmt.parseInt(u256, trimmed, 16) catch {
                        break :blk try self.createNamedPlaceholderOp("ora.constant_decl", constant.name, constant.range, result_type);
                    };
                    var decimal_buf: [80]u8 = undefined;
                    const decimal_text = std.fmt.bufPrint(&decimal_buf, "{}", .{parsed}) catch {
                        break :blk try self.createNamedPlaceholderOp("ora.constant_decl", constant.name, constant.range, result_type);
                    };
                    const value_attr = mlir.oraIntegerAttrGetFromString(result_type, strRef(decimal_text));
                    if (mlir.oraAttributeIsNull(value_attr)) {
                        break :blk try self.createNamedPlaceholderOp("ora.constant_decl", constant.name, constant.range, result_type);
                    }
                    const created = mlir.oraConstOpCreate(self.context, self.location(constant.range), strRef(constant.name), value_attr, result_type);
                    if (mlir.oraOperationIsNull(created)) {
                        break :blk try self.createNamedPlaceholderOp("ora.constant_decl", constant.name, constant.range, result_type);
                    }
                    break :blk created;
                },
                else => try self.createNamedPlaceholderOp("ora.constant_decl", constant.name, constant.range, result_type),
            };
            if (constant.is_ghost) @This().attachGhostAttrs(self, op, "ghost_constant");
            appendOp(parent_block, op);
            try self.appendItemHandle(item_id, .constant, constant.name, constant.range, op);
        }

        pub fn lowerGhostBlock(self: *Lowerer, ghost_block: ast.GhostBlockItem, parent_block: mlir.MlirBlock) anyerror!void {
            var function_lowerer = FunctionLowerer.initContractContext(self, parent_block);
            function_lowerer.in_ghost_context = true;
            var locals = try function_lowerer.cloneLocals(&function_lowerer.locals);
            _ = try function_lowerer.lowerBody(ghost_block.body, &locals);
        }

        fn constValueAttr(self: *Lowerer, value: sema.ConstValue, result_type: mlir.MlirType) anyerror!?mlir.MlirAttribute {
            return switch (value) {
                .integer => |integer| blk: {
                    if (integer.toInt(i64)) |small| {
                        break :blk mlir.oraIntegerAttrCreateI64FromType(result_type, small);
                    } else |_| {}

                    const text = try integer.toString(self.allocator, 10, .lower);
                    break :blk mlir.oraIntegerAttrGetFromString(result_type, strRef(text));
                },
                .boolean => |boolean| mlir.oraBoolAttrCreate(self.context, boolean),
                .string => |text| mlir.oraStringAttrCreate(self.context, strRef(text)),
            };
        }

        pub fn lowerStructDecl(self: *Lowerer, item_id: ast.ItemId, struct_item: ast.StructItem, parent_block: mlir.MlirBlock) anyerror!void {
            if (struct_item.is_generic) return;
            const loc = self.location(struct_item.range);
            const op = mlir.oraStructDeclOpCreate(self.context, loc, strRef(struct_item.name));
            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;

            if (struct_item.fields.len > 0) {
                const field_names = try self.allocator.alloc(mlir.MlirAttribute, struct_item.fields.len);
                const field_types = try self.allocator.alloc(mlir.MlirAttribute, struct_item.fields.len);
                for (struct_item.fields, 0..) |field, index| {
                    field_names[index] = mlir.oraStringAttrCreate(self.context, strRef(field.name));
                    field_types[index] = mlir.oraTypeAttrCreateFromType(self.lowerTypeExpr(field.type_expr));
                }
                mlir.oraOperationSetAttributeByName(op, strRef("ora.field_names"), mlir.oraArrayAttrCreate(self.context, @intCast(field_names.len), field_names.ptr));
                mlir.oraOperationSetAttributeByName(op, strRef("ora.field_types"), mlir.oraArrayAttrCreate(self.context, @intCast(field_types.len), field_types.ptr));
            }
            mlir.oraOperationSetAttributeByName(op, strRef("ora.struct_decl"), mlir.oraBoolAttrCreate(self.context, true));

            appendOp(parent_block, op);
            try self.appendItemHandle(item_id, .struct_, struct_item.name, struct_item.range, op);
        }

        pub fn lowerEnumDecl(self: *Lowerer, item_id: ast.ItemId, enum_item: ast.EnumItem, parent_block: mlir.MlirBlock) anyerror!void {
            if (enum_item.is_generic) return;
            const loc = self.location(enum_item.range);
            const repr_type = defaultIntegerType(self.context);
            const op = mlir.oraEnumDeclOpCreate(self.context, loc, strRef(enum_item.name), repr_type);
            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;

            if (enum_item.variants.len > 0) {
                const variant_names = try self.allocator.alloc(mlir.MlirAttribute, enum_item.variants.len);
                const variant_values = try self.allocator.alloc(mlir.MlirAttribute, enum_item.variants.len);
                for (enum_item.variants, 0..) |variant, index| {
                    variant_names[index] = mlir.oraStringAttrCreate(self.context, strRef(variant.name));
                    variant_values[index] = mlir.oraIntegerAttrCreateI64FromType(repr_type, @intCast(index));
                }
                mlir.oraOperationSetAttributeByName(op, strRef("ora.variant_names"), mlir.oraArrayAttrCreate(self.context, @intCast(variant_names.len), variant_names.ptr));
                mlir.oraOperationSetAttributeByName(op, strRef("ora.variant_values"), mlir.oraArrayAttrCreate(self.context, @intCast(variant_values.len), variant_values.ptr));
            }
            mlir.oraOperationSetAttributeByName(op, strRef("ora.enum_decl"), mlir.oraBoolAttrCreate(self.context, true));
            mlir.oraOperationSetAttributeByName(op, strRef("ora.has_explicit_values"), mlir.oraBoolAttrCreate(self.context, false));

            appendOp(parent_block, op);
            try self.appendItemHandle(item_id, .enum_, enum_item.name, enum_item.range, op);
        }

        pub fn lowerLogDecl(self: *Lowerer, item_id: ast.ItemId, log_decl: ast.LogDeclItem, parent_block: mlir.MlirBlock) anyerror!void {
            const loc = self.location(log_decl.range);
            var attrs: std.ArrayList(mlir.MlirNamedAttribute) = .{};
            try attrs.append(self.allocator, namedStringAttr(self.context, "sym_name", log_decl.name));
            try attrs.append(self.allocator, namedBoolAttr(self.context, "ora.log_decl", true));

            if (log_decl.fields.len > 0) {
                const field_names = try self.allocator.alloc(mlir.MlirAttribute, log_decl.fields.len);
                const field_types = try self.allocator.alloc(mlir.MlirAttribute, log_decl.fields.len);
                const field_indexed = try self.allocator.alloc(mlir.MlirAttribute, log_decl.fields.len);
                for (log_decl.fields, 0..) |field, index| {
                    field_names[index] = mlir.oraStringAttrCreate(self.context, strRef(field.name));
                    field_types[index] = mlir.oraTypeAttrCreateFromType(self.lowerTypeExpr(field.type_expr));
                    field_indexed[index] = mlir.oraBoolAttrCreate(self.context, field.indexed);
                }
                try attrs.append(self.allocator, mlir.oraNamedAttributeGet(identifier(self.context, "ora.field_names"), mlir.oraArrayAttrCreate(self.context, @intCast(field_names.len), field_names.ptr)));
                try attrs.append(self.allocator, mlir.oraNamedAttributeGet(identifier(self.context, "ora.field_types"), mlir.oraArrayAttrCreate(self.context, @intCast(field_types.len), field_types.ptr)));
                try attrs.append(self.allocator, mlir.oraNamedAttributeGet(identifier(self.context, "ora.field_indexed"), mlir.oraArrayAttrCreate(self.context, @intCast(field_indexed.len), field_indexed.ptr)));
            }

            const op = mlir.oraLogDeclOpCreate(self.context, loc, if (attrs.items.len == 0) null else attrs.items.ptr, attrs.items.len);
            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
            appendOp(parent_block, op);
            try self.appendItemHandle(item_id, .log_decl, log_decl.name, log_decl.range, op);
        }

        pub fn lowerErrorDecl(self: *Lowerer, item_id: ast.ItemId, error_decl: ast.ErrorDeclItem, parent_block: mlir.MlirBlock) anyerror!void {
            const loc = self.location(error_decl.range);
            var attrs: std.ArrayList(mlir.MlirNamedAttribute) = .{};
            try attrs.append(self.allocator, namedStringAttr(self.context, "sym_name", error_decl.name));
            try attrs.append(self.allocator, namedBoolAttr(self.context, "ora.error_decl", true));

            if (error_decl.parameters.len > 0) {
                const param_names = try self.allocator.alloc(mlir.MlirAttribute, error_decl.parameters.len);
                const param_types = try self.allocator.alloc(mlir.MlirAttribute, error_decl.parameters.len);
                for (error_decl.parameters, 0..) |param, index| {
                    const pattern = self.file.pattern(param.pattern).*;
                    const param_name = switch (pattern) {
                        .Name => |name| name.name,
                        else => "",
                    };
                    param_names[index] = mlir.oraStringAttrCreate(self.context, strRef(param_name));
                    param_types[index] = mlir.oraTypeAttrCreateFromType(self.lowerTypeExpr(param.type_expr));
                }
                try attrs.append(self.allocator, mlir.oraNamedAttributeGet(identifier(self.context, "ora.param_names"), mlir.oraArrayAttrCreate(self.context, @intCast(param_names.len), param_names.ptr)));
                try attrs.append(self.allocator, mlir.oraNamedAttributeGet(identifier(self.context, "ora.param_types"), mlir.oraArrayAttrCreate(self.context, @intCast(param_types.len), param_types.ptr)));
            }

            const op = mlir.oraErrorDeclOpCreate(
                self.context,
                loc,
                null,
                0,
                if (attrs.items.len == 0) null else attrs.items.ptr,
                attrs.items.len,
            );
            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
            appendOp(parent_block, op);
            try self.appendItemHandle(item_id, .error_decl, error_decl.name, error_decl.range, op);
        }

        pub fn attachBitfieldParamMetadata(self: *Lowerer, func_op: mlir.MlirOperation, type_expr_id: ast.TypeExprId, index: c_uint) !void {
            const metadata = self.bitfieldMetadataForTypeExpr(type_expr_id) orelse return;
            _ = mlir.oraFuncSetArgAttr(func_op, index, strRef("ora.bitfield"), mlir.oraStringAttrCreate(self.context, strRef(metadata.name)));
        }

        pub fn attachBitfieldParamMetadataForType(self: *Lowerer, func_op: mlir.MlirOperation, ty: sema.Type, index: c_uint) !void {
            const metadata = (try self.bitfieldMetadataForType(ty)) orelse return;
            _ = mlir.oraFuncSetArgAttr(func_op, index, strRef("ora.bitfield"), mlir.oraStringAttrCreate(self.context, strRef(metadata.name)));
        }

        pub fn attachBitfieldOpMetadata(self: *Lowerer, op: mlir.MlirOperation, type_expr_id: ast.TypeExprId) !void {
            const metadata = self.bitfieldMetadataForTypeExpr(type_expr_id) orelse return;
            mlir.oraOperationSetAttributeByName(op, strRef("ora.bitfield"), mlir.oraStringAttrCreate(self.context, strRef(metadata.name)));
            if (metadata.layout.len > 0) {
                mlir.oraOperationSetAttributeByName(op, strRef("ora.bitfield_layout"), mlir.oraStringAttrCreate(self.context, strRef(metadata.layout)));
            }
        }

        pub fn attachBitfieldOpMetadataForType(self: *Lowerer, op: mlir.MlirOperation, ty: sema.Type) !void {
            const metadata = (try self.bitfieldMetadataForType(ty)) orelse return;
            mlir.oraOperationSetAttributeByName(op, strRef("ora.bitfield"), mlir.oraStringAttrCreate(self.context, strRef(metadata.name)));
            if (metadata.layout.len > 0) {
                mlir.oraOperationSetAttributeByName(op, strRef("ora.bitfield_layout"), mlir.oraStringAttrCreate(self.context, strRef(metadata.layout)));
            }
        }

        const BitfieldMetadata = struct {
            name: []const u8,
            layout: []const u8,
        };

        pub fn bitfieldMetadataForTypeExpr(self: *Lowerer, type_expr_id: ast.TypeExprId) ?BitfieldMetadata {
            const type_expr = self.file.typeExpr(type_expr_id).*;
            const name = switch (type_expr) {
                .Path => |path| path.name,
                else => return null,
            };
            const bitfield = self.bitfieldItemByName(name) orelse return null;
            const layout = self.buildBitfieldLayout(bitfield) catch return null;
            return .{
                .name = bitfield.name,
                .layout = layout,
            };
        }

        pub fn bitfieldMetadataForType(self: *Lowerer, ty: sema.Type) !?BitfieldMetadata {
            if (ty.kind() != .bitfield) return null;
            const name = ty.name() orelse return null;
            if (self.instantiatedBitfieldByName(name)) |bitfield| {
                const layout = try self.buildInstantiatedBitfieldLayout(bitfield);
                return .{
                    .name = bitfield.mangled_name,
                    .layout = layout,
                };
            }
            const template = self.bitfieldItemByName(name) orelse return null;
            const layout = try self.buildBitfieldLayout(template);
            return .{
                .name = template.name,
                .layout = layout,
            };
        }

        pub fn buildBitfieldLayout(self: *Lowerer, bitfield: ast.BitfieldItem) ![]const u8 {
            var buffer: std.ArrayList(u8) = .{};
            for (bitfield.fields) |field| {
                const resolved = self.resolveBitfieldField(bitfield.name, field.name) orelse continue;
                try buffer.writer(self.allocator).print("{s}:{d}:{d}:{c};", .{ field.name, resolved.offset, resolved.width, resolved.sign });
            }
            return buffer.toOwnedSlice(self.allocator);
        }

        pub fn buildInstantiatedBitfieldLayout(self: *Lowerer, bitfield: sema.InstantiatedBitfield) ![]const u8 {
            var buffer: std.ArrayList(u8) = .{};
            var next_offset: u32 = 0;
            for (bitfield.fields) |field| {
                const width = field.width orelse self.bitfieldFieldWidthFromType(field.ty);
                const offset = field.offset orelse next_offset;
                const sign = self.bitfieldFieldSignFromType(field.ty);
                try buffer.writer(self.allocator).print("{s}:{d}:{d}:{c};", .{ field.name, offset, width, sign });
                next_offset = offset + width;
            }
            return buffer.toOwnedSlice(self.allocator);
        }

        pub fn bitfieldFieldSign(self: *const Lowerer, type_expr_id: ast.TypeExprId) u8 {
            return switch (self.file.typeExpr(type_expr_id).*) {
                .Path => |path| blk: {
                    const trimmed = std.mem.trim(u8, path.name, " \t\n\r");
                    if (trimmed.len > 1 and trimmed[0] == 'i' and support.parseSignedIntegerType(trimmed) != null) break :blk 's';
                    break :blk 'u';
                },
                else => 'u',
            };
        }

        pub fn lowerDeclPlaceholder(
            self: *Lowerer,
            item_id: ast.ItemId,
            kind: HirSymbolKind,
            name: []const u8,
            range: source.TextRange,
            op_name: []const u8,
            parent_block: mlir.MlirBlock,
        ) anyerror!void {
            const op = try self.createNamedPlaceholderOp(op_name, name, range, mlir.oraNoneTypeCreate(self.context));
            appendOp(parent_block, op);
            try self.appendItemHandle(item_id, kind, name, range, op);
        }

        pub fn createNamedPlaceholderOp(self: *Lowerer, op_name: []const u8, name: []const u8, range: source.TextRange, ty: mlir.MlirType) anyerror!mlir.MlirOperation {
            const loc = self.location(range);
            var attrs: [2]mlir.MlirNamedAttribute = .{
                namedStringAttr(self.context, "sym_name", name),
                namedTypeAttr(self.context, "ora.type", ty),
            };
            return self.createPlaceholderOp(op_name, loc, &attrs);
        }

        pub fn createPlaceholderOp(self: *Lowerer, op_name: []const u8, loc: mlir.MlirLocation, attrs: []const mlir.MlirNamedAttribute) anyerror!mlir.MlirOperation {
            const op = mlir.oraOperationCreate(
                self.context,
                loc,
                strRef(op_name),
                null,
                0,
                null,
                0,
                if (attrs.len == 0) null else attrs.ptr,
                attrs.len,
                0,
                false,
            );
            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
            return op;
        }

        pub fn appendItemHandle(self: *Lowerer, item_id: ast.ItemId, kind: HirSymbolKind, name: []const u8, range: source.TextRange, op: mlir.MlirOperation) anyerror!void {
            try self.items.append(self.allocator, .{
                .item_id = item_id,
                .kind = kind,
                .symbol_name = name,
                .location = .{ .file_id = self.file.file_id, .range = range },
                .raw_operation = op,
            });
        }

        fn attachGhostAttrs(self: *Lowerer, op: mlir.MlirOperation, context: []const u8) void {
            mlir.oraOperationSetAttributeByName(op, strRef("ora.ghost"), namedBoolAttr(self.context, "ora.ghost", true).attribute);
            mlir.oraOperationSetAttributeByName(op, strRef("ora.verification"), namedBoolAttr(self.context, "ora.verification", true).attribute);
            mlir.oraOperationSetAttributeByName(op, strRef("ora.formal"), namedBoolAttr(self.context, "ora.formal", true).attribute);
            mlir.oraOperationSetAttributeByName(op, strRef("ora.verification_context"), namedStringAttr(self.context, "ora.verification_context", context).attribute);
        }

        fn collectContractLockedStorageRoots(self: *Lowerer, contract: ast.ContractItem) anyerror!std.StringHashMap(void) {
            var roots = std.StringHashMap(void).init(self.allocator);
            for (contract.members) |member_id| {
                switch (self.file.item(member_id).*) {
                    .Function => |function| try collectLockedRootsFromBody(self, function.body, &roots),
                    else => {},
                }
            }
            return roots;
        }

        fn collectLockedRootsFromBody(self: *Lowerer, body_id: ast.BodyId, roots: *std.StringHashMap(void)) anyerror!void {
            const body = self.file.body(body_id).*;
            for (body.statements) |statement_id| {
                try collectLockedRootsFromStmt(self, statement_id, roots);
            }
        }

        fn collectLockedRootsFromStmt(self: *Lowerer, statement_id: ast.StmtId, roots: *std.StringHashMap(void)) anyerror!void {
            switch (self.file.statement(statement_id).*) {
                .Lock => |lock_stmt| {
                    const root = lockRootName(self.file, lock_stmt.path) orelse return;
                    try roots.put(root, {});
                },
                .If => |if_stmt| {
                    try collectLockedRootsFromBody(self, if_stmt.then_body, roots);
                    if (if_stmt.else_body) |else_body| try collectLockedRootsFromBody(self, else_body, roots);
                },
                .While => |while_stmt| try collectLockedRootsFromBody(self, while_stmt.body, roots),
                .For => |for_stmt| try collectLockedRootsFromBody(self, for_stmt.body, roots),
                .Switch => |switch_stmt| {
                    for (switch_stmt.arms) |arm| try collectLockedRootsFromBody(self, arm.body, roots);
                    if (switch_stmt.else_body) |else_body| try collectLockedRootsFromBody(self, else_body, roots);
                },
                .Try => |try_stmt| {
                    try collectLockedRootsFromBody(self, try_stmt.try_body, roots);
                    if (try_stmt.catch_clause) |catch_clause| try collectLockedRootsFromBody(self, catch_clause.body, roots);
                },
                .Block => |block_stmt| try collectLockedRootsFromBody(self, block_stmt.body, roots),
                .LabeledBlock => |block_stmt| try collectLockedRootsFromBody(self, block_stmt.body, roots),
                else => {},
            }
        }

        fn lockRootName(file: *const ast.AstFile, expr_id: ast.ExprId) ?[]const u8 {
            return switch (file.expression(expr_id).*) {
                .Name => |name| name.name,
                .Index => |index| lockRootName(file, index.base),
                .Group => |group| lockRootName(file, group.expr),
                else => null,
            };
        }
    };
}
