const std = @import("std");
const mlir = @import("mlir_c_api").c;
const ast = @import("../ast/mod.zig");
const sema = @import("../sema/mod.zig");
const abi_support = @import("abi.zig");
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
            for (self.typecheck.instantiated_structs) |instantiated| {
                if (self.enclosingContractForItem(instantiated.template_item_id)) |contract_id| {
                    if (contract_id.index() == item_id.index()) {
                        try self.lowerInstantiatedStructDecl(instantiated, body);
                    }
                }
            }
            for (self.typecheck.instantiated_enums) |instantiated| {
                if (self.enclosingContractForItem(instantiated.template_item_id)) |contract_id| {
                    if (contract_id.index() == item_id.index()) {
                        try self.lowerInstantiatedEnumDecl(instantiated, body);
                    }
                }
            }
            for (self.typecheck.instantiated_bitfields) |instantiated| {
                if (self.enclosingContractForItem(instantiated.template_item_id)) |contract_id| {
                    if (contract_id.index() == item_id.index()) {
                        try self.lowerInstantiatedBitfieldDecl(instantiated, body);
                    }
                }
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
            _ = impl_item_id;
            _ = parent_block;
            const impl_parent_block = @This().implParentBlock(self, impl_item.target_name);

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
            }

        }

        fn lowerImplGhostBlock(self: *Lowerer, impl_item: ast.ImplItem, ghost_block: ast.GhostBlockItem, parent_block: mlir.MlirBlock) anyerror!void {
            var function_lowerer = FunctionLowerer.initContractContext(self, parent_block);
            function_lowerer.in_ghost_context = true;

            var locals = try function_lowerer.cloneLocals(&function_lowerer.locals);
            if (@This().firstImplSelfPattern(self, impl_item)) |pattern_id| {
                function_lowerer.trait_ghost_self_pattern = pattern_id;
                const self_value = try @This().lowerImplGhostSelfValue(self, impl_item, ghost_block.range, &function_lowerer);
                try locals.bindPattern(self.file, pattern_id, self_value);
            }

            _ = try function_lowerer.lowerBody(ghost_block.body, &locals);
        }

        fn lowerImplGhostSelfValue(self: *Lowerer, impl_item: ast.ImplItem, range: source.TextRange, function_lowerer: *FunctionLowerer) anyerror!mlir.MlirValue {
            const target_item_id = self.item_index.lookup(impl_item.target_name) orelse return error.MlirOperationCreationFailed;
            const self_type = mlir.oraStructTypeGet(self.context, strRef(impl_item.target_name));

            switch (self.file.item(target_item_id).*) {
                .Contract => {
                    const created = mlir.oraStructInstantiateOpCreate(self.context, self.location(range), strRef(impl_item.target_name), null, 0, self_type);
                    if (!mlir.oraOperationIsNull(created)) return support.appendValueOp(function_lowerer.block, created);
                },
                .Struct => |struct_item| {
                    var operands: std.ArrayList(mlir.MlirValue) = .{};
                    for (struct_item.fields) |field| {
                        try operands.append(self.allocator, try function_lowerer.defaultValue(self.lowerTypeExpr(field.type_expr), field.range));
                    }
                    const created = mlir.oraStructInstantiateOpCreate(
                        self.context,
                        self.location(range),
                        strRef(impl_item.target_name),
                        if (operands.items.len == 0) null else operands.items.ptr,
                        operands.items.len,
                        self_type,
                    );
                    if (!mlir.oraOperationIsNull(created)) return support.appendValueOp(function_lowerer.block, created);
                },
                else => {},
            }

            return error.MlirOperationCreationFailed;
        }

        pub fn enclosingContractForItem(self: *const Lowerer, item_id: ast.ItemId) ?ast.ItemId {
            for (self.file.items, 0..) |item, index| {
                if (item != .Contract) continue;
                for (item.Contract.members) |member_id| {
                    if (member_id.index() == item_id.index()) return ast.ItemId.fromIndex(index);
                }
            }
            return null;
        }

        fn firstImplSelfPattern(self: *Lowerer, impl_item: ast.ImplItem) ?ast.PatternId {
            for (impl_item.methods) |method_item_id| {
                const item = self.file.item(method_item_id).*;
                if (item != .Function) continue;
                for (item.Function.parameters) |parameter| {
                    if (parameter.is_comptime) continue;
                    const pattern = self.file.pattern(parameter.pattern).*;
                    if (pattern != .Name) break;
                    if (std.mem.eql(u8, pattern.Name.name, "self")) return parameter.pattern;
                    break;
                }
            }
            return null;
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
            if (@This().enclosingImplForMethod(self, item_id)) |impl_item| {
                const symbol_name = try @This().ensureLoweredImplMethod(self, item_id, function, impl_item.target_name, call, null);
                return symbol_name;
            }
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

        fn enclosingImplForMethod(self: *Lowerer, method_item_id: ast.ItemId) ?ast.ImplItem {
            for (self.file.items) |item| {
                if (item != .Impl) continue;
                for (item.Impl.methods) |candidate_id| {
                    if (candidate_id.index() == method_item_id.index()) return item.Impl;
                }
            }
            return null;
        }

        pub fn ensureLoweredImplMethod(
            self: *Lowerer,
            method_item_id: ast.ItemId,
            function: ast.FunctionItem,
            target_name: []const u8,
            call: ?ast.CallExpr,
            parent_block_override: ?mlir.MlirBlock,
        ) anyerror![]const u8 {
            const base_symbol_name = try @This().implMethodSymbolName(self, target_name, function.name);
            const parent_block = parent_block_override orelse @This().implParentBlock(self, target_name);
            if (function.is_generic) {
                const method_call = call orelse return base_symbol_name;
                const bindings = (try self.genericTypeBindingsForCall(function, method_call)) orelse return base_symbol_name;
                const runtime_parameters = try self.runtimeFunctionParameters(function);
                const symbol_name = try self.mangleGenericFunctionName(base_symbol_name, bindings);
                if (!self.monomorphized_function_names.contains(symbol_name)) {
                    try @This().lowerConcreteFunction(self, method_item_id, function, symbol_name, runtime_parameters, parent_block, bindings);
                    try self.monomorphized_function_names.put(symbol_name, {});
                }
                return symbol_name;
            }

            const symbol_name = base_symbol_name;
            if (!self.monomorphized_function_names.contains(symbol_name)) {
                try @This().lowerConcreteFunction(self, method_item_id, function, symbol_name, function.parameters, parent_block, &.{});
                try self.monomorphized_function_names.put(symbol_name, {});
            }
            return symbol_name;
        }

        fn implParentBlock(self: *Lowerer, target_name: []const u8) mlir.MlirBlock {
            if (self.item_index.lookup(target_name)) |target_item_id| {
                if (self.file.item(target_item_id).* == .Contract) {
                    const block = self.contract_body_blocks[target_item_id.index()];
                    if (!mlir.oraBlockIsNull(block)) return block;
                }
            }
            return self.module_body;
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
            if (std.mem.eql(u8, function.name, "init")) {
                try attrs.append(self.allocator, namedBoolAttr(self.context, "ora.init", true));
            }
            if (function.visibility == .public and function.parent_contract != null) {
                @This().attachPublicAbiAttrs(self, &attrs, function, parameters) catch |err| switch (err) {
                    error.UnsupportedAbiType => {},
                    else => return err,
                };
            }

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
                if (self.errorUnionRequiresWideCarrier(self.typecheck.pattern_types[parameter.pattern.index()].type)) {
                    _ = mlir.oraFuncSetArgAttr(op, @intCast(index), strRef("ora.force_wide_error_union"), mlir.oraBoolAttrCreate(self.context, true));
                }
            }

            if (return_type != null and self.errorUnionRequiresWideCarrier(self.typecheck.body_types[function.body.index()])) {
                mlir.oraOperationSetAttributeByName(op, strRef("ora.force_wide_error_union"), mlir.oraBoolAttrCreate(self.context, true));
            }

            appendOp(parent_block, op);
            try self.appendItemHandle(item_id, .function, symbol_name, function.range, op);

            var specialized_function = function;
            specialized_function.name = symbol_name;
            specialized_function.is_generic = false;
            specialized_function.parameters = @constCast(function.parameters);
            var function_lowerer = FunctionLowerer.init(self, item_id, specialized_function, op, return_type);
            function_lowerer.extra_verification_clauses = try @This().traitGhostClausesForImplMethod(self, item_id);
            try function_lowerer.lower();
        }

        fn traitGhostClausesForImplMethod(self: *Lowerer, method_item_id: ast.ItemId) anyerror![]const FunctionLowerer.ExtraVerificationClause {
            const impl_item = @This().enclosingImplForMethod(self, method_item_id) orelse return &.{};
            const trait_item_id = self.item_index.lookup(impl_item.trait_name) orelse return &.{};
            const trait_item = switch (self.file.item(trait_item_id).*) {
                .Trait => |trait_item| trait_item,
                else => return &.{},
            };
            const ghost_id = trait_item.ghost_block orelse return &.{};
            const ghost_block = self.file.item(ghost_id).GhostBlock;
            const body = self.file.body(ghost_block.body).*;

            var clauses: std.ArrayList(FunctionLowerer.ExtraVerificationClause) = .{};
            for (body.statements) |stmt_id| {
                switch (self.file.statement(stmt_id).*) {
                    .Assert => |assert_stmt| {
                        try clauses.append(self.allocator, .{
                            .kind = .ensures,
                            .expr = assert_stmt.condition,
                            .range = assert_stmt.range,
                        });
                    },
                    .Assume => |assume_stmt| {
                        try clauses.append(self.allocator, .{
                            .kind = .requires,
                            .expr = assume_stmt.condition,
                            .range = assume_stmt.range,
                        });
                    },
                    else => {},
                }
            }
            return clauses.toOwnedSlice(self.allocator);
        }

        fn attachPublicAbiAttrs(
            self: *Lowerer,
            attrs: *std.ArrayList(mlir.MlirNamedAttribute),
            function: ast.FunctionItem,
            parameters: []const ast.Parameter,
        ) anyerror!void {
            var signature_parts: std.ArrayList([]const u8) = .{};
            defer {
                for (signature_parts.items) |part| self.allocator.free(part);
                signature_parts.deinit(self.allocator);
            }
            var abi_param_attrs: std.ArrayList(mlir.MlirAttribute) = .{};
            defer abi_param_attrs.deinit(self.allocator);

            for (parameters) |parameter| {
                const abi_type = try abi_support.canonicalAbiType(self.allocator, self.typecheck.pattern_types[parameter.pattern.index()].type);
                defer self.allocator.free(abi_type);
                try signature_parts.append(self.allocator, try self.allocator.dupe(u8, abi_type));
                abi_param_attrs.append(self.allocator, mlir.oraStringAttrCreate(self.context, strRef(abi_type))) catch return error.OutOfMemory;
            }

            if (abi_param_attrs.items.len != 0) {
                const abi_params_attr = mlir.oraArrayAttrCreate(self.context, @intCast(abi_param_attrs.items.len), abi_param_attrs.items.ptr);
                try attrs.append(self.allocator, .{
                    .name = identifier(self.context, "ora.abi_params"),
                    .attribute = abi_params_attr,
                });
            }

            if (!std.mem.eql(u8, function.name, "init")) {
                const joined = try std.mem.join(self.allocator, ",", signature_parts.items);
                defer self.allocator.free(joined);
                const signature = try std.fmt.allocPrint(self.allocator, "{s}({s})", .{ function.name, joined });
                defer self.allocator.free(signature);
                const selector = try abi_support.keccakSelectorHex(self.allocator, signature);
                defer self.allocator.free(selector);

                try attrs.append(self.allocator, namedStringAttr(self.context, "ora.selector", selector));
            }

            if (function.return_type) |_| {
                const body_type = self.typecheck.body_types[function.body.index()];
                const abi_return_type = switch (body_type) {
                    .error_union => |error_union| error_union.payload_type.*,
                    else => body_type,
                };
                const abi_return = try abi_support.externReturnAbiType(self.allocator, abi_return_type);
                defer self.allocator.free(abi_return);
                try attrs.append(self.allocator, namedStringAttr(self.context, "ora.abi_return", abi_return));
                if (@This().abiLayoutForType(self, abi_return_type)) |layout| {
                    defer self.allocator.free(layout);
                    try attrs.append(self.allocator, namedStringAttr(self.context, "ora.abi_return_layout", layout));
                } else |_| {}
                if (@This().staticAbiWordCountForType(self, abi_return_type)) |word_count| {
                    try attrs.append(self.allocator, .{
                        .name = identifier(self.context, "ora.abi_return_words"),
                        .attribute = mlir.oraIntegerAttrCreateI64FromType(defaultIntegerType(self.context), @intCast(word_count)),
                    });
                }

                if (function.return_type) |return_type_id| {
                    if (self.file.typeExpr(return_type_id).* == .ErrorUnion) {
                        try attrs.append(self.allocator, namedBoolAttr(self.context, "ora.returns_error_union", true));
                    }
                }
            }
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
            try attrs.append(self.allocator, .{
                .name = identifier(self.context, "ora.error_id"),
                .attribute = mlir.oraIntegerAttrCreateI64FromType(defaultIntegerType(self.context), @intCast(item_id.index() + 1)),
            });

            const param_types = try self.allocator.alloc(sema.Type, error_decl.parameters.len);
            defer self.allocator.free(param_types);
            for (error_decl.parameters, 0..) |param, index| {
                param_types[index] = self.typecheck.pattern_types[param.pattern.index()].type;
            }
            const maybe_signature = abi_support.signatureForMethod(self.allocator, error_decl.name, false, param_types) catch |err| switch (err) {
                error.UnsupportedAbiType => null,
                else => return err,
            };
            if (maybe_signature) |signature| {
                defer self.allocator.free(signature);
                const selector = try abi_support.keccakSelectorHex(self.allocator, signature);
                defer self.allocator.free(selector);
                try attrs.append(self.allocator, namedStringAttr(self.context, "ora.error_selector", selector));
            }

            if (error_decl.parameters.len > 0) {
                const param_names = try self.allocator.alloc(mlir.MlirAttribute, error_decl.parameters.len);
                const param_type_attrs = try self.allocator.alloc(mlir.MlirAttribute, error_decl.parameters.len);
                for (error_decl.parameters, 0..) |param, index| {
                    const pattern = self.file.pattern(param.pattern).*;
                    const param_name = switch (pattern) {
                        .Name => |name| name.name,
                        else => "",
                    };
                    param_names[index] = mlir.oraStringAttrCreate(self.context, strRef(param_name));
                    param_type_attrs[index] = mlir.oraTypeAttrCreateFromType(self.lowerTypeExpr(param.type_expr));
                }
                try attrs.append(self.allocator, mlir.oraNamedAttributeGet(identifier(self.context, "ora.param_names"), mlir.oraArrayAttrCreate(self.context, @intCast(param_names.len), param_names.ptr)));
                try attrs.append(self.allocator, mlir.oraNamedAttributeGet(identifier(self.context, "ora.param_types"), mlir.oraArrayAttrCreate(self.context, @intCast(param_type_attrs.len), param_type_attrs.ptr)));
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

        fn staticAbiWordCountForType(self: *Lowerer, ty: sema.Type) ?usize {
            return switch (ty) {
                .bool, .address, .integer => 1,
                .refinement => |refinement| @This().staticAbiWordCountForType(self, refinement.base_type.*),
                .tuple => |elements| blk: {
                    var total: usize = 0;
                    for (elements) |element| {
                        const words = @This().staticAbiWordCountForType(self, element) orelse return null;
                        total += words;
                    }
                    break :blk total;
                },
                .array => |array| blk: {
                    const len = array.len orelse return null;
                    const element_words = @This().staticAbiWordCountForType(self, array.element_type.*) orelse return null;
                    break :blk element_words * len;
                },
                .struct_ => |named| @This().staticAbiWordCountForNamedStruct(self, named.name),
                .contract => |named| @This().staticAbiWordCountForNamedStruct(self, named.name),
                .named => |named| @This().staticAbiWordCountForNamedStruct(self, named.name),
                else => null,
            };
        }

        fn abiLayoutForPrimitiveName(self: *Lowerer, name: []const u8) ![]const u8 {
            if (std.mem.eql(u8, name, "bool") or std.mem.eql(u8, name, "address") or std.mem.eql(u8, name, "string") or std.mem.eql(u8, name, "bytes")) {
                return self.allocator.dupe(u8, name);
            }
            if (std.mem.eql(u8, name, "u256")) return self.allocator.dupe(u8, "uint256");
            if (std.mem.eql(u8, name, "i256")) return self.allocator.dupe(u8, "int256");
            if (std.mem.startsWith(u8, name, "u")) return std.fmt.allocPrint(self.allocator, "uint{s}", .{name[1..]});
            if (std.mem.startsWith(u8, name, "i")) return std.fmt.allocPrint(self.allocator, "int{s}", .{name[1..]});
            return error.UnsupportedAbiType;
        }

        fn abiLayoutForType(self: *Lowerer, ty: sema.Type) anyerror![]const u8 {
            return switch (ty) {
                .bool, .address, .string, .bytes, .integer => abi_support.canonicalAbiType(self.allocator, ty),
                .refinement => |refinement| @This().abiLayoutForType(self, refinement.base_type.*),
                .array => |array| blk: {
                    const element = try @This().abiLayoutForType(self, array.element_type.*);
                    defer self.allocator.free(element);
                    if (array.len) |len| break :blk std.fmt.allocPrint(self.allocator, "{s}[{d}]", .{ element, len });
                    break :blk std.fmt.allocPrint(self.allocator, "{s}[]", .{element});
                },
                .slice => |slice| blk: {
                    const element = try @This().abiLayoutForType(self, slice.element_type.*);
                    defer self.allocator.free(element);
                    break :blk std.fmt.allocPrint(self.allocator, "{s}[]", .{element});
                },
                .tuple => |elements| blk: {
                    var parts: std.ArrayList([]const u8) = .{};
                    defer {
                        for (parts.items) |part| self.allocator.free(part);
                        parts.deinit(self.allocator);
                    }
                    for (elements) |element| try parts.append(self.allocator, try @This().abiLayoutForType(self, element));
                    const joined = try std.mem.join(self.allocator, ",", parts.items);
                    defer self.allocator.free(joined);
                    break :blk std.fmt.allocPrint(self.allocator, "({s})", .{joined});
                },
                .struct_ => |named| @This().abiLayoutForNamedStruct(self, named.name),
                .contract => |named| @This().abiLayoutForNamedStruct(self, named.name),
                .named => |named| @This().abiLayoutForNamedStruct(self, named.name),
                else => error.UnsupportedAbiType,
            };
        }

        fn abiLayoutForNamedStruct(self: *Lowerer, name: []const u8) anyerror![]const u8 {
            if (self.typecheck.instantiatedStructByName(name)) |instantiated| {
                var parts: std.ArrayList([]const u8) = .{};
                defer {
                    for (parts.items) |part| self.allocator.free(part);
                    parts.deinit(self.allocator);
                }
                for (instantiated.fields) |field| try parts.append(self.allocator, try @This().abiLayoutForType(self, field.ty));
                const joined = try std.mem.join(self.allocator, ",", parts.items);
                defer self.allocator.free(joined);
                return std.fmt.allocPrint(self.allocator, "({s})", .{joined});
            }

            const item_id = self.item_index.lookup(name) orelse return error.UnsupportedAbiType;
            return switch (self.file.item(item_id).*) {
                .Struct => |struct_item| blk: {
                    var parts: std.ArrayList([]const u8) = .{};
                    defer {
                        for (parts.items) |part| self.allocator.free(part);
                        parts.deinit(self.allocator);
                    }
                    for (struct_item.fields) |field| try parts.append(self.allocator, try @This().abiLayoutForTypeExpr(self, field.type_expr));
                    const joined = try std.mem.join(self.allocator, ",", parts.items);
                    defer self.allocator.free(joined);
                    break :blk std.fmt.allocPrint(self.allocator, "({s})", .{joined});
                },
                else => error.UnsupportedAbiType,
            };
        }

        fn abiLayoutForTypeExpr(self: *Lowerer, type_expr_id: ast.TypeExprId) anyerror![]const u8 {
            return switch (self.file.typeExpr(type_expr_id).*) {
                .Path => |path| blk: {
                    const trimmed = std.mem.trim(u8, path.name, " \t\n\r");
                    break :blk @This().abiLayoutForPrimitiveName(self, trimmed) catch @This().abiLayoutForNamedStruct(self, trimmed);
                },
                .Generic => |generic| blk: {
                    if (support.isRefinementTypeName(generic.name) and generic.args.len > 0) {
                        break :blk switch (generic.args[0]) {
                            .Type => |type_expr| @This().abiLayoutForTypeExpr(self, type_expr),
                            else => error.UnsupportedAbiType,
                        };
                    }
                    if (std.mem.eql(u8, generic.name, "map")) break :blk error.UnsupportedAbiType;
                    if ((std.mem.eql(u8, generic.name, "slice") or std.mem.eql(u8, generic.name, "array")) and generic.args.len > 0) {
                        break :blk switch (generic.args[0]) {
                            .Type => |type_expr| blk2: {
                                const element = try @This().abiLayoutForTypeExpr(self, type_expr);
                                defer self.allocator.free(element);
                                break :blk2 std.fmt.allocPrint(self.allocator, "{s}[]", .{element});
                            },
                            else => error.UnsupportedAbiType,
                        };
                    }
                    break :blk @This().abiLayoutForNamedStruct(self, generic.name);
                },
                .Tuple => |tuple| blk: {
                    var parts: std.ArrayList([]const u8) = .{};
                    defer {
                        for (parts.items) |part| self.allocator.free(part);
                        parts.deinit(self.allocator);
                    }
                    for (tuple.elements) |element| try parts.append(self.allocator, try @This().abiLayoutForTypeExpr(self, element));
                    const joined = try std.mem.join(self.allocator, ",", parts.items);
                    defer self.allocator.free(joined);
                    break :blk std.fmt.allocPrint(self.allocator, "({s})", .{joined});
                },
                .Array => |array| blk: {
                    const element = try @This().abiLayoutForTypeExpr(self, array.element);
                    defer self.allocator.free(element);
                    const size_text = switch (array.size) {
                        .Integer => |literal| std.mem.trim(u8, literal.text, " \t\n\r"),
                        else => "",
                    };
                    if (size_text.len == 0) break :blk std.fmt.allocPrint(self.allocator, "{s}[]", .{element});
                    break :blk std.fmt.allocPrint(self.allocator, "{s}[{s}]", .{ element, size_text });
                },
                else => error.UnsupportedAbiType,
            };
        }

        fn staticAbiWordCountForNamedStruct(self: *Lowerer, name: []const u8) ?usize {
            if (self.typecheck.instantiatedStructByName(name)) |instantiated| {
                var total: usize = 0;
                for (instantiated.fields) |field| {
                    const words = @This().staticAbiWordCountForType(self, field.ty) orelse return null;
                    total += words;
                }
                return total;
            }

            const item_id = self.item_index.lookup(name) orelse return null;
            return switch (self.file.item(item_id).*) {
                .Struct => |struct_item| blk: {
                    var total: usize = 0;
                    for (struct_item.fields) |field| {
                        const words = @This().staticAbiWordCountForTypeExpr(self, field.type_expr) orelse return null;
                        total += words;
                    }
                    break :blk total;
                },
                else => null,
            };
        }

        fn staticAbiWordCountForTypeExpr(self: *Lowerer, type_expr_id: ast.TypeExprId) ?usize {
            return switch (self.file.typeExpr(type_expr_id).*) {
                .Path => |path| blk: {
                    const trimmed = std.mem.trim(u8, path.name, " \t\n\r");
                    if (std.mem.eql(u8, trimmed, "bool") or std.mem.eql(u8, trimmed, "address")) break :blk 1;
                    if (support.parseSignedIntegerType(trimmed) != null) break :blk 1;
                    break :blk @This().staticAbiWordCountForNamedStruct(self, trimmed);
                },
                .Generic => |generic| blk: {
                    if (support.isRefinementTypeName(generic.name) and generic.args.len > 0) {
                        break :blk switch (generic.args[0]) {
                            .Type => |type_expr| @This().staticAbiWordCountForTypeExpr(self, type_expr),
                            else => null,
                        };
                    }
                    if (std.mem.eql(u8, generic.name, "map")) break :blk null;
                    if (generic.args.len > 0) {
                        break :blk switch (generic.args[0]) {
                            .Type => |type_expr| @This().staticAbiWordCountForTypeExpr(self, type_expr),
                            else => null,
                        };
                    }
                    break :blk @This().staticAbiWordCountForNamedStruct(self, generic.name);
                },
                .Tuple => |tuple| blk: {
                    var total: usize = 0;
                    for (tuple.elements) |element| {
                        const words = @This().staticAbiWordCountForTypeExpr(self, element) orelse return null;
                        total += words;
                    }
                    break :blk total;
                },
                .Array => |array| blk: {
                    const len = switch (array.size) {
                        .Integer => |literal| std.fmt.parseInt(usize, std.mem.trim(u8, literal.text, " \t\n\r"), 10) catch return null,
                        else => return null,
                    };
                    const element_words = @This().staticAbiWordCountForTypeExpr(self, array.element) orelse return null;
                    break :blk element_words * len;
                },
                else => null,
            };
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
