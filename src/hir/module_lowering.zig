const std = @import("std");
const mlir = @import("mlir_c_api").c;
const ast = @import("../ast/mod.zig");
const sema = @import("../sema/mod.zig");
const ora_types = @import("ora_types");
const ConstValue = ora_types.ConstValue;
const refinements = ora_types.refinement_semantics;
const type_descriptors = @import("../sema/type_descriptors.zig");
const abi_layout_context = @import("../abi/layout_context.zig");
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
                .Import => {},
                .Contract => |contract| try self.lowerContract(item_id, contract, parent_block),
                .Function => |function| try self.lowerFunction(item_id, function, parent_block),
                .Struct => |struct_item| try self.lowerStructDecl(item_id, struct_item, parent_block),
                .Bitfield => |bitfield| {
                    if (bitfield.is_generic) return;
                    try self.lowerDeclPlaceholder(item_id, .bitfield, bitfield.name, bitfield.range, "ora.bitfield_decl", parent_block);
                },
                .Enum => |enum_item| try self.lowerEnumDecl(item_id, enum_item, parent_block),
                .Trait => {},
                .Impl => |impl_item| try @This().lowerImpl(self, impl_item, parent_block),
                .TypeAlias => {},
                .LogDecl => |log_decl| try self.lowerLogDecl(item_id, log_decl, parent_block),
                .ErrorDecl => |error_decl| try self.lowerErrorDecl(item_id, error_decl, parent_block),
                .GhostBlock => |ghost_block| try @This().lowerGhostBlock(self, item_id, ghost_block, parent_block),
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

            for (self.itemVerificationFactEntries(item_id)) |entry| {
                const fact = self.verificationFact(entry);
                if (fact.kind != .contract_invariant) continue;
                try contract_lowerer.lowerInvariant(fact.*);
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
            if (function.is_generic or function.is_comptime) return;
            const parameters = try self.runtimeFunctionParameters(function);
            try @This().lowerConcreteFunction(self, item_id, function, function.name, parameters, parent_block, &.{}, null, null);
        }

        pub fn lowerImpl(self: *Lowerer, impl_item: ast.ImplItem, parent_block: mlir.MlirBlock) anyerror!void {
            const impl_parent_block = @This().implParentBlock(self, impl_item.target_name, parent_block);

            for (impl_item.methods) |method_item_id| {
                const function = switch (self.file.item(method_item_id).*) {
                    .Function => |function| function,
                    else => continue,
                };
                if (function.is_generic or function.is_comptime) continue;
                const symbol_name = try @This().implMethodSymbolName(self, impl_item.trait_name, impl_item.target_name, function.name);
                if (self.monomorphized_function_names.contains(symbol_name)) continue;
                try @This().lowerConcreteFunction(self, method_item_id, function, symbol_name, function.parameters, impl_parent_block, &.{}, null, null);
                try self.monomorphized_function_names.put(symbol_name, {});
            }
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
            if (@This().instantiatedEnumHasPayload(instantiated)) return;
            const template_item = self.file.item(instantiated.template_item_id).Enum;
            const loc = self.location(template_item.range);
            const repr_type = if (instantiated.repr_type) |resolved|
                self.lowerSemaType(resolved, template_item.range)
            else
                defaultIntegerType(self.context);
            const op = mlir.oraEnumDeclOpCreate(self.context, loc, strRef(instantiated.mangled_name), repr_type);
            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;

            if (template_item.variants.len > 0) {
                const variant_names = try self.allocator.alloc(mlir.MlirAttribute, template_item.variants.len);
                const variant_values = try self.allocator.alloc(mlir.MlirAttribute, template_item.variants.len);
                var has_explicit_values = false;
                var next_value: i64 = 0;
                for (template_item.variants, 0..) |variant, index| {
                    variant_names[index] = mlir.oraStringAttrCreate(self.context, strRef(variant.name));
                    const value_attr = try @This().lowerInstantiatedEnumVariantValue(self, instantiated, variant.name, instantiated.variants[index].explicit_value, repr_type, &next_value, &has_explicit_values);
                    variant_values[index] = value_attr;
                }
                mlir.oraOperationSetAttributeByName(op, strRef("ora.variant_names"), mlir.oraArrayAttrCreate(self.context, @intCast(variant_names.len), variant_names.ptr));
                mlir.oraOperationSetAttributeByName(op, strRef("ora.variant_values"), mlir.oraArrayAttrCreate(self.context, @intCast(variant_values.len), variant_values.ptr));
                mlir.oraOperationSetAttributeByName(op, strRef("ora.has_explicit_values"), mlir.oraBoolAttrCreate(self.context, has_explicit_values));
            } else {
                mlir.oraOperationSetAttributeByName(op, strRef("ora.has_explicit_values"), mlir.oraBoolAttrCreate(self.context, false));
            }
            mlir.oraOperationSetAttributeByName(op, strRef("ora.enum_decl"), mlir.oraBoolAttrCreate(self.context, true));

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

        pub fn ensureMonomorphizedFunction(
            self: *Lowerer,
            item_id: ast.ItemId,
            function: ast.FunctionItem,
            call: ast.CallExpr,
            parameters: []const ast.Parameter,
            caller_contract_id: ?ast.ItemId,
        ) anyerror!?[]const u8 {
            if (@This().enclosingImplForMethod(self, item_id)) |impl_item| {
                const symbol_name = try @This().ensureLoweredImplMethod(self, item_id, function, impl_item.trait_name, impl_item.target_name, call, null);
                return symbol_name;
            }
            const scoped_contract_id = if (function.parent_contract == null) caller_contract_id else null;
            const base_symbol_name = if (scoped_contract_id) |contract_id|
                try @This().scopedFunctionSymbolName(self, contract_id, function.name)
            else
                function.name;
            const parent_block = if (function.parent_contract) |contract_id|
                self.contract_body_blocks[contract_id.index()]
            else if (scoped_contract_id) |contract_id|
                self.contract_body_blocks[contract_id.index()]
            else
                self.module_body;

            if (!function.is_generic) {
                if (scoped_contract_id == null) return function.name;
                if (!self.monomorphized_function_names.contains(base_symbol_name)) {
                    try @This().lowerConcreteFunction(self, item_id, function, base_symbol_name, parameters, parent_block, &.{}, null, null);
                    try self.monomorphized_function_names.put(base_symbol_name, {});
                }
                return base_symbol_name;
            }

            const bindings = (try self.genericTypeBindingsForCall(function, call)) orelse return null;
            const mangled_name = try self.mangleGenericFunctionName(base_symbol_name, bindings);
            if (!self.monomorphized_function_names.contains(mangled_name)) {
                try @This().lowerConcreteFunction(self, item_id, function, mangled_name, parameters, parent_block, bindings, null, null);
                try self.monomorphized_function_names.put(mangled_name, {});
            }
            return mangled_name;
        }

        fn scopedFunctionSymbolName(self: *Lowerer, contract_id: ast.ItemId, function_name: []const u8) anyerror![]const u8 {
            const contract = self.file.item(contract_id).Contract;
            return std.fmt.allocPrint(self.allocator, "{s}.{s}", .{ contract.name, function_name });
        }

        pub fn ensureImportedFunctionSymbol(
            self: *Lowerer,
            target_module_id: source.ModuleId,
            item_id: ast.ItemId,
            function: ast.FunctionItem,
            call: ast.CallExpr,
            resolved_call: ?sema.ResolvedCall,
            parent_block: mlir.MlirBlock,
        ) anyerror!?[]const u8 {
            if (function.parent_contract != null or function.is_comptime) return null;

            const query = self.module_query.?;
            const target_file = try query.astFile(target_module_id);
            const target_item_index = try query.itemIndex(target_module_id);
            const target_resolution = try query.nameResolution(target_module_id);
            const target_typecheck = try query.moduleTypeCheck(target_module_id);
            const target_verification_facts = try query.moduleVerificationFacts(target_module_id);
            const target_verification_fact_lookup = try self.makeVerificationFactLookup(target_verification_facts.facts);
            const target_verification_trait_method_fact_lookup = try self.makeVerificationTraitMethodFactLookup(target_verification_facts.facts);
            const target_verification_statement_fact_lookup = try self.makeVerificationStatementFactLookup(target_verification_facts.facts);
            const target_const_eval = try query.constEval(target_module_id);

            const module_name = try self.allocator.dupe(u8, self.sources.module(target_module_id).name);
            for (module_name) |*ch| {
                if (ch.* == '/') ch.* = '.';
            }
            const base_symbol_name = try std.fmt.allocPrint(self.allocator, "{s}.{s}", .{ module_name, function.name });

            const imported_contract_body_blocks = try self.allocator.alloc(mlir.MlirBlock, target_file.items.len);
            @memset(imported_contract_body_blocks, std.mem.zeroes(mlir.MlirBlock));

            var imported_lowerer = Lowerer{
                .allocator = self.allocator,
                .context = self.context,
                .module_id = target_module_id,
                .sources = self.sources,
                .file = target_file,
                .item_index = target_item_index,
                .resolution = target_resolution,
                .const_eval = target_const_eval,
                .typecheck = target_typecheck,
                .verification_facts = target_verification_facts,
                .verification_fact_lookup = target_verification_fact_lookup,
                .verification_trait_method_fact_lookup = target_verification_trait_method_fact_lookup,
                .verification_statement_fact_lookup = target_verification_statement_fact_lookup,
                .module_query = self.module_query,
                .module_body = self.module_body,
                .items = self.items,
                .type_fallbacks = self.type_fallbacks,
                .placeholder_count = self.placeholder_count,
                .default_value_count = self.default_value_count,
                .diagnostics = self.diagnostics,
                .contract_body_blocks = imported_contract_body_blocks,
                .monomorphized_function_names = std.StringHashMap(void).init(self.allocator),
            };
            imported_lowerer.monomorphized_function_names = self.monomorphized_function_names;

            for (target_file.root_items) |root_item_id| {
                switch (target_file.item(root_item_id).*) {
                    .ErrorDecl => |error_decl| {
                        const error_symbol_name = try std.fmt.allocPrint(self.allocator, "{s}.{s}", .{ module_name, error_decl.name });
                        if (!@This().hasLoweredItemSymbol(&imported_lowerer, .error_decl, error_symbol_name)) {
                            try @This().lowerErrorDeclNamed(&imported_lowerer, root_item_id, error_decl, error_symbol_name, parent_block);
                        }
                    },
                    else => {},
                }
            }

            if (function.is_generic) {
                const runtime_parameters = try imported_lowerer.runtimeFunctionParameters(function);
                const bindings = if (resolved_call) |resolved|
                    try @This().hirBindingsFromSema(self, resolved.generic_bindings)
                else blk: {
                    var binding_lowerer = imported_lowerer;
                    binding_lowerer.typecheck = self.typecheck;
                    break :blk (try binding_lowerer.genericTypeBindingsForCall(function, call)) orelse return null;
                };
                const symbol_name = try imported_lowerer.mangleGenericFunctionName(base_symbol_name, bindings);
                if (!self.monomorphized_function_names.contains(symbol_name)) {
                    try @This().lowerConcreteFunction(
                        &imported_lowerer,
                        item_id,
                        function,
                        symbol_name,
                        runtime_parameters,
                        parent_block,
                        bindings,
                        if (resolved_call) |resolved| resolved.runtime_parameter_types else null,
                        if (resolved_call) |resolved| resolved.return_type else null,
                    );
                    try self.monomorphized_function_names.put(symbol_name, {});
                }
                self.items = imported_lowerer.items;
                self.type_fallbacks = imported_lowerer.type_fallbacks;
                self.placeholder_count = imported_lowerer.placeholder_count;
                self.default_value_count = imported_lowerer.default_value_count;
                return symbol_name;
            }

            if (!self.monomorphized_function_names.contains(base_symbol_name)) {
                try @This().lowerConcreteFunction(&imported_lowerer, item_id, function, base_symbol_name, function.parameters, parent_block, &.{}, null, null);
                try self.monomorphized_function_names.put(base_symbol_name, {});
            }
            self.items = imported_lowerer.items;
            self.type_fallbacks = imported_lowerer.type_fallbacks;
            self.placeholder_count = imported_lowerer.placeholder_count;
            self.default_value_count = imported_lowerer.default_value_count;
            return base_symbol_name;
        }

        fn hasLoweredItemSymbol(self: *const Lowerer, kind: HirSymbolKind, symbol_name: []const u8) bool {
            for (self.items.items) |handle| {
                if (handle.kind == kind and std.mem.eql(u8, handle.symbol_name, symbol_name)) return true;
            }
            return false;
        }

        fn hirBindingsFromSema(self: *Lowerer, bindings: []const sema.GenericTypeBinding) anyerror![]const Lowerer.GenericTypeBinding {
            const lowered = try self.allocator.alloc(Lowerer.GenericTypeBinding, bindings.len);
            for (bindings, 0..) |binding, index| {
                lowered[index] = .{
                    .name = binding.name,
                    .value = switch (binding.value) {
                        .ty => |ty| .{ .ty = ty },
                        .integer => |text| .{ .integer = text },
                    },
                    .mangle_name = switch (binding.value) {
                        .ty => |ty| try self.typeMangleName(ty),
                        .integer => |text| text,
                    },
                };
            }
            return lowered;
        }

        fn enclosingImplForMethod(self: *Lowerer, method_item_id: ast.ItemId) ?ast.ImplItem {
            const impl_item_id = self.item_index.lookupImplContainingMethod(method_item_id) orelse return null;
            return self.file.item(impl_item_id).Impl;
        }

        pub fn ensureLoweredImplMethod(
            self: *Lowerer,
            method_item_id: ast.ItemId,
            function: ast.FunctionItem,
            trait_name: []const u8,
            target_name: []const u8,
            call: ?ast.CallExpr,
            parent_block_override: ?mlir.MlirBlock,
        ) anyerror![]const u8 {
            const base_symbol_name = try @This().implMethodSymbolName(self, trait_name, target_name, function.name);
            const parent_block = parent_block_override orelse @This().implParentBlock(self, target_name, self.module_body);
            if (function.is_generic) {
                const method_call = call orelse return base_symbol_name;
                const bindings = (try self.genericTypeBindingsForCall(function, method_call)) orelse return base_symbol_name;
                const runtime_parameters = try self.runtimeFunctionParameters(function);
                const symbol_name = try self.mangleGenericFunctionName(base_symbol_name, bindings);
                if (!self.monomorphized_function_names.contains(symbol_name)) {
                    try @This().lowerConcreteFunction(self, method_item_id, function, symbol_name, runtime_parameters, parent_block, bindings, null, null);
                    try self.monomorphized_function_names.put(symbol_name, {});
                }
                return symbol_name;
            }

            const symbol_name = base_symbol_name;
            if (!self.monomorphized_function_names.contains(symbol_name)) {
                try @This().lowerConcreteFunction(self, method_item_id, function, symbol_name, function.parameters, parent_block, &.{}, null, null);
                try self.monomorphized_function_names.put(symbol_name, {});
            }
            return symbol_name;
        }

        fn implParentBlock(self: *Lowerer, target_name: []const u8, fallback_block: mlir.MlirBlock) mlir.MlirBlock {
            if (self.item_index.lookup(target_name)) |target_item_id| {
                if (self.file.item(target_item_id).* == .Contract) {
                    const block = self.contract_body_blocks[target_item_id.index()];
                    if (!mlir.oraBlockIsNull(block)) return block;
                }
            }
            return fallback_block;
        }

        fn implMethodSymbolName(self: *Lowerer, trait_name: []const u8, target_name: []const u8, method_name: []const u8) anyerror![]const u8 {
            var name = std.ArrayList(u8){};
            try name.appendSlice(self.allocator, trait_name);
            try name.appendSlice(self.allocator, ".");
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
            concrete_runtime_parameter_types: ?[]const sema.Type,
            concrete_return_type: ?sema.Type,
        ) anyerror!void {
            const previous_type_bindings = self.active_type_bindings;
            self.active_type_bindings = type_bindings;
            defer self.active_type_bindings = previous_type_bindings;

            var attrs: std.ArrayList(mlir.MlirNamedAttribute) = .{};
            const return_type = if (function.return_type) |_| blk: {
                if (concrete_return_type) |resolved| break :blk self.lowerSemaType(resolved, function.range);
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
            if (function.visibility == .private and std.mem.indexOfScalar(u8, symbol_name, '.') != null) {
                // Imported private helpers are inlined before OraToSIR call conversion so
                // generic/library code lowers through the caller's real value representation.
                try attrs.append(self.allocator, namedBoolAttr(self.context, "ora.inline", true));
            }
            if (std.mem.eql(u8, function.name, "init")) {
                try attrs.append(self.allocator, namedBoolAttr(self.context, "ora.init", true));
            }
            if (function.visibility == .public and function.parent_contract != null) {
                @This().attachPublicAbiAttrs(self, &attrs, function, parameters) catch |err| switch (err) {
                    error.UnsupportedAbiType => return err,
                    else => return err,
                };
                if (function.abi_decode_permissive) {
                    try attrs.append(self.allocator, namedStringAttr(self.context, "ora.abi_decode_mode", "permissive"));
                }
            }
            if (function.visibility == .private) {
                if (return_type) |ret_type| {
                    if (!mlir.oraTypeIsNull(mlir.oraErrorUnionTypeGetSuccessType(ret_type))) {
                        try attrs.append(self.allocator, namedBoolAttr(self.context, "ora.returns_error_union", true));
                    }
                }
            }
            try @This().attachEffectSummaryAttrs(self, &attrs, item_id);
            try @This().attachModifiesSummaryAttrs(self, &attrs, item_id);

            var param_types: std.ArrayList(mlir.MlirType) = .{};
            var param_locs: std.ArrayList(mlir.MlirLocation) = .{};
            for (parameters, 0..) |parameter, index| {
                const param_type = if (concrete_runtime_parameter_types) |resolved|
                    self.lowerSemaType(resolved[index], parameter.range)
                else if (type_bindings.len == 0)
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
            if (self.ghostDeclarationContextName(item_id)) |context| @This().attachGhostAttrs(self, op, context);

            for (parameters, 0..) |parameter, index| {
                const param_type = if (concrete_runtime_parameter_types) |resolved|
                    resolved[index]
                else
                    self.typecheck.pattern_types[parameter.pattern.index()].type;
                try self.attachBitfieldParamMetadataForType(op, param_type, @intCast(index));
                if (self.errorUnionRequiresWideCarrier(param_type)) {
                    _ = mlir.oraFuncSetArgAttr(op, @intCast(index), strRef("ora.force_wide_error_union"), mlir.oraBoolAttrCreate(self.context, true));
                }
                if (try @This().publicResultInputErrorId(self, param_type)) |error_id| {
                    _ = mlir.oraFuncSetArgAttr(
                        op,
                        @intCast(index),
                        strRef("ora.result_input_error_id"),
                        mlir.oraIntegerAttrCreateI64FromType(defaultIntegerType(self.context), error_id),
                    );
                }
            }

            // Attach ora.param_names so the SMT encoder can use readable variable names.
            if (parameters.len > 0) {
                const param_name_attrs = try self.allocator.alloc(mlir.MlirAttribute, parameters.len);
                defer self.allocator.free(param_name_attrs);
                for (parameters, 0..) |parameter, index| {
                    const param_name = switch (self.file.pattern(parameter.pattern).*) {
                        .Name => |name| name.name,
                        else => "",
                    };
                    param_name_attrs[index] = mlir.oraStringAttrCreate(self.context, strRef(param_name));
                }
                mlir.oraOperationSetAttributeByName(
                    op,
                    strRef("ora.param_names"),
                    mlir.oraArrayAttrCreate(self.context, @intCast(param_name_attrs.len), param_name_attrs.ptr),
                );
            }

            if (return_type != null and self.errorUnionRequiresWideCarrier(concrete_return_type orelse self.typecheck.body_types[function.body.index()])) {
                mlir.oraOperationSetAttributeByName(op, strRef("ora.force_wide_error_union"), mlir.oraBoolAttrCreate(self.context, true));
            }

            appendOp(parent_block, op);
            try self.appendItemHandle(item_id, .function, symbol_name, function.range, op);

            var specialized_function = function;
            specialized_function.name = symbol_name;
            specialized_function.is_generic = false;
            specialized_function.parameters = @constCast(function.parameters);
            var function_lowerer = FunctionLowerer.init(self, item_id, specialized_function, op, return_type);
            function_lowerer.extra_verification_clauses = try @This().traitVerificationClausesForImplMethod(self, item_id);
            try function_lowerer.lower();
        }

        fn attachEffectSummaryAttrs(
            self: *Lowerer,
            attrs: *std.ArrayList(mlir.MlirNamedAttribute),
            item_id: ast.ItemId,
        ) anyerror!void {
            if (item_id.index() >= self.typecheck.item_effects.len) return;

            const effect = self.typecheck.item_effects[item_id.index()];
            switch (effect) {
                .pure => return,
                .external, .side_effects => return,
                .reads => |read_effect| {
                    if (try @This().appendEffectSlotAttrs(self, attrs, "ora.read_slots", read_effect.slots)) {
                        try attrs.append(self.allocator, namedStringAttr(self.context, "ora.effect", "reads"));
                    }
                },
                .writes => |write_effect| {
                    if (try @This().appendEffectSlotAttrs(self, attrs, "ora.write_slots", write_effect.slots)) {
                        try attrs.append(self.allocator, namedStringAttr(self.context, "ora.effect", "writes"));
                    }
                },
                .reads_writes => |read_write| {
                    const has_reads = try @This().appendEffectSlotAttrs(self, attrs, "ora.read_slots", read_write.reads);
                    const has_writes = try @This().appendEffectSlotAttrs(self, attrs, "ora.write_slots", read_write.writes);
                    if (has_reads and has_writes) {
                        try attrs.append(self.allocator, namedStringAttr(self.context, "ora.effect", "readwrites"));
                    } else if (has_reads) {
                        try attrs.append(self.allocator, namedStringAttr(self.context, "ora.effect", "reads"));
                    } else if (has_writes) {
                        try attrs.append(self.allocator, namedStringAttr(self.context, "ora.effect", "writes"));
                    }
                },
            }
        }

        fn attachModifiesSummaryAttrs(
            self: *Lowerer,
            attrs: *std.ArrayList(mlir.MlirNamedAttribute),
            item_id: ast.ItemId,
        ) anyerror!void {
            if (!self.itemHasVerificationFact(item_id, .modifies)) return;
            if (item_id.index() >= self.typecheck.item_modifies.len) return;
            const slots = self.typecheck.item_modifies[item_id.index()] orelse return;

            var slot_attrs: std.ArrayList(mlir.MlirAttribute) = .{};
            defer slot_attrs.deinit(self.allocator);

            for (slots) |slot| {
                if (slot.region != .storage) continue;
                const slot_path = try sema.formatEffectSlotPath(self.allocator, slot);
                defer self.allocator.free(slot_path);
                try slot_attrs.append(
                    self.allocator,
                    mlir.oraStringAttrCreate(self.context, strRef(slot_path)),
                );
            }

            try attrs.append(self.allocator, .{
                .name = identifier(self.context, "ora.modifies_slots"),
                .attribute = mlir.oraArrayAttrCreate(
                    self.context,
                    @intCast(slot_attrs.items.len),
                    if (slot_attrs.items.len == 0) null else slot_attrs.items.ptr,
                ),
            });
        }

        fn appendEffectSlotAttrs(
            self: *Lowerer,
            attrs: *std.ArrayList(mlir.MlirNamedAttribute),
            attr_name: []const u8,
            slots: []const sema.EffectSlot,
        ) anyerror!bool {
            var slot_attrs: std.ArrayList(mlir.MlirAttribute) = .{};
            defer slot_attrs.deinit(self.allocator);

            for (slots) |slot| {
                switch (slot.region) {
                    .storage => try slot_attrs.append(
                        self.allocator,
                        mlir.oraStringAttrCreate(self.context, strRef(slot.name)),
                    ),
                    .transient => {
                        const slot_name = try std.fmt.allocPrint(self.allocator, "transient:{s}", .{slot.name});
                        defer self.allocator.free(slot_name);
                        try slot_attrs.append(
                            self.allocator,
                            mlir.oraStringAttrCreate(self.context, strRef(slot_name)),
                        );
                    },
                    else => {},
                }
            }

            if (slot_attrs.items.len == 0) return false;
            try attrs.append(self.allocator, .{
                .name = identifier(self.context, attr_name),
                .attribute = mlir.oraArrayAttrCreate(self.context, @intCast(slot_attrs.items.len), slot_attrs.items.ptr),
            });
            return true;
        }

        const TraitMethodMatch = struct {
            method: ast.nodes.TraitMethod,
            owner: sema.VerificationTraitMethodOwner,
        };

        fn traitVerificationClausesForImplMethod(self: *Lowerer, method_item_id: ast.ItemId) anyerror![]const FunctionLowerer.ExtraVerificationClause {
            const impl_item = @This().enclosingImplForMethod(self, method_item_id) orelse return &.{};
            const trait_item_id = self.item_index.lookup(impl_item.trait_name) orelse return &.{};

            var clauses: std.ArrayList(FunctionLowerer.ExtraVerificationClause) = .{};
            if (@This().matchingTraitMethodForImplMethod(self, trait_item_id, method_item_id)) |matched| {
                const aliases = try @This().traitMethodPatternAliases(self, matched.method, self.file.item(method_item_id).Function);
                for (self.traitMethodVerificationFactEntries(matched.owner)) |entry| {
                    const fact = self.traitMethodVerificationFact(entry);
                    try @This().appendExtraVerificationClauseFromFact(self, &clauses, fact.*, "trait_method_contract", aliases);
                }
            }

            try @This().appendTraitGhostFactClauses(self, trait_item_id, &clauses);
            return clauses.toOwnedSlice(self.allocator);
        }

        fn appendExtraVerificationClauseFromFact(
            self: *Lowerer,
            clauses: *std.ArrayList(FunctionLowerer.ExtraVerificationClause),
            fact: sema.VerificationFact,
            verification_context: []const u8,
            pattern_aliases: []const FunctionLowerer.PatternAlias,
        ) anyerror!void {
            const kind = fact.kind.specClauseKind() orelse return;
            const expr = fact.expr orelse return error.InvalidVerificationFact;
            try clauses.append(self.allocator, .{
                .kind = kind,
                .expr = expr,
                .range = fact.range,
                .verification_context = verification_context,
                .pattern_aliases = pattern_aliases,
            });
        }

        fn appendTraitGhostFactClauses(
            self: *Lowerer,
            trait_item_id: ast.ItemId,
            clauses: *std.ArrayList(FunctionLowerer.ExtraVerificationClause),
        ) anyerror!void {
            for (self.itemVerificationFactEntries(trait_item_id)) |entry| {
                const fact = self.verificationFact(entry);
                if (fact.context != .trait_ghost_block) continue;
                const kind: ast.SpecClauseKind = switch (fact.kind) {
                    .assert => .ensures,
                    .assume => .requires,
                    .ghost_axiom => continue,
                    else => unreachable,
                };
                const expr = fact.expr orelse return error.InvalidVerificationFact;
                try clauses.append(self.allocator, .{
                    .kind = kind,
                    .expr = expr,
                    .range = fact.range,
                    .verification_context = "ghost_axiom",
                });
            }
        }

        fn matchingTraitMethodForImplMethod(self: *Lowerer, trait_item_id: ast.ItemId, method_item_id: ast.ItemId) ?TraitMethodMatch {
            const method = self.file.item(method_item_id).Function;
            const method_index = self.item_index.lookupTraitMethodIndex(trait_item_id, method.name) orelse return null;
            const trait_item = switch (self.file.item(trait_item_id).*) {
                .Trait => |trait_item| trait_item,
                else => return null,
            };
            if (method_index >= trait_item.methods.len) return null;
            return .{
                .method = trait_item.methods[method_index],
                .owner = .{
                    .trait_item = trait_item_id,
                    .method_index = method_index,
                },
            };
        }

        fn traitMethodPatternAliases(self: *Lowerer, trait_method: ast.nodes.TraitMethod, impl_method: ast.FunctionItem) anyerror![]const FunctionLowerer.PatternAlias {
            const offset: usize = if (self.functionHasRuntimeSelf(impl_method)) 1 else 0;
            if (impl_method.parameters.len < offset + trait_method.parameters.len) return &.{};
            const aliases = try self.allocator.alloc(FunctionLowerer.PatternAlias, trait_method.parameters.len);
            for (trait_method.parameters, 0..) |trait_param, index| {
                aliases[index] = .{
                    .source = trait_param.pattern,
                    .target = impl_method.parameters[index + offset].pattern,
                };
            }
            return aliases;
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
            var result_input_mode_attrs: std.ArrayList(mlir.MlirAttribute) = .{};
            defer result_input_mode_attrs.deinit(self.allocator);
            var result_input_error_id_attrs: std.ArrayList(mlir.MlirAttribute) = .{};
            defer result_input_error_id_attrs.deinit(self.allocator);
            var abi_param_enum_count_attrs: std.ArrayList(mlir.MlirAttribute) = .{};
            defer abi_param_enum_count_attrs.deinit(self.allocator);
            var abi_param_refinement_attrs: std.ArrayList(mlir.MlirAttribute) = .{};
            defer abi_param_refinement_attrs.deinit(self.allocator);

            for (parameters) |parameter| {
                const param_type = self.typecheck.pattern_types[parameter.pattern.index()].type;
                const abi_type = @This().abiLayoutForType(self, param_type) catch |err| switch (err) {
                    error.UnsupportedAbiType => try @This().abiLayoutForTypeExpr(self, parameter.type_expr),
                    else => return err,
                };
                defer self.allocator.free(abi_type);
                try signature_parts.append(self.allocator, try self.allocator.dupe(u8, abi_type));
                abi_param_attrs.append(self.allocator, mlir.oraStringAttrCreate(self.context, strRef(abi_type))) catch return error.OutOfMemory;
                result_input_mode_attrs.append(
                    self.allocator,
                    mlir.oraStringAttrCreate(self.context, strRef(switch (@This().publicResultInputMode(self, param_type)) {
                        .none => "",
                        .narrow_payloadless => "narrow_payloadless",
                        .wide_payloadless => "wide_payloadless",
                        .wide_single_error => "wide_single_error",
                    })),
                ) catch return error.OutOfMemory;
                result_input_error_id_attrs.append(
                    self.allocator,
                    mlir.oraIntegerAttrCreateI64FromType(defaultIntegerType(self.context), @intCast((try @This().publicResultInputErrorId(self, param_type)) orelse 0)),
                ) catch return error.OutOfMemory;
                abi_param_enum_count_attrs.append(
                    self.allocator,
                    mlir.oraIntegerAttrCreateI64FromType(defaultIntegerType(self.context), @intCast(@This().enumVariantCountForType(self, param_type) orelse 0)),
                ) catch return error.OutOfMemory;
                const refinement_spec = try @This().abiParamRefinementSpec(self, param_type);
                defer self.allocator.free(refinement_spec);
                abi_param_refinement_attrs.append(
                    self.allocator,
                    mlir.oraStringAttrCreate(self.context, strRef(refinement_spec)),
                ) catch return error.OutOfMemory;
            }

            if (abi_param_attrs.items.len != 0) {
                const abi_params_attr = mlir.oraArrayAttrCreate(self.context, @intCast(abi_param_attrs.items.len), abi_param_attrs.items.ptr);
                try attrs.append(self.allocator, .{
                    .name = identifier(self.context, "ora.abi_params"),
                    .attribute = abi_params_attr,
                });
                try attrs.append(self.allocator, .{
                    .name = identifier(self.context, "ora.result_input_modes"),
                    .attribute = mlir.oraArrayAttrCreate(self.context, @intCast(result_input_mode_attrs.items.len), result_input_mode_attrs.items.ptr),
                });
                try attrs.append(self.allocator, .{
                    .name = identifier(self.context, "ora.result_input_error_ids"),
                    .attribute = mlir.oraArrayAttrCreate(self.context, @intCast(result_input_error_id_attrs.items.len), result_input_error_id_attrs.items.ptr),
                });
                try attrs.append(self.allocator, .{
                    .name = identifier(self.context, "ora.abi_param_enum_counts"),
                    .attribute = mlir.oraArrayAttrCreate(self.context, @intCast(abi_param_enum_count_attrs.items.len), abi_param_enum_count_attrs.items.ptr),
                });
                try attrs.append(self.allocator, .{
                    .name = identifier(self.context, "ora.abi_param_refinements"),
                    .attribute = mlir.oraArrayAttrCreate(self.context, @intCast(abi_param_refinement_attrs.items.len), abi_param_refinement_attrs.items.ptr),
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
                if (abi_return_type.kind() != .void) {
                    const abi_return = @This().publicReturnAbiTypeForType(self, abi_return_type) catch |err| switch (err) {
                        error.UnsupportedAbiType => blk: {
                            const return_type_id = function.return_type.?;
                            const layout = @This().abiLayoutForType(self, abi_return_type) catch try @This().abiLayoutForTypeExpr(self, return_type_id);
                            break :blk layout;
                        },
                        else => return err,
                    };
                    defer self.allocator.free(abi_return);
                    try attrs.append(self.allocator, namedStringAttr(self.context, "ora.abi_return", abi_return));
                    if (@This().abiLayoutForType(self, abi_return_type) catch |err| switch (err) {
                        error.UnsupportedAbiType => if (function.return_type) |return_type_id|
                            @This().abiLayoutForTypeExpr(self, return_type_id)
                        else
                            return err,
                        else => return err,
                    }) |layout| {
                        defer self.allocator.free(layout);
                        try attrs.append(self.allocator, namedStringAttr(self.context, "ora.abi_return_layout", layout));
                    } else |_| {}
                    if (@This().staticAbiWordCountForType(self, abi_return_type)) |word_count| {
                        try attrs.append(self.allocator, .{
                            .name = identifier(self.context, "ora.abi_return_words"),
                            .attribute = mlir.oraIntegerAttrCreateI64FromType(defaultIntegerType(self.context), @intCast(word_count)),
                        });
                    }
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
                if (self.ghostDeclarationContextName(item_id)) |context| @This().attachGhostAttrs(self, op, context);

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
            if (self.ghostDeclarationContextName(item_id)) |context| @This().attachGhostAttrs(self, op, context);

            if (field.type_expr) |_| {
                try self.attachBitfieldOpMetadataForType(op, self.typecheck.item_types[item_id.index()]);
            }

            appendOp(parent_block, op);
            try self.appendItemHandle(item_id, .field, field.name, field.range, op);
        }

        pub fn lowerConstant(self: *Lowerer, item_id: ast.ItemId, constant: ast.ConstantItem, parent_block: mlir.MlirBlock) anyerror!void {
            const expr = self.file.expression(constant.value).*;
            const sema_result_type = self.typecheck.item_types[item_id.index()];
            const declared_type = if (constant.type_expr) |_|
                self.lowerSemaType(sema_result_type, constant.range)
            else
                self.lowerExprType(constant.value);
            const result_type = if (mlir.oraTypeIsAddressType(declared_type))
                mlir.oraIntegerTypeCreate(self.context, 160)
            else
                declared_type;
            if (self.const_eval.values[constant.value.index()]) |value| {
                if (try @This().constValueAttr(self, value, sema_result_type, result_type)) |value_attr| {
                    const created = mlir.oraConstOpCreate(self.context, self.location(constant.range), strRef(constant.name), value_attr, result_type);
                    if (!mlir.oraOperationIsNull(created)) {
                        if (self.ghostDeclarationContextName(item_id)) |context| @This().attachGhostAttrs(self, created, context);
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
            if (self.ghostDeclarationContextName(item_id)) |context| @This().attachGhostAttrs(self, op, context);
            appendOp(parent_block, op);
            try self.appendItemHandle(item_id, .constant, constant.name, constant.range, op);
        }

        pub fn lowerGhostBlock(self: *Lowerer, item_id: ast.ItemId, ghost_block: ast.GhostBlockItem, parent_block: mlir.MlirBlock) anyerror!void {
            var function_lowerer = FunctionLowerer.initContractContext(self, parent_block);
            function_lowerer.item_id = item_id;
            function_lowerer.in_ghost_context = self.itemHasVerificationFact(item_id, .ghost_block);
            var locals = try function_lowerer.cloneLocals(&function_lowerer.locals);
            _ = try function_lowerer.lowerBody(ghost_block.body, &locals);
        }

        fn constValueAttr(self: *Lowerer, value: ConstValue, sema_type: sema.Type, result_type: mlir.MlirType) anyerror!?mlir.MlirAttribute {
            return switch (value) {
                .integer => |integer| blk: {
                    if (integer.toInt(i64)) |small| {
                        break :blk mlir.oraIntegerAttrCreateI64FromType(result_type, small);
                    } else |_| {}

                    const text = try integer.toString(self.allocator, 10, .lower);
                    break :blk mlir.oraIntegerAttrGetFromString(result_type, strRef(text));
                },
                .fixed_bytes => |bytes| blk: {
                    if (sema_type != .fixed_bytes or sema_type.fixed_bytes.len != bytes.len) break :blk null;
                    var value_int = try std.math.big.int.Managed.initSet(self.allocator, 0);
                    for (bytes) |byte| {
                        var shifted = try std.math.big.int.Managed.init(self.allocator);
                        try std.math.big.int.Managed.shiftLeft(&shifted, &value_int, 8);
                        var byte_int = try std.math.big.int.Managed.initSet(self.allocator, byte);
                        var next = try std.math.big.int.Managed.init(self.allocator);
                        try std.math.big.int.Managed.bitOr(&next, &shifted, &byte_int);
                        value_int = next;
                    }
                    if (value_int.toInt(i64)) |small| {
                        break :blk mlir.oraIntegerAttrCreateI64FromType(result_type, small);
                    } else |_| {}

                    const text = try value_int.toString(self.allocator, 10, .lower);
                    break :blk mlir.oraIntegerAttrGetFromString(result_type, strRef(text));
                },
                .boolean => |boolean| mlir.oraBoolAttrCreate(self.context, boolean),
                .address => |address| blk: {
                    if (!mlir.oraTypeEqual(result_type, support.addressType(self.context))) break :blk null;
                    const i160_type = mlir.oraIntegerTypeCreate(self.context, 160);
                    var decimal_buf: [80]u8 = undefined;
                    const decimal_text = try std.fmt.bufPrint(&decimal_buf, "{}", .{address});
                    break :blk mlir.oraIntegerAttrGetFromString(i160_type, strRef(decimal_text));
                },
                .string => |text| mlir.oraStringAttrCreate(self.context, strRef(text)),
                .tuple => |elements| blk: {
                    if (sema_type != .tuple or sema_type.tuple.len != elements.len) break :blk null;

                    const attrs = try self.allocator.alloc(mlir.MlirAttribute, elements.len);
                    for (elements, sema_type.tuple, 0..) |element_value, element_type, index| {
                        const element_mlir_type = self.lowerSemaType(element_type, source.TextRange.empty(0));
                        attrs[index] = (try @This().constValueAttr(self, element_value, element_type, element_mlir_type)) orelse break :blk null;
                    }
                    break :blk mlir.oraArrayAttrCreate(self.context, @intCast(attrs.len), attrs.ptr);
                },
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
                    field_types[index] = mlir.oraTypeAttrCreateFromType(
                        self.lowerSemaType(
                            try type_descriptors.descriptorFromTypeExpr(self.allocator, self.file, self.item_index, field.type_expr),
                            field.range,
                        ),
                    );
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
            if (@This().enumItemHasPayload(enum_item)) return;
            const loc = self.location(enum_item.range);
            const repr_type = try @This().lowerEnumReprType(self, enum_item);
            const op = mlir.oraEnumDeclOpCreate(self.context, loc, strRef(enum_item.name), repr_type);
            if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;

            if (enum_item.variants.len > 0) {
                const variant_names = try self.allocator.alloc(mlir.MlirAttribute, enum_item.variants.len);
                const variant_values = try self.allocator.alloc(mlir.MlirAttribute, enum_item.variants.len);
                var has_explicit_values = false;
                var next_value: i64 = 0;
                for (enum_item.variants, 0..) |variant, index| {
                    variant_names[index] = mlir.oraStringAttrCreate(self.context, strRef(variant.name));
                    variant_values[index] = try @This().lowerEnumVariantValue(self, enum_item, variant, repr_type, &next_value, &has_explicit_values);
                }
                mlir.oraOperationSetAttributeByName(op, strRef("ora.variant_names"), mlir.oraArrayAttrCreate(self.context, @intCast(variant_names.len), variant_names.ptr));
                mlir.oraOperationSetAttributeByName(op, strRef("ora.variant_values"), mlir.oraArrayAttrCreate(self.context, @intCast(variant_values.len), variant_values.ptr));
                mlir.oraOperationSetAttributeByName(op, strRef("ora.has_explicit_values"), mlir.oraBoolAttrCreate(self.context, has_explicit_values));
            } else {
                mlir.oraOperationSetAttributeByName(op, strRef("ora.has_explicit_values"), mlir.oraBoolAttrCreate(self.context, false));
            }
            mlir.oraOperationSetAttributeByName(op, strRef("ora.enum_decl"), mlir.oraBoolAttrCreate(self.context, true));

            appendOp(parent_block, op);
            try self.appendItemHandle(item_id, .enum_, enum_item.name, enum_item.range, op);
        }

        fn lowerEnumReprType(self: *Lowerer, enum_item: ast.EnumItem) anyerror!mlir.MlirType {
            const type_expr = enum_item.base_type orelse return defaultIntegerType(self.context);
            return self.lowerSemaType(
                try type_descriptors.descriptorFromTypeExpr(self.allocator, self.file, self.item_index, type_expr),
                enum_item.range,
            );
        }

        fn lowerEnumVariantValue(
            self: *Lowerer,
            enum_item: ast.EnumItem,
            variant: ast.EnumVariant,
            repr_type: mlir.MlirType,
            next_value: *i64,
            has_explicit_values: *bool,
        ) anyerror!mlir.MlirAttribute {
            if (mlir.oraTypeEqual(repr_type, support.stringType(self.context))) {
                const text = if (variant.value) |expr_id| blk: {
                    has_explicit_values.* = true;
                    break :blk @This().enumStringValue(self, expr_id) orelse "";
                } else try std.fmt.allocPrint(self.allocator, "{s}.{s}", .{ enum_item.name, variant.name });
                return mlir.oraStringAttrCreate(self.context, strRef(text));
            }
            if (mlir.oraTypeEqual(repr_type, support.bytesType(self.context))) {
                const text = if (variant.value) |expr_id| blk: {
                    has_explicit_values.* = true;
                    break :blk @This().enumBytesValue(self, expr_id) orelse "";
                } else "";
                return mlir.oraStringAttrCreate(self.context, strRef(text));
            }

            const resolved_value = if (variant.value) |expr_id| blk: {
                has_explicit_values.* = true;
                break :blk @This().enumIntegerValue(self, expr_id) orelse 0;
            } else blk: {
                break :blk next_value.*;
            };
            next_value.* = resolved_value + 1;
            return mlir.oraIntegerAttrCreateI64FromType(repr_type, resolved_value);
        }

        fn enumIntegerValue(self: *Lowerer, expr_id: ast.ExprId) ?i64 {
            const value = self.const_eval.values[expr_id.index()] orelse return null;
            return switch (value) {
                .integer => |integer| integer.toInt(i64) catch null,
                .boolean => |boolean| if (boolean) 1 else 0,
                else => null,
            };
        }

        fn enumStringValue(self: *Lowerer, expr_id: ast.ExprId) ?[]const u8 {
            const value = self.const_eval.values[expr_id.index()] orelse return null;
            return switch (value) {
                .string => |string| string,
                else => null,
            };
        }

        fn enumBytesValue(self: *Lowerer, expr_id: ast.ExprId) ?[]const u8 {
            return switch (self.file.expression(expr_id).*) {
                .BytesLiteral => |literal| literal.text,
                .Group => |group| @This().enumBytesValue(self, group.expr),
                else => null,
            };
        }

        fn lowerInstantiatedEnumVariantValue(
            self: *Lowerer,
            instantiated: sema.InstantiatedEnum,
            variant_name: []const u8,
            explicit_value: ?sema.ExplicitEnumValue,
            repr_type: mlir.MlirType,
            next_value: *i64,
            has_explicit_values: *bool,
        ) anyerror!mlir.MlirAttribute {
            if (mlir.oraTypeEqual(repr_type, support.stringType(self.context))) {
                const text = if (explicit_value) |value| blk: {
                    has_explicit_values.* = true;
                    break :blk switch (value) {
                        .string => |literal| literal,
                        else => "",
                    };
                } else try std.fmt.allocPrint(self.allocator, "{s}.{s}", .{ instantiated.mangled_name, variant_name });
                return mlir.oraStringAttrCreate(self.context, strRef(text));
            }
            if (mlir.oraTypeEqual(repr_type, support.bytesType(self.context))) {
                const text = if (explicit_value) |value| blk: {
                    has_explicit_values.* = true;
                    break :blk switch (value) {
                        .bytes => |literal| literal,
                        else => "",
                    };
                } else "";
                return mlir.oraStringAttrCreate(self.context, strRef(text));
            }

            const resolved_value = if (explicit_value) |value| blk: {
                has_explicit_values.* = true;
                break :blk switch (value) {
                    .integer => |literal| literal,
                    else => 0,
                };
            } else blk: {
                break :blk next_value.*;
            };
            next_value.* = resolved_value + 1;
            return mlir.oraIntegerAttrCreateI64FromType(repr_type, resolved_value);
        }

        fn enumItemHasPayload(enum_item: ast.EnumItem) bool {
            for (enum_item.variants) |variant| {
                switch (variant.payload) {
                    .none => {},
                    else => return true,
                }
            }
            return false;
        }

        fn instantiatedEnumHasPayload(instantiated: sema.InstantiatedEnum) bool {
            for (instantiated.variants) |variant| {
                if (variant.payload_type != null) return true;
            }
            return false;
        }

        pub fn lowerLogDecl(self: *Lowerer, item_id: ast.ItemId, log_decl: ast.LogDeclItem, parent_block: mlir.MlirBlock) anyerror!void {
            const loc = self.location(log_decl.range);
            var attrs: std.ArrayList(mlir.MlirNamedAttribute) = .{};
            const event_name = abi_support.eventWireNameFromLogDecl(self.file, log_decl) orelse log_decl.name;
            try attrs.append(self.allocator, namedStringAttr(self.context, "sym_name", log_decl.name));
            try attrs.append(self.allocator, namedStringAttr(self.context, "ora.event_name", event_name));
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
            return @This().lowerErrorDeclNamed(self, item_id, error_decl, error_decl.name, parent_block);
        }

        fn lowerErrorDeclNamed(self: *Lowerer, item_id: ast.ItemId, error_decl: ast.ErrorDeclItem, symbol_name: []const u8, parent_block: mlir.MlirBlock) anyerror!void {
            const loc = self.location(error_decl.range);
            var attrs: std.ArrayList(mlir.MlirNamedAttribute) = .{};
            try attrs.append(self.allocator, namedStringAttr(self.context, "sym_name", symbol_name));
            try attrs.append(self.allocator, namedBoolAttr(self.context, "ora.error_decl", true));
            try attrs.append(self.allocator, .{
                .name = identifier(self.context, "ora.error_id"),
                .attribute = mlir.oraIntegerAttrCreateI64FromType(defaultIntegerType(self.context), try @This().errorDeclRuntimeId(self, error_decl)),
            });

            const param_types = try self.allocator.alloc(sema.Type, error_decl.parameters.len);
            defer self.allocator.free(param_types);
            for (error_decl.parameters, 0..) |param, index| {
                param_types[index] = self.typecheck.pattern_types[param.pattern.index()].type;
            }
            const maybe_signature = @This().abiSignatureForMethod(self, error_decl.name, false, param_types) catch |err| switch (err) {
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
            try self.appendItemHandle(item_id, .error_decl, symbol_name, error_decl.range, op);
        }

        pub fn errorDeclRuntimeIdForItem(self: *Lowerer, item_id: ast.ItemId) anyerror!i64 {
            const item = self.file.item(item_id).*;
            if (item != .ErrorDecl) return error.UnsupportedAbiType;
            return @This().errorDeclRuntimeId(self, item.ErrorDecl);
        }

        fn errorDeclRuntimeId(self: *Lowerer, error_decl: ast.ErrorDeclItem) anyerror!i64 {
            const param_types = try self.allocator.alloc(sema.Type, error_decl.parameters.len);
            defer self.allocator.free(param_types);
            for (error_decl.parameters, 0..) |param, index| {
                param_types[index] = self.typecheck.pattern_types[param.pattern.index()].type;
            }

            const signature = @This().abiSignatureForMethod(self, error_decl.name, false, param_types) catch |err| switch (err) {
                error.UnsupportedAbiType => try std.fmt.allocPrint(self.allocator, "{s}#ora-internal-error/{d}", .{ error_decl.name, error_decl.parameters.len }),
                else => return err,
            };
            defer self.allocator.free(signature);
            return @intCast(abi_support.keccakSelectorValue(signature));
        }

        pub fn attachBitfieldParamMetadata(self: *Lowerer, func_op: mlir.MlirOperation, type_expr_id: ast.TypeExprId, index: c_uint) !void {
            const metadata = (try self.bitfieldMetadataForTypeExpr(type_expr_id)) orelse return;
            _ = mlir.oraFuncSetArgAttr(func_op, index, strRef("ora.bitfield"), mlir.oraStringAttrCreate(self.context, strRef(metadata.name)));
        }

        pub fn attachBitfieldParamMetadataForType(self: *Lowerer, func_op: mlir.MlirOperation, ty: sema.Type, index: c_uint) !void {
            const metadata = (try self.bitfieldMetadataForType(ty)) orelse return;
            _ = mlir.oraFuncSetArgAttr(func_op, index, strRef("ora.bitfield"), mlir.oraStringAttrCreate(self.context, strRef(metadata.name)));
        }

        pub fn attachBitfieldOpMetadata(self: *Lowerer, op: mlir.MlirOperation, type_expr_id: ast.TypeExprId) !void {
            const metadata = (try self.bitfieldMetadataForTypeExpr(type_expr_id)) orelse return;
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

        pub fn bitfieldMetadataForTypeExpr(self: *Lowerer, type_expr_id: ast.TypeExprId) !?BitfieldMetadata {
            const type_expr = self.file.typeExpr(type_expr_id).*;
            const name = switch (type_expr) {
                .Path => |path| path.name,
                else => return null,
            };
            const bitfield = self.bitfieldItemByName(name) orelse return null;
            const layout = try self.buildBitfieldLayout(bitfield);
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
                const resolved = (try self.resolveBitfieldField(bitfield.name, field.name)) orelse continue;
                try buffer.writer(self.allocator).print("{s}:{d}:{d}:{c};", .{ field.name, resolved.offset, resolved.width, resolved.sign });
            }
            return buffer.toOwnedSlice(self.allocator);
        }

        pub fn buildInstantiatedBitfieldLayout(self: *Lowerer, bitfield: sema.InstantiatedBitfield) ![]const u8 {
            var buffer: std.ArrayList(u8) = .{};
            var next_offset: u32 = 0;
            for (bitfield.fields) |field| {
                const width = field.width orelse self.bitfieldFieldWidthFromType(field.ty) orelse return error.InvalidBitfieldFieldType;
                const offset = field.offset orelse next_offset;
                const sign = self.bitfieldFieldSignFromType(field.ty);
                try buffer.writer(self.allocator).print("{s}:{d}:{d}:{c};", .{ field.name, offset, width, sign });
                next_offset = offset + width;
            }
            return buffer.toOwnedSlice(self.allocator);
        }

        fn abiLayoutContext(self: *Lowerer) abi_layout_context.LayoutContext {
            return .{
                .allocator = self.allocator,
                .provider = sema.abiLayoutProvider(self.file, self.item_index, self.typecheck),
            };
        }

        pub fn staticAbiWordCountForType(self: *Lowerer, ty: sema.Type) ?usize {
            const ctx = @This().abiLayoutContext(self);
            return ctx.staticWordCountForType(ty);
        }

        pub fn abiLayoutForType(self: *Lowerer, ty: sema.Type) anyerror![]const u8 {
            const ctx = @This().abiLayoutContext(self);
            return ctx.canonicalAbiTypeForType(ty);
        }

        fn publicReturnAbiTypeForType(self: *Lowerer, ty: sema.Type) anyerror![]const u8 {
            const ctx = @This().abiLayoutContext(self);
            return ctx.publicReturnAbiTypeForType(ty);
        }

        fn abiSignatureForMethod(self: *Lowerer, name: []const u8, has_self: bool, param_types: []const sema.Type) anyerror![]const u8 {
            const ctx = @This().abiLayoutContext(self);
            return ctx.signatureForMethod(name, has_self, param_types);
        }

        fn publicResultInputMode(self: *Lowerer, ty: sema.Type) abi_layout_context.ResultInputMode {
            const ctx = @This().abiLayoutContext(self);
            return ctx.publicResultInputMode(ty);
        }

        fn publicResultInputErrorId(self: *Lowerer, ty: sema.Type) anyerror!?i64 {
            if (@This().publicResultInputMode(self, ty) != .narrow_payloadless) return null;
            const error_union = switch (ty) {
                .error_union => |error_union| error_union,
                else => return null,
            };

            const error_name = error_union.error_types[0].name() orelse return null;
            if (self.item_index.lookup(error_name)) |item_id| {
                if (self.file.item(item_id).* != .ErrorDecl) return null;
                return try @This().errorDeclRuntimeIdForItem(self, item_id);
            }

            var signature_buf: [256]u8 = undefined;
            const signature = std.fmt.bufPrint(&signature_buf, "{s}()", .{error_name}) catch return null;
            return @intCast(abi_support.keccakSelectorValue(signature));
        }

        fn enumVariantCountForType(self: *Lowerer, ty: sema.Type) ?usize {
            const enum_name = ty.name() orelse return null;
            for (self.typecheck.instantiated_enums) |instantiated| {
                if (std.mem.eql(u8, instantiated.mangled_name, enum_name)) return instantiated.variants.len;
            }
            const item_id = self.item_index.lookup(enum_name) orelse return null;
            const enum_item = switch (self.file.item(item_id).*) {
                .Enum => |enum_item| enum_item,
                else => return null,
            };
            return enum_item.variants.len;
        }

        fn abiParamRefinementSpec(self: *Lowerer, ty: sema.Type) ![]const u8 {
            var parts: std.ArrayList([]const u8) = .{};
            defer {
                for (parts.items) |part| self.allocator.free(part);
                parts.deinit(self.allocator);
            }
            try @This().appendAbiParamRefinementSpec(self, &parts, ty);
            if (parts.items.len == 0) return self.allocator.dupe(u8, "");
            return std.mem.join(self.allocator, ";", parts.items);
        }

        fn appendAbiParamRefinementSpec(self: *Lowerer, parts: *std.ArrayList([]const u8), ty: sema.Type) !void {
            switch (ty) {
                .refinement => |refinement| {
                    try @This().appendAbiParamRefinementSpec(self, parts, refinement.base_type.*);
                    if (refinements.kindForName(refinement.name) == .non_zero_address) {
                        try parts.append(self.allocator, try self.allocator.dupe(u8, "nonzero_address"));
                        return;
                    }
                    if (refinements.bounds(refinement)) |bounds| {
                        const signed = try @This().refinementBaseIsSignedInteger(bounds.base_type);
                        if (bounds.min_text) |min_text| {
                            const min_value = try @This().parseAbiRefinementBound(min_text, signed);
                            try @This().appendAbiRefinementBound(self, parts, "min", signed, min_value);
                        }
                        if (bounds.max_text) |max_text| {
                            const max_value = try @This().parseAbiRefinementBound(max_text, signed);
                            try @This().appendAbiRefinementBound(self, parts, "max", signed, max_value);
                        }
                    }
                },
                else => {},
            }
        }

        fn appendAbiRefinementBound(self: *Lowerer, parts: *std.ArrayList([]const u8), tag: []const u8, signed: bool, value: u256) !void {
            const high_high: u64 = @truncate(value >> 192);
            const high_low: u64 = @truncate(value >> 128);
            const low_high: u64 = @truncate(value >> 64);
            const low_low: u64 = @truncate(value);
            try parts.append(self.allocator, try std.fmt.allocPrint(
                self.allocator,
                "{s}:{s}:{d}:{d}:{d}:{d}",
                .{ tag, if (signed) "s" else "u", high_high, high_low, low_high, low_low },
            ));
        }

        fn parseAbiRefinementBound(text: []const u8, signed: bool) !u256 {
            if (signed) {
                const parsed = std.fmt.parseInt(i256, text, 10) catch return error.MlirOperationCreationFailed;
                return @bitCast(parsed);
            }
            return std.fmt.parseInt(u256, text, 10) catch error.MlirOperationCreationFailed;
        }

        fn refinementBaseIsSignedInteger(ty: sema.Type) anyerror!bool {
            return (try support.resolvedIntegerSignedness(ty)) orelse error.MlirOperationCreationFailed;
        }

        pub fn abiLayoutForTypeExpr(self: *Lowerer, type_expr_id: ast.TypeExprId) anyerror![]const u8 {
            const ctx = @This().abiLayoutContext(self);
            return ctx.canonicalAbiTypeForTypeExpr(type_expr_id);
        }

        pub fn staticAbiWordCountForTypeExpr(self: *Lowerer, type_expr_id: ast.TypeExprId) ?usize {
            const ctx = @This().abiLayoutContext(self);
            return ctx.staticWordCountForTypeExpr(type_expr_id);
        }

        pub fn bitfieldFieldSign(self: *const Lowerer, type_expr_id: ast.TypeExprId) u8 {
            return switch (self.file.typeExpr(type_expr_id).*) {
                .Path => |path| blk: {
                    const trimmed = std.mem.trim(u8, path.name, " \t\n\r");
                    if (support.parseBitfieldIntegerType(trimmed)) |int_info| {
                        if (int_info.signed) break :blk 's';
                    }
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
            if (std.mem.endsWith(u8, op_name, "_placeholder")) {
                self.recordPlaceholder();
            }
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
