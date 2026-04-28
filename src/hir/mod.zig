const std = @import("std");
const mlir = @import("mlir_c_api").c;
const ast = @import("../ast/mod.zig");
const sema = @import("../sema/mod.zig");
const sema_model = @import("../sema/model.zig");
const source = @import("../source/mod.zig");
const contract_lowering = @import("contract_lowering.zig");
const control_flow = @import("control_flow.zig");
const expr_lowering = @import("expr_lowering.zig");
const function_core = @import("function_core.zig");
const hir_locals = @import("locals.zig");
const module_lowering = @import("module_lowering.zig");
const refinement_cleanup = @import("refinement_cleanup.zig");
const support = @import("support.zig");
const type_descriptors = @import("../sema/type_descriptors.zig");

pub const abi = @import("abi.zig");

pub const ModuleQuery = struct {
    context: *anyopaque,
    ast_file: *const fn (context: *anyopaque, module_id: source.ModuleId) anyerror!*const ast.AstFile,
    item_index: *const fn (context: *anyopaque, module_id: source.ModuleId) anyerror!*const sema.ItemIndexResult,
    resolution: *const fn (context: *anyopaque, module_id: source.ModuleId) anyerror!*const sema.NameResolutionResult,
    module_typecheck: *const fn (context: *anyopaque, module_id: source.ModuleId) anyerror!*const sema.TypeCheckResult,
    const_eval: *const fn (context: *anyopaque, module_id: source.ModuleId) anyerror!*const sema.ConstEvalResult,
    lookup_item: *const fn (context: *anyopaque, module_id: source.ModuleId, name: []const u8) anyerror!?ast.ItemId,
    resolve_import_alias: *const fn (context: *anyopaque, module_id: source.ModuleId, alias: []const u8) anyerror!?source.ModuleId,
};

const descriptorFromPathName = sema.descriptorFromPathName;
const appendSemaTypeMangleName = sema.appendTypeMangleName;

pub const HirSymbolKind = enum {
    contract,
    function,
    struct_,
    bitfield,
    enum_,
    log_decl,
    error_decl,
    field,
    constant,
    unknown,
};

pub const HirItemHandle = struct {
    item_id: ast.ItemId,
    kind: HirSymbolKind,
    symbol_name: []const u8,
    location: source.SourceLocation,
    raw_operation: mlir.MlirOperation = std.mem.zeroes(mlir.MlirOperation),
};

pub const HirModuleHandle = struct {
    module_id: source.ModuleId,
    file_id: source.FileId,
    raw_module: mlir.MlirModule = std.mem.zeroes(mlir.MlirModule),
};

pub const TypeFallbackReason = enum {
    sema_unknown,
    unsupported_function_sema_type,
    unsupported_tuple_sema_type,
    syntax_error_type,
    unsupported_syntax_type,
    invalid_generic_type_arg,
};

pub const TypeFallbackRecord = struct {
    reason: TypeFallbackReason,
    location: source.SourceLocation,
};

const SyntheticFrame = struct {
    index: u32,
    count: u32,
};

pub const LoweringResult = struct {
    arena: std.heap.ArenaAllocator,
    context: mlir.MlirContext = std.mem.zeroes(mlir.MlirContext),
    module: HirModuleHandle,
    items: []HirItemHandle,
    type_fallback_count: usize = 0,
    type_fallbacks: []const TypeFallbackRecord = &.{},

    pub fn deinit(self: *LoweringResult) void {
        if (!mlir.oraModuleIsNull(self.module.raw_module)) {
            mlir.oraModuleDestroy(self.module.raw_module);
        }
        if (self.context.ptr != null) {
            mlir.oraContextDestroy(self.context);
        }
        self.arena.deinit();
    }

    pub fn renderText(self: *const LoweringResult, allocator: std.mem.Allocator) ![]u8 {
        const module_op = mlir.oraModuleGetOperation(self.module.raw_module);
        const text_ref = mlir.oraOperationPrintToString(module_op);
        defer if (text_ref.data != null) mlir.oraStringRefFree(text_ref);

        if (text_ref.data == null or text_ref.length == 0) {
            return allocator.dupe(u8, "");
        }
        return allocator.dupe(u8, text_ref.data[0..text_ref.length]);
    }

    pub fn cleanupRefinementGuards(self: *LoweringResult, proven_guard_ids: *const std.StringHashMap(void)) void {
        refinement_cleanup.cleanupRefinementGuards(self.context, self.module.raw_module, proven_guard_ids);
    }
};

pub fn lowerModule(
    allocator: std.mem.Allocator,
    sources: *const source.SourceStore,
    module_id: source.ModuleId,
    file: *const ast.AstFile,
    item_index: *const sema.ItemIndexResult,
    resolution: *const sema.NameResolutionResult,
    const_eval: *const sema.ConstEvalResult,
    typecheck: *const sema.TypeCheckResult,
    module_query: ?ModuleQuery,
) !LoweringResult {
    var result = LoweringResult{
        .arena = std.heap.ArenaAllocator.init(allocator),
        .module = .{
            .module_id = module_id,
            .file_id = file.file_id,
        },
        .items = &[_]HirItemHandle{},
        .type_fallbacks = &.{},
    };
    errdefer result.deinit();

    result.context = try support.createContext();

    const root_range = if (file.root_items.len > 0)
        support.itemRange(file, file.root_items[0])
    else
        source.TextRange.empty(0);
    result.module.raw_module = mlir.oraModuleCreateEmpty(support.locationFromRange(result.context, sources, file.file_id, root_range));
    if (mlir.oraModuleIsNull(result.module.raw_module)) return error.MlirModuleCreationFailed;

    var lowerer = Lowerer{
        .allocator = result.arena.allocator(),
        .context = result.context,
        .module_id = module_id,
        .sources = sources,
        .file = file,
        .item_index = item_index,
        .resolution = resolution,
        .const_eval = const_eval,
        .typecheck = typecheck,
        .module_query = module_query,
        .module_body = mlir.oraModuleGetBody(result.module.raw_module),
        .items = .{},
        .type_fallbacks = .{},
        .contract_body_blocks = try result.arena.allocator().alloc(mlir.MlirBlock, file.items.len),
        .monomorphized_function_names = std.StringHashMap(void).init(result.arena.allocator()),
    };
    @memset(lowerer.contract_body_blocks, std.mem.zeroes(mlir.MlirBlock));

    for (typecheck.instantiated_structs) |instantiated| {
        if (lowerer.enclosingContractForItem(instantiated.template_item_id) != null) continue;
        try lowerer.lowerInstantiatedStructDecl(instantiated, lowerer.module_body);
    }
    for (typecheck.instantiated_enums) |instantiated| {
        if (lowerer.enclosingContractForItem(instantiated.template_item_id) != null) continue;
        try lowerer.lowerInstantiatedEnumDecl(instantiated, lowerer.module_body);
    }
    for (typecheck.instantiated_bitfields) |instantiated| {
        if (lowerer.enclosingContractForItem(instantiated.template_item_id) != null) continue;
        try lowerer.lowerInstantiatedBitfieldDecl(instantiated, lowerer.module_body);
    }

    for (file.root_items) |item_id| {
        try lowerer.lowerItem(item_id, lowerer.module_body);
    }

    result.items = try lowerer.items.toOwnedSlice(result.arena.allocator());
    result.type_fallbacks = try lowerer.type_fallbacks.toOwnedSlice(result.arena.allocator());
    result.type_fallback_count = result.type_fallbacks.len;
    return result;
}

const Lowerer = struct {
    pub const GenericBindingValue = union(enum) {
        ty: sema.Type,
        integer: []const u8,
    };

    pub const GenericTypeBinding = struct {
        name: []const u8,
        value: GenericBindingValue,
        mangle_name: []const u8,
    };

    pub const ResolvedBitfieldField = struct {
        field: ast.BitfieldField,
        field_type: ?sema.Type = null,
        offset: u32,
        width: u32,
        sign: u8,
    };

    allocator: std.mem.Allocator,
    context: mlir.MlirContext,
    module_id: source.ModuleId,
    sources: *const source.SourceStore,
    file: *const ast.AstFile,
    item_index: *const sema.ItemIndexResult,
    resolution: *const sema.NameResolutionResult,
    const_eval: *const sema.ConstEvalResult,
    typecheck: *const sema.TypeCheckResult,
    module_query: ?ModuleQuery,
    module_body: mlir.MlirBlock,
    items: std.ArrayList(HirItemHandle),
    type_fallbacks: std.ArrayList(TypeFallbackRecord),
    guarded_storage_roots: ?*const std.StringHashMap(void) = null,
    contract_body_blocks: []mlir.MlirBlock,
    monomorphized_function_names: std.StringHashMap(void),
    active_type_bindings: []const GenericTypeBinding = &.{},
    current_statement_id: ?ast.StmtId = null,
    current_synthetic_index: ?u32 = null,
    current_synthetic_count: ?u32 = null,
    synthetic_stack_len: u8 = 0,
    synthetic_stack: [16]SyntheticFrame = undefined,

    const ModuleLowering = module_lowering.mixin(Lowerer, ContractLowerer, FunctionLowerer, HirSymbolKind);
    pub const lowerItem = ModuleLowering.lowerItem;
    pub const lowerContract = ModuleLowering.lowerContract;
    pub const lowerFunction = ModuleLowering.lowerFunction;
    pub const lowerInstantiatedStructDecl = ModuleLowering.lowerInstantiatedStructDecl;
    pub const lowerInstantiatedEnumDecl = ModuleLowering.lowerInstantiatedEnumDecl;
    pub const lowerInstantiatedBitfieldDecl = ModuleLowering.lowerInstantiatedBitfieldDecl;
    pub const lowerStructDecl = ModuleLowering.lowerStructDecl;
    pub const lowerEnumDecl = ModuleLowering.lowerEnumDecl;
    pub const lowerField = ModuleLowering.lowerField;
    pub const lowerConstant = ModuleLowering.lowerConstant;
    pub const lowerLogDecl = ModuleLowering.lowerLogDecl;
    pub const lowerErrorDecl = ModuleLowering.lowerErrorDecl;
    pub const lowerDeclPlaceholder = ModuleLowering.lowerDeclPlaceholder;
    pub const createNamedPlaceholderOp = ModuleLowering.createNamedPlaceholderOp;
    pub const createPlaceholderOp = ModuleLowering.createPlaceholderOp;
    pub const appendItemHandle = ModuleLowering.appendItemHandle;
    pub const enclosingContractForItem = ModuleLowering.enclosingContractForItem;
    pub const ensureMonomorphizedFunction = ModuleLowering.ensureMonomorphizedFunction;
    pub const ensureImportedFunctionSymbol = ModuleLowering.ensureImportedFunctionSymbol;
    pub const ensureLoweredImplMethod = ModuleLowering.ensureLoweredImplMethod;
    pub const attachBitfieldParamMetadata = ModuleLowering.attachBitfieldParamMetadata;
    pub const attachBitfieldParamMetadataForType = ModuleLowering.attachBitfieldParamMetadataForType;
    pub const attachBitfieldOpMetadata = ModuleLowering.attachBitfieldOpMetadata;
    pub const attachBitfieldOpMetadataForType = ModuleLowering.attachBitfieldOpMetadataForType;
    pub const bitfieldMetadataForTypeExpr = ModuleLowering.bitfieldMetadataForTypeExpr;
    pub const bitfieldMetadataForType = ModuleLowering.bitfieldMetadataForType;
    pub const buildBitfieldLayout = ModuleLowering.buildBitfieldLayout;
    pub const buildInstantiatedBitfieldLayout = ModuleLowering.buildInstantiatedBitfieldLayout;
    pub const bitfieldFieldSign = ModuleLowering.bitfieldFieldSign;

    pub fn location(self: *const Lowerer, range: source.TextRange) mlir.MlirLocation {
        var loc = support.locationFromRangeWithStmt(self.context, self.sources, self.file.file_id, range, self.current_statement_id);
        var i: usize = 0;
        while (i < self.synthetic_stack_len) : (i += 1) {
            const frame = self.synthetic_stack[i];
            loc = mlir.oraLocationSyntheticTaggedGet(
                self.context,
                loc,
                frame.index,
                frame.count,
            );
        }
        return loc;
    }

    pub fn pushSyntheticFrame(self: *Lowerer, index: u32, count: u32) void {
        // This stack should never overflow under the current nested-unroll policy.
        // Keep the bound explicit so future budget changes do not silently truncate provenance.
        std.debug.assert(self.synthetic_stack_len < self.synthetic_stack.len);
        self.synthetic_stack[self.synthetic_stack_len] = .{ .index = index, .count = count };
        self.synthetic_stack_len += 1;
        self.current_synthetic_index = index;
        self.current_synthetic_count = count;
    }

    pub fn popSyntheticFrame(self: *Lowerer) void {
        if (self.synthetic_stack_len == 0) {
            self.current_synthetic_index = null;
            self.current_synthetic_count = null;
            return;
        }
        self.synthetic_stack_len -= 1;
        if (self.synthetic_stack_len == 0) {
            self.current_synthetic_index = null;
            self.current_synthetic_count = null;
            return;
        }
        const frame = self.synthetic_stack[self.synthetic_stack_len - 1];
        self.current_synthetic_index = frame.index;
        self.current_synthetic_count = frame.count;
    }

    pub fn substitutedType(self: *const Lowerer, name: []const u8) ?sema.Type {
        for (self.active_type_bindings) |binding| {
            if (!std.mem.eql(u8, binding.name, name)) continue;
            switch (binding.value) {
                .ty => |ty| return ty,
                .integer => {},
            }
        }
        return null;
    }

    pub fn substitutedInteger(self: *const Lowerer, name: []const u8) ?[]const u8 {
        for (self.active_type_bindings) |binding| {
            if (!std.mem.eql(u8, binding.name, name)) continue;
            switch (binding.value) {
                .integer => |text| return text,
                .ty => {},
            }
        }
        return null;
    }

    pub fn lowerTypeExpr(self: *Lowerer, type_expr_id: ast.TypeExprId) mlir.MlirType {
        return switch (self.file.typeExpr(type_expr_id).*) {
            .Path => |path| self.lowerNamedPathType(path.name),
            .Generic => |generic| lowerGenericType(self, generic),
            .Tuple => |tuple| blk: {
                var element_types: std.ArrayList(mlir.MlirType) = .{};
                for (tuple.elements) |element| {
                    element_types.append(self.allocator, self.lowerTypeExpr(element)) catch
                        break :blk self.recordTypeFallback(.unsupported_syntax_type, self.typeExprRange(type_expr_id));
                }
                break :blk mlir.oraTupleTypeGet(
                    self.context,
                    element_types.items.len,
                    if (element_types.items.len == 0) null else element_types.items.ptr,
                );
            },
            .AnonymousStruct => |struct_type| blk: {
                var field_types: std.ArrayList(mlir.MlirType) = .{};
                for (struct_type.fields) |field| {
                    field_types.append(self.allocator, self.lowerTypeExpr(field.type_expr)) catch
                        break :blk self.recordTypeFallback(.unsupported_syntax_type, self.typeExprRange(type_expr_id));
                }
                break :blk mlir.oraTupleTypeGet(
                    self.context,
                    field_types.items.len,
                    if (field_types.items.len == 0) null else field_types.items.ptr,
                );
            },
            .Array => |array| support.arrayMemRefType(self.context, self.lowerTypeExpr(array.element), self.lowerArraySize(array.size) orelse 0),
            .Slice => |slice| support.sliceMemRefType(self.context, self.lowerTypeExpr(slice.element)),
            .ErrorUnion => |error_union| blk: {
                const payload_type = self.lowerTypeExpr(error_union.payload);
                var error_types: std.ArrayList(mlir.MlirType) = .{};
                for (error_union.errors) |error_type_expr| {
                    error_types.append(self.allocator, self.lowerTypeExpr(error_type_expr)) catch
                        break :blk self.recordTypeFallback(.unsupported_syntax_type, self.typeExprRange(type_expr_id));
                }
                break :blk mlir.oraErrorUnionTypeGetWithErrors(
                    self.context,
                    payload_type,
                    error_types.items.len,
                    if (error_types.items.len == 0) null else error_types.items.ptr,
                );
            },
            .Error => self.recordTypeFallback(.syntax_error_type, self.typeExprRange(type_expr_id)),
        };
    }

    fn lowerArraySize(self: *const Lowerer, size: ast.TypeArraySize) ?u32 {
        return switch (size) {
            .Integer => |literal| support.parseArrayLen(literal.text),
            .Name => |name| if (self.substitutedInteger(std.mem.trim(u8, name.name, " \t\n\r"))) |text|
                support.parseArrayLen(text)
            else
                null,
        };
    }

    pub fn lowerExprType(self: *Lowerer, expr_id: ast.ExprId) mlir.MlirType {
        return self.lowerSemaType(self.typecheck.exprType(expr_id), support.exprRange(self.file, expr_id));
    }

    pub fn errorTypeHasPayload(self: *const Lowerer, ty: sema.Type) bool {
        const error_name = ty.name() orelse return false;
        const item_id = self.item_index.lookup(error_name) orelse return false;
        return switch (self.file.item(item_id).*) {
            .ErrorDecl => |error_decl| error_decl.parameters.len != 0,
            else => false,
        };
    }

    pub fn errorUnionRequiresWideCarrier(self: *const Lowerer, ty: sema.Type) bool {
        return switch (ty) {
            .error_union => |error_union| blk: {
                for (error_union.error_types) |error_type| {
                    if (self.errorTypeHasPayload(error_type)) break :blk true;
                }
                break :blk false;
            },
            else => false,
        };
    }

    pub fn lowerSemaType(self: *Lowerer, ty: sema.Type, range: source.TextRange) mlir.MlirType {
        return switch (ty) {
            .bool => support.boolType(self.context),
            .integer => |integer| if (integer.spelling) |name| support.lowerPathType(self.context, name) else support.defaultIntegerType(self.context),
            .address => support.addressType(self.context),
            .string => support.stringType(self.context),
            .bytes => support.bytesType(self.context),
            .external_proxy => support.addressType(self.context),
            .void => mlir.oraNoneTypeCreate(self.context),
            .array => |array| support.arrayMemRefType(self.context, self.lowerSemaType(array.element_type.*, range), array.len orelse 0),
            .slice => |slice| support.sliceMemRefType(self.context, self.lowerSemaType(slice.element_type.*, range)),
            .map => |map| mlir.oraMapTypeGet(
                self.context,
                if (map.key_type) |key| self.lowerSemaType(key.*, range) else support.defaultIntegerType(self.context),
                if (map.value_type) |value| self.lowerSemaType(value.*, range) else support.defaultIntegerType(self.context),
            ),
            .refinement => |refinement| blk: {
                const base_type = refinement.base_type.*;
                if (base_type == .named) {
                    if (self.substitutedType(base_type.named.name)) |substituted_base| {
                        var base_copy = substituted_base;
                        var refinement_copy = refinement;
                        refinement_copy.base_type = &base_copy;
                        break :blk support.lowerRefinementType(self.context, refinement_copy, self.allocator) catch
                            return self.recordTypeFallback(.unsupported_syntax_type, range);
                    }
                }
                break :blk support.lowerRefinementType(self.context, refinement, self.allocator) catch
                    return self.recordTypeFallback(.unsupported_syntax_type, range);
            },
            .struct_ => |named| mlir.oraStructTypeGet(self.context, support.strRef(named.name)),
            .contract => |named| mlir.oraStructTypeGet(self.context, support.strRef(named.name)),
            // Bitfields are lowered as packed integer wire values with attrs carrying layout metadata.
            .bitfield => support.defaultIntegerType(self.context),
            .enum_ => |named| self.lowerEnumSemaType(named.name, range),
            .named => |named| if (self.substitutedType(named.name)) |substituted|
                self.lowerSemaType(substituted, range)
            else blk: {
                if (self.item_index.lookup(named.name)) |item_id| {
                    if (self.file.item(item_id).* == .Enum) break :blk self.lowerEnumSemaType(named.name, range);
                }
                break :blk mlir.oraStructTypeGet(self.context, support.strRef(named.name));
            },
            .error_union => |error_union| self.lowerErrorUnionSemaType(error_union, range),
            .unknown => self.recordTypeFallback(.sema_unknown, range),
            .function => |function| blk: {
                var param_types: std.ArrayList(mlir.MlirType) = .{};
                for (function.param_types) |param_type| {
                    param_types.append(self.allocator, self.lowerSemaType(param_type, range)) catch
                        break :blk self.recordTypeFallback(.unsupported_function_sema_type, range);
                }

                var result_types: std.ArrayList(mlir.MlirType) = .{};
                for (function.return_types) |return_type| {
                    result_types.append(self.allocator, self.lowerSemaType(return_type, range)) catch
                        break :blk self.recordTypeFallback(.unsupported_function_sema_type, range);
                }

                break :blk mlir.oraOraFunctionTypeGet(
                    self.context,
                    param_types.items.len,
                    if (param_types.items.len == 0) null else param_types.items.ptr,
                    result_types.items.len,
                    if (result_types.items.len == 0) null else result_types.items.ptr,
                );
            },
            .tuple => |elements| blk: {
                var element_types: std.ArrayList(mlir.MlirType) = .{};
                for (elements) |element| {
                    element_types.append(self.allocator, self.lowerSemaType(element, range)) catch
                        break :blk self.recordTypeFallback(.unsupported_tuple_sema_type, range);
                }
                break :blk mlir.oraTupleTypeGet(
                    self.context,
                    element_types.items.len,
                    if (element_types.items.len == 0) null else element_types.items.ptr,
                );
            },
            .anonymous_struct => |struct_type| blk: {
                var field_names: std.ArrayList(mlir.MlirStringRef) = .{};
                var field_types: std.ArrayList(mlir.MlirType) = .{};
                for (struct_type.fields) |field| {
                    field_names.append(self.allocator, support.strRef(field.name)) catch
                        break :blk self.recordTypeFallback(.unsupported_tuple_sema_type, range);
                    field_types.append(self.allocator, self.lowerSemaType(field.ty, range)) catch
                        break :blk self.recordTypeFallback(.unsupported_tuple_sema_type, range);
                }
                break :blk mlir.oraAnonymousStructTypeGet(
                    self.context,
                    field_types.items.len,
                    if (field_names.items.len == 0) null else field_names.items.ptr,
                    if (field_types.items.len == 0) null else field_types.items.ptr,
                );
            },
        };
    }

    fn lowerEnumSemaType(self: *Lowerer, name: []const u8, range: source.TextRange) mlir.MlirType {
        if (self.typecheck.instantiatedEnumByName(name)) |instantiated| {
            if (enumInstHasPayload(instantiated)) return self.lowerInstantiatedEnumAdtType(instantiated, range);
            if (instantiated.repr_type) |repr| return self.lowerSemaType(repr, range);
            return support.defaultIntegerType(self.context);
        }
        const item_id = self.item_index.lookup(name) orelse return support.defaultIntegerType(self.context);
        const enum_item = switch (self.file.item(item_id).*) {
            .Enum => |enum_item| enum_item,
            else => return support.defaultIntegerType(self.context),
        };
        if (enumItemHasPayload(enum_item)) return self.lowerEnumAdtType(enum_item, range);
        const base_type = enum_item.base_type orelse return support.defaultIntegerType(self.context);
        const repr = type_descriptors.descriptorFromTypeExpr(self.allocator, self.file, self.item_index, base_type) catch
            return self.recordTypeFallback(.unsupported_syntax_type, range);
        return self.lowerSemaType(repr, range);
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

    fn enumInstHasPayload(instantiated: sema.InstantiatedEnum) bool {
        for (instantiated.variants) |variant| {
            if (variant.payload_type != null) return true;
        }
        return false;
    }

    fn lowerInstantiatedEnumAdtType(self: *Lowerer, instantiated: sema.InstantiatedEnum, range: source.TextRange) mlir.MlirType {
        var variant_names: std.ArrayList(mlir.MlirStringRef) = .{};
        var payload_types: std.ArrayList(mlir.MlirType) = .{};
        for (instantiated.variants) |variant| {
            variant_names.append(self.allocator, support.strRef(variant.name)) catch
                return self.recordTypeFallback(.unsupported_syntax_type, range);
            payload_types.append(
                self.allocator,
                if (variant.payload_type) |payload| self.lowerSemaType(payload, range) else mlir.oraNoneTypeCreate(self.context),
            ) catch return self.recordTypeFallback(.unsupported_syntax_type, range);
        }
        return mlir.oraAdtTypeGet(
            self.context,
            support.strRef(instantiated.mangled_name),
            variant_names.items.len,
            if (variant_names.items.len == 0) null else variant_names.items.ptr,
            if (payload_types.items.len == 0) null else payload_types.items.ptr,
        );
    }

    fn lowerEnumAdtType(self: *Lowerer, enum_item: ast.EnumItem, range: source.TextRange) mlir.MlirType {
        var variant_names: std.ArrayList(mlir.MlirStringRef) = .{};
        var payload_types: std.ArrayList(mlir.MlirType) = .{};
        for (enum_item.variants) |variant| {
            variant_names.append(self.allocator, support.strRef(variant.name)) catch
                return self.recordTypeFallback(.unsupported_syntax_type, range);
            payload_types.append(self.allocator, self.lowerEnumVariantPayloadType(variant.payload, variant.range)) catch
                return self.recordTypeFallback(.unsupported_syntax_type, range);
        }
        return mlir.oraAdtTypeGet(
            self.context,
            support.strRef(enum_item.name),
            variant_names.items.len,
            if (variant_names.items.len == 0) null else variant_names.items.ptr,
            if (payload_types.items.len == 0) null else payload_types.items.ptr,
        );
    }

    fn lowerEnumVariantPayloadType(self: *Lowerer, payload: ast.EnumVariantPayload, range: source.TextRange) mlir.MlirType {
        return switch (payload) {
            .none => mlir.oraNoneTypeCreate(self.context),
            .positional => |types| blk: {
                if (types.len == 0) break :blk mlir.oraNoneTypeCreate(self.context);
                if (types.len == 1) break :blk self.lowerEnumPayloadTypeExpr(types[0], range);
                var element_types: std.ArrayList(mlir.MlirType) = .{};
                for (types) |type_expr| {
                    element_types.append(self.allocator, self.lowerEnumPayloadTypeExpr(type_expr, range)) catch
                        break :blk self.recordTypeFallback(.unsupported_syntax_type, range);
                }
                break :blk mlir.oraTupleTypeGet(
                    self.context,
                    element_types.items.len,
                    if (element_types.items.len == 0) null else element_types.items.ptr,
                );
            },
            .named => |fields| blk: {
                if (fields.len == 0) break :blk mlir.oraNoneTypeCreate(self.context);
                var field_names: std.ArrayList(mlir.MlirStringRef) = .{};
                var field_types: std.ArrayList(mlir.MlirType) = .{};
                for (fields) |field| {
                    field_names.append(self.allocator, support.strRef(field.name)) catch
                        break :blk self.recordTypeFallback(.unsupported_syntax_type, range);
                    field_types.append(self.allocator, self.lowerEnumPayloadTypeExpr(field.type_expr, field.range)) catch
                        break :blk self.recordTypeFallback(.unsupported_syntax_type, range);
                }
                break :blk mlir.oraAnonymousStructTypeGet(
                    self.context,
                    field_types.items.len,
                    if (field_names.items.len == 0) null else field_names.items.ptr,
                    if (field_types.items.len == 0) null else field_types.items.ptr,
                );
            },
        };
    }

    fn lowerEnumPayloadTypeExpr(self: *Lowerer, type_expr: ast.TypeExprId, range: source.TextRange) mlir.MlirType {
        const ty = type_descriptors.descriptorFromTypeExpr(self.allocator, self.file, self.item_index, type_expr) catch
            return self.recordTypeFallback(.unsupported_syntax_type, range);
        return self.lowerSemaType(ty, range);
    }

    fn lowerErrorUnionSemaType(self: *Lowerer, error_union: sema_model.ErrorUnionType, range: source.TextRange) mlir.MlirType {
        const payload_type = self.lowerSemaType(error_union.payload_type.*, range);
        var error_types: std.ArrayList(mlir.MlirType) = .{};
        for (error_union.error_types) |error_type| {
            error_types.append(self.allocator, self.lowerSemaType(error_type, range)) catch
                return self.recordTypeFallback(.unsupported_syntax_type, range);
        }
        return mlir.oraErrorUnionTypeGetWithErrors(
            self.context,
            payload_type,
            error_types.items.len,
            if (error_types.items.len == 0) null else error_types.items.ptr,
        );
    }

    pub fn lowerNamedPathType(self: *Lowerer, name: []const u8) mlir.MlirType {
        if (self.substitutedType(name)) |substituted| {
            return self.lowerSemaType(substituted, source.TextRange.empty(0));
        }
        if (std.mem.eql(u8, std.mem.trim(u8, name, " \t\n\r"), "NonZeroAddress")) {
            return mlir.oraNonZeroAddressTypeGet(self.context);
        }
        if (self.item_index.lookup(name)) |item_id| {
            switch (self.file.item(item_id).*) {
                .Bitfield => return support.defaultIntegerType(self.context),
                else => {},
            }
        }
        return support.lowerPathType(self.context, name);
    }

    pub fn bitfieldItemByName(self: *const Lowerer, name: []const u8) ?ast.BitfieldItem {
        const item_id = self.item_index.lookup(name) orelse return null;
        return switch (self.file.item(item_id).*) {
            .Bitfield => |bitfield| bitfield,
            else => null,
        };
    }

    pub fn instantiatedBitfieldByName(self: *const Lowerer, name: []const u8) ?sema.InstantiatedBitfield {
        return self.typecheck.instantiatedBitfieldByName(name);
    }

    pub fn bitfieldFieldWidth(self: *const Lowerer, type_expr_id: ast.TypeExprId) u32 {
        return switch (self.file.typeExpr(type_expr_id).*) {
            .Path => |path| blk: {
                const trimmed = std.mem.trim(u8, path.name, " \t\n\r");
                if (std.mem.eql(u8, trimmed, "bool")) break :blk 1;
                if (std.mem.eql(u8, trimmed, "address")) break :blk 160;
                if (support.parseSignedIntegerType(trimmed)) |int_info| break :blk int_info.bits;
                break :blk 256;
            },
            else => 256,
        };
    }

    pub fn bitfieldFieldWidthFromType(self: *const Lowerer, ty: sema.Type) u32 {
        _ = self;
        return switch (ty) {
            .bool => 1,
            .address => 160,
            .integer => |integer| integer.bits orelse 256,
            else => 256,
        };
    }

    pub fn bitfieldFieldSignFromType(self: *const Lowerer, ty: sema.Type) u8 {
        _ = self;
        return switch (ty) {
            .integer => |integer| if (integer.signed == true) 's' else 'u',
            else => 'u',
        };
    }

    pub fn resolveBitfieldField(self: *const Lowerer, bitfield_name: []const u8, field_name: []const u8) ?ResolvedBitfieldField {
        if (self.instantiatedBitfieldByName(bitfield_name)) |bitfield| {
            const template = self.file.item(bitfield.template_item_id).Bitfield;
            var next_offset: u32 = 0;
            for (bitfield.fields, 0..) |field, index| {
                const width = field.width orelse self.bitfieldFieldWidthFromType(field.ty);
                const offset = field.offset orelse next_offset;
                const sign = self.bitfieldFieldSignFromType(field.ty);
                if (std.mem.eql(u8, field.name, field_name)) {
                    return .{
                        .field = template.fields[index],
                        .field_type = field.ty,
                        .offset = offset,
                        .width = width,
                        .sign = sign,
                    };
                }
                next_offset = offset + width;
            }
            return null;
        }
        const bitfield = self.bitfieldItemByName(bitfield_name) orelse return null;
        var next_offset: u32 = 0;
        for (bitfield.fields) |field| {
            const width = field.width orelse self.bitfieldFieldWidth(field.type_expr);
            const offset = field.offset orelse next_offset;
            const sign = self.bitfieldFieldSign(field.type_expr);
            if (std.mem.eql(u8, field.name, field_name)) {
                return .{
                    .field = field,
                    .field_type = null,
                    .offset = offset,
                    .width = width,
                    .sign = sign,
                };
            }
            next_offset = offset + width;
        }
        return null;
    }

    pub fn isGenericTypeParameter(self: *const Lowerer, parameter: ast.Parameter) bool {
        if (!parameter.is_comptime) return false;
        return switch (self.file.typeExpr(parameter.type_expr).*) {
            .Path => |path| std.mem.eql(u8, path.name, "type"),
            else => false,
        };
    }

    pub fn runtimeFunctionParameters(self: *const Lowerer, function: ast.FunctionItem) ![]ast.Parameter {
        var parameters: std.ArrayList(ast.Parameter) = .{};
        for (function.parameters) |parameter| {
            if (parameter.is_comptime) continue;
            try parameters.append(self.allocator, parameter);
        }
        return parameters.toOwnedSlice(self.allocator);
    }

    pub fn resolvedRuntimeParameterTypeForCall(self: *Lowerer, function: ast.FunctionItem, parameter: ast.Parameter, call: ast.CallExpr) !sema.Type {
        if (!function.is_generic) {
            return self.typecheck.pattern_types[parameter.pattern.index()].type;
        }

        const bindings = (try self.genericTypeBindingsForCall(function, call)) orelse {
            return self.typecheck.pattern_types[parameter.pattern.index()].type;
        };

        const type_expr = parameter.type_expr;
        const type_expr_node = self.file.typeExpr(type_expr).*;
        switch (type_expr_node) {
            .Path => |path| {
                const trimmed = std.mem.trim(u8, path.name, " \t\n\r");
                for (bindings) |binding| {
                    if (!std.mem.eql(u8, binding.name, trimmed)) continue;
                    if (hirGenericBindingType(binding)) |bound_type| return bound_type;
                    break;
                }
            },
            else => {},
        }

        return self.typecheck.pattern_types[parameter.pattern.index()].type;
    }

    pub fn genericTypeBindingsForCall(self: *Lowerer, function: ast.FunctionItem, call: ast.CallExpr) !?[]const GenericTypeBinding {
        const comptime_count = self.leadingComptimeParameterCount(function);
        const inferable_type_count = self.leadingGenericTypeParameterCount(function);
        if (comptime_count == 0) return &.{};
        const method_receiver_supplied = self.callSuppliesMethodReceiver(call.callee) and self.functionHasRuntimeSelf(function);
        var runtime_count: usize = 0;
        for (function.parameters) |parameter| {
            if (!parameter.is_comptime) runtime_count += 1;
        }
        const effective_runtime_count = runtime_count - @as(usize, if (method_receiver_supplied) 1 else 0);

        if (call.args.len >= comptime_count + effective_runtime_count) {
            var bindings: std.ArrayList(GenericTypeBinding) = .{};
            for (function.parameters[0..comptime_count], 0..) |parameter, index| {
                const type_name = self.patternName(parameter.pattern) orelse return null;
                const binding = (try self.genericBindingForCallArg(parameter, type_name, call.args[index])) orelse return null;
                try bindings.append(self.allocator, .{
                    .name = binding.name,
                    .value = binding.value,
                    .mangle_name = binding.mangle_name,
                });
            }
            return @constCast(try bindings.toOwnedSlice(self.allocator));
        }

        if (call.args.len != effective_runtime_count) return null;
        if (comptime_count != inferable_type_count) return null;

        var bindings: std.ArrayList(GenericTypeBinding) = .{};
        for (function.parameters[0..comptime_count]) |param| {
            const name = self.patternName(param.pattern) orelse return null;
            try bindings.append(self.allocator, .{
                .name = name,
                .value = .{ .ty = .{ .unknown = {} } },
                .mangle_name = "",
            });
        }

        var runtime_index: usize = 0;
        for (function.parameters) |parameter| {
            if (parameter.is_comptime) continue;
            if (method_receiver_supplied and runtime_index == 0 and std.mem.eql(u8, self.patternName(parameter.pattern) orelse "", "self")) continue;
            if (runtime_index >= call.args.len) break;
            const arg_type = self.typecheck.exprType(call.args[runtime_index]);
            if (arg_type.kind() != .unknown) {
                try self.inferHirBinding(function, inferable_type_count, parameter, arg_type, bindings.items);
            }
            runtime_index += 1;
        }

        for (bindings.items) |*binding| {
            if (binding.mangle_name.len == 0) {
                const ty = hirGenericBindingType(binding.*) orelse return null;
                binding.mangle_name = try self.typeMangleName(ty);
            }
        }

        return @constCast(try bindings.toOwnedSlice(self.allocator));
    }

    pub fn stripGenericCallArgs(self: *Lowerer, function: ast.FunctionItem, call: ast.CallExpr) []const ast.ExprId {
        const comptime_count = self.leadingComptimeParameterCount(function);
        var runtime_count: usize = 0;
        for (function.parameters) |parameter| {
            if (!parameter.is_comptime) runtime_count += 1;
        }
        if (call.args.len == runtime_count) return call.args;
        if (comptime_count >= call.args.len) return &.{};
        return call.args[comptime_count..];
    }

    pub fn leadingComptimeParameterCount(self: *const Lowerer, function: ast.FunctionItem) usize {
        _ = self;
        var count: usize = 0;
        for (function.parameters) |parameter| {
            if (!parameter.is_comptime) break;
            count += 1;
        }
        return count;
    }

    pub fn leadingGenericTypeParameterCount(self: *const Lowerer, function: ast.FunctionItem) usize {
        var count: usize = 0;
        for (function.parameters) |parameter| {
            if (!parameter.is_comptime) break;
            if (!self.isGenericTypeParameter(parameter)) break;
            count += 1;
        }
        return count;
    }

    pub fn functionHasRuntimeSelf(self: *const Lowerer, function: ast.FunctionItem) bool {
        for (function.parameters) |parameter| {
            if (parameter.is_comptime) continue;
            return std.mem.eql(u8, self.patternName(parameter.pattern) orelse "", "self");
        }
        return false;
    }

    pub fn callSuppliesMethodReceiver(self: *const Lowerer, expr_id: ast.ExprId) bool {
        if (expr_id.index() >= self.file.expressions.len) return false;
        return switch (self.file.expression(expr_id).*) {
            .Field => true,
            .Group => |group| self.callSuppliesMethodReceiver(group.expr),
            else => false,
        };
    }

    pub fn mangleGenericFunctionName(self: *Lowerer, base_name: []const u8, bindings: []const GenericTypeBinding) ![]const u8 {
        var name = std.ArrayList(u8){};
        try name.appendSlice(self.allocator, base_name);
        for (bindings) |binding| {
            try name.appendSlice(self.allocator, "__");
            for (binding.mangle_name) |ch| {
                try name.append(self.allocator, if (std.ascii.isAlphanumeric(ch)) ch else '_');
            }
        }
        return name.toOwnedSlice(self.allocator);
    }

    pub fn patternName(self: *const Lowerer, pattern_id: ast.PatternId) ?[]const u8 {
        return switch (self.file.pattern(pattern_id).*) {
            .Name => |name| name.name,
            else => null,
        };
    }

    fn typeArgNameFromExpr(self: *const Lowerer, expr_id: ast.ExprId) ?[]const u8 {
        return switch (self.file.expression(expr_id).*) {
            .Name => |name| name.name,
            .TypeValue => |type_value| self.typeArgNameFromTypeExpr(type_value.type_expr),
            .Group => |group| self.typeArgNameFromExpr(group.expr),
            else => null,
        };
    }

    fn typeArgNameFromTypeExpr(self: *const Lowerer, type_expr_id: ast.TypeExprId) ?[]const u8 {
        return switch (self.file.typeExpr(type_expr_id).*) {
            .Path => |path| path.name,
            else => null,
        };
    }

    fn typeArgTypeFromExpr(self: *const Lowerer, expr_id: ast.ExprId) ?sema.Type {
        return switch (self.file.expression(expr_id).*) {
            .TypeValue => self.typecheck.exprType(expr_id),
            .Group => |group| self.typeArgTypeFromExpr(group.expr),
            else => null,
        };
    }

    fn integerArgText(self: *const Lowerer, expr_id: ast.ExprId) ?[]const u8 {
        return switch (self.file.expression(expr_id).*) {
            .IntegerLiteral => |literal| std.mem.trim(u8, literal.text, " \t\n\r"),
            .Group => |group| self.integerArgText(group.expr),
            else => null,
        };
    }

    fn genericBindingForCallArg(self: *Lowerer, parameter: ast.Parameter, name: []const u8, arg_expr: ast.ExprId) !?GenericTypeBinding {
        if (self.isGenericTypeParameter(parameter)) {
            if (self.typeArgTypeFromExpr(arg_expr)) |arg_type| {
                if (arg_type.kind() != .unknown) {
                    return .{
                        .name = name,
                        .value = .{ .ty = arg_type },
                        .mangle_name = try self.typeMangleName(arg_type),
                    };
                }
            }
            const concrete_name = self.typeArgNameFromExpr(arg_expr) orelse return null;
            return .{
                .name = name,
                .value = .{ .ty = descriptorFromPathName(self.file, self.item_index, concrete_name) },
                .mangle_name = try self.allocator.dupe(u8, std.mem.trim(u8, concrete_name, " \t\n\r")),
            };
        }
        if (parameter.is_comptime) {
            const integer_text = self.integerArgText(arg_expr) orelse return null;
            return .{
                .name = name,
                .value = .{ .integer = integer_text },
                .mangle_name = try self.allocator.dupe(u8, integer_text),
            };
        }
        return null;
    }

    fn inferHirBinding(
        self: *Lowerer,
        function: ast.FunctionItem,
        generic_count: usize,
        param: ast.Parameter,
        arg_type: sema.Type,
        bindings: []GenericTypeBinding,
    ) !void {
        try self.inferHirBindingsFromTypeExpr(function, generic_count, param.type_expr, arg_type, bindings);
    }

    fn inferHirBindingsFromTypeExpr(
        self: *Lowerer,
        function: ast.FunctionItem,
        generic_count: usize,
        type_expr_id: ast.TypeExprId,
        arg_type: sema.Type,
        bindings: []GenericTypeBinding,
    ) !void {
        switch (self.file.typeExpr(type_expr_id).*) {
            .Path => |path| {
                const name = path.name;
                for (function.parameters[0..generic_count], 0..) |generic_param, index| {
                    const param_name = self.patternName(generic_param.pattern) orelse continue;
                    if (!std.mem.eql(u8, name, param_name)) continue;
                    if (!self.isGenericTypeParameter(generic_param)) continue;

                    if (bindings[index].mangle_name.len > 0) {
                        const existing = hirGenericBindingType(bindings[index]) orelse return;
                        if (shouldDeferLiteralMangleBinding(existing, arg_type)) return;
                        const existing_mangle = try self.typeMangleName(existing);
                        const arg_mangle = try self.typeMangleName(arg_type);
                        if (!std.mem.eql(u8, existing_mangle, arg_mangle)) {
                            bindings[index].mangle_name = "";
                            bindings[index].value = .{ .ty = .{ .unknown = {} } };
                        }
                        return;
                    }

                    bindings[index].value = .{ .ty = arg_type };
                    bindings[index].mangle_name = try self.typeMangleName(arg_type);
                    return;
                }
            },
            .Generic => |generic| {
                const base_name = std.mem.trim(u8, generic.name, " \t\n\r");
                if (std.mem.eql(u8, base_name, "Result") and generic.args.len == 2 and arg_type.kind() == .error_union) {
                    if (generic.args[0] == .Type) {
                        try self.inferHirBindingsFromTypeExpr(function, generic_count, generic.args[0].Type, arg_type.error_union.payload_type.*, bindings);
                    }
                    if (generic.args[1] == .Type and arg_type.error_union.error_types.len == 1) {
                        try self.inferHirBindingsFromTypeExpr(function, generic_count, generic.args[1].Type, arg_type.error_union.error_types[0], bindings);
                    }
                }
            },
            .ErrorUnion => |error_union| {
                if (arg_type.kind() != .error_union) return;
                try self.inferHirBindingsFromTypeExpr(function, generic_count, error_union.payload, arg_type.error_union.payload_type.*, bindings);
                if (error_union.errors.len == 1 and arg_type.error_union.error_types.len == 1) {
                    try self.inferHirBindingsFromTypeExpr(function, generic_count, error_union.errors[0], arg_type.error_union.error_types[0], bindings);
                }
            },
            else => {},
        }
    }

    fn hirGenericBindingType(binding: GenericTypeBinding) ?sema.Type {
        return switch (binding.value) {
            .ty => |ty| ty,
            .integer => null,
        };
    }

    fn shouldDeferLiteralMangleBinding(existing: sema.Type, candidate: sema.Type) bool {
        if (existing.kind() != .integer or candidate.kind() != .integer) return false;
        return candidate.integer.spelling == null;
    }

    pub fn typeMangleName(self: *Lowerer, ty: sema.Type) ![]const u8 {
        var name = std.ArrayList(u8){};
        try appendSemaTypeMangleName(self.allocator, &name, ty);
        return name.toOwnedSlice(self.allocator);
    }

    fn recordTypeFallback(self: *Lowerer, reason: TypeFallbackReason, range: source.TextRange) mlir.MlirType {
        self.type_fallbacks.append(self.allocator, .{
            .reason = reason,
            .location = .{
                .file_id = self.file.file_id,
                .range = range,
            },
        }) catch {};
        return support.defaultIntegerType(self.context);
    }

    fn typeExprRange(self: *const Lowerer, type_expr_id: ast.TypeExprId) source.TextRange {
        return switch (self.file.typeExpr(type_expr_id).*) {
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
};

const ContractLowerer = struct {
    parent: *Lowerer,
    block: mlir.MlirBlock,

    const ContractLowering = contract_lowering.mixin(ContractLowerer, Lowerer, FunctionLowerer);
    pub const lowerInvariant = ContractLowering.lowerInvariant;
};

const FunctionLowerer = struct {
    const DeferredReturnKind = enum {
        none,
        ora_yield,
        scf_yield,
    };

    pub const ExtraVerificationClause = struct {
        kind: ast.SpecClauseKind,
        expr: ast.ExprId,
        range: source.TextRange,
        verification_context: ?[]const u8 = null,
    };

    parent: *Lowerer,
    item_id: ?ast.ItemId,
    function: ?ast.FunctionItem,
    op: mlir.MlirOperation,
    block: mlir.MlirBlock,
    locals: hir_locals.LocalEnv,
    return_type: ?mlir.MlirType,
    current_return_value: ?mlir.MlirValue = null,
    deferred_return_flag: ?mlir.MlirValue = null,
    deferred_return_value_slot: ?mlir.MlirValue = null,
    deferred_return_kind: DeferredReturnKind = .none,
    deferred_return_carried_locals: []const hir_locals.LocalId = &.{},
    in_try_block: bool = false,
    in_ghost_context: bool = false,
    extra_verification_clauses: []const ExtraVerificationClause = &.{},
    trait_ghost_self_pattern: ?ast.PatternId = null,
    current_scf_carried_locals: ?[]const hir_locals.LocalId = null,
    loop_context: ?*const support.LoopContext = null,
    block_context: ?*const support.BlockContext = null,
    switch_context: ?*const support.SwitchContext = null,
    unrolled_loop_context: ?*support.UnrolledLoopContext = null,
    unroll_multiplier: u64 = 1,

    const FunctionCore = function_core.mixin(FunctionLowerer, Lowerer);
    const ControlFlow = control_flow.mixin(FunctionLowerer, Lowerer);
    const ExprLowering = expr_lowering.mixin(FunctionLowerer, Lowerer);

    pub const init = FunctionCore.init;
    pub const initContractContext = FunctionCore.initContractContext;
    pub const lower = FunctionCore.lower;
    pub const cloneLocals = FunctionCore.cloneLocals;
    pub const lowerBody = FunctionCore.lowerBody;
    pub const lowerStmt = FunctionCore.lowerStmt;
    pub const ensureDeferredReturnSlots = FunctionCore.ensureDeferredReturnSlots;
    pub const appendDeferredReturnTerminator = FunctionCore.appendDeferredReturnTerminator;
    pub const appendDeferredReturnCheck = FunctionCore.appendDeferredReturnCheck;
    pub const bindPatternValue = FunctionCore.bindPatternValue;
    pub const storePattern = FunctionCore.storePattern;
    pub const convertValueForFlow = FunctionCore.convertValueForFlow;
    pub const convertValueForSemaFlow = FunctionCore.convertValueForSemaFlow;
    pub const lowerCheckedPower = FunctionCore.lowerCheckedPower;
    pub const lowerPowerWithOverflow = FunctionCore.lowerPowerWithOverflow;
    pub const createBitfieldFieldExtract = FunctionCore.createBitfieldFieldExtract;
    pub const createBitfieldFieldUpdate = FunctionCore.createBitfieldFieldUpdate;
    pub const appendUnsupportedControlPlaceholder = FunctionCore.appendUnsupportedControlPlaceholder;
    pub const filterCarriedLocals = FunctionCore.filterCarriedLocals;
    pub const buildCarriedResultTypes = FunctionCore.buildCarriedResultTypes;
    pub const materializeCarriedLocalValue = FunctionCore.materializeCarriedLocalValue;
    pub const evalKnownIntExpr = FunctionCore.evalKnownIntExpr;
    pub const evalKnownBoolExpr = FunctionCore.evalKnownBoolExpr;
    pub const evalKnownExprValue = FunctionCore.evalKnownExprValue;
    pub const appendOraYieldFromLocals = FunctionCore.appendOraYieldFromLocals;
    pub const appendScfYieldFromLocals = FunctionCore.appendScfYieldFromLocals;
    pub const writeBackCarriedLocals = FunctionCore.writeBackCarriedLocals;
    pub const annotateCarriedLocalResults = FunctionCore.annotateCarriedLocalResults;
    pub const emitOverflowAssert = FunctionCore.emitOverflowAssert;

    pub const lowerIfStmt = ControlFlow.lowerIfStmt;
    pub const lowerLabeledBlockStmt = ControlFlow.lowerLabeledBlockStmt;
    pub const lowerTryStmt = ControlFlow.lowerTryStmt;
    pub const lowerWhileStmt = ControlFlow.lowerWhileStmt;
    pub const lowerSwitchStmt = ControlFlow.lowerSwitchStmt;
    pub const buildSwitchPatternData = ControlFlow.buildSwitchPatternData;
    pub const buildSwitchExprPatternData = ControlFlow.buildSwitchExprPatternData;
    pub const appendSwitchPatternData = ControlFlow.appendSwitchPatternData;
    pub const lowerSwitchCaseBlock = ControlFlow.lowerSwitchCaseBlock;
    pub const lowerSwitchExpr = ControlFlow.lowerSwitchExpr;
    pub const switchPatternValue = ControlFlow.switchPatternValue;

    pub const lowerExpr = ExprLowering.lowerExpr;
    pub const lowerNameExpr = ExprLowering.lowerNameExpr;
    pub const lowerUnary = ExprLowering.lowerUnary;
    pub const lowerBinary = ExprLowering.lowerBinary;
    pub const maybeEmitCheckedBinaryOverflowAssert = ExprLowering.maybeEmitCheckedBinaryOverflowAssert;
    pub const createCompareOp = ExprLowering.createCompareOp;
    pub const lowerCall = ExprLowering.lowerCall;
    pub const lowerBuiltin = ExprLowering.lowerBuiltin;
    pub const createValuePlaceholder = ExprLowering.createValuePlaceholder;
    pub const createAggregatePlaceholder = ExprLowering.createAggregatePlaceholder;
    pub const defaultValue = ExprLowering.defaultValue;
    pub const typeIsVoid = ExprLowering.typeIsVoid;
};

fn lowerGenericType(lowerer: *Lowerer, generic: ast.GenericTypeExpr) mlir.MlirType {
    if (std.mem.eql(u8, generic.name, "map") and generic.args.len == 2) {
        const key_type = switch (generic.args[0]) {
            .Type => |type_expr| lowerer.lowerTypeExpr(type_expr),
            else => lowerer.recordTypeFallback(.invalid_generic_type_arg, generic.range),
        };
        const value_type = switch (generic.args[1]) {
            .Type => |type_expr| lowerer.lowerTypeExpr(type_expr),
            else => lowerer.recordTypeFallback(.invalid_generic_type_arg, generic.range),
        };
        return mlir.oraMapTypeGet(lowerer.context, key_type, value_type);
    }
    if (std.mem.eql(u8, generic.name, "Result") and generic.args.len == 2) {
        const payload_type = switch (generic.args[0]) {
            .Type => |type_expr| lowerer.lowerTypeExpr(type_expr),
            else => return lowerer.recordTypeFallback(.invalid_generic_type_arg, generic.range),
        };
        const error_type = switch (generic.args[1]) {
            .Type => |type_expr| lowerer.lowerTypeExpr(type_expr),
            else => return lowerer.recordTypeFallback(.invalid_generic_type_arg, generic.range),
        };
        var error_types = [_]mlir.MlirType{error_type};
        return mlir.oraErrorUnionTypeGetWithErrors(lowerer.context, payload_type, error_types.len, &error_types);
    }
    if (support.isRefinementTypeName(generic.name) and generic.args.len > 0) {
        const resolved_args = lowerRefinementArgs(lowerer, generic.args) orelse generic.args;
        const base_type = switch (generic.args[0]) {
            .Type => |type_expr| lowerer.lowerTypeExpr(type_expr),
            else => return lowerer.recordTypeFallback(.invalid_generic_type_arg, generic.range),
        };
        return support.buildRefinementType(lowerer.context, generic.name, base_type, resolved_args) orelse
            lowerer.recordTypeFallback(.invalid_generic_type_arg, generic.range);
    }
    if (generic.args.len > 0) {
        return switch (generic.args[0]) {
            .Type => |type_expr| lowerer.lowerTypeExpr(type_expr),
            else => lowerer.recordTypeFallback(.invalid_generic_type_arg, generic.range),
        };
    }
    if (std.mem.eql(u8, generic.name, "NonZeroAddress")) {
        return mlir.oraNonZeroAddressTypeGet(lowerer.context);
    }
    return support.lowerPathType(lowerer.context, generic.name);
}

fn lowerRefinementArgs(lowerer: *Lowerer, args: []const ast.TypeArg) ?[]const ast.TypeArg {
    var changed = false;
    const resolved = lowerer.allocator.alloc(ast.TypeArg, args.len) catch return null;
    for (args, 0..) |arg, index| {
        resolved[index] = switch (arg) {
            .Integer => arg,
            .Type => |type_expr| if (typeExprIntegerBinding(lowerer, type_expr)) |integer| blk: {
                changed = true;
                break :blk .{ .Integer = .{
                    .range = lowerer.typeExprRange(type_expr),
                    .text = integer,
                } };
            } else arg,
        };
    }
    if (!changed) return null;
    return resolved;
}

fn typeExprIntegerBinding(lowerer: *const Lowerer, type_expr: ast.TypeExprId) ?[]const u8 {
    return switch (lowerer.file.typeExpr(type_expr).*) {
        .Path => |path| lowerer.substitutedInteger(std.mem.trim(u8, path.name, " \t\n\r")),
        else => null,
    };
}
