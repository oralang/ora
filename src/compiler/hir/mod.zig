const std = @import("std");
const mlir = @import("mlir_c_api").c;
const ast = @import("../ast/mod.zig");
const sema = @import("../sema/mod.zig");
const source = @import("../source/mod.zig");
const contract_lowering = @import("contract_lowering.zig");
const control_flow = @import("control_flow.zig");
const expr_lowering = @import("expr_lowering.zig");
const function_core = @import("function_core.zig");
const hir_locals = @import("locals.zig");
const module_lowering = @import("module_lowering.zig");
const support = @import("support.zig");

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
        .sources = sources,
        .file = file,
        .item_index = item_index,
        .resolution = resolution,
        .const_eval = const_eval,
        .typecheck = typecheck,
        .module_body = mlir.oraModuleGetBody(result.module.raw_module),
        .items = .{},
        .type_fallbacks = .{},
    };

    for (file.root_items) |item_id| {
        try lowerer.lowerItem(item_id, lowerer.module_body);
    }

    result.items = try lowerer.items.toOwnedSlice(result.arena.allocator());
    result.type_fallbacks = try lowerer.type_fallbacks.toOwnedSlice(result.arena.allocator());
    result.type_fallback_count = result.type_fallbacks.len;
    return result;
}

const Lowerer = struct {
    allocator: std.mem.Allocator,
    context: mlir.MlirContext,
    sources: *const source.SourceStore,
    file: *const ast.AstFile,
    item_index: *const sema.ItemIndexResult,
    resolution: *const sema.NameResolutionResult,
    const_eval: *const sema.ConstEvalResult,
    typecheck: *const sema.TypeCheckResult,
    module_body: mlir.MlirBlock,
    items: std.ArrayList(HirItemHandle),
    type_fallbacks: std.ArrayList(TypeFallbackRecord),
    guarded_storage_roots: ?*const std.StringHashMap(void) = null,

    const ModuleLowering = module_lowering.mixin(Lowerer, ContractLowerer, FunctionLowerer, HirSymbolKind);
    pub const lowerItem = ModuleLowering.lowerItem;
    pub const lowerContract = ModuleLowering.lowerContract;
    pub const lowerFunction = ModuleLowering.lowerFunction;
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
    pub const attachBitfieldParamMetadata = ModuleLowering.attachBitfieldParamMetadata;
    pub const attachBitfieldOpMetadata = ModuleLowering.attachBitfieldOpMetadata;
    pub const bitfieldMetadataForTypeExpr = ModuleLowering.bitfieldMetadataForTypeExpr;
    pub const buildBitfieldLayout = ModuleLowering.buildBitfieldLayout;
    pub const bitfieldFieldSign = ModuleLowering.bitfieldFieldSign;

    pub fn location(self: *const Lowerer, range: source.TextRange) mlir.MlirLocation {
        return support.locationFromRange(self.context, self.sources, self.file.file_id, range);
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
            .Array => |array| support.arrayMemRefType(self.context, self.lowerTypeExpr(array.element), support.parseArrayLen(array.size.text) orelse 0),
            .Slice => |slice| support.sliceMemRefType(self.context, self.lowerTypeExpr(slice.element)),
            .ErrorUnion => |error_union| mlir.oraErrorUnionTypeGet(self.context, self.lowerTypeExpr(error_union.payload)),
            .Error => self.recordTypeFallback(.syntax_error_type, self.typeExprRange(type_expr_id)),
        };
    }

    pub fn lowerExprType(self: *Lowerer, expr_id: ast.ExprId) mlir.MlirType {
        return self.lowerSemaType(self.typecheck.exprType(expr_id), support.exprRange(self.file, expr_id));
    }

    pub fn lowerSemaType(self: *Lowerer, ty: sema.Type, range: source.TextRange) mlir.MlirType {
        return switch (ty) {
            .bool => support.boolType(self.context),
            .integer => |integer| if (integer.spelling) |name| support.lowerPathType(self.context, name) else support.defaultIntegerType(self.context),
            .address => support.addressType(self.context),
            .string => support.stringType(self.context),
            .bytes => support.bytesType(self.context),
            .void => mlir.oraNoneTypeCreate(self.context),
            .array => |array| support.arrayMemRefType(self.context, self.lowerSemaType(array.element_type.*, range), array.len orelse 0),
            .slice => |slice| support.sliceMemRefType(self.context, self.lowerSemaType(slice.element_type.*, range)),
            .map => |map| mlir.oraMapTypeGet(
                self.context,
                if (map.key_type) |key| self.lowerSemaType(key.*, range) else support.defaultIntegerType(self.context),
                if (map.value_type) |value| self.lowerSemaType(value.*, range) else support.defaultIntegerType(self.context),
            ),
            .refinement => |refinement| support.lowerRefinementType(self.context, refinement),
            .struct_ => |named| mlir.oraStructTypeGet(self.context, support.strRef(named.name)),
            .contract => |named| mlir.oraStructTypeGet(self.context, support.strRef(named.name)),
            // Bitfields are lowered as packed integer wire values with attrs carrying layout metadata.
            .bitfield => support.defaultIntegerType(self.context),
            .enum_ => |named| mlir.oraStructTypeGet(self.context, support.strRef(named.name)),
            .named => |named| mlir.oraStructTypeGet(self.context, support.strRef(named.name)),
            .error_union => |error_union| mlir.oraErrorUnionTypeGet(self.context, self.lowerSemaType(error_union.payload_type.*, range)),
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
        };
    }

    pub fn lowerNamedPathType(self: *Lowerer, name: []const u8) mlir.MlirType {
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
    parent: *Lowerer,
    item_id: ?ast.ItemId,
    function: ?ast.FunctionItem,
    op: mlir.MlirOperation,
    block: mlir.MlirBlock,
    locals: hir_locals.LocalEnv,
    return_type: ?mlir.MlirType,
    current_return_value: ?mlir.MlirValue = null,
    in_try_block: bool = false,
    in_ghost_context: bool = false,
    loop_context: ?*const support.LoopContext = null,
    switch_context: ?*const support.SwitchContext = null,

    const FunctionCore = function_core.mixin(FunctionLowerer, Lowerer);
    const ControlFlow = control_flow.mixin(FunctionLowerer, Lowerer);
    const ExprLowering = expr_lowering.mixin(FunctionLowerer, Lowerer);

    pub const init = FunctionCore.init;
    pub const initContractContext = FunctionCore.initContractContext;
    pub const lower = FunctionCore.lower;
    pub const cloneLocals = FunctionCore.cloneLocals;
    pub const lowerBody = FunctionCore.lowerBody;
    pub const lowerStmt = FunctionCore.lowerStmt;
    pub const bindPatternValue = FunctionCore.bindPatternValue;
    pub const storePattern = FunctionCore.storePattern;
    pub const lowerCheckedPower = FunctionCore.lowerCheckedPower;
    pub const appendUnsupportedControlPlaceholder = FunctionCore.appendUnsupportedControlPlaceholder;
    pub const buildCarriedResultTypes = FunctionCore.buildCarriedResultTypes;
    pub const appendOraYieldFromLocals = FunctionCore.appendOraYieldFromLocals;
    pub const appendScfYieldFromLocals = FunctionCore.appendScfYieldFromLocals;
    pub const writeBackCarriedLocals = FunctionCore.writeBackCarriedLocals;

    pub const lowerIfStmt = ControlFlow.lowerIfStmt;
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
    if (support.isRefinementTypeName(generic.name) and generic.args.len > 0) {
        const base_type = switch (generic.args[0]) {
            .Type => |type_expr| lowerer.lowerTypeExpr(type_expr),
            else => return lowerer.recordTypeFallback(.invalid_generic_type_arg, generic.range),
        };
        return support.buildRefinementType(lowerer.context, generic.name, base_type, generic.args) orelse
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
