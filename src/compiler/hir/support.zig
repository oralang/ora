const std = @import("std");
const mlir = @import("mlir_c_api").c;
const ast = @import("../ast/mod.zig");
const sema = @import("../sema/mod.zig");
const source = @import("../source/mod.zig");

pub const LoopContext = struct {
    parent: ?*const LoopContext,
    break_flag: mlir.MlirValue,
    carried_locals: []const ast.PatternId,
};

pub const SwitchContext = struct {
    parent: ?*const SwitchContext,
};

pub const SwitchPatternData = struct {
    case_values: std.ArrayList(i64),
    range_starts: std.ArrayList(i64),
    range_ends: std.ArrayList(i64),
    case_kinds: std.ArrayList(i64),
    default_case_index: i64 = -1,
    total_cases: usize,
};

pub fn createContext() !mlir.MlirContext {
    const ctx = mlir.oraContextCreate();
    if (ctx.ptr == null) return error.MlirContextCreationFailed;

    const registry = mlir.oraDialectRegistryCreate();
    defer mlir.oraDialectRegistryDestroy(registry);
    mlir.oraRegisterAllDialects(registry);
    mlir.oraContextAppendDialectRegistry(ctx, registry);
    mlir.oraContextLoadAllAvailableDialects(ctx);
    if (!mlir.oraDialectRegister(ctx)) return error.MlirDialectRegistrationFailed;
    return ctx;
}

pub fn locationFromRange(ctx: mlir.MlirContext, sources: *const source.SourceStore, file_id: source.FileId, range: source.TextRange) mlir.MlirLocation {
    const file = sources.file(file_id);
    if (file.path.len == 0) return mlir.oraLocationUnknownGet(ctx);
    const line_column = sources.lineColumn(.{ .file_id = file_id, .range = range });
    return mlir.oraLocationFileLineColGet(ctx, strRef(file.path), line_column.line, line_column.column);
}

pub fn lowerPathType(ctx: mlir.MlirContext, name: []const u8) mlir.MlirType {
    const trimmed = std.mem.trim(u8, name, " \t\n\r");
    if (std.mem.eql(u8, trimmed, "bool")) return boolType(ctx);
    if (std.mem.eql(u8, trimmed, "address")) return addressType(ctx);
    if (std.mem.eql(u8, trimmed, "string")) return stringType(ctx);
    if (std.mem.eql(u8, trimmed, "bytes")) return bytesType(ctx);
    if (std.mem.eql(u8, trimmed, "void")) return mlir.oraNoneTypeCreate(ctx);
    if (parseSignedIntegerType(trimmed)) |int_info| {
        return mlir.oraIntegerTypeCreate(ctx, int_info.bits);
    }
    return mlir.oraStructTypeGet(ctx, strRef(trimmed));
}

pub fn lowerTypeDescriptor(ctx: mlir.MlirContext, descriptor: sema.Type) mlir.MlirType {
    return switch (descriptor) {
        .bool => boolType(ctx),
        .integer => |integer| if (integer.spelling) |name| lowerPathType(ctx, name) else defaultIntegerType(ctx),
        .address => addressType(ctx),
        .string => stringType(ctx),
        .bytes => bytesType(ctx),
        .void => mlir.oraNoneTypeCreate(ctx),
        .array => |array| memRefType(ctx, lowerTypeDescriptor(ctx, array.element_type.*)),
        .slice => |slice| memRefType(ctx, lowerTypeDescriptor(ctx, slice.element_type.*)),
        .map => |map| mlir.oraMapTypeGet(
            ctx,
            if (map.key_type) |key| lowerTypeDescriptor(ctx, key.*) else defaultIntegerType(ctx),
            if (map.value_type) |value| lowerTypeDescriptor(ctx, value.*) else defaultIntegerType(ctx),
        ),
        .struct_ => |named| mlir.oraStructTypeGet(ctx, strRef(named.name)),
        .contract => |named| mlir.oraStructTypeGet(ctx, strRef(named.name)),
        // Bitfields are carried on the wire as the base packed integer plus attrs.
        .bitfield => defaultIntegerType(ctx),
        .enum_ => |named| mlir.oraStructTypeGet(ctx, strRef(named.name)),
        .named => |named| mlir.oraStructTypeGet(ctx, strRef(named.name)),
        .error_union => |error_union| mlir.oraErrorUnionTypeGet(ctx, lowerTypeDescriptor(ctx, error_union.payload_type.*)),
        else => defaultIntegerType(ctx),
    };
}

pub fn appendOp(block: mlir.MlirBlock, op: mlir.MlirOperation) void {
    mlir.oraBlockAppendOwnedOperation(block, op);
}

pub fn appendValueOp(block: mlir.MlirBlock, op: mlir.MlirOperation) mlir.MlirValue {
    appendOp(block, op);
    return mlir.oraOperationGetResult(op, 0);
}

pub fn appendEmptyYield(ctx: mlir.MlirContext, block: mlir.MlirBlock, loc: mlir.MlirLocation) !void {
    const op = mlir.oraYieldOpCreate(ctx, loc, null, 0);
    if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
    appendOp(block, op);
}

pub fn appendOraYieldValues(ctx: mlir.MlirContext, block: mlir.MlirBlock, loc: mlir.MlirLocation, values: []const mlir.MlirValue) !void {
    const op = mlir.oraYieldOpCreate(ctx, loc, if (values.len == 0) null else values.ptr, values.len);
    if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
    appendOp(block, op);
}

pub fn appendEmptyScfYield(ctx: mlir.MlirContext, block: mlir.MlirBlock, loc: mlir.MlirLocation) !void {
    const op = mlir.oraScfYieldOpCreate(ctx, loc, null, 0);
    if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
    appendOp(block, op);
}

pub fn appendScfYieldValues(ctx: mlir.MlirContext, block: mlir.MlirBlock, loc: mlir.MlirLocation, values: []const mlir.MlirValue) !void {
    const op = mlir.oraScfYieldOpCreate(ctx, loc, if (values.len == 0) null else values.ptr, values.len);
    if (mlir.oraOperationIsNull(op)) return error.MlirOperationCreationFailed;
    appendOp(block, op);
}

pub fn blockEndsWithTerminator(block: mlir.MlirBlock) bool {
    return !mlir.oraOperationIsNull(mlir.oraBlockGetTerminator(block));
}

pub fn createIntegerConstant(ctx: mlir.MlirContext, loc: mlir.MlirLocation, ty: mlir.MlirType, value: i64) mlir.MlirOperation {
    const concrete_type = if (mlir.oraTypeIsAddressType(ty)) defaultIntegerType(ctx) else ty;
    const attr = mlir.oraIntegerAttrCreateI64FromType(concrete_type, value);
    return mlir.oraArithConstantOpCreate(ctx, loc, concrete_type, attr);
}

pub fn zeroInitAttr(ty: mlir.MlirType) mlir.MlirAttribute {
    if (mlir.oraTypeIsAddressType(ty)) {
        return mlir.oraNullAttrCreate();
    }
    if (mlir.oraTypeIsAInteger(ty) or mlir.oraTypeIsIntegerType(ty)) {
        return mlir.oraIntegerAttrCreateI64FromType(ty, 0);
    }
    return mlir.oraNullAttrCreate();
}

pub fn defaultIntegerType(ctx: mlir.MlirContext) mlir.MlirType {
    return mlir.oraIntegerTypeCreate(ctx, 256);
}

pub fn boolType(ctx: mlir.MlirContext) mlir.MlirType {
    return mlir.oraIntegerTypeCreate(ctx, 1);
}

pub fn addressType(ctx: mlir.MlirContext) mlir.MlirType {
    return mlir.oraAddressTypeGet(ctx);
}

pub fn stringType(ctx: mlir.MlirContext) mlir.MlirType {
    return mlir.oraStringTypeGet(ctx);
}

pub fn bytesType(ctx: mlir.MlirContext) mlir.MlirType {
    return mlir.oraBytesTypeGet(ctx);
}

pub fn memRefType(ctx: mlir.MlirContext, element_type: mlir.MlirType) mlir.MlirType {
    return mlir.oraMemRefTypeCreate(ctx, element_type, 0, null, mlir.oraNullAttrCreate(), mlir.oraNullAttrCreate());
}

pub fn parseSignedIntegerType(name: []const u8) ?struct { bits: u32, signed: bool } {
    if (name.len < 2) return null;
    const signed = switch (name[0]) {
        'u' => false,
        'i' => true,
        else => return null,
    };
    const bits = std.fmt.parseInt(u32, name[1..], 10) catch return null;
    return .{ .bits = bits, .signed = signed };
}

pub fn parseIntLiteral(text: []const u8) ?i64 {
    const base: u8 = if (std.mem.startsWith(u8, text, "0x")) 16 else if (std.mem.startsWith(u8, text, "0b")) 2 else 10;
    const digits = if (base == 10) text else text[2..];
    return std.fmt.parseInt(i64, digits, base) catch null;
}

pub fn exprRange(file: *const ast.AstFile, expr_id: ast.ExprId) source.TextRange {
    return switch (file.expression(expr_id).*) {
        .IntegerLiteral => |node| node.range,
        .StringLiteral => |node| node.range,
        .BoolLiteral => |node| node.range,
        .AddressLiteral => |node| node.range,
        .BytesLiteral => |node| node.range,
        .Tuple => |node| node.range,
        .ArrayLiteral => |node| node.range,
        .StructLiteral => |node| node.range,
        .Switch => |node| node.range,
        .Comptime => |node| node.range,
        .ErrorReturn => |node| node.range,
        .Name => |node| node.range,
        .Result => |node| node.range,
        .Unary => |node| node.range,
        .Binary => |node| node.range,
        .Call => |node| node.range,
        .Builtin => |node| node.range,
        .Field => |node| node.range,
        .Index => |node| node.range,
        .Group => |node| node.range,
        .Old => |node| node.range,
        .Quantified => |node| node.range,
        .Error => |node| node.range,
    };
}

pub fn itemRange(file: *const ast.AstFile, item_id: ast.ItemId) source.TextRange {
    return switch (file.item(item_id).*) {
        .Import => |node| node.range,
        .Contract => |node| node.range,
        .Function => |node| node.range,
        .Struct => |node| node.range,
        .Bitfield => |node| node.range,
        .Enum => |node| node.range,
        .LogDecl => |node| node.range,
        .ErrorDecl => |node| node.range,
        .GhostBlock => |node| node.range,
        .Field => |node| node.range,
        .Constant => |node| node.range,
        .Error => |node| node.range,
    };
}

pub fn cmpPredicate(predicate: []const u8) i64 {
    if (std.mem.eql(u8, predicate, "eq")) return 0;
    if (std.mem.eql(u8, predicate, "ne")) return 1;
    if (std.mem.eql(u8, predicate, "slt")) return 2;
    if (std.mem.eql(u8, predicate, "sle")) return 3;
    if (std.mem.eql(u8, predicate, "sgt")) return 4;
    if (std.mem.eql(u8, predicate, "sge")) return 5;
    return 0;
}

pub fn strRef(bytes: []const u8) mlir.MlirStringRef {
    return mlir.oraStringRefCreate(bytes.ptr, bytes.len);
}

pub fn nullStringRef() mlir.MlirStringRef {
    return .{ .data = null, .length = 0 };
}

pub fn identifier(ctx: mlir.MlirContext, name: []const u8) mlir.MlirIdentifier {
    return mlir.oraIdentifierGet(ctx, strRef(name));
}

pub fn namedStringAttr(ctx: mlir.MlirContext, name: []const u8, value: []const u8) mlir.MlirNamedAttribute {
    return mlir.oraNamedAttributeGet(identifier(ctx, name), mlir.oraStringAttrCreate(ctx, strRef(value)));
}

pub fn namedBoolAttr(ctx: mlir.MlirContext, name: []const u8, value: bool) mlir.MlirNamedAttribute {
    return mlir.oraNamedAttributeGet(identifier(ctx, name), mlir.oraBoolAttrCreate(ctx, value));
}

pub fn namedTypeAttr(ctx: mlir.MlirContext, name: []const u8, ty: mlir.MlirType) mlir.MlirNamedAttribute {
    return mlir.oraNamedAttributeGet(identifier(ctx, name), mlir.oraTypeAttrCreateFromType(ty));
}
