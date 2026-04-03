const std = @import("std");
const mlir = @import("mlir_c_api").c;
const ast = @import("../ast/mod.zig");
const sema = @import("../sema/mod.zig");
const source = @import("../source/mod.zig");
const hir_locals = @import("locals.zig");

pub const LoopContext = struct {
    parent: ?*const LoopContext,
    label: ?[]const u8 = null,
    break_flag: mlir.MlirValue,
    continue_flag: mlir.MlirValue = std.mem.zeroes(mlir.MlirValue),
    carried_locals: []const ast.PatternId,
};

pub const UnrolledLoopSignal = enum {
    none,
    break_loop,
    continue_loop,
};

pub const UnrolledLoopContext = struct {
    parent: ?*UnrolledLoopContext,
    label: ?[]const u8 = null,
    signal: UnrolledLoopSignal = .none,
};

pub const runtime_loop_unroll_limit: u64 = 8;
pub const runtime_total_unroll_budget: u64 = 16;

pub const BlockContext = struct {
    parent: ?*const BlockContext,
    label: ?[]const u8 = null,
    continue_flag: mlir.MlirValue,
    carried_locals: []const hir_locals.LocalId = &.{},
};

pub const SwitchContext = struct {
    parent: ?*const SwitchContext,
    label: ?[]const u8 = null,
    continue_flag: ?mlir.MlirValue = null,
    value_slot: ?mlir.MlirValue = null,
    value_type: ?mlir.MlirType = null,
    carried_locals: []const hir_locals.LocalId = &.{},
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

pub fn locationFromRangeWithStmt(ctx: mlir.MlirContext, sources: *const source.SourceStore, file_id: source.FileId, range: source.TextRange, stmt_id: ?ast.StmtId) mlir.MlirLocation {
    const base = locationFromRange(ctx, sources, file_id, range);
    if (stmt_id) |id| {
        const tagged_origin = mlir.oraLocationOriginStmtTaggedGet(ctx, base, @intCast(id.index()));
        return mlir.oraLocationStmtTaggedGet(ctx, tagged_origin, @intCast(id.index()));
    }
    return base;
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
        .array => |array| arrayMemRefType(ctx, lowerTypeDescriptor(ctx, array.element_type.*), array.len orelse 0),
        .slice => |slice| sliceMemRefType(ctx, lowerTypeDescriptor(ctx, slice.element_type.*)),
        .map => |map| mlir.oraMapTypeGet(
            ctx,
            if (map.key_type) |key| lowerTypeDescriptor(ctx, key.*) else defaultIntegerType(ctx),
            if (map.value_type) |value| lowerTypeDescriptor(ctx, value.*) else defaultIntegerType(ctx),
        ),
        .refinement => |refinement| lowerRefinementType(ctx, refinement),
        .struct_ => |named| mlir.oraStructTypeGet(ctx, strRef(named.name)),
        .contract => |named| mlir.oraStructTypeGet(ctx, strRef(named.name)),
        // Bitfields are carried on the wire as the base packed integer plus attrs.
        .bitfield => defaultIntegerType(ctx),
        .enum_ => defaultIntegerType(ctx),
        .named => |named| lowerPathType(ctx, named.name),
        .error_union => |error_union| mlir.oraErrorUnionTypeGet(ctx, lowerTypeDescriptor(ctx, error_union.payload_type.*)),
        else => defaultIntegerType(ctx),
    };
}

pub fn lowerRefinementType(ctx: mlir.MlirContext, refinement: sema.RefinementType) mlir.MlirType {
    const base_type = lowerTypeDescriptor(ctx, refinement.base_type.*);
    return buildRefinementType(ctx, refinement.name, base_type, refinement.args) orelse base_type;
}

pub fn isRefinementTypeName(name: []const u8) bool {
    return std.mem.eql(u8, name, "MinValue") or
        std.mem.eql(u8, name, "MaxValue") or
        std.mem.eql(u8, name, "InRange") or
        std.mem.eql(u8, name, "Scaled") or
        std.mem.eql(u8, name, "Exact") or
        std.mem.eql(u8, name, "NonZeroAddress");
}

pub fn buildRefinementType(ctx: mlir.MlirContext, name: []const u8, base_type: mlir.MlirType, args: []const ast.TypeArg) ?mlir.MlirType {
    if (std.mem.eql(u8, name, "MinValue")) {
        const value = parseRefinementIntArg(args, 1) orelse return null;
        const words = splitU256IntoU64Words(value);
        return mlir.oraMinValueTypeGet(ctx, base_type, words.high_high, words.high_low, words.low_high, words.low_low);
    }
    if (std.mem.eql(u8, name, "MaxValue")) {
        const value = parseRefinementIntArg(args, 1) orelse return null;
        const words = splitU256IntoU64Words(value);
        return mlir.oraMaxValueTypeGet(ctx, base_type, words.high_high, words.high_low, words.low_high, words.low_low);
    }
    if (std.mem.eql(u8, name, "InRange")) {
        const min_value = parseRefinementIntArg(args, 1) orelse return null;
        const max_value = parseRefinementIntArg(args, 2) orelse return null;
        const min_words = splitU256IntoU64Words(min_value);
        const max_words = splitU256IntoU64Words(max_value);
        return mlir.oraInRangeTypeGet(ctx, base_type, min_words.high_high, min_words.high_low, min_words.low_high, min_words.low_low, max_words.high_high, max_words.high_low, max_words.low_high, max_words.low_low);
    }
    if (std.mem.eql(u8, name, "Scaled")) {
        const decimals = parseRefinementIntArg(args, 1) orelse return null;
        return mlir.oraScaledTypeGet(ctx, base_type, @intCast(decimals));
    }
    if (std.mem.eql(u8, name, "Exact")) {
        return mlir.oraExactTypeGet(ctx, base_type);
    }
    if (std.mem.eql(u8, name, "NonZeroAddress")) {
        return mlir.oraNonZeroAddressTypeGet(ctx);
    }
    return null;
}

fn parseRefinementIntArg(args: []const ast.TypeArg, index: usize) ?u256 {
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

fn splitU256IntoU64Words(x: u256) struct {
    high_high: u64,
    high_low: u64,
    low_high: u64,
    low_low: u64,
} {
    return .{
        .high_high = @truncate(x >> 192),
        .high_low = @truncate(x >> 128),
        .low_high = @truncate(x >> 64),
        .low_low = @truncate(x),
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

pub fn clearKnownTerminator(block: mlir.MlirBlock) void {
    const term = mlir.oraBlockGetTerminator(block);
    if (mlir.oraOperationIsNull(term)) return;

    const name_ref = mlir.oraOperationGetName(term);
    if (name_ref.data == null) return;
    const op_name = name_ref.data[0..name_ref.length];

    const is_terminator = std.mem.eql(u8, op_name, "ora.yield") or
        std.mem.eql(u8, op_name, "scf.yield") or
        std.mem.eql(u8, op_name, "ora.return") or
        std.mem.eql(u8, op_name, "func.return") or
        std.mem.eql(u8, op_name, "cf.br") or
        std.mem.eql(u8, op_name, "cf.cond_br");
    if (is_terminator) {
        mlir.oraOperationErase(term);
    }
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

pub fn arrayMemRefType(ctx: mlir.MlirContext, element_type: mlir.MlirType, len: u32) mlir.MlirType {
    const shape: [1]i64 = .{@intCast(len)};
    return mlir.oraMemRefTypeCreate(ctx, element_type, shape.len, &shape, mlir.oraNullAttrCreate(), mlir.oraNullAttrCreate());
}

pub fn sliceMemRefType(ctx: mlir.MlirContext, element_type: mlir.MlirType) mlir.MlirType {
    const shape: [1]i64 = .{mlir.oraShapedTypeDynamicSize()};
    return mlir.oraMemRefTypeCreate(ctx, element_type, shape.len, &shape, mlir.oraNullAttrCreate(), mlir.oraNullAttrCreate());
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

pub fn parseArrayLen(text: []const u8) ?u32 {
    return std.fmt.parseInt(u32, std.mem.trim(u8, text, " \t\n\r"), 10) catch null;
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
        .TypeValue => |node| node.range,
        .Tuple => |node| node.range,
        .ArrayLiteral => |node| node.range,
        .StructLiteral => |node| node.range,
        .Switch => |node| node.range,
        .ExternalProxy => |node| node.range,
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

pub fn stmtRange(file: *const ast.AstFile, stmt_id: ast.StmtId) source.TextRange {
    return switch (file.statement(stmt_id).*) {
        .VariableDecl => |node| node.range,
        .Return => |node| node.range,
        .Expr => |node| node.range,
        .Assign => |node| node.range,
        .If => |node| node.range,
        .While => |node| node.range,
        .For => |node| node.range,
        .Switch => |node| node.range,
        .Try => |node| node.range,
        .Break => |node| node.range,
        .Continue => |node| node.range,
        .Assert => |node| node.range,
        .Assume => |node| node.range,
        .Havoc => |node| node.range,
        .Log => |node| node.range,
        .Lock => |node| node.range,
        .Unlock => |node| node.range,
        .Block => |node| node.range,
        .LabeledBlock => |node| node.range,
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
        .Trait => |node| node.range,
        .Impl => |node| node.range,
        .TypeAlias => |node| node.range,
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
    if (std.mem.eql(u8, predicate, "ult")) return 6;
    if (std.mem.eql(u8, predicate, "ule")) return 7;
    if (std.mem.eql(u8, predicate, "ugt")) return 8;
    if (std.mem.eql(u8, predicate, "uge")) return 9;
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
