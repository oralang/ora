const std = @import("std");
const mlir = @import("mlir_c_api").c;
const ast = @import("../ast/mod.zig");
const source = @import("../source/mod.zig");
const ora_types = @import("ora_types");
const type_builtin = ora_types.builtin;
const integer_constants = ora_types.integer_constants;
const refinements = ora_types.refinement_semantics;
const hir_locals = @import("locals.zig");
const Type = ora_types.SemanticType;
const IntegerType = ora_types.IntegerType;
const RefinementType = ora_types.RefinementType;
const RefinementArg = ora_types.RefinementArg;

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
    exit_flag: mlir.MlirValue,
    carried_locals: []const hir_locals.LocalId = &.{},
};

pub const SwitchContext = struct {
    parent: ?*const SwitchContext,
    label: ?[]const u8 = null,
    continue_flag: ?mlir.MlirValue = null,
    value_slot: ?mlir.MlirValue = null,
    value_type: ?mlir.MlirType = null,
    condition_local: ?hir_locals.LocalId = null,
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
    if (type_builtin.parseFixedBytesName(trimmed) != null) return reprIntegerType(ctx);
    if (parseBuiltinIntegerType(trimmed)) |int_info| {
        return mlir.oraIntegerTypeCreate(ctx, int_info.bits);
    }
    return mlir.oraStructTypeGet(ctx, strRef(trimmed));
}

pub fn lowerTypeDescriptor(ctx: mlir.MlirContext, descriptor: Type, allocator: std.mem.Allocator) anyerror!mlir.MlirType {
    return switch (descriptor) {
        .bool => boolType(ctx),
        .integer => |integer| lowerIntegerType(ctx, integer),
        .address => addressType(ctx),
        .string => stringType(ctx),
        .bytes => bytesType(ctx),
        .fixed_bytes => reprIntegerType(ctx),
        .storage_slot => reprIntegerType(ctx),
        .storage_range => arrayMemRefType(ctx, reprIntegerType(ctx), 2),
        .void => mlir.oraNoneTypeCreate(ctx),
        .array => |array| blk: {
            const len = array.len orelse return error.UnresolvedArrayLength;
            if (len > std.math.maxInt(u32)) return error.ArrayLengthOutOfRange;
            break :blk arrayMemRefType(ctx, try lowerTypeDescriptor(ctx, array.element_type.*, allocator), @intCast(len));
        },
        .slice => |slice| sliceMemRefType(ctx, try lowerTypeDescriptor(ctx, slice.element_type.*, allocator)),
        .map => |map| mlir.oraMapTypeGet(
            ctx,
            if (map.key_type) |key| try lowerTypeDescriptor(ctx, key.*, allocator) else return error.UnresolvedMapKeyType,
            if (map.value_type) |value| try lowerTypeDescriptor(ctx, value.*, allocator) else return error.UnresolvedMapValueType,
        ),
        .refinement => |refinement| try lowerRefinementType(ctx, refinement, allocator),
        .struct_ => |named| mlir.oraStructTypeGet(ctx, strRef(named.name)),
        .contract => |named| mlir.oraStructTypeGet(ctx, strRef(named.name)),
        // Bitfields are carried on the wire as the base packed integer plus attrs.
        .bitfield => reprIntegerType(ctx),
        .enum_ => reprIntegerType(ctx),
        .named => |named| lowerPathType(ctx, named.name),
        .error_union => |error_union| blk: {
            const error_types = try allocator.alloc(mlir.MlirType, error_union.error_types.len);
            defer allocator.free(error_types);
            for (error_union.error_types, 0..) |error_type, index| {
                error_types[index] = try lowerTypeDescriptor(ctx, error_type, allocator);
            }
            break :blk mlir.oraErrorUnionTypeGetWithErrors(
                ctx,
                try lowerTypeDescriptor(ctx, error_union.payload_type.*, allocator),
                error_union.error_types.len,
                if (error_union.error_types.len == 0) null else error_types.ptr,
            );
        },
        else => error.UnsupportedTypeDescriptor,
    };
}

pub fn lowerRefinementType(ctx: mlir.MlirContext, refinement: RefinementType, allocator: std.mem.Allocator) anyerror!mlir.MlirType {
    const base_type = try lowerTypeDescriptor(ctx, refinement.base_type.*, allocator);
    return buildRefinementType(ctx, refinement.name, base_type, refinement.args) orelse base_type;
}

pub fn isRefinementTypeName(name: []const u8) bool {
    return refinements.hasNativeMlirTypeName(name);
}

pub fn buildRefinementType(ctx: mlir.MlirContext, name: []const u8, base_type: mlir.MlirType, args: []const RefinementArg) ?mlir.MlirType {
    const entry = refinements.entryForName(name) orelse return null;
    if (!entry.has_native_mlir_type) return null;

    return switch (entry.kind) {
        .min_value => blk: {
            const value = parseRefinementIntArg(args, 1) orelse return null;
            const words = splitU256IntoU64Words(value);
            break :blk mlir.oraMinValueTypeGet(ctx, base_type, words.high_high, words.high_low, words.low_high, words.low_low);
        },
        .max_value => blk: {
            const value = parseRefinementIntArg(args, 1) orelse return null;
            const words = splitU256IntoU64Words(value);
            break :blk mlir.oraMaxValueTypeGet(ctx, base_type, words.high_high, words.high_low, words.low_high, words.low_low);
        },
        .in_range => blk: {
            const min_value = parseRefinementIntArg(args, 1) orelse return null;
            const max_value = parseRefinementIntArg(args, 2) orelse return null;
            const min_words = splitU256IntoU64Words(min_value);
            const max_words = splitU256IntoU64Words(max_value);
            break :blk mlir.oraInRangeTypeGet(ctx, base_type, min_words.high_high, min_words.high_low, min_words.low_high, min_words.low_low, max_words.high_high, max_words.high_low, max_words.low_high, max_words.low_low);
        },
        .scaled => blk: {
            const decimals = parseRefinementIntArg(args, 1) orelse return null;
            break :blk mlir.oraScaledTypeGet(ctx, base_type, @intCast(decimals));
        },
        .exact => mlir.oraExactTypeGet(ctx, base_type),
        .non_zero_address => mlir.oraNonZeroAddressTypeGet(ctx),
        .non_zero, .basis_points => null,
    };
}

fn parseRefinementIntArg(args: []const RefinementArg, index: usize) ?u256 {
    if (index >= args.len) return null;
    return switch (args[index]) {
        .Integer => |literal| parseRefinementIntLiteral(literal.text),
        else => null,
    };
}

fn parseRefinementIntLiteral(text: []const u8) ?u256 {
    if (integer_constants.lookup(text)) |constant| {
        return parseRefinementIntLiteral(constant);
    }
    if (std.mem.startsWith(u8, text, "-")) {
        const magnitude = parseU256Literal(text[1..]) orelse return null;
        return @as(u256, 0) -% magnitude;
    }
    if (std.mem.startsWith(u8, text, "+")) {
        return parseU256Literal(text[1..]);
    }
    return parseU256Literal(text);
}

fn parseU256Literal(text: []const u8) ?u256 {
    const base: u8 = if (std.mem.startsWith(u8, text, "0x")) 16 else if (std.mem.startsWith(u8, text, "0b")) 2 else 10;
    const digits = if (base == 10) text else text[2..];
    return std.fmt.parseInt(u256, digits, base) catch null;
}

test "HIR refinement type builder follows registry native type classification" {
    const ctx = try createContext();
    defer mlir.oraContextDestroy(ctx);

    const base_type = reprIntegerType(ctx);
    const type_arg: RefinementArg = .{ .Type = {} };
    const min_arg: RefinementArg = .{ .Integer = .{ .text = "1" } };

    try std.testing.expect(buildRefinementType(ctx, "NonZero", base_type, &.{type_arg}) == null);

    const min_type = buildRefinementType(ctx, "MinValue", base_type, &.{ type_arg, min_arg }) orelse return error.TestUnexpectedResult;
    try std.testing.expect(!mlir.oraTypeIsNull(min_type));
}

test "HIR type descriptor lowering separates representation integers from unresolved map fallbacks" {
    const ctx = try createContext();
    defer mlir.oraContextDestroy(ctx);

    const fixed_bytes_type: Type = .{ .fixed_bytes = .{ .len = 4, .spelling = "bytes4" } };
    const fixed_bytes_repr = try lowerTypeDescriptor(ctx, fixed_bytes_type, std.testing.allocator);
    try std.testing.expect(mlir.oraTypeIsAInteger(fixed_bytes_repr));

    const u256_type: Type = .{ .integer = .{ .signed = false, .bits = 256, .spelling = "u256" } };
    const missing_key_map: Type = .{ .map = .{ .key_type = null, .value_type = &u256_type } };
    try std.testing.expectError(error.UnresolvedMapKeyType, lowerTypeDescriptor(ctx, missing_key_map, std.testing.allocator));

    const missing_value_map: Type = .{ .map = .{ .key_type = &u256_type, .value_type = null } };
    try std.testing.expectError(error.UnresolvedMapValueType, lowerTypeDescriptor(ctx, missing_value_map, std.testing.allocator));
}

test "HIR refinement type builder preserves negative bounds as u256 two's-complement limbs" {
    const ctx = try createContext();
    defer mlir.oraContextDestroy(ctx);

    const base_type = mlir.oraIntegerTypeCreate(ctx, 8);
    const type_arg: RefinementArg = .{ .Type = {} };
    const min_arg: RefinementArg = .{ .Integer = .{ .text = "-5" } };

    const min_type = buildRefinementType(ctx, "MinValue", base_type, &.{ type_arg, min_arg }) orelse return error.TestUnexpectedResult;
    try std.testing.expect(!mlir.oraTypeIsNull(min_type));
}

test "HIR integer type parsing uses canonical builtin widths" {
    const u160_info = parseBuiltinIntegerType("u160") orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u16, 160), u160_info.bits);
    try std.testing.expect(!u160_info.signed);

    const i256_info = parseBuiltinIntegerType("i256") orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u16, 256), i256_info.bits);
    try std.testing.expect(i256_info.signed);

    try std.testing.expect(parseBuiltinIntegerType("u24") == null);
    try std.testing.expect(parseBuiltinIntegerType("i96") == null);
    try std.testing.expect(parseBuiltinIntegerType("u1_6") == null);
    try std.testing.expect(parseBuiltinIntegerType("u+8") == null);

    const bitfield_u1 = parseBitfieldIntegerType("u1") orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u32, 1), bitfield_u1.bits);
    try std.testing.expect(!bitfield_u1.signed);
}

test "HIR integer signedness helper fails closed on unresolved integer facts" {
    const i8_type: Type = .{ .integer = .{ .bits = 8, .signed = true, .spelling = "i8" } };
    const signed = try resolvedIntegerSignedness(i8_type);
    try std.testing.expectEqual(true, signed orelse return error.TestUnexpectedResult);

    const refinement_type: Type = .{ .refinement = .{ .name = "SignedByte", .base_type = &i8_type } };
    const refinement_signed = try resolvedIntegerSignedness(refinement_type);
    try std.testing.expectEqual(true, refinement_signed orelse return error.TestUnexpectedResult);

    try std.testing.expect((try resolvedIntegerSignedness(.{ .bool = {} })) == null);
    try std.testing.expectError(error.MlirOperationCreationFailed, resolvedIntegerSignedness(.{ .comptime_integer = .{} }));
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
    return operationIsKnownBlockTerminator(blockLastOperation(block));
}

pub fn clearKnownTerminator(block: mlir.MlirBlock) void {
    const term = blockLastOperation(block);
    if (operationIsKnownBlockTerminator(term)) {
        mlir.oraOperationErase(term);
    }
}

fn blockLastOperation(block: mlir.MlirBlock) mlir.MlirOperation {
    var current = mlir.oraBlockGetFirstOperation(block);
    var last = std.mem.zeroes(mlir.MlirOperation);
    while (!mlir.oraOperationIsNull(current)) {
        last = current;
        current = mlir.oraOperationGetNextInBlock(current);
    }
    return last;
}

fn operationIsKnownBlockTerminator(op: mlir.MlirOperation) bool {
    if (mlir.oraOperationIsNull(op)) return false;

    const name_ref = mlir.oraOperationGetName(op);
    if (name_ref.data == null) return false;
    const op_name = name_ref.data[0..name_ref.length];

    return std.mem.eql(u8, op_name, "ora.yield") or
        std.mem.eql(u8, op_name, "scf.yield") or
        std.mem.eql(u8, op_name, "ora.return") or
        std.mem.eql(u8, op_name, "func.return") or
        std.mem.eql(u8, op_name, "cf.br") or
        std.mem.eql(u8, op_name, "cf.cond_br");
}

pub fn createIntegerConstant(ctx: mlir.MlirContext, loc: mlir.MlirLocation, ty: mlir.MlirType, value: i64) mlir.MlirOperation {
    const concrete_type = if (mlir.oraTypeIsAddressType(ty)) reprIntegerType(ctx) else ty;
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

pub fn unknownTypeFallbackI256(ctx: mlir.MlirContext) mlir.MlirType {
    return reprIntegerType(ctx);
}

pub fn reprIntegerType(ctx: mlir.MlirContext) mlir.MlirType {
    return mlir.oraIntegerTypeCreate(ctx, 256);
}

pub fn lowerIntegerType(ctx: mlir.MlirContext, integer: IntegerType) mlir.MlirType {
    return mlir.oraIntegerTypeCreate(ctx, integer.bits);
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

pub fn parseBuiltinIntegerType(name: []const u8) ?struct { bits: u16, signed: bool } {
    const spec = type_builtin.parseIntegerBuiltin(name) orelse return null;
    return .{
        .bits = spec.bit_width orelse return null,
        .signed = spec.signed orelse return null,
    };
}

pub fn parseBitfieldIntegerType(name: []const u8) ?struct { bits: u32, signed: bool } {
    if (name.len < 2) return null;
    const signed = switch (name[0]) {
        'u' => false,
        'i' => true,
        else => return null,
    };
    const bits = std.fmt.parseInt(u32, name[1..], 10) catch return null;
    return .{ .bits = bits, .signed = signed };
}

pub fn unwrapRefinementSemaType(ty: Type) Type {
    return if (ty.refinementBaseType()) |base| base.* else ty;
}

pub fn resolvedIntegerSignedness(ty: Type) anyerror!?bool {
    return switch (unwrapRefinementSemaType(ty)) {
        .integer => |integer| integer.signed,
        .comptime_integer => error.MlirOperationCreationFailed,
        else => null,
    };
}

pub fn mlirIntegerTypeIsSigned(ty: mlir.MlirType) bool {
    return mlir.oraTypeIsAInteger(ty) and mlir.oraIntegerTypeIsSigned(ty);
}

pub fn mlirIntegerValueIsSigned(value: mlir.MlirValue) bool {
    return mlirIntegerTypeIsSigned(mlir.oraValueGetType(value));
}

pub fn parseArrayLen(text: []const u8) ?u32 {
    return std.fmt.parseInt(u32, std.mem.trim(u8, text, " \t\n\r"), 10) catch null;
}

pub fn parseI64Literal(text: []const u8) ?i64 {
    const base: u8 = if (std.mem.startsWith(u8, text, "0x")) 16 else if (std.mem.startsWith(u8, text, "0b")) 2 else 10;
    const digits = if (base == 10) text else text[2..];
    return std.fmt.parseInt(i64, digits, base) catch null;
}

pub fn parseUnsignedIntegerLiteral(comptime T: type, text: []const u8) ?T {
    const trimmed = std.mem.trim(u8, text, " \t\n\r");
    const base: u8 = if (std.mem.startsWith(u8, trimmed, "0x")) 16 else if (std.mem.startsWith(u8, trimmed, "0b")) 2 else 10;
    const digits = if (base == 10) trimmed else trimmed[2..];
    var result: T = 0;
    var digit_count: usize = 0;
    for (digits) |c| {
        if (c == '_') continue;
        const digit = std.fmt.charToDigit(c, base) catch return null;
        const shifted = @mulWithOverflow(result, @as(T, @intCast(base)));
        if (shifted[1] != 0) return null;
        const summed = @addWithOverflow(shifted[0], @as(T, @intCast(digit)));
        if (summed[1] != 0) return null;
        result = summed[0];
        digit_count += 1;
    }
    if (digit_count == 0) return null;
    return result;
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
        .Resource => |node| node.range,
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
