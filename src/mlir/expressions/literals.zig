// ============================================================================
// Literal Expression Lowering
// ============================================================================
// Lowering for literal expressions (integers, strings, addresses, etc.)

const std = @import("std");
const c = @import("mlir_c_api").c;
const lib = @import("ora_lib");
const constants = @import("../lower.zig");
const h = @import("../helpers.zig");
const TypeMapper = @import("../types.zig").TypeMapper;
const LocationTracker = @import("../locations.zig").LocationTracker;
const OraDialect = @import("../dialect.zig").OraDialect;
const expr_helpers = @import("helpers.zig");
const log = @import("log");

/// Lower literal expressions
pub fn lowerLiteral(
    ctx: c.MlirContext,
    block: c.MlirBlock,
    type_mapper: *const TypeMapper,
    ora_dialect: *OraDialect,
    locations: LocationTracker,
    literal: *const lib.ast.Expressions.LiteralExpr,
) c.MlirValue {
    return switch (literal.*) {
        .Integer => |int| lowerIntegerLiteral(ctx, block, type_mapper, locations, int),
        .Bool => |bool_lit| lowerBoolLiteral(ctx, block, ora_dialect, locations, bool_lit),
        .String => |string_lit| lowerStringLiteral(ctx, block, ora_dialect, locations, string_lit),
        .Address => |addr_lit| lowerAddressLiteral(ctx, block, type_mapper, locations, addr_lit),
        .Hex => |hex_lit| lowerHexLiteral(ctx, block, ora_dialect, locations, hex_lit),
        .Binary => |bin_lit| lowerBinaryLiteral(ctx, block, locations, bin_lit),
        .Character => |char_lit| lowerCharacterLiteral(ctx, block, ora_dialect, locations, char_lit),
        .Bytes => |bytes_lit| lowerBytesLiteral(ctx, block, ora_dialect, locations, bytes_lit),
    };
}

fn lowerIntegerLiteral(
    ctx: c.MlirContext,
    block: c.MlirBlock,
    type_mapper: *const TypeMapper,
    locations: LocationTracker,
    int: lib.ast.Expressions.IntegerLiteral,
) c.MlirValue {
    const ty = if (int.type_info.ora_type) |ora_ty| blk: {
        const base_ora_ty = switch (ora_ty) {
            .min_value => |mv| mv.base.*,
            .max_value => |mv| mv.base.*,
            .in_range => |ir| ir.base.*,
            .scaled => |s| s.base.*,
            .exact => |e| e.*,
            else => ora_ty,
        };
        const mapped = type_mapper.toMlirType(.{ .ora_type = base_ora_ty });
        if (!c.oraTypeIsNull(mapped)) break :blk mapped;
        break :blk c.oraIntegerTypeCreate(ctx, constants.DEFAULT_INTEGER_BITS);
    } else c.oraIntegerTypeCreate(ctx, constants.DEFAULT_INTEGER_BITS);

    var parsed: u256 = 0;
    var digit_count: u32 = 0;

    for (int.value) |char| {
        if (char == '_') continue;
        if (char < '0' or char > '9') continue;

        if (digit_count >= 78) {
            log.err("Integer literal '{s}' is too large for u256\n", .{int.value});
            parsed = 0;
            break;
        }

        const mul_overflow = @mulWithOverflow(parsed, 10);
        if (mul_overflow[1] != 0) {
            log.err("Integer literal '{s}' overflow during multiplication\n", .{int.value});
            parsed = 0;
            break;
        }
        parsed = mul_overflow[0];

        const digit_value = char - '0';
        const add_overflow = @addWithOverflow(parsed, digit_value);
        if (add_overflow[1] != 0) {
            log.err("Integer literal '{s}' overflow during addition\n", .{int.value});
            parsed = 0;
            break;
        }
        parsed = add_overflow[0];
        digit_count += 1;
    }

    const loc = locations.createLocation(int.span);
    const attr = if (parsed <= std.math.maxInt(i64)) blk: {
        break :blk c.oraIntegerAttrCreateI64FromType(ty, @intCast(parsed));
    } else blk: {
        const value_str_ref = h.strRef(int.value);
        const big_attr = c.oraIntegerAttrGetFromString(ty, value_str_ref);
        if (c.oraAttributeIsNull(big_attr)) {
            log.err("Failed to create integer attribute from string '{s}'\n", .{int.value});
            break :blk c.oraIntegerAttrCreateI64FromType(ty, 0);
        }
        break :blk big_attr;
    };

    const op = c.oraArithConstantOpCreate(ctx, loc, ty, attr);
    if (op.ptr == null) {
        @panic("Failed to create integer literal arith.constant");
    }
    h.appendOp(block, op);
    return h.getResult(op, 0);
}

fn lowerBoolLiteral(
    _: c.MlirContext,
    block: c.MlirBlock,
    ora_dialect: *OraDialect,
    locations: LocationTracker,
    bool_lit: lib.ast.Expressions.BoolLiteral,
) c.MlirValue {
    const loc = locations.createLocation(bool_lit.span);
    const op = ora_dialect.createArithConstantBool(bool_lit.value, loc);
    h.appendOp(block, op);
    return h.getResult(op, 0);
}

fn lowerStringLiteral(
    ctx: c.MlirContext,
    block: c.MlirBlock,
    ora_dialect: *OraDialect,
    locations: LocationTracker,
    string_lit: lib.ast.Expressions.StringLiteral,
) c.MlirValue {
    const string_len = string_lit.value.len;
    const ty = c.oraStringTypeGet(ctx);
    const loc = locations.createLocation(string_lit.span);
    const op = ora_dialect.createStringConstant(string_lit.value, ty, loc);
    const length_attr = h.intAttr(ctx, c.oraIntegerTypeCreate(ctx, 32), @intCast(string_len));
    const length_name = h.strRef("length");
    c.oraOperationSetAttributeByName(op, length_name, length_attr);
    h.appendOp(block, op);
    return h.getResult(op, 0);
}

fn lowerAddressLiteral(
    ctx: c.MlirContext,
    block: c.MlirBlock,
    type_mapper: *const TypeMapper,
    locations: LocationTracker,
    addr_lit: lib.ast.Expressions.AddressLiteral,
) c.MlirValue {
    _ = type_mapper;

    const loc = locations.createLocation(addr_lit.span);

    const addr_str = if (std.mem.startsWith(u8, addr_lit.value, "0x"))
        addr_lit.value[2..]
    else
        addr_lit.value;

    if (addr_str.len != 40) {
        log.err("Invalid address length '{d}' (expected 40 hex characters): {s}\n", .{ addr_str.len, addr_lit.value });
    }

    for (addr_str) |char| {
        if (!((char >= '0' and char <= '9') or (char >= 'a' and char <= 'f') or (char >= 'A' and char <= 'F'))) {
            log.err("Invalid hex character '{c}' in address '{s}'\n", .{ char, addr_lit.value });
            break;
        }
    }

    var parsed: u256 = 0;
    for (addr_str) |char| {
        if (char >= '0' and char <= '9') {
            parsed = parsed * 16 + (char - '0');
        } else if (char >= 'a' and char <= 'f') {
            parsed = parsed * 16 + (char - 'a' + 10);
        } else if (char >= 'A' and char <= 'F') {
            parsed = parsed * 16 + (char - 'A' + 10);
        }
    }

    const i160_ty = c.oraIntegerTypeCreate(ctx, 160);
    const attr = if (parsed <= std.math.maxInt(i64)) blk: {
        break :blk c.oraIntegerAttrCreateI64FromType(i160_ty, @intCast(parsed));
    } else blk: {
        var decimal_buf: [80]u8 = undefined;
        const decimal_str = std.fmt.bufPrint(&decimal_buf, "{}", .{parsed}) catch {
            log.err("Failed to format address as decimal\n", .{});
            break :blk c.oraIntegerAttrCreateI64FromType(i160_ty, 0);
        };
        const addr_str_ref = h.strRef(decimal_str);
        const big_attr = c.oraIntegerAttrGetFromString(i160_ty, addr_str_ref);
        if (c.oraAttributeIsNull(big_attr)) {
            log.err("Failed to create address attribute from string '{s}'\n", .{addr_lit.value});
            break :blk c.oraIntegerAttrCreateI64FromType(i160_ty, 0);
        }
        break :blk big_attr;
    };

    const const_op = c.oraArithConstantOpCreate(ctx, loc, i160_ty, attr);
    if (const_op.ptr == null) {
        @panic("Failed to create address literal arith.constant");
    }
    h.appendOp(block, const_op);
    const i160_value = h.getResult(const_op, 0);
    const addr_op = c.oraI160ToAddrOpCreate(ctx, loc, i160_value);
    if (addr_op.ptr == null) {
        @panic("Failed to create address literal ora.i160.to.addr");
    }
    h.appendOp(block, addr_op);
    return h.getResult(addr_op, 0);
}

fn lowerHexLiteral(
    ctx: c.MlirContext,
    block: c.MlirBlock,
    ora_dialect: *OraDialect,
    locations: LocationTracker,
    hex_lit: lib.ast.Expressions.HexLiteral,
) c.MlirValue {
    const ty = blk: {
        if (hex_lit.type_info.category == .Bytes) {
            const bytes_ty = c.oraBytesTypeGet(ctx);
            if (bytes_ty.ptr != null) break :blk bytes_ty;
        }
        break :blk c.oraIntegerTypeCreate(ctx, constants.DEFAULT_INTEGER_BITS);
    };
    const loc = locations.createLocation(hex_lit.span);
    const op = if (hex_lit.type_info.category == .Bytes)
        ora_dialect.createBytesConstant(hex_lit.value, ty, loc)
    else
        ora_dialect.createHexConstant(hex_lit.value, ty, loc);
    h.appendOp(block, op);
    return h.getResult(op, 0);
}

fn lowerBinaryLiteral(
    ctx: c.MlirContext,
    block: c.MlirBlock,
    locations: LocationTracker,
    bin_lit: lib.ast.Expressions.BinaryLiteral,
) c.MlirValue {
    const ty = c.oraIntegerTypeCreate(ctx, constants.DEFAULT_INTEGER_BITS);
    const loc = locations.createLocation(bin_lit.span);
    const bin_str = if (std.mem.startsWith(u8, bin_lit.value, "0b"))
        bin_lit.value[2..]
    else
        bin_lit.value;

    for (bin_str) |char| {
        if (char != '0' and char != '1') {
            log.err("Invalid binary character '{c}' in binary literal '{s}'\n", .{ char, bin_lit.value });
            break;
        }
    }

    if (bin_str.len > 64) {
        log.warn("Binary literal '{s}' may overflow i64 (length: {d})\n", .{ bin_lit.value, bin_str.len });
    }

    const parsed: i64 = std.fmt.parseInt(i64, bin_str, 2) catch |err| blk: {
        log.err("Failed to parse binary literal '{s}': {s}\n", .{ bin_lit.value, @errorName(err) });
        break :blk 0;
    };
    const attr = c.oraIntegerAttrCreateI64FromType(ty, parsed);

    const value_id = h.identifier(ctx, "value");
    const binary_id = h.identifier(ctx, "ora.binary");
    const binary_ref = c.oraStringRefCreate(bin_lit.value.ptr, bin_lit.value.len);
    const binary_attr = c.oraStringAttrCreate(ctx, binary_ref);
    const length_id = h.identifier(ctx, "length");
    const length_attr = c.oraIntegerAttrCreateI64FromType(c.oraIntegerTypeCreate(ctx, 32), @intCast(bin_str.len));

    var attrs = [_]c.MlirNamedAttribute{
        c.oraNamedAttributeGet(value_id, attr),
        c.oraNamedAttributeGet(binary_id, binary_attr),
        c.oraNamedAttributeGet(length_id, length_attr),
    };
    const op = c.oraBinaryConstantOpCreate(
        ctx,
        loc,
        ty,
        attrs[0..].ptr,
        attrs.len,
    );
    if (op.ptr == null) {
        @panic("Failed to create ora.binary.constant operation");
    }
    h.appendOp(block, op);
    return h.getResult(op, 0);
}

fn lowerCharacterLiteral(
    ctx: c.MlirContext,
    block: c.MlirBlock,
    ora_dialect: *OraDialect,
    locations: LocationTracker,
    char_lit: lib.ast.Expressions.CharacterLiteral,
) c.MlirValue {
    const ty = c.oraIntegerTypeCreate(ctx, 8);
    const loc = locations.createLocation(char_lit.span);

    if (char_lit.value > 127) {
        log.err("Invalid character value '{d}' (not ASCII)\n", .{char_lit.value});
        return expr_helpers.createConstant(ctx, block, ora_dialect, locations, 0, char_lit.span);
    }

    const character_id = h.identifier(ctx, "ora.character_literal");
    const custom_attrs = [_]c.MlirNamedAttribute{c.oraNamedAttributeGet(character_id, h.boolAttr(ctx, 1))};
    const op = ora_dialect.createArithConstantWithAttrs(@intCast(char_lit.value), ty, &custom_attrs, loc);
    h.appendOp(block, op);
    return h.getResult(op, 0);
}

fn lowerBytesLiteral(
    ctx: c.MlirContext,
    block: c.MlirBlock,
    ora_dialect: *OraDialect,
    locations: LocationTracker,
    bytes_lit: lib.ast.Expressions.BytesLiteral,
) c.MlirValue {
    const loc = locations.createLocation(bytes_lit.span);
    const bytes_str = if (std.mem.startsWith(u8, bytes_lit.value, "0x"))
        bytes_lit.value[2..]
    else
        bytes_lit.value;

    if (bytes_str.len % 2 != 0) {
        log.err("Invalid bytes length '{d}' (must be even number of hex digits): {s}\n", .{ bytes_str.len, bytes_lit.value });
        return expr_helpers.createConstant(ctx, block, ora_dialect, locations, 0, bytes_lit.span);
    }

    for (bytes_str) |char| {
        if (!((char >= '0' and char <= '9') or (char >= 'a' and char <= 'f') or (char >= 'A' and char <= 'F'))) {
            log.err("Invalid hex character '{c}' in bytes '{s}'\n", .{ char, bytes_lit.value });
            return expr_helpers.createConstant(ctx, block, ora_dialect, locations, 0, bytes_lit.span);
        }
    }

    const bytes_len = bytes_str.len / 2;
    const ty = c.oraBytesTypeGet(ctx);
    const op = ora_dialect.createBytesConstant(bytes_lit.value, ty, loc);
    const length_attr = h.intAttr(ctx, c.oraIntegerTypeCreate(ctx, 32), @intCast(bytes_len));
    const length_name = h.strRef("length");
    c.oraOperationSetAttributeByName(op, length_name, length_attr);
    h.appendOp(block, op);
    return h.getResult(op, 0);
}

/// Extract integer value from a literal expression
pub fn extractIntegerFromLiteral(literal: *const lib.ast.Expressions.LiteralExpr) ?i64 {
    return switch (literal.*) {
        .Integer => |int| blk: {
            const parsed = std.fmt.parseInt(i64, int.value, 0) catch return null;
            break :blk parsed;
        },
        else => null,
    };
}

/// Extract integer value from an expression
pub fn extractIntegerFromExpr(expr: *const lib.ast.Expressions.ExprNode) ?i64 {
    return switch (expr.*) {
        .Literal => |lit| extractIntegerFromLiteral(&lit),
        else => null,
    };
}
