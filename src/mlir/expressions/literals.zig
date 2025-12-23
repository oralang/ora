// ============================================================================
// Literal Expression Lowering
// ============================================================================
// Lowering for literal expressions (integers, strings, addresses, etc.)

const std = @import("std");
const c = @import("../c.zig").c;
const lib = @import("ora_lib");
const constants = @import("../lower.zig");
const h = @import("../helpers.zig");
const TypeMapper = @import("../types.zig").TypeMapper;
const LocationTracker = @import("../locations.zig").LocationTracker;
const OraDialect = @import("../dialect.zig").OraDialect;
const expr_helpers = @import("helpers.zig");

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
    const ty = if (int.type_info.ora_type) |_|
        type_mapper.toMlirType(int.type_info)
    else
        c.mlirIntegerTypeGet(ctx, constants.DEFAULT_INTEGER_BITS);

    var parsed: u256 = 0;
    var digit_count: u32 = 0;

    for (int.value) |char| {
        if (char == '_') continue;
        if (char < '0' or char > '9') continue;

        if (digit_count >= 78) {
            std.debug.print("ERROR: Integer literal '{s}' is too large for u256\n", .{int.value});
            parsed = 0;
            break;
        }

        const mul_overflow = @mulWithOverflow(parsed, 10);
        if (mul_overflow[1] != 0) {
            std.debug.print("ERROR: Integer literal '{s}' overflow during multiplication\n", .{int.value});
            parsed = 0;
            break;
        }
        parsed = mul_overflow[0];

        const digit_value = char - '0';
        const add_overflow = @addWithOverflow(parsed, digit_value);
        if (add_overflow[1] != 0) {
            std.debug.print("ERROR: Integer literal '{s}' overflow during addition\n", .{int.value});
            parsed = 0;
            break;
        }
        parsed = add_overflow[0];
        digit_count += 1;
    }

    const loc = locations.createLocation(int.span);
    var state = h.opState("arith.constant", loc);
    c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ty));

    if (parsed <= std.math.maxInt(i64)) {
        const attr = c.mlirIntegerAttrGet(ty, @intCast(parsed));
        const value_id = h.identifier(ctx, "value");
        const gas_cost_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(ctx, 64), 0);
        const gas_cost_id = h.identifier(ctx, "gas_cost");
        var attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(value_id, attr),
            c.mlirNamedAttributeGet(gas_cost_id, gas_cost_attr),
        };
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
    } else {
        const value_str_ref = h.strRef(int.value);
        const attr = c.oraIntegerAttrGetFromString(ty, value_str_ref);

        if (c.mlirAttributeIsNull(attr)) {
            std.debug.print("ERROR: Failed to create integer attribute from string '{s}'\n", .{int.value});
            const fallback_attr = c.mlirIntegerAttrGet(ty, 0);
            const value_id = h.identifier(ctx, "value");
            const gas_cost_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(ctx, 64), 0);
            const gas_cost_id = h.identifier(ctx, "gas_cost");
            var attrs = [_]c.MlirNamedAttribute{
                c.mlirNamedAttributeGet(value_id, fallback_attr),
                c.mlirNamedAttributeGet(gas_cost_id, gas_cost_attr),
            };
            c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
        } else {
            const value_id = h.identifier(ctx, "value");
            const gas_cost_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(ctx, 64), 0);
            const gas_cost_id = h.identifier(ctx, "gas_cost");
            var attrs = [_]c.MlirNamedAttribute{
                c.mlirNamedAttributeGet(value_id, attr),
                c.mlirNamedAttributeGet(gas_cost_id, gas_cost_attr),
            };
            c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
        }
    }

    const op = c.mlirOperationCreate(&state);
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
    const ty = c.mlirIntegerTypeGet(ctx, @intCast(string_len * 8));
    const loc = locations.createLocation(string_lit.span);
    const op = ora_dialect.createStringConstant(string_lit.value, ty, loc);
    const length_attr = h.intAttr(ctx, c.mlirIntegerTypeGet(ctx, 32), @intCast(string_len));
    const length_name = h.strRef("length");
    c.mlirOperationSetAttributeByName(op, length_name, length_attr);
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
    const addr_ty = c.oraAddressTypeGet(ctx);
    const loc = locations.createLocation(addr_lit.span);

    const addr_str = if (std.mem.startsWith(u8, addr_lit.value, "0x"))
        addr_lit.value[2..]
    else
        addr_lit.value;

    if (addr_str.len != 40) {
        std.debug.print("ERROR: Invalid address length '{d}' (expected 40 hex characters): {s}\n", .{ addr_str.len, addr_lit.value });
    }

    for (addr_str) |char| {
        if (!((char >= '0' and char <= '9') or (char >= 'a' and char <= 'f') or (char >= 'A' and char <= 'F'))) {
            std.debug.print("ERROR: Invalid hex character '{c}' in address '{s}'\n", .{ char, addr_lit.value });
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

    const i160_ty = c.mlirIntegerTypeGet(ctx, 160);
    var state = h.opState("arith.constant", loc);
    c.mlirOperationStateAddResults(&state, 1, @ptrCast(&i160_ty));

    if (parsed <= std.math.maxInt(i64)) {
        const attr = c.mlirIntegerAttrGet(i160_ty, @intCast(parsed));
        const value_id = h.identifier(ctx, "value");
        const gas_cost_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(ctx, 64), 0);
        const gas_cost_id = h.identifier(ctx, "gas_cost");
        var attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(value_id, attr),
            c.mlirNamedAttributeGet(gas_cost_id, gas_cost_attr),
        };
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
    } else {
        var decimal_buf: [80]u8 = undefined;
        const decimal_str = std.fmt.bufPrint(&decimal_buf, "{}", .{parsed}) catch {
            std.debug.print("ERROR: Failed to format address as decimal\n", .{});
            const fallback_attr = c.mlirIntegerAttrGet(i160_ty, 0);
            const value_id = h.identifier(ctx, "value");
            const gas_cost_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(ctx, 64), 0);
            const gas_cost_id = h.identifier(ctx, "gas_cost");
            var attrs = [_]c.MlirNamedAttribute{
                c.mlirNamedAttributeGet(value_id, fallback_attr),
                c.mlirNamedAttributeGet(gas_cost_id, gas_cost_attr),
            };
            c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
            const op = c.mlirOperationCreate(&state);
            h.appendOp(block, op);
            return h.getResult(op, 0);
        };
        const addr_str_ref = h.strRef(decimal_str);
        const attr = c.oraIntegerAttrGetFromString(i160_ty, addr_str_ref);

        if (c.mlirAttributeIsNull(attr)) {
            std.debug.print("ERROR: Failed to create address attribute from string '{s}'\n", .{addr_lit.value});
            const fallback_attr = c.mlirIntegerAttrGet(i160_ty, 0);
            const value_id = h.identifier(ctx, "value");
            const gas_cost_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(ctx, 64), 0);
            const gas_cost_id = h.identifier(ctx, "gas_cost");
            var attrs = [_]c.MlirNamedAttribute{
                c.mlirNamedAttributeGet(value_id, fallback_attr),
                c.mlirNamedAttributeGet(gas_cost_id, gas_cost_attr),
            };
            c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
        } else {
            const value_id = h.identifier(ctx, "value");
            const gas_cost_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(ctx, 64), 0);
            const gas_cost_id = h.identifier(ctx, "gas_cost");
            var attrs = [_]c.MlirNamedAttribute{
                c.mlirNamedAttributeGet(value_id, attr),
                c.mlirNamedAttributeGet(gas_cost_id, gas_cost_attr),
            };
            c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
        }
    }

    const const_op = c.mlirOperationCreate(&state);
    h.appendOp(block, const_op);
    const i160_value = h.getResult(const_op, 0);
    const converted = type_mapper.createConversionOp(block, i160_value, addr_ty, addr_lit.span);
    return converted;
}

fn lowerHexLiteral(
    ctx: c.MlirContext,
    block: c.MlirBlock,
    ora_dialect: *OraDialect,
    locations: LocationTracker,
    hex_lit: lib.ast.Expressions.HexLiteral,
) c.MlirValue {
    const ty = c.mlirIntegerTypeGet(ctx, constants.DEFAULT_INTEGER_BITS);
    const loc = locations.createLocation(hex_lit.span);
    const op = ora_dialect.createHexConstant(hex_lit.value, ty, loc);
    h.appendOp(block, op);
    return h.getResult(op, 0);
}

fn lowerBinaryLiteral(
    ctx: c.MlirContext,
    block: c.MlirBlock,
    locations: LocationTracker,
    bin_lit: lib.ast.Expressions.BinaryLiteral,
) c.MlirValue {
    const ty = c.mlirIntegerTypeGet(ctx, constants.DEFAULT_INTEGER_BITS);
    const loc = locations.createLocation(bin_lit.span);
    var state = h.opState("ora.binary.constant", loc);
    c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ty));

    const bin_str = if (std.mem.startsWith(u8, bin_lit.value, "0b"))
        bin_lit.value[2..]
    else
        bin_lit.value;

    for (bin_str) |char| {
        if (char != '0' and char != '1') {
            std.debug.print("ERROR: Invalid binary character '{c}' in binary literal '{s}'\n", .{ char, bin_lit.value });
            break;
        }
    }

    if (bin_str.len > 64) {
        std.debug.print("WARNING: Binary literal '{s}' may overflow i64 (length: {d})\n", .{ bin_lit.value, bin_str.len });
    }

    const parsed: i64 = std.fmt.parseInt(i64, bin_str, 2) catch |err| blk: {
        std.debug.print("ERROR: Failed to parse binary literal '{s}': {s}\n", .{ bin_lit.value, @errorName(err) });
        break :blk 0;
    };
    const attr = c.mlirIntegerAttrGet(ty, parsed);

    const value_id = h.identifier(ctx, "value");
    const binary_id = h.identifier(ctx, "ora.binary");
    const binary_ref = c.mlirStringRefCreate(bin_lit.value.ptr, bin_lit.value.len);
    const binary_attr = c.mlirStringAttrGet(ctx, binary_ref);
    const length_id = h.identifier(ctx, "length");
    const length_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(ctx, 32), @intCast(bin_str.len));

    var attrs = [_]c.MlirNamedAttribute{
        c.mlirNamedAttributeGet(value_id, attr),
        c.mlirNamedAttributeGet(binary_id, binary_attr),
        c.mlirNamedAttributeGet(length_id, length_attr),
    };
    c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
    const op = c.mlirOperationCreate(&state);
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
    const ty = c.mlirIntegerTypeGet(ctx, 8);
    const loc = locations.createLocation(char_lit.span);

    if (char_lit.value > 127) {
        std.debug.print("ERROR: Invalid character value '{d}' (not ASCII)\n", .{char_lit.value});
        return expr_helpers.createConstant(ctx, block, ora_dialect, locations, 0, char_lit.span);
    }

    const character_id = h.identifier(ctx, "ora.character_literal");
    const custom_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(character_id, c.mlirBoolAttrGet(ctx, 1))};
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
    const bytes_len = bytes_lit.value.len;
    const ty = c.mlirIntegerTypeGet(ctx, @intCast(bytes_len * 8));
    const loc = locations.createLocation(bytes_lit.span);
    var state = h.opState("arith.constant", loc);
    c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ty));

    const bytes_str = if (std.mem.startsWith(u8, bytes_lit.value, "0x"))
        bytes_lit.value[2..]
    else
        bytes_lit.value;

    if (bytes_str.len % 2 != 0) {
        std.debug.print("ERROR: Invalid bytes length '{d}' (must be even number of hex digits): {s}\n", .{ bytes_str.len, bytes_lit.value });
        return expr_helpers.createConstant(ctx, block, ora_dialect, locations, 0, bytes_lit.span);
    }

    for (bytes_str) |char| {
        if (!((char >= '0' and char <= '9') or (char >= 'a' and char <= 'f') or (char >= 'A' and char <= 'F'))) {
            std.debug.print("ERROR: Invalid hex character '{c}' in bytes '{s}'\n", .{ char, bytes_lit.value });
            return expr_helpers.createConstant(ctx, block, ora_dialect, locations, 0, bytes_lit.span);
        }
    }

    const parsed: i64 = std.fmt.parseInt(i64, bytes_str, 16) catch |err| {
        std.debug.print("ERROR: Failed to parse bytes literal '{s}': {s}\n", .{ bytes_lit.value, @errorName(err) });
        return expr_helpers.createConstant(ctx, block, ora_dialect, locations, 0, bytes_lit.span);
    };

    const attr = c.mlirIntegerAttrGet(ty, parsed);
    const value_id = h.identifier(ctx, "value");
    const bytes_id = h.identifier(ctx, "ora.bytes_literal");
    var attrs = [_]c.MlirNamedAttribute{ c.mlirNamedAttributeGet(value_id, attr), c.mlirNamedAttributeGet(bytes_id, c.mlirBoolAttrGet(ctx, 1)) };
    c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

    const op = c.mlirOperationCreate(&state);
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
