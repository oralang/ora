const std = @import("std");
const testing = std.testing;
const types = @import("types.zig");

pub const ArgValue = types.ArgValue;

const StaticWireKind = enum {
    bool,
    address,
};

const static_wire_map = std.StaticStringMap(StaticWireKind).initComptime(.{
    .{ "bool", .bool },
    .{ "address", .address },
});

const fixed_bytes_map = blk: {
    const KV = struct { []const u8, u8 };
    var entries: [32]KV = undefined;
    for (0..32) |i| {
        entries[i] = .{ std.fmt.comptimePrint("bytes{d}", .{i + 1}), @intCast(i + 1) };
    }
    break :blk std.StaticStringMap(u8).initComptime(entries);
};

pub fn parseString(value: []const u8) ![]const u8 {
    if (value.len < 2 or value[0] != '"' or value[value.len - 1] != '"') return error.InvalidString;
    const inner = value[1 .. value.len - 1];
    if (std.mem.indexOfScalar(u8, inner, '\\') != null) return error.UnsupportedEscape;
    return inner;
}

pub fn parseArgArray(allocator: std.mem.Allocator, value: []const u8) ![]ArgValue {
    const trimmed = std.mem.trim(u8, value, " \t");
    if (trimmed.len < 2 or trimmed[0] != '[' or trimmed[trimmed.len - 1] != ']') return error.InvalidArgs;
    const tokens = try splitTopLevelItems(allocator, trimmed[1 .. trimmed.len - 1]);
    defer allocator.free(tokens);

    var args = std.ArrayList(ArgValue){};
    errdefer args.deinit(allocator);
    for (tokens) |token| {
        if (token.len == 0) return error.InvalidArgs;
        try args.append(allocator, try parseArgValue(token));
    }
    return try args.toOwnedSlice(allocator);
}

pub fn parseArgValue(token: []const u8) !ArgValue {
    if (token.len == 0) return error.InvalidArgs;
    if (token[0] == '"') {
        const s = try parseString(token);
        // `"@name"` is a cross-contract address reference, resolved at run time.
        if (s.len > 1 and s[0] == '@') return .{ .contract_ref = s[1..] };
        return .{ .literal = s };
    }
    if (std.mem.eql(u8, token, "true")) return .{ .boolean = true };
    if (std.mem.eql(u8, token, "false")) return .{ .boolean = false };
    if (std.mem.indexOfAny(u8, token, "{}") != null) return error.InvalidArgs;
    return .{ .literal = token };
}

pub fn parseU256(value: []const u8) !u256 {
    const token = if (value.len > 0 and value[0] == '"') try parseString(value) else value;
    const trimmed = std.mem.trim(u8, token, " \t");
    if (trimmed.len == 0) return error.InvalidNumber;
    return try std.fmt.parseInt(u256, trimmed, 0);
}

pub fn parseSelector(text: []const u8) ![4]u8 {
    return try parseHexBytesFixed(4, text);
}

pub fn parseHexBytesFixed(comptime len: usize, text: []const u8) ![len]u8 {
    var out: [len]u8 = undefined;
    const trimmed = stripHexPrefix(std.mem.trim(u8, text, " \t\r\n"));
    if (trimmed.len != len * 2) return error.InvalidHex;
    _ = try std.fmt.hexToBytes(&out, trimmed);
    return out;
}

pub fn parseHexBytes(allocator: std.mem.Allocator, text: []const u8) ![]u8 {
    const trimmed = stripHexPrefix(std.mem.trim(u8, text, " \t\r\n"));
    if (trimmed.len % 2 != 0) return error.InvalidHex;
    const out = try allocator.alloc(u8, trimmed.len / 2);
    errdefer allocator.free(out);
    _ = try std.fmt.hexToBytes(out, trimmed);
    return out;
}

pub fn stripHexPrefix(text: []const u8) []const u8 {
    if (std.mem.startsWith(u8, text, "0x") or std.mem.startsWith(u8, text, "0X")) return text[2..];
    return text;
}

pub fn encodeArgs(allocator: std.mem.Allocator, wires: []const []const u8, args: []const ArgValue) anyerror![]u8 {
    return try encodeAbiSequence(allocator, wires, args);
}

pub fn isSingleStaticWord(allocator: std.mem.Allocator, wire_type: []const u8) anyerror!bool {
    const words = (try staticAbiWordCount(allocator, wire_type)) orelse return false;
    return words == 1;
}

fn encodeAbiSequence(allocator: std.mem.Allocator, wires: []const []const u8, args: []const ArgValue) anyerror![]u8 {
    if (wires.len != args.len) return error.ArgumentCountMismatch;

    var head_size: usize = 0;
    for (wires) |wire| {
        if (try staticAbiWordCount(allocator, wire)) |words| {
            head_size += words * 32;
        } else {
            head_size += 32;
        }
    }

    var head = std.ArrayList(u8){};
    errdefer head.deinit(allocator);
    var tail = std.ArrayList(u8){};
    defer tail.deinit(allocator);

    for (wires, args) |wire, arg| {
        if (try staticAbiWordCount(allocator, wire)) |_| {
            const encoded = try encodeAbiValue(allocator, wire, arg);
            defer allocator.free(encoded);
            try head.appendSlice(allocator, encoded);
        } else {
            try appendU256Word(&head, allocator, @intCast(head_size + tail.items.len));
            const encoded = try encodeAbiValue(allocator, wire, arg);
            defer allocator.free(encoded);
            try tail.appendSlice(allocator, encoded);
        }
    }

    try head.appendSlice(allocator, tail.items);
    return try head.toOwnedSlice(allocator);
}

fn encodeAbiValue(allocator: std.mem.Allocator, wire_type: []const u8, arg: ArgValue) anyerror![]u8 {
    if (arrayElementType(wire_type)) |element_type| return try encodeDynamicArray(allocator, element_type, arg);
    if (std.mem.eql(u8, wire_type, "bytes")) {
        const bytes = try parseHexBytes(allocator, try argAsLiteral(arg));
        defer allocator.free(bytes);
        return try encodeDynamicBytes(allocator, bytes);
    }
    if (std.mem.eql(u8, wire_type, "string")) return try encodeDynamicBytes(allocator, try argAsLiteral(arg));
    if (tupleInner(wire_type) != null) return try encodeTuple(allocator, wire_type, arg);

    const out = try allocator.alloc(u8, 32);
    errdefer allocator.free(out);
    try encodeStaticAbiWord(out[0..32], wire_type, arg);
    return out;
}

fn encodeDynamicArray(allocator: std.mem.Allocator, element_type: []const u8, arg: ArgValue) anyerror![]u8 {
    const elements = try parseDelimitedItems(allocator, try argAsLiteral(arg), '[', ']');
    defer allocator.free(elements);

    const wires = try allocator.alloc([]const u8, elements.len);
    defer allocator.free(wires);
    const values = try allocator.alloc(ArgValue, elements.len);
    defer allocator.free(values);

    for (elements, 0..) |element, i| {
        wires[i] = element_type;
        values[i] = try parseArgValue(element);
    }

    const body = try encodeAbiSequence(allocator, wires, values);
    defer allocator.free(body);

    var out = std.ArrayList(u8){};
    errdefer out.deinit(allocator);
    try appendU256Word(&out, allocator, @intCast(elements.len));
    try out.appendSlice(allocator, body);
    return try out.toOwnedSlice(allocator);
}

fn encodeTuple(allocator: std.mem.Allocator, wire_type: []const u8, arg: ArgValue) anyerror![]u8 {
    const fields = try parseDelimitedItems(allocator, wire_type, '(', ')');
    defer allocator.free(fields);
    const values_text = try parseDelimitedItems(allocator, try argAsLiteral(arg), '(', ')');
    defer allocator.free(values_text);
    if (fields.len != values_text.len) return error.ArgumentCountMismatch;

    const values = try allocator.alloc(ArgValue, values_text.len);
    defer allocator.free(values);
    for (values_text, 0..) |value, i| values[i] = try parseArgValue(value);
    return try encodeAbiSequence(allocator, fields, values);
}

fn encodeDynamicBytes(allocator: std.mem.Allocator, bytes: []const u8) ![]u8 {
    var out = std.ArrayList(u8){};
    errdefer out.deinit(allocator);
    try appendU256Word(&out, allocator, @intCast(bytes.len));
    try out.appendSlice(allocator, bytes);
    try out.appendNTimes(allocator, 0, paddedAbiByteLen(bytes.len) - bytes.len);
    return try out.toOwnedSlice(allocator);
}

fn appendU256Word(out: *std.ArrayList(u8), allocator: std.mem.Allocator, value: u256) !void {
    var word: [32]u8 = undefined;
    std.mem.writeInt(u256, &word, value, .big);
    try out.appendSlice(allocator, &word);
}

fn paddedAbiByteLen(len: usize) usize {
    return ((len + 31) / 32) * 32;
}

fn staticAbiWordCount(allocator: std.mem.Allocator, wire_type: []const u8) anyerror!?usize {
    if (arrayElementType(wire_type) != null) return null;
    if (std.mem.eql(u8, wire_type, "bytes") or std.mem.eql(u8, wire_type, "string")) return null;
    if (tupleInner(wire_type) != null) {
        const fields = try parseDelimitedItems(allocator, wire_type, '(', ')');
        defer allocator.free(fields);
        var total: usize = 0;
        for (fields) |field| {
            const count = (try staticAbiWordCount(allocator, field)) orelse return null;
            total += count;
        }
        return total;
    }
    if (isStaticAbiScalar(wire_type)) return 1;
    return error.UnsupportedArgType;
}

fn isStaticAbiScalar(wire_type: []const u8) bool {
    if (static_wire_map.get(wire_type) != null) return true;
    if (parseFixedBytesWireType(wire_type) != null) return true;
    if (std.mem.startsWith(u8, wire_type, "uint")) return (parseAbiIntBits(wire_type, "uint") catch null) != null;
    if (std.mem.startsWith(u8, wire_type, "int")) return (parseAbiIntBits(wire_type, "int") catch null) != null;
    return false;
}

fn arrayElementType(wire_type: []const u8) ?[]const u8 {
    if (!std.mem.endsWith(u8, wire_type, "[]")) return null;
    return wire_type[0 .. wire_type.len - 2];
}

fn tupleInner(wire_type: []const u8) ?[]const u8 {
    if (wire_type.len < 2 or wire_type[0] != '(' or wire_type[wire_type.len - 1] != ')') return null;
    return wire_type[1 .. wire_type.len - 1];
}

fn parseDelimitedItems(allocator: std.mem.Allocator, value: []const u8, open: u8, close: u8) ![][]const u8 {
    const trimmed = std.mem.trim(u8, value, " \t");
    if (trimmed.len < 2 or trimmed[0] != open or trimmed[trimmed.len - 1] != close) return error.InvalidArgs;
    return try splitTopLevelItems(allocator, trimmed[1 .. trimmed.len - 1]);
}

pub fn encodeStaticAbiWord(dest: []u8, wire_type: []const u8, arg: ArgValue) !void {
    if (dest.len != 32) return error.InvalidAbiWord;
    @memset(dest, 0);

    if (static_wire_map.get(wire_type)) |kind| switch (kind) {
        .bool => {
            dest[31] = if (try argAsBool(arg)) 1 else 0;
            return;
        },
        .address => {
            const bytes = try parseHexBytesFixed(20, try argAsLiteral(arg));
            @memcpy(dest[12..32], &bytes);
            return;
        },
    };

    if (std.mem.startsWith(u8, wire_type, "uint")) {
        const bits = try parseAbiIntBits(wire_type, "uint");
        const value = try parseUnsignedArg(arg);
        if (value > maxUnsignedValue(bits)) return error.ValueOutOfRange;
        std.mem.writeInt(u256, dest[0..32], value, .big);
        return;
    }

    if (std.mem.startsWith(u8, wire_type, "int")) {
        const bits = try parseAbiIntBits(wire_type, "int");
        const value = try parseSignedArg(arg);
        const bounds = signedBounds(bits);
        if (value < bounds.min or value > bounds.max) return error.ValueOutOfRange;
        const encoded: u256 = @bitCast(value);
        std.mem.writeInt(u256, dest[0..32], encoded, .big);
        return;
    }

    if (parseFixedBytesWireType(wire_type)) |len| {
        try writeFixedBytesArg(dest, len, try argAsLiteral(arg));
        return;
    }

    return error.UnsupportedArgType;
}

pub fn expectStaticReturn(wire_type: []const u8, expected: ArgValue, actual_word: []const u8) !void {
    if (actual_word.len != 32) return error.InvalidAbiWord;

    if (static_wire_map.get(wire_type)) |kind| switch (kind) {
        .bool => {
            try testing.expectEqual(try argAsBool(expected), std.mem.readInt(u256, actual_word[0..32], .big) != 0);
            return;
        },
        .address => {
            const expected_bytes = try parseHexBytesFixed(20, try argAsLiteral(expected));
            try testing.expect(std.mem.allEqual(u8, actual_word[0..12], 0));
            try testing.expectEqualSlices(u8, &expected_bytes, actual_word[12..32]);
            return;
        },
    };

    if (std.mem.startsWith(u8, wire_type, "uint")) {
        const bits = try parseAbiIntBits(wire_type, "uint");
        const expected_value = try parseUnsignedArg(expected);
        if (expected_value > maxUnsignedValue(bits)) return error.ValueOutOfRange;
        try testing.expectEqual(expected_value, std.mem.readInt(u256, actual_word[0..32], .big) & maxUnsignedValue(bits));
        return;
    }

    if (std.mem.startsWith(u8, wire_type, "int")) {
        const bits = try parseAbiIntBits(wire_type, "int");
        const expected_value = try parseSignedArg(expected);
        const bounds = signedBounds(bits);
        if (expected_value < bounds.min or expected_value > bounds.max) return error.ValueOutOfRange;
        var expected_word: [32]u8 = undefined;
        try encodeStaticAbiWord(&expected_word, wire_type, expected);
        if (!std.mem.eql(u8, &expected_word, actual_word[0..32])) return error.NonCanonicalAbiReturn;
        return;
    }

    if (parseFixedBytesWireType(wire_type)) |len| {
        var expected_word: [32]u8 = undefined;
        try encodeStaticAbiWord(&expected_word, wire_type, expected);
        try testing.expectEqualSlices(u8, expected_word[0..len], actual_word[0..len]);
        return;
    }

    return error.UnsupportedReturnType;
}

pub fn decodeSignedWord(word: []const u8, bits: u16) i256 {
    const value = std.mem.readInt(u256, word[0..32], .big);
    if (bits == 256) return @bitCast(value);

    const mask = maxUnsignedValue(bits);
    const low = value & mask;
    const sign_bit = @as(u256, 1) << @intCast(bits - 1);
    if ((low & sign_bit) == 0) return @intCast(low);
    return @as(i256, @intCast(low)) - (@as(i256, 1) << @intCast(bits));
}

pub fn specTypeMatchesAbiWire(spec_type: []const u8, abi_wire: []const u8) bool {
    if (std.mem.eql(u8, spec_type, abi_wire)) return true;
    if (spec_type.len > 1 and spec_type[0] == 'u') return abiWireMatchesShortInteger(abi_wire, "uint", spec_type[1..]);
    if (spec_type.len > 1 and spec_type[0] == 'i') return abiWireMatchesShortInteger(abi_wire, "int", spec_type[1..]);
    return false;
}

fn abiWireMatchesShortInteger(abi_wire: []const u8, prefix: []const u8, bits: []const u8) bool {
    if (bits.len == 0) return false;
    for (bits) |c| if (c < '0' or c > '9') return false;
    return std.mem.startsWith(u8, abi_wire, prefix) and std.mem.eql(u8, abi_wire[prefix.len..], bits);
}

fn parseAbiIntBits(wire_type: []const u8, prefix: []const u8) !u16 {
    if (wire_type.len <= prefix.len) return error.UnsupportedArgType;
    const bits = try std.fmt.parseInt(u16, wire_type[prefix.len..], 10);
    if (bits == 0 or bits > 256 or bits % 8 != 0) return error.UnsupportedArgType;
    return bits;
}

fn maxUnsignedValue(bits: u16) u256 {
    if (bits == 256) return std.math.maxInt(u256);
    return (@as(u256, 1) << @intCast(bits)) - 1;
}

const SignedBounds = struct { min: i256, max: i256 };

fn signedBounds(bits: u16) SignedBounds {
    if (bits == 256) return .{ .min = std.math.minInt(i256), .max = std.math.maxInt(i256) };
    const shift: u8 = @intCast(bits - 1);
    return .{
        .min = -(@as(i256, 1) << shift),
        .max = (@as(i256, 1) << shift) - 1,
    };
}

pub fn parseFixedBytesWireType(wire_type: []const u8) ?u8 {
    return fixed_bytes_map.get(wire_type);
}

fn writeFixedBytesArg(dest: []u8, len: u8, text: []const u8) !void {
    const hex = stripHexPrefix(std.mem.trim(u8, text, " \t\r\n"));
    if (hex.len != @as(usize, len) * 2) return error.InvalidFixedBytesLiteral;
    _ = try std.fmt.hexToBytes(dest[0..len], hex);
}

pub fn argAsLiteral(arg: ArgValue) ![]const u8 {
    return switch (arg) {
        .literal => |literal| literal,
        .boolean => error.InvalidArgumentType,
        .contract_ref => error.UnresolvedContractRef,
    };
}

pub fn argAsBool(arg: ArgValue) !bool {
    return switch (arg) {
        .boolean => |value| value,
        .contract_ref => error.InvalidArgumentType,
        .literal => |literal| blk: {
            const trimmed = std.mem.trim(u8, literal, " \t");
            if (std.mem.eql(u8, trimmed, "true") or std.mem.eql(u8, trimmed, "1")) break :blk true;
            if (std.mem.eql(u8, trimmed, "false") or std.mem.eql(u8, trimmed, "0")) break :blk false;
            return error.InvalidBoolLiteral;
        },
    };
}

fn parseUnsignedArg(arg: ArgValue) !u256 {
    return try std.fmt.parseInt(u256, std.mem.trim(u8, try argAsLiteral(arg), " \t"), 0);
}

fn parseSignedArg(arg: ArgValue) !i256 {
    return try std.fmt.parseInt(i256, std.mem.trim(u8, try argAsLiteral(arg), " \t"), 0);
}

pub fn splitTopLevelArgs(text: []const u8, out: [][]const u8) !usize {
    var paren_depth: usize = 0;
    var bracket_depth: usize = 0;
    var start: usize = 0;
    var count: usize = 0;
    var in_string = false;
    var escaped = false;

    for (text, 0..) |c, i| {
        if (escaped) {
            escaped = false;
            continue;
        }
        if (in_string and c == '\\') {
            escaped = true;
            continue;
        }
        if (c == '"') {
            in_string = !in_string;
            continue;
        }
        if (in_string) continue;

        switch (c) {
            '(' => paren_depth += 1,
            '[' => bracket_depth += 1,
            ')' => {
                if (paren_depth == 0) return error.InvalidSlotExpression;
                paren_depth -= 1;
            },
            ']' => {
                if (bracket_depth == 0) return error.InvalidSlotExpression;
                bracket_depth -= 1;
            },
            ',' => if (paren_depth == 0 and bracket_depth == 0) {
                if (count >= out.len) return error.InvalidSlotExpression;
                out[count] = std.mem.trim(u8, text[start..i], " \t");
                if (out[count].len == 0) return error.InvalidSlotExpression;
                count += 1;
                start = i + 1;
            },
            else => {},
        }
    }

    if (in_string or paren_depth != 0 or bracket_depth != 0 or count >= out.len) return error.InvalidSlotExpression;
    out[count] = std.mem.trim(u8, text[start..], " \t");
    if (out[count].len == 0) return error.InvalidSlotExpression;
    return count + 1;
}

pub fn splitTopLevelItems(allocator: std.mem.Allocator, text: []const u8) ![][]const u8 {
    const trimmed = std.mem.trim(u8, text, " \t");
    if (trimmed.len == 0) return allocator.alloc([]const u8, 0);

    var items = std.ArrayList([]const u8){};
    errdefer items.deinit(allocator);

    var paren_depth: usize = 0;
    var bracket_depth: usize = 0;
    var start: usize = 0;
    var in_string = false;
    var escaped = false;

    for (trimmed, 0..) |c, i| {
        if (escaped) {
            escaped = false;
            continue;
        }
        if (in_string and c == '\\') {
            escaped = true;
            continue;
        }
        if (c == '"') {
            in_string = !in_string;
            continue;
        }
        if (in_string) continue;

        switch (c) {
            '(' => paren_depth += 1,
            '[' => bracket_depth += 1,
            ')' => {
                if (paren_depth == 0) return error.InvalidArgs;
                paren_depth -= 1;
            },
            ']' => {
                if (bracket_depth == 0) return error.InvalidArgs;
                bracket_depth -= 1;
            },
            ',' => if (paren_depth == 0 and bracket_depth == 0) {
                const token = std.mem.trim(u8, trimmed[start..i], " \t");
                if (token.len == 0) return error.InvalidArgs;
                try items.append(allocator, token);
                start = i + 1;
            },
            else => {},
        }
    }

    if (in_string or paren_depth != 0 or bracket_depth != 0) return error.InvalidArgs;
    const token = std.mem.trim(u8, trimmed[start..], " \t");
    if (token.len == 0) return error.InvalidArgs;
    try items.append(allocator, token);
    return try items.toOwnedSlice(allocator);
}
