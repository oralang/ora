const std = @import("std");
const abi_layout = @import("layout.zig");
const comptime_mod = @import("../comptime/mod.zig");
const sema = @import("../sema/mod.zig");
const refinements = @import("../sema/refinements.zig");

const CtAggregate = comptime_mod.CtAggregate;
const CtHeap = comptime_mod.CtHeap;
const CtValue = comptime_mod.CtValue;

pub const MAX_DECODE_DEPTH: usize = 16;
pub const MAX_BUFFER_SIZE: usize = 1024 * 1024;
pub const MAX_ARRAY_LENGTH: usize = MAX_BUFFER_SIZE / 32;
pub const MAX_STRING_LENGTH: usize = MAX_BUFFER_SIZE;

pub const DecodeError = enum(u32) {
    truncated_buffer,
    oversize_buffer,
    buffer_size_exceeded,
    non_canonical_padding,
    invalid_bool_value,
    invalid_address,
    invalid_fixed_bytes,
    enum_out_of_range,
    depth_limit_exceeded,
    array_length_exceeded,
    refinement_violation,
    non_canonical_encoding,
    invalid_offset,
    length_overflow,
    string_length_exceeded,
};

pub const DecodeResult = union(enum) {
    ok: CtValue,
    err: DecodeError,
};

pub const DecodeMode = enum {
    strict,
    permissive,
};

pub const TypeResolver = struct {
    context: *anyopaque,
    typeIdForType: *const fn (*anyopaque, sema.Type) anyerror!?u32,
    structFields: *const fn (*anyopaque, []const u8) anyerror!?[]const sema.AnonymousStructField,
    enumVariantCount: *const fn (*anyopaque, []const u8) anyerror!?usize,
};

pub fn decodeComptimeValue(
    allocator: std.mem.Allocator,
    heap: *CtHeap,
    resolver: TypeResolver,
    layout: abi_layout.LayoutNode,
    target_type: sema.Type,
    bytes: []const u8,
) !DecodeResult {
    return decodeComptimeValueWithMode(allocator, heap, resolver, layout, target_type, bytes, .strict);
}

pub fn decodeComptimeValuePermissive(
    allocator: std.mem.Allocator,
    heap: *CtHeap,
    resolver: TypeResolver,
    layout: abi_layout.LayoutNode,
    target_type: sema.Type,
    bytes: []const u8,
) !DecodeResult {
    return decodeComptimeValueWithMode(allocator, heap, resolver, layout, target_type, bytes, .permissive);
}

pub fn decodeComptimeValueWithMode(
    allocator: std.mem.Allocator,
    heap: *CtHeap,
    resolver: TypeResolver,
    layout: abi_layout.LayoutNode,
    target_type: sema.Type,
    bytes: []const u8,
    mode: DecodeMode,
) !DecodeResult {
    if (bytes.len > MAX_BUFFER_SIZE) return .{ .err = .buffer_size_exceeded };

    const decoded = if (layout.isDynamic() and topLevelWrapsSingleArgument(target_type))
        try decodeSingleTopLevelArgument(allocator, heap, resolver, layout, target_type, bytes, mode)
    else
        try decodeNodeAt(allocator, heap, resolver, layout, target_type, bytes, 0, 0, mode);

    const value = switch (decoded) {
        .value => |value| blk: {
            if (mode == .strict and value.size < bytes.len) return .{ .err = .oversize_buffer };
            if (value.size > bytes.len) return .{ .err = .truncated_buffer };
            break :blk value.value;
        },
        .err => |err| return .{ .err = err },
    };
    return .{ .ok = value };
}

fn DecodeStep(comptime T: type) type {
    return union(enum) {
        value: T,
        err: DecodeError,
    };
}

const DecodeValue = struct {
    value: CtValue,
    size: usize,
};

fn topLevelWrapsSingleArgument(target_type: sema.Type) bool {
    // ABI encodes a single non-tuple argument as a one-element argument list:
    // bare `string` and `(string,)` both start with a 32-byte offset frame.
    return switch (unwrapRefinement(target_type)) {
        .tuple => false,
        else => true,
    };
}

fn decodeSingleTopLevelArgument(
    allocator: std.mem.Allocator,
    heap: *CtHeap,
    resolver: TypeResolver,
    layout: abi_layout.LayoutNode,
    target_type: sema.Type,
    bytes: []const u8,
    mode: DecodeMode,
) anyerror!DecodeStep(DecodeValue) {
    const head_size: usize = 32;
    const offset = switch (readUsizeWord(bytes, 0, .invalid_offset)) {
        .value => |value| value,
        .err => |err| return .{ .err = err },
    };
    if (mode == .strict and offset != head_size) return .{ .err = .non_canonical_encoding };

    const decoded = try decodeNodeAt(allocator, heap, resolver, layout, target_type, bytes, offset, 1, mode);
    return switch (decoded) {
        .value => |value| blk: {
            const tail_end = checkedAdd(offset, value.size) orelse return .{ .err = .length_overflow };
            break :blk .{ .value = .{
                .value = value.value,
                .size = if (mode == .strict) head_size + value.size else @max(head_size, tail_end),
            } };
        },
        .err => |err| .{ .err = err },
    };
}

fn decodeNodeAt(
    allocator: std.mem.Allocator,
    heap: *CtHeap,
    resolver: TypeResolver,
    layout: abi_layout.LayoutNode,
    target_type: sema.Type,
    bytes: []const u8,
    offset: usize,
    depth: usize,
    mode: DecodeMode,
) anyerror!DecodeStep(DecodeValue) {
    if (depth > MAX_DECODE_DEPTH) return .{ .err = .depth_limit_exceeded };
    const unwrapped_type = unwrapRefinement(target_type);
    const decoded: DecodeStep(DecodeValue) = switch (layout) {
        .static_word => |word| blk: {
            const value = try decodeStaticWordAt(heap, resolver, word.encoding, unwrapped_type, bytes, offset, mode);
            break :blk switch (value) {
                .value => |decoded_word| .{ .value = .{
                    .value = decoded_word,
                    .size = 32,
                } },
                .err => |err| .{ .err = err },
            };
        },
        .dynamic_bytes => |dynamic_bytes| try decodeDynamicBytesAt(heap, dynamic_bytes.kind, unwrapped_type, bytes, offset, mode),
        .dynamic_array => |array| try decodeDynamicArrayAt(allocator, heap, resolver, array, unwrapped_type, bytes, offset, depth, mode),
        .fixed_array => |array| try decodeFixedArrayAt(allocator, heap, resolver, array, unwrapped_type, bytes, offset, depth, mode),
        .tuple => |tuple| try decodeTupleAt(allocator, heap, resolver, tuple, unwrapped_type, bytes, offset, depth, mode),
    };
    return switch (try validateDecodedValue(target_type, decoded)) {
        .value => |value| .{ .value = value },
        .err => |err| .{ .err = err },
    };
}

fn decodeStaticWordAt(
    heap: *CtHeap,
    resolver: TypeResolver,
    encoding: abi_layout.StaticEncoding,
    target_type: sema.Type,
    bytes: []const u8,
    offset: usize,
    mode: DecodeMode,
) anyerror!DecodeStep(CtValue) {
    const word = readWordAt(bytes, offset) orelse return .{ .err = .truncated_buffer };
    return switch (encoding) {
        .uint => |bits| try decodeUnsignedWord(resolver, target_type, word, bits, mode),
        .int => |bits| decodeSignedWord(target_type, word, bits, mode),
        .bool => decodeBoolWord(word, mode),
        .address => decodeAddressWord(word, mode),
        .fixed_bytes => |len| decodeFixedBytesWord(heap, word, len, mode),
    };
}

fn decodeDynamicBytesAt(
    heap: *CtHeap,
    kind: abi_layout.DynamicBytesKind,
    target_type: sema.Type,
    bytes: []const u8,
    offset: usize,
    mode: DecodeMode,
) anyerror!DecodeStep(DecodeValue) {
    const len = switch (readUsizeWord(bytes, offset, .length_overflow)) {
        .value => |value| value,
        .err => |err| return .{ .err = err },
    };
    if (len > MAX_STRING_LENGTH) return .{ .err = .string_length_exceeded };

    const content_start = checkedAdd(offset, 32) orelse return .{ .err = .length_overflow };
    const padded_len = paddedByteLen(len) orelse return .{ .err = .length_overflow };
    const content_end = checkedAdd(content_start, len) orelse return .{ .err = .length_overflow };
    const padded_end = checkedAdd(content_start, padded_len) orelse return .{ .err = .length_overflow };
    if (padded_end > bytes.len) return .{ .err = .length_overflow };
    if (mode == .strict) {
        for (bytes[content_end..padded_end]) |byte| {
            if (byte != 0) return .{ .err = .non_canonical_encoding };
        }
    }

    const value: CtValue = switch (kind) {
        .string => switch (target_type) {
            .string => .{ .string_ref = try heap.allocString(bytes[content_start..content_end]) },
            else => return error.AbiDecoderInternalShapeMismatch,
        },
        .bytes => switch (target_type) {
            .bytes => .{ .bytes_ref = try heap.allocBytes(bytes[content_start..content_end]) },
            else => return error.AbiDecoderInternalShapeMismatch,
        },
    };
    return .{ .value = .{
        .value = value,
        .size = checkedAdd(32, padded_len) orelse return .{ .err = .length_overflow },
    } };
}

fn decodeUnsignedWord(resolver: TypeResolver, target_type: sema.Type, word: *const [32]u8, bits: u16, mode: DecodeMode) anyerror!DecodeStep(CtValue) {
    var value = std.mem.readInt(u256, word, .big);
    if (bits < 256) {
        const mask = (@as(u256, 1) << @intCast(bits)) - 1;
        if (mode == .strict and (value & ~mask) != 0) return .{ .err = .non_canonical_padding };
        value &= mask;
    }
    if (target_type == .enum_) {
        // TODO(decoder): explicit enum values are not represented here yet.
        // Encoding and decoding both use the positional variant id.
        const count = (try resolver.enumVariantCount(resolver.context, target_type.enum_.name)) orelse return error.AbiDecoderInternalShapeMismatch;
        if (value >= count) return .{ .err = .enum_out_of_range };
        const type_id = (try resolver.typeIdForType(resolver.context, target_type)) orelse return error.AbiDecoderInternalShapeMismatch;
        return .{ .value = .{ .adt_val = .{
            .type_id = type_id,
            .variant_id = @intCast(value),
            .payload = null,
        } } };
    }
    return .{ .value = .{ .integer = value } };
}

fn decodeSignedWord(target_type: sema.Type, word: *const [32]u8, bits: u16, mode: DecodeMode) DecodeStep(CtValue) {
    _ = target_type;
    var value = std.mem.readInt(u256, word, .big);
    if (bits < 256) {
        const sign_bit = @as(u256, 1) << @intCast(bits - 1);
        const mask = (@as(u256, 1) << @intCast(bits)) - 1;
        const expected = if ((value & sign_bit) == 0) value & mask else (value & mask) | ~mask;
        if (mode == .strict and value != expected) return .{ .err = .non_canonical_padding };
        value = expected;
    }
    return .{ .value = .{ .integer = value } };
}

fn decodeBoolWord(word: *const [32]u8, mode: DecodeMode) DecodeStep(CtValue) {
    if (mode == .permissive) return .{ .value = .{ .boolean = std.mem.readInt(u256, word, .big) != 0 } };
    var prefix: [31]u8 = [_]u8{0} ** 31;
    if (!std.mem.eql(u8, word[0..31], &prefix)) return .{ .err = .invalid_bool_value };
    return switch (word[31]) {
        0 => .{ .value = .{ .boolean = false } },
        1 => .{ .value = .{ .boolean = true } },
        else => .{ .err = .invalid_bool_value },
    };
}

fn decodeAddressWord(word: *const [32]u8, mode: DecodeMode) DecodeStep(CtValue) {
    var prefix: [12]u8 = [_]u8{0} ** 12;
    if (mode == .strict and !std.mem.eql(u8, word[0..12], &prefix)) return .{ .err = .invalid_address };
    return .{ .value = .{ .address = std.mem.readInt(u160, word[12..32], .big) } };
}

fn decodeFixedBytesWord(heap: *CtHeap, word: *const [32]u8, len: u8, mode: DecodeMode) anyerror!DecodeStep(CtValue) {
    if (len > 32) return error.AbiDecoderInternalShapeMismatch;
    if (mode == .strict) {
        for (word[len..32]) |byte| {
            if (byte != 0) return .{ .err = .invalid_fixed_bytes };
        }
    }
    return .{ .value = .{ .bytes_ref = try heap.allocBytes(word[0..len]) } };
}

fn decodeDynamicArrayAt(
    allocator: std.mem.Allocator,
    heap: *CtHeap,
    resolver: TypeResolver,
    array: abi_layout.DynamicArrayLayout,
    target_type: sema.Type,
    bytes: []const u8,
    offset: usize,
    depth: usize,
    mode: DecodeMode,
) anyerror!DecodeStep(DecodeValue) {
    const element_type = switch (target_type) {
        .array => |array_ty| array_ty.element_type.*,
        .slice => |slice_ty| slice_ty.element_type.*,
        else => return error.AbiDecoderInternalShapeMismatch,
    };
    const len = switch (readUsizeWord(bytes, offset, .length_overflow)) {
        .value => |value| value,
        .err => |err| return .{ .err = err },
    };
    const elements_base = checkedAdd(offset, 32) orelse return .{ .err = .length_overflow };
    const decoded = try decodeArrayElementsAt(allocator, heap, resolver, array.element.*, element_type, len, bytes, elements_base, depth + 1, mode);
    return switch (decoded) {
        .value => |elements| .{ .value = .{
            .value = .{ .slice_ref = try heap.allocSlice(elements.values) },
            .size = checkedAdd(32, elements.size) orelse return .{ .err = .length_overflow },
        } },
        .err => |err| .{ .err = err },
    };
}

fn decodeFixedArrayAt(
    allocator: std.mem.Allocator,
    heap: *CtHeap,
    resolver: TypeResolver,
    array: abi_layout.FixedArrayLayout,
    target_type: sema.Type,
    bytes: []const u8,
    offset: usize,
    depth: usize,
    mode: DecodeMode,
) anyerror!DecodeStep(DecodeValue) {
    const element_type = switch (target_type) {
        .array => |array_ty| array_ty.element_type.*,
        else => return error.AbiDecoderInternalShapeMismatch,
    };
    if (array.len > MAX_ARRAY_LENGTH) return .{ .err = .array_length_exceeded };
    const decoded = try decodeArrayElementsAt(allocator, heap, resolver, array.element.*, element_type, array.len, bytes, offset, depth + 1, mode);
    return switch (decoded) {
        .value => |elements| .{ .value = .{
            .value = .{ .array_ref = try heap.allocArray(elements.values) },
            .size = elements.size,
        } },
        .err => |err| .{ .err = err },
    };
}

const DecodedElements = struct {
    values: []const CtValue,
    size: usize,
};

fn decodeArrayElementsAt(
    allocator: std.mem.Allocator,
    heap: *CtHeap,
    resolver: TypeResolver,
    element_layout: abi_layout.LayoutNode,
    element_type: sema.Type,
    len: usize,
    bytes: []const u8,
    offset: usize,
    depth: usize,
    mode: DecodeMode,
) anyerror!DecodeStep(DecodedElements) {
    switch (validateArrayHeadBudget(element_layout, len, bytes, offset)) {
        .value => {},
        .err => |err| return .{ .err = err },
    }
    const elems = try allocator.alloc(CtValue, len);
    if (!element_layout.isDynamic()) {
        var cursor = offset;
        for (0..len) |index| {
            const decoded = try decodeNodeAt(allocator, heap, resolver, element_layout, element_type, bytes, cursor, depth + 1, mode);
            switch (decoded) {
                .value => |value| {
                    elems[index] = value.value;
                    cursor = checkedAdd(cursor, value.size) orelse return .{ .err = .length_overflow };
                },
                .err => |err| return .{ .err = err },
            }
        }
        return .{ .value = .{ .values = elems, .size = cursor - offset } };
    }

    const element_head_size = checkedMul(element_layout.headSlotWordCount(), 32) orelse return .{ .err = .length_overflow };
    const head_size = checkedMul(element_head_size, len) orelse return .{ .err = .length_overflow };
    var head_cursor = offset;
    var tail_cursor = head_size;
    for (0..len) |index| {
        const element_offset = switch (readUsizeWord(bytes, head_cursor, .invalid_offset)) {
            .value => |value| value,
            .err => |err| return .{ .err = err },
        };
        if (mode == .strict and element_offset != tail_cursor) return .{ .err = .non_canonical_encoding };
        head_cursor = checkedAdd(head_cursor, 32) orelse return .{ .err = .length_overflow };

        const absolute = checkedAdd(offset, element_offset) orelse return .{ .err = .length_overflow };
        const decoded = try decodeNodeAt(allocator, heap, resolver, element_layout, element_type, bytes, absolute, depth + 1, mode);
        switch (decoded) {
            .value => |value| {
                elems[index] = value.value;
                tail_cursor = if (mode == .strict)
                    checkedAdd(tail_cursor, value.size) orelse return .{ .err = .length_overflow }
                else blk: {
                    const element_end = checkedAdd(element_offset, value.size) orelse return .{ .err = .length_overflow };
                    break :blk @max(tail_cursor, element_end);
                };
            },
            .err => |err| return .{ .err = err },
        }
    }
    return .{ .value = .{ .values = elems, .size = tail_cursor } };
}

fn validateArrayHeadBudget(
    element_layout: abi_layout.LayoutNode,
    len: usize,
    bytes: []const u8,
    offset: usize,
) DecodeStep(void) {
    if (len > MAX_ARRAY_LENGTH) return .{ .err = .array_length_exceeded };
    if (offset > bytes.len) return .{ .err = .truncated_buffer };

    const element_head_size = checkedMul(element_layout.headSlotWordCount(), 32) orelse return .{ .err = .length_overflow };
    // Empty-tuple/void-like elements have no byte footprint; MAX_ARRAY_LENGTH
    // still bounds allocation even though available bytes cannot further cap it.
    if (element_head_size == 0) return .{ .value = {} };

    const available = bytes.len - offset;
    if (len > available / element_head_size) return .{ .err = .truncated_buffer };
    return .{ .value = {} };
}

fn decodeTupleAt(
    allocator: std.mem.Allocator,
    heap: *CtHeap,
    resolver: TypeResolver,
    tuple: abi_layout.TupleLayout,
    target_type: sema.Type,
    bytes: []const u8,
    offset: usize,
    depth: usize,
    mode: DecodeMode,
) anyerror!DecodeStep(DecodeValue) {
    if (tuple.elements.len == 0) return .{ .value = .{ .value = .void_val, .size = 0 } };
    switch (target_type) {
        .tuple => |elements| {
            if (elements.len != tuple.elements.len) return error.AbiDecoderInternalShapeMismatch;
            const decoded = switch (try decodeTupleElementsAt(allocator, heap, resolver, tuple.elements, elements, bytes, offset, depth + 1, mode)) {
                .value => |decoded| decoded,
                .err => |err| return .{ .err = err },
            };
            return .{ .value = .{
                .value = .{ .tuple_ref = try heap.allocTuple(decoded.values) },
                .size = decoded.size,
            } };
        },
        .anonymous_struct => |struct_type| {
            if (struct_type.fields.len != tuple.elements.len) return error.AbiDecoderInternalShapeMismatch;
            const types = try allocator.alloc(sema.Type, struct_type.fields.len);
            for (struct_type.fields, 0..) |field, index| {
                types[index] = field.ty;
            }
            const decoded = switch (try decodeTupleElementsAt(allocator, heap, resolver, tuple.elements, types, bytes, offset, depth + 1, mode)) {
                .value => |decoded| decoded,
                .err => |err| return .{ .err = err },
            };
            return .{ .value = .{
                .value = .{ .tuple_ref = try heap.allocTuple(decoded.values) },
                .size = decoded.size,
            } };
        },
        .struct_ => |named| {
            const fields = (try resolver.structFields(resolver.context, named.name)) orelse return error.AbiDecoderInternalShapeMismatch;
            if (fields.len != tuple.elements.len) return error.AbiDecoderInternalShapeMismatch;
            const type_id = (try resolver.typeIdForType(resolver.context, target_type)) orelse return error.AbiDecoderInternalShapeMismatch;
            const types = try allocator.alloc(sema.Type, fields.len);
            for (fields, 0..) |field, index| {
                types[index] = field.ty;
            }
            const decoded = switch (try decodeTupleElementsAt(allocator, heap, resolver, tuple.elements, types, bytes, offset, depth + 1, mode)) {
                .value => |decoded| decoded,
                .err => |err| return .{ .err = err },
            };
            const values = try allocator.alloc(CtAggregate.StructField, fields.len);
            for (decoded.values, 0..) |value, index| {
                values[index] = .{ .field_id = @intCast(index), .value = value };
            }
            return .{ .value = .{
                .value = .{ .struct_ref = try heap.allocStruct(type_id, values) },
                .size = decoded.size,
            } };
        },
        else => return error.AbiDecoderInternalShapeMismatch,
    }
}

fn decodeTupleElementsAt(
    allocator: std.mem.Allocator,
    heap: *CtHeap,
    resolver: TypeResolver,
    layouts: []const abi_layout.LayoutNode,
    types: []const sema.Type,
    bytes: []const u8,
    offset: usize,
    depth: usize,
    mode: DecodeMode,
) anyerror!DecodeStep(DecodedElements) {
    if (layouts.len != types.len) return error.AbiDecoderInternalShapeMismatch;
    const values = try allocator.alloc(CtValue, layouts.len);
    var head_size: usize = 0;
    for (layouts) |layout| {
        const head_bytes = checkedMul(layout.headSlotWordCount(), 32) orelse return .{ .err = .length_overflow };
        head_size = checkedAdd(head_size, head_bytes) orelse return .{ .err = .length_overflow };
    }

    var head_cursor = offset;
    var tail_cursor = head_size;
    for (layouts, types, 0..) |layout, ty, index| {
        if (layout.isDynamic()) {
            const element_offset = switch (readUsizeWord(bytes, head_cursor, .invalid_offset)) {
                .value => |value| value,
                .err => |err| return .{ .err = err },
            };
            if (mode == .strict and element_offset != tail_cursor) return .{ .err = .non_canonical_encoding };
            head_cursor = checkedAdd(head_cursor, 32) orelse return .{ .err = .length_overflow };

            const absolute = checkedAdd(offset, element_offset) orelse return .{ .err = .length_overflow };
            const decoded = try decodeNodeAt(allocator, heap, resolver, layout, ty, bytes, absolute, depth + 1, mode);
            switch (decoded) {
                .value => |value| {
                    values[index] = value.value;
                    tail_cursor = if (mode == .strict)
                        checkedAdd(tail_cursor, value.size) orelse return .{ .err = .length_overflow }
                    else blk: {
                        const element_end = checkedAdd(element_offset, value.size) orelse return .{ .err = .length_overflow };
                        break :blk @max(tail_cursor, element_end);
                    };
                },
                .err => |err| return .{ .err = err },
            }
        } else {
            const decoded = try decodeNodeAt(allocator, heap, resolver, layout, ty, bytes, head_cursor, depth + 1, mode);
            switch (decoded) {
                .value => |value| {
                    values[index] = value.value;
                    head_cursor = checkedAdd(head_cursor, value.size) orelse return .{ .err = .length_overflow };
                },
                .err => |err| return .{ .err = err },
            }
        }
    }

    return .{ .value = .{ .values = values, .size = tail_cursor } };
}

fn readWordAt(bytes: []const u8, offset: usize) ?*const [32]u8 {
    if (offset > bytes.len or bytes.len - offset < 32) return null;
    const slice = bytes[offset .. offset + 32];
    return slice[0..32];
}

fn readUsizeWord(bytes: []const u8, offset: usize, overflow_err: DecodeError) DecodeStep(usize) {
    const word = readWordAt(bytes, offset) orelse return .{ .err = .truncated_buffer };
    const value = std.mem.readInt(u256, word, .big);
    if (value > std.math.maxInt(usize)) return .{ .err = overflow_err };
    return .{ .value = @intCast(value) };
}

fn checkedAdd(a: usize, b: usize) ?usize {
    return std.math.add(usize, a, b) catch null;
}

fn checkedMul(a: usize, b: usize) ?usize {
    return std.math.mul(usize, a, b) catch null;
}

fn paddedByteLen(len: usize) ?usize {
    const padded = std.mem.alignForward(usize, len, 32);
    if (padded < len) return null;
    return padded;
}

fn unwrapRefinement(ty: sema.Type) sema.Type {
    return switch (ty) {
        .refinement => |refinement| unwrapRefinement(refinement.base_type.*),
        else => ty,
    };
}

fn validateDecodedValue(target_type: sema.Type, decoded: DecodeStep(DecodeValue)) anyerror!DecodeStep(DecodeValue) {
    return switch (decoded) {
        .err => |err| .{ .err = err },
        .value => |value| if (try decodedValueSatisfiesType(target_type, value.value))
            .{ .value = value }
        else
            .{ .err = .refinement_violation },
    };
}

fn decodedValueSatisfiesType(ty: sema.Type, value: CtValue) anyerror!bool {
    return switch (ty) {
        .refinement => |refinement| (try refinementSatisfied(refinement, value)) and (try decodedValueSatisfiesType(refinement.base_type.*, value)),
        else => true,
    };
}

fn refinementSatisfied(refinement: sema.RefinementType, value: CtValue) anyerror!bool {
    if (refinements.bounds(refinement)) |info| {
        const integer = switch (value) {
            .integer => |integer| integer,
            else => return error.AbiDecoderInternalShapeMismatch,
        };
        if (refinementBaseIsSignedInteger(info.base_type)) {
            const signed_integer: i256 = @bitCast(integer);
            if (try parseI256Text(info.min_text)) |min| {
                if (signed_integer < min) return false;
            }
            if (try parseI256Text(info.max_text)) |max| {
                if (signed_integer > max) return false;
            }
        } else {
            if (try parseU256Text(info.min_text)) |min| {
                if (integer < min) return false;
            }
            if (try parseU256Text(info.max_text)) |max| {
                if (integer > max) return false;
            }
        }
        return true;
    }
    if (refinements.kindForName(refinement.name) == .non_zero_address) {
        const address = switch (value) {
            .address => |address| address,
            else => return error.AbiDecoderInternalShapeMismatch,
        };
        return address != 0;
    }
    return error.AbiDecoderInternalShapeMismatch;
}

fn refinementBaseIsSignedInteger(ty: sema.Type) bool {
    return switch (ty) {
        .refinement => |refinement| refinementBaseIsSignedInteger(refinement.base_type.*),
        .integer => |integer| integer.signed orelse signedFromIntegerSpelling(integer.spelling) orelse false,
        else => false,
    };
}

fn signedFromIntegerSpelling(spelling: ?[]const u8) ?bool {
    const text = spelling orelse return null;
    if (std.mem.startsWith(u8, text, "u")) return false;
    if (std.mem.startsWith(u8, text, "i")) return true;
    return null;
}

fn parseU256Text(text: ?[]const u8) anyerror!?u256 {
    const raw = text orelse return null;
    if (std.mem.startsWith(u8, raw, "-")) return error.AbiDecoderInternalShapeMismatch;
    return std.fmt.parseInt(u256, raw, 10) catch error.AbiDecoderInternalShapeMismatch;
}

fn parseI256Text(text: ?[]const u8) anyerror!?i256 {
    const raw = text orelse return null;
    return std.fmt.parseInt(i256, raw, 10) catch error.AbiDecoderInternalShapeMismatch;
}
