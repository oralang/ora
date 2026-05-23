const std = @import("std");
const abi_layout = @import("layout.zig");
const comptime_mod = @import("../comptime/mod.zig");
const sema_model = @import("../sema/model.zig");

const CtAggregate = comptime_mod.CtAggregate;
const CtHeap = comptime_mod.CtHeap;
const CtValue = comptime_mod.CtValue;
const BigInt = std.math.big.int.Managed;

const EncodeError = error{
    AbiEncoderInternalShapeMismatch,
};

pub const ComptimeAbiValue = union(enum) {
    ct: CtValue,
    constant: sema_model.ConstValue,
};

pub fn encodeStaticComptimeValue(
    allocator: std.mem.Allocator,
    heap: *const CtHeap,
    layout: abi_layout.LayoutNode,
    value: ComptimeAbiValue,
) ![]const u8 {
    if (layout.isDynamic()) return error.UnsupportedAbiType;
    return encodeComptimeValue(allocator, heap, layout, value);
}

pub fn encodeComptimeValue(
    allocator: std.mem.Allocator,
    heap: *const CtHeap,
    layout: abi_layout.LayoutNode,
    value: ComptimeAbiValue,
) ![]const u8 {
    var out: std.ArrayList(u8) = .{};
    defer out.deinit(allocator);
    if (!layout.isDynamic()) {
        try appendStaticEncoding(allocator, heap, &out, layout, value);
    } else switch (layout) {
        .dynamic_bytes, .dynamic_array, .fixed_array => {
            const elements = [_]abi_layout.LayoutNode{layout};
            try appendTupleEncoding(allocator, heap, &out, &elements, value);
        },
        .tuple => |tuple| {
            if (isStructValue(value)) {
                try appendSingleElementTupleEncoding(allocator, heap, &out, layout, value);
            } else {
                try appendTupleEncoding(allocator, heap, &out, tuple.elements, value);
            }
        },
        .static_word => unreachable,
    }
    return try out.toOwnedSlice(allocator);
}

fn appendSingleElementTupleEncoding(
    allocator: std.mem.Allocator,
    heap: *const CtHeap,
    out: *std.ArrayList(u8),
    element_layout: abi_layout.LayoutNode,
    value: ComptimeAbiValue,
) anyerror!void {
    var head: std.ArrayList(u8) = .{};
    defer head.deinit(allocator);
    var tail: std.ArrayList(u8) = .{};
    defer tail.deinit(allocator);

    const head_size = element_layout.headSlotWordCount() * 32;
    try appendTupleElementEncoding(allocator, heap, &head, &tail, head_size, element_layout, value);
    try out.appendSlice(allocator, head.items);
    try out.appendSlice(allocator, tail.items);
}

fn isStructValue(value: ComptimeAbiValue) bool {
    return switch (value) {
        .ct => |ct_value| switch (ct_value) {
            .struct_ref => true,
            else => false,
        },
        .constant => false,
    };
}

fn appendEncoding(
    allocator: std.mem.Allocator,
    heap: *const CtHeap,
    out: *std.ArrayList(u8),
    layout: abi_layout.LayoutNode,
    value: ComptimeAbiValue,
) anyerror!void {
    if (!layout.isDynamic()) {
        try appendStaticEncoding(allocator, heap, out, layout, value);
        return;
    }

    switch (layout) {
        .dynamic_bytes => |bytes| try appendDynamicBytes(allocator, heap, out, bytes.kind, value),
        .dynamic_array => |array| try appendDynamicArray(allocator, heap, out, array, value),
        .tuple => |tuple| try appendTupleEncoding(allocator, heap, out, tuple.elements, value),
        .fixed_array => |array| try appendFixedArray(allocator, heap, out, array, value),
        .static_word => unreachable,
    }
}

fn appendStaticEncoding(
    allocator: std.mem.Allocator,
    heap: *const CtHeap,
    out: *std.ArrayList(u8),
    layout: abi_layout.LayoutNode,
    value: ComptimeAbiValue,
) anyerror!void {
    switch (layout) {
        .static_word => |word| try appendStaticWord(allocator, heap, out, word.encoding, value),
        .fixed_array => |array| {
            const elems = try arrayElements(heap, value);
            if (elems.len != array.len) return error.AbiEncoderInternalShapeMismatch;
            for (elems) |elem| try appendStaticEncoding(allocator, heap, out, array.element.*, .{ .ct = elem });
        },
        .tuple => |tuple| {
            if (tuple.elements.len == 0) return;
            switch (value) {
                .ct => |ct_value| switch (ct_value) {
                    .tuple_ref => |heap_id| {
                        const tuple_data = heap.getTuple(heap_id);
                        if (tuple_data.elems.len != tuple.elements.len) return error.AbiEncoderInternalShapeMismatch;
                        for (tuple.elements, tuple_data.elems) |element_layout, elem_value| {
                            try appendStaticEncoding(allocator, heap, out, element_layout, .{ .ct = elem_value });
                        }
                    },
                    .struct_ref => |heap_id| {
                        const struct_data = heap.getStruct(heap_id);
                        for (tuple.elements, 0..) |element_layout, index| {
                            const field_value = structFieldValue(struct_data, index) orelse return error.AbiEncoderInternalShapeMismatch;
                            try appendStaticEncoding(allocator, heap, out, element_layout, .{ .ct = field_value });
                        }
                    },
                    .void_val => if (tuple.elements.len != 0) return error.AbiEncoderInternalShapeMismatch,
                    else => return error.AbiEncoderInternalShapeMismatch,
                },
                .constant => |const_value| switch (const_value) {
                    .tuple => |elems| {
                        if (elems.len != tuple.elements.len) return error.AbiEncoderInternalShapeMismatch;
                        for (tuple.elements, elems) |element_layout, elem_value| {
                            try appendStaticEncoding(allocator, heap, out, element_layout, .{ .constant = elem_value });
                        }
                    },
                    else => return error.AbiEncoderInternalShapeMismatch,
                },
            }
        },
        .dynamic_bytes, .dynamic_array => return error.UnsupportedAbiType,
    }
}

fn appendTupleEncoding(
    allocator: std.mem.Allocator,
    heap: *const CtHeap,
    out: *std.ArrayList(u8),
    elements: []const abi_layout.LayoutNode,
    value: ComptimeAbiValue,
) anyerror!void {
    if (elements.len == 0) return;

    var head: std.ArrayList(u8) = .{};
    defer head.deinit(allocator);
    var tail: std.ArrayList(u8) = .{};
    defer tail.deinit(allocator);

    var head_size: usize = 0;
    for (elements) |element| head_size += element.headSlotWordCount() * 32;

    switch (value) {
        .ct => |ct_value| switch (ct_value) {
            .tuple_ref => |heap_id| {
                const tuple_data = heap.getTuple(heap_id);
                if (tuple_data.elems.len != elements.len) return error.AbiEncoderInternalShapeMismatch;
                for (elements, tuple_data.elems) |element_layout, elem_value| {
                    try appendTupleElementEncoding(allocator, heap, &head, &tail, head_size, element_layout, .{ .ct = elem_value });
                }
            },
            .struct_ref => |heap_id| {
                const struct_data = heap.getStruct(heap_id);
                for (elements, 0..) |element_layout, index| {
                    const field_value = structFieldValue(struct_data, index) orelse return error.AbiEncoderInternalShapeMismatch;
                    try appendTupleElementEncoding(allocator, heap, &head, &tail, head_size, element_layout, .{ .ct = field_value });
                }
            },
            .void_val => if (elements.len != 0) return error.AbiEncoderInternalShapeMismatch,
            else => {
                if (elements.len != 1) return error.AbiEncoderInternalShapeMismatch;
                try appendTupleElementEncoding(allocator, heap, &head, &tail, head_size, elements[0], .{ .ct = ct_value });
            },
        },
        .constant => |const_value| switch (const_value) {
            .tuple => |elems| {
                if (elems.len != elements.len) return error.AbiEncoderInternalShapeMismatch;
                for (elements, elems) |element_layout, elem_value| {
                    try appendTupleElementEncoding(allocator, heap, &head, &tail, head_size, element_layout, .{ .constant = elem_value });
                }
            },
            else => {
                if (elements.len != 1) return error.AbiEncoderInternalShapeMismatch;
                try appendTupleElementEncoding(allocator, heap, &head, &tail, head_size, elements[0], .{ .constant = const_value });
            },
        },
    }

    try out.appendSlice(allocator, head.items);
    try out.appendSlice(allocator, tail.items);
}

fn appendTupleElementEncoding(
    allocator: std.mem.Allocator,
    heap: *const CtHeap,
    head: *std.ArrayList(u8),
    tail: *std.ArrayList(u8),
    head_size: usize,
    element_layout: abi_layout.LayoutNode,
    value: ComptimeAbiValue,
) anyerror!void {
    if (element_layout.isDynamic()) {
        try appendAbiWord(allocator, head, @as(u256, @intCast(head_size + tail.items.len)));
        try appendEncoding(allocator, heap, tail, element_layout, value);
    } else {
        try appendStaticEncoding(allocator, heap, head, element_layout, value);
    }
}

fn appendDynamicBytes(
    allocator: std.mem.Allocator,
    heap: *const CtHeap,
    out: *std.ArrayList(u8),
    kind: abi_layout.DynamicBytesKind,
    value: ComptimeAbiValue,
) anyerror!void {
    const bytes = try dynamicBytesValue(heap, kind, value);
    try appendAbiWord(allocator, out, @as(u256, @intCast(bytes.len)));
    try out.appendSlice(allocator, bytes);

    const padded_len = std.mem.alignForward(usize, bytes.len, 32);
    var index = bytes.len;
    while (index < padded_len) : (index += 1) {
        try out.append(allocator, 0);
    }
}

fn appendDynamicArray(
    allocator: std.mem.Allocator,
    heap: *const CtHeap,
    out: *std.ArrayList(u8),
    array: abi_layout.DynamicArrayLayout,
    value: ComptimeAbiValue,
) anyerror!void {
    const elems = try arrayElements(heap, value);
    try appendAbiWord(allocator, out, @as(u256, @intCast(elems.len)));
    try appendArrayElementsEncoding(allocator, heap, out, array.element.*, elems);
}

fn appendFixedArray(
    allocator: std.mem.Allocator,
    heap: *const CtHeap,
    out: *std.ArrayList(u8),
    array: abi_layout.FixedArrayLayout,
    value: ComptimeAbiValue,
) anyerror!void {
    const elems = try arrayElements(heap, value);
    if (elems.len != array.len) return error.AbiEncoderInternalShapeMismatch;
    try appendArrayElementsEncoding(allocator, heap, out, array.element.*, elems);
}

fn appendArrayElementsEncoding(
    allocator: std.mem.Allocator,
    heap: *const CtHeap,
    out: *std.ArrayList(u8),
    element_layout: abi_layout.LayoutNode,
    elems: []const CtValue,
) anyerror!void {
    if (!element_layout.isDynamic()) {
        for (elems) |elem| {
            try appendStaticEncoding(allocator, heap, out, element_layout, .{ .ct = elem });
        }
        return;
    }

    var head: std.ArrayList(u8) = .{};
    defer head.deinit(allocator);
    var tail: std.ArrayList(u8) = .{};
    defer tail.deinit(allocator);

    const head_size = elems.len * element_layout.headSlotWordCount() * 32;
    for (elems) |elem| {
        try appendTupleElementEncoding(allocator, heap, &head, &tail, head_size, element_layout, .{ .ct = elem });
    }

    try out.appendSlice(allocator, head.items);
    try out.appendSlice(allocator, tail.items);
}

fn appendStaticWord(
    allocator: std.mem.Allocator,
    heap: *const CtHeap,
    out: *std.ArrayList(u8),
    encoding: abi_layout.StaticEncoding,
    value: ComptimeAbiValue,
) !void {
    var word: [32]u8 = [_]u8{0} ** 32;
    switch (encoding) {
        .uint => |bits| {
            const integer = try unsignedIntegerValue(value);
            std.mem.writeInt(u256, word[0..32], truncateUnsignedInteger(integer, bits), .big);
        },
        .int => |bits| {
            if (constIntegerValue(value)) |integer| {
                try writeSignedBigIntAbiWord(allocator, &word, integer);
            } else {
                const integer = try unsignedIntegerValue(value);
                std.mem.writeInt(u256, word[0..32], signExtendIntegerBits(integer, bits), .big);
            }
        },
        .bool => word[31] = if (try boolValue(value)) 1 else 0,
        .address => std.mem.writeInt(u160, word[12..32], try addressValue(value), .big),
        .fixed_bytes => |len| {
            const bytes = try bytesValue(heap, value);
            if (bytes.len != len) return error.AbiEncoderInternalShapeMismatch;
            @memcpy(word[0..bytes.len], bytes);
        },
    }
    try out.appendSlice(allocator, &word);
}

fn appendAbiWord(allocator: std.mem.Allocator, out: *std.ArrayList(u8), value: u256) !void {
    var word: [32]u8 = [_]u8{0} ** 32;
    std.mem.writeInt(u256, word[0..32], value, .big);
    try out.appendSlice(allocator, &word);
}

fn arrayElements(heap: *const CtHeap, value: ComptimeAbiValue) EncodeError![]const CtValue {
    return switch (value) {
        .ct => |ct_value| switch (ct_value) {
            .array_ref => |heap_id| heap.getArray(heap_id).elems,
            .slice_ref => |heap_id| heap.getSlice(heap_id).elems,
            .tuple_ref => |heap_id| heap.getTuple(heap_id).elems,
            else => error.AbiEncoderInternalShapeMismatch,
        },
        .constant => error.AbiEncoderInternalShapeMismatch,
    };
}

fn structFieldValue(struct_data: CtAggregate.StructData, field_index: usize) ?CtValue {
    const field_id: comptime_mod.FieldId = @intCast(field_index);
    for (struct_data.fields) |field| {
        if (field.field_id == field_id) return field.value;
    }
    return null;
}

fn unsignedIntegerValue(value: ComptimeAbiValue) EncodeError!u256 {
    return switch (value) {
        .ct => |ct_value| switch (ct_value) {
            .integer => |integer| integer,
            .adt_val => |enum_value| enum_value.variant_id,
            else => error.AbiEncoderInternalShapeMismatch,
        },
        .constant => |const_value| switch (const_value) {
            .integer => |integer| integer.toInt(u256) catch return error.AbiEncoderInternalShapeMismatch,
            else => error.AbiEncoderInternalShapeMismatch,
        },
    };
}

fn constIntegerValue(value: ComptimeAbiValue) ?BigInt {
    return switch (value) {
        .constant => |const_value| switch (const_value) {
            .integer => |integer| integer,
            else => null,
        },
        .ct => null,
    };
}

fn boolValue(value: ComptimeAbiValue) EncodeError!bool {
    return switch (value) {
        .ct => |ct_value| switch (ct_value) {
            .boolean => |boolean| boolean,
            else => error.AbiEncoderInternalShapeMismatch,
        },
        .constant => |const_value| switch (const_value) {
            .boolean => |boolean| boolean,
            else => error.AbiEncoderInternalShapeMismatch,
        },
    };
}

fn addressValue(value: ComptimeAbiValue) EncodeError!u160 {
    return switch (value) {
        .ct => |ct_value| switch (ct_value) {
            .address => |address| address,
            else => error.AbiEncoderInternalShapeMismatch,
        },
        .constant => |const_value| switch (const_value) {
            .address => |address| address,
            else => error.AbiEncoderInternalShapeMismatch,
        },
    };
}

fn bytesValue(heap: *const CtHeap, value: ComptimeAbiValue) EncodeError![]const u8 {
    return switch (value) {
        .ct => |ct_value| switch (ct_value) {
            .bytes_ref => |heap_id| heap.getBytes(heap_id),
            else => error.AbiEncoderInternalShapeMismatch,
        },
        .constant => |const_value| switch (const_value) {
            .fixed_bytes => |bytes| bytes,
            else => error.AbiEncoderInternalShapeMismatch,
        },
    };
}

fn dynamicBytesValue(heap: *const CtHeap, kind: abi_layout.DynamicBytesKind, value: ComptimeAbiValue) EncodeError![]const u8 {
    return switch (kind) {
        .string => switch (value) {
            .ct => |ct_value| switch (ct_value) {
                .string_ref => |heap_id| heap.getString(heap_id),
                else => error.AbiEncoderInternalShapeMismatch,
            },
            .constant => |const_value| switch (const_value) {
                .string => |string| string,
                else => error.AbiEncoderInternalShapeMismatch,
            },
        },
        .bytes => switch (value) {
            .ct => |ct_value| switch (ct_value) {
                .bytes_ref => |heap_id| heap.getBytes(heap_id),
                else => error.AbiEncoderInternalShapeMismatch,
            },
            .constant => |const_value| switch (const_value) {
                .fixed_bytes => |bytes| bytes,
                else => error.AbiEncoderInternalShapeMismatch,
            },
        },
    };
}

fn truncateUnsignedInteger(integer: u256, bits: u16) u256 {
    if (bits >= 256) return integer;
    if (bits == 0) return 0;
    const shift: u8 = @intCast(bits);
    const mask = (@as(u256, 1) << shift) - 1;
    return integer & mask;
}

fn signExtendIntegerBits(integer: u256, bits: u16) u256 {
    if (bits >= 256) return integer;
    if (bits == 0) return 0;
    const width_shift: u8 = @intCast(bits);
    const sign_shift: u8 = @intCast(bits - 1);
    const mask = (@as(u256, 1) << width_shift) - 1;
    const truncated = integer & mask;
    const sign_bit = @as(u256, 1) << sign_shift;
    if ((truncated & sign_bit) == 0) return truncated;
    return truncated | ~mask;
}

fn writeSignedBigIntAbiWord(allocator: std.mem.Allocator, word: *[32]u8, integer: BigInt) !void {
    if (integer.isPositive() or integer.eqlZero()) {
        const as_u256 = integer.toInt(u256) catch return error.AbiEncoderInternalShapeMismatch;
        std.mem.writeInt(u256, word[0..32], as_u256, .big);
        return;
    }

    var modulus = try BigInt.initSet(allocator, 1);
    try BigInt.shiftLeft(&modulus, &modulus, 256);
    var encoded = try BigInt.init(allocator);
    try BigInt.add(&encoded, &modulus, &integer);
    const as_u256 = encoded.toInt(u256) catch return error.AbiEncoderInternalShapeMismatch;
    std.mem.writeInt(u256, word[0..32], as_u256, .big);
}
