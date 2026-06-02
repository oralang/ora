const std = @import("std");
const builtin = @import("ora_types").builtin;
const sema = @import("../sema/mod.zig");
const sema_model = @import("../sema/model.zig");
const abi_type_names = @import("type_names.zig");

pub const LayoutError = error{
    UnsupportedAbiType,
    InvalidIntegerWidth,
    InvalidFixedBytesWidth,
};

pub const DynamicBytesKind = enum {
    bytes,
    string,
};

pub const ValuePathSegment = union(enum) {
    tuple_index: u32,
    struct_field: []const u8,
    each_element,
};

pub const ValuePath = struct {
    segments: []const ValuePathSegment = &.{},

    pub fn deinit(self: ValuePath, allocator: std.mem.Allocator) void {
        allocator.free(self.segments);
    }
};

pub const StaticEncoding = union(enum) {
    uint: u16,
    int: u16,
    bool,
    address,
    fixed_bytes: u8,
};

pub const StaticWordLayout = struct {
    path: ValuePath,
    encoding: StaticEncoding,
};

pub const DynamicBytesLayout = struct {
    path: ValuePath,
    kind: DynamicBytesKind,
};

pub const DynamicArrayLayout = struct {
    path: ValuePath,
    element: *LayoutNode,
};

pub const FixedArrayLayout = struct {
    path: ValuePath,
    element: *LayoutNode,
    len: u32,
};

pub const TupleLayout = struct {
    path: ValuePath,
    elements: []const LayoutNode,
};

pub const LayoutNode = union(enum) {
    static_word: StaticWordLayout,
    dynamic_bytes: DynamicBytesLayout,
    dynamic_array: DynamicArrayLayout,
    fixed_array: FixedArrayLayout,
    tuple: TupleLayout,

    pub fn deinit(self: LayoutNode, allocator: std.mem.Allocator) void {
        switch (self) {
            .static_word => |word| word.path.deinit(allocator),
            .dynamic_bytes => |bytes| bytes.path.deinit(allocator),
            .dynamic_array => |array| {
                array.path.deinit(allocator);
                array.element.deinit(allocator);
                allocator.destroy(array.element);
            },
            .fixed_array => |array| {
                array.path.deinit(allocator);
                array.element.deinit(allocator);
                allocator.destroy(array.element);
            },
            .tuple => |tuple| {
                tuple.path.deinit(allocator);
                for (tuple.elements) |element| element.deinit(allocator);
                allocator.free(tuple.elements);
            },
        }
    }

    pub fn isDynamic(self: LayoutNode) bool {
        return switch (self) {
            .static_word => false,
            .dynamic_bytes, .dynamic_array => true,
            .fixed_array => |array| array.element.isDynamic(),
            .tuple => |tuple| blk: {
                for (tuple.elements) |element| {
                    if (element.isDynamic()) break :blk true;
                }
                break :blk false;
            },
        };
    }

    pub fn headSlotWordCount(self: LayoutNode) usize {
        return if (self.isDynamic()) 1 else self.staticWordCount().?;
    }

    pub fn staticWordCount(self: LayoutNode) ?usize {
        return switch (self) {
            .static_word => 1,
            .dynamic_bytes, .dynamic_array => null,
            .fixed_array => |array| (array.element.staticWordCount() orelse return null) * array.len,
            .tuple => |tuple| blk: {
                var total: usize = 0;
                for (tuple.elements) |element| total += element.staticWordCount() orelse return null;
                break :blk total;
            },
        };
    }
};

pub fn fromType(allocator: std.mem.Allocator, ty: sema.Type) anyerror!LayoutNode {
    return fromTypeAtPath(allocator, ty, &.{});
}

fn fromTypeAtPath(allocator: std.mem.Allocator, ty: sema.Type, path: []const ValuePathSegment) anyerror!LayoutNode {
    return switch (ty) {
        .void => .{ .tuple = .{ .path = try clonePath(allocator, path), .elements = try allocator.alloc(LayoutNode, 0) } },
        .bool => staticWordNode(allocator, path, .bool),
        .address => staticWordNode(allocator, path, .address),
        .string => .{ .dynamic_bytes = .{ .path = try clonePath(allocator, path), .kind = .string } },
        .bytes => .{ .dynamic_bytes = .{ .path = try clonePath(allocator, path), .kind = .bytes } },
        .fixed_bytes => |fixed_bytes| staticWordNode(allocator, path, .{ .fixed_bytes = try fixedBytesLen(fixed_bytes) }),
        .integer => |integer| staticWordNode(allocator, path, try integerEncoding(integer)),
        // Legacy context-free behavior. Context-aware lowering resolves named bitfields to their base type.
        .bitfield => staticWordNode(allocator, path, .{ .uint = 256 }),
        .enum_ => error.UnsupportedAbiType,
        .named => |named| blk: {
            if (parseFixedBytesSpelling(named.name)) |len| {
                break :blk staticWordNode(allocator, path, .{ .fixed_bytes = len });
            }
            break :blk error.UnsupportedAbiType;
        },
        .refinement => |refinement| fromTypeAtPath(allocator, refinement.base_type.*, path),
        // Result/error-union ABI carrier shape depends on item-index context
        // to distinguish payloadless named errors from payload-bearing errors.
        // Callers that need error-union layout must use abi/layout_context.zig.
        .error_union => error.UnsupportedAbiType,
        .tuple => |elements| blk: {
            var nodes: std.ArrayList(LayoutNode) = .{};
            errdefer {
                for (nodes.items) |node| node.deinit(allocator);
                nodes.deinit(allocator);
            }
            for (elements, 0..) |element, index| {
                const child_path = try childPath(allocator, path, .{ .tuple_index = @intCast(index) });
                defer allocator.free(child_path);
                try nodes.append(allocator, try fromTypeAtPath(allocator, element, child_path));
            }
            break :blk .{ .tuple = .{ .path = try clonePath(allocator, path), .elements = try nodes.toOwnedSlice(allocator) } };
        },
        .anonymous_struct => |struct_type| blk: {
            var nodes: std.ArrayList(LayoutNode) = .{};
            errdefer {
                for (nodes.items) |node| node.deinit(allocator);
                nodes.deinit(allocator);
            }
            for (struct_type.fields) |field| {
                const child_path = try childPath(allocator, path, .{ .struct_field = field.name });
                defer allocator.free(child_path);
                try nodes.append(allocator, try fromTypeAtPath(allocator, field.ty, child_path));
            }
            break :blk .{ .tuple = .{ .path = try clonePath(allocator, path), .elements = try nodes.toOwnedSlice(allocator) } };
        },
        .array => |array| blk: {
            const element = try allocator.create(LayoutNode);
            errdefer allocator.destroy(element);
            const element_path = try childPath(allocator, path, .each_element);
            defer allocator.free(element_path);
            element.* = try fromTypeAtPath(allocator, array.element_type.*, element_path);
            if (array.len) |len| {
                break :blk .{ .fixed_array = .{ .path = try clonePath(allocator, path), .element = element, .len = len } };
            }
            break :blk .{ .dynamic_array = .{ .path = try clonePath(allocator, path), .element = element } };
        },
        .slice => |slice| blk: {
            const element = try allocator.create(LayoutNode);
            errdefer allocator.destroy(element);
            const element_path = try childPath(allocator, path, .each_element);
            defer allocator.free(element_path);
            element.* = try fromTypeAtPath(allocator, slice.element_type.*, element_path);
            break :blk .{ .dynamic_array = .{ .path = try clonePath(allocator, path), .element = element } };
        },
        else => error.UnsupportedAbiType,
    };
}

pub fn canonicalAbiType(allocator: std.mem.Allocator, node: LayoutNode) ![]const u8 {
    return switch (node) {
        .static_word => |word| allocator.dupe(u8, try staticEncodingAbiName(word.encoding)),
        .dynamic_bytes => |bytes| allocator.dupe(u8, dynamicBytesAbiName(bytes.kind)),
        .dynamic_array => |array| blk: {
            const element_text = try canonicalAbiType(allocator, array.element.*);
            defer allocator.free(element_text);
            break :blk std.fmt.allocPrint(allocator, "{s}[]", .{element_text});
        },
        .fixed_array => |array| blk: {
            const element_text = try canonicalAbiType(allocator, array.element.*);
            defer allocator.free(element_text);
            break :blk std.fmt.allocPrint(allocator, "{s}[{d}]", .{ element_text, array.len });
        },
        .tuple => |tuple| blk: {
            var parts: std.ArrayList([]const u8) = .{};
            defer {
                for (parts.items) |part| allocator.free(part);
                parts.deinit(allocator);
            }
            for (tuple.elements) |element| try parts.append(allocator, try canonicalAbiType(allocator, element));
            const joined = try std.mem.join(allocator, ",", parts.items);
            defer allocator.free(joined);
            break :blk std.fmt.allocPrint(allocator, "({s})", .{joined});
        },
    };
}

pub fn canonicalAbiTypeFromType(allocator: std.mem.Allocator, ty: sema.Type) ![]const u8 {
    const layout = try fromType(allocator, ty);
    defer layout.deinit(allocator);
    return canonicalAbiType(allocator, layout);
}

pub fn staticWordCountFromType(allocator: std.mem.Allocator, ty: sema.Type) !?usize {
    const layout = try fromType(allocator, ty);
    defer layout.deinit(allocator);
    return layout.staticWordCount();
}

pub fn staticWordCountForType(ty: sema.Type) ?usize {
    return switch (ty) {
        .bool, .address, .enum_, .bitfield => 1,
        .fixed_bytes => |fixed_bytes| if (fixedBytesLen(fixed_bytes)) |_| 1 else |_| null,
        .integer => |integer| if (integerEncoding(integer)) |_| 1 else |_| null,
        .named => |named| if (parseFixedBytesSpelling(named.name) != null) 1 else null,
        .refinement => |refinement| staticWordCountForType(refinement.base_type.*),
        // Keep the legacy API boundary: `hir.abi.staticAbiWordCount` did not
        // classify Result carriers. Layout trees still count supported carriers
        // through `LayoutNode.staticWordCount`.
        .error_union => null,
        .tuple => |elements| blk: {
            var total: usize = 0;
            for (elements) |element| total += staticWordCountForType(element) orelse return null;
            break :blk total;
        },
        .anonymous_struct => |struct_type| blk: {
            var total: usize = 0;
            for (struct_type.fields) |field| total += staticWordCountForType(field.ty) orelse return null;
            break :blk total;
        },
        .array => |array| blk: {
            const len = array.len orelse return null;
            const element_words = staticWordCountForType(array.element_type.*) orelse return null;
            break :blk element_words * len;
        },
        .slice => null,
        else => null,
    };
}

pub fn serializeForMlirAttr(allocator: std.mem.Allocator, node: LayoutNode) ![]const u8 {
    return switch (node) {
        .static_word => |word| std.fmt.allocPrint(allocator, "static({s})", .{try staticEncodingAbiName(word.encoding)}),
        .dynamic_bytes => |bytes| std.fmt.allocPrint(allocator, "dynamic({s})", .{dynamicBytesAbiName(bytes.kind)}),
        .dynamic_array => |array| blk: {
            const element = try serializeForMlirAttr(allocator, array.element.*);
            defer allocator.free(element);
            break :blk std.fmt.allocPrint(allocator, "array(dynamic,{s})", .{element});
        },
        .fixed_array => |array| blk: {
            const element = try serializeForMlirAttr(allocator, array.element.*);
            defer allocator.free(element);
            break :blk std.fmt.allocPrint(allocator, "array({d},{s})", .{ array.len, element });
        },
        .tuple => |tuple| blk: {
            var parts: std.ArrayList([]const u8) = .{};
            defer {
                for (parts.items) |part| allocator.free(part);
                parts.deinit(allocator);
            }
            for (tuple.elements) |element| try parts.append(allocator, try serializeForMlirAttr(allocator, element));
            const joined = try std.mem.join(allocator, ",", parts.items);
            defer allocator.free(joined);
            break :blk std.fmt.allocPrint(allocator, "tuple({s})", .{joined});
        },
    };
}

pub fn parseFixedBytesSpelling(name: []const u8) ?u8 {
    return builtin.parseFixedBytesName(name);
}

fn fixedBytesLen(fixed_bytes: sema.FixedBytesType) !u8 {
    if (fixed_bytes.len < builtin.fixed_bytes_min_len or fixed_bytes.len > builtin.fixed_bytes_max_len) {
        return error.InvalidFixedBytesWidth;
    }
    return fixed_bytes.len;
}

fn staticWordNode(allocator: std.mem.Allocator, path: []const ValuePathSegment, encoding: StaticEncoding) !LayoutNode {
    return .{ .static_word = .{ .path = try clonePath(allocator, path), .encoding = encoding } };
}

fn staticEncodingAbiName(encoding: StaticEncoding) LayoutError![]const u8 {
    return switch (encoding) {
        .uint => |bits| abi_type_names.integerAbiName(false, bits) orelse error.InvalidIntegerWidth,
        .int => |bits| abi_type_names.integerAbiName(true, bits) orelse error.InvalidIntegerWidth,
        .bool => abi_type_names.builtinAbiName(.bool),
        .address => abi_type_names.builtinAbiName(.address),
        .fixed_bytes => |len| abi_type_names.fixedBytesAbiName(len) orelse error.InvalidFixedBytesWidth,
    };
}

fn dynamicBytesAbiName(kind: DynamicBytesKind) []const u8 {
    return switch (kind) {
        .bytes => abi_type_names.builtinAbiName(.bytes),
        .string => abi_type_names.builtinAbiName(.string),
    };
}

fn clonePath(allocator: std.mem.Allocator, path: []const ValuePathSegment) !ValuePath {
    return .{ .segments = try allocator.dupe(ValuePathSegment, path) };
}

fn childPath(allocator: std.mem.Allocator, parent: []const ValuePathSegment, segment: ValuePathSegment) ![]const ValuePathSegment {
    var path = try allocator.alloc(ValuePathSegment, parent.len + 1);
    @memcpy(path[0..parent.len], parent);
    path[parent.len] = segment;
    return path;
}

fn integerEncoding(integer: sema_model.IntegerType) !StaticEncoding {
    const signed = integer.signed orelse return error.InvalidIntegerWidth;
    const bits = integer.bits orelse return error.InvalidIntegerWidth;
    const spec = builtin.lookupIntegerBuiltin(signed, bits) orelse return error.InvalidIntegerWidth;
    const width = spec.bit_width orelse return error.InvalidIntegerWidth;
    return if (spec.signed orelse return error.InvalidIntegerWidth)
        .{ .int = width }
    else
        .{ .uint = width };
}
