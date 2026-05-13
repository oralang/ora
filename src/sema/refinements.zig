const std = @import("std");
const ast = @import("../ast/mod.zig");
const model = @import("model.zig");

pub const Kind = enum {
    min_value,
    max_value,
    in_range,
    non_zero_address,
    non_zero,
    basis_points,
    exact,
    scaled,
};

pub const Entry = struct {
    kind: Kind,
    name: []const u8,
    has_runtime_guard: bool,
    compile_time_only: bool = false,
    has_native_mlir_type: bool,
    path_form: bool = false,
};

pub const entries = [_]Entry{
    .{ .kind = .min_value, .name = "MinValue", .has_runtime_guard = true, .has_native_mlir_type = true },
    .{ .kind = .max_value, .name = "MaxValue", .has_runtime_guard = true, .has_native_mlir_type = true },
    .{ .kind = .in_range, .name = "InRange", .has_runtime_guard = true, .has_native_mlir_type = true },
    .{ .kind = .non_zero_address, .name = "NonZeroAddress", .has_runtime_guard = true, .has_native_mlir_type = true, .path_form = true },
    .{ .kind = .non_zero, .name = "NonZero", .has_runtime_guard = true, .has_native_mlir_type = false },
    .{ .kind = .basis_points, .name = "BasisPoints", .has_runtime_guard = true, .has_native_mlir_type = false },
    .{ .kind = .exact, .name = "Exact", .has_runtime_guard = false, .compile_time_only = true, .has_native_mlir_type = true },
    .{ .kind = .scaled, .name = "Scaled", .has_runtime_guard = false, .compile_time_only = true, .has_native_mlir_type = true },
};

pub fn entryForName(name: []const u8) ?Entry {
    const trimmed = std.mem.trim(u8, name, " \t\n\r");
    for (entries) |entry| {
        if (std.mem.eql(u8, trimmed, entry.name)) return entry;
    }
    return null;
}

pub fn kindForName(name: []const u8) ?Kind {
    return if (entryForName(name)) |entry| entry.kind else null;
}

pub fn isKnownName(name: []const u8) bool {
    return entryForName(name) != null;
}

pub fn isPathFormName(name: []const u8) bool {
    return if (entryForName(name)) |entry| entry.path_form else false;
}

pub fn hasNativeMlirTypeName(name: []const u8) bool {
    return if (entryForName(name)) |entry| entry.has_native_mlir_type else false;
}

pub fn supportsRuntimeGuard(refinement: model.RefinementType) bool {
    return if (entryForName(refinement.name)) |entry| entry.has_runtime_guard else false;
}

pub fn isCompileTimeOnly(refinement: model.RefinementType) bool {
    return if (entryForName(refinement.name)) |entry| entry.compile_time_only else false;
}

pub const Bounds = struct {
    base_type: model.Type,
    min_text: ?[]const u8 = null,
    max_text: ?[]const u8 = null,
};

pub fn bounds(refinement: model.RefinementType) ?Bounds {
    const kind = kindForName(refinement.name) orelse return null;
    return switch (kind) {
        .min_value => {
            if (refinement.args.len < 2 or refinement.args[1] != .Integer) return null;
            return .{
                .base_type = refinement.base_type.*,
                .min_text = refinement.args[1].Integer.text,
            };
        },
        .max_value => {
            if (refinement.args.len < 2 or refinement.args[1] != .Integer) return null;
            return .{
                .base_type = refinement.base_type.*,
                .max_text = refinement.args[1].Integer.text,
            };
        },
        .in_range => {
            if (refinement.args.len < 3 or refinement.args[1] != .Integer or refinement.args[2] != .Integer) return null;
            return .{
                .base_type = refinement.base_type.*,
                .min_text = refinement.args[1].Integer.text,
                .max_text = refinement.args[2].Integer.text,
            };
        },
        .basis_points => .{
            .base_type = refinement.base_type.*,
            .min_text = "0",
            .max_text = "10000",
        },
        .non_zero => .{
            .base_type = refinement.base_type.*,
            .min_text = "1",
        },
        .exact => null,
        .scaled => null,
        .non_zero_address => null,
    };
}

pub fn refinementType(
    allocator: std.mem.Allocator,
    name: []const u8,
    base_type: model.Type,
    args: []const ast.TypeArg,
) !model.Type {
    return .{ .refinement = .{
        .name = name,
        .base_type = try storeType(allocator, base_type),
        .args = args,
    } };
}

fn storeType(allocator: std.mem.Allocator, ty: model.Type) !*model.Type {
    const ptr = try allocator.create(model.Type);
    ptr.* = ty;
    return ptr;
}

test "refinement registry classifies runtime and compile-time-only refinements" {
    try std.testing.expect(isKnownName("MinValue"));
    try std.testing.expect(isKnownName("BasisPoints"));
    try std.testing.expect(isPathFormName("NonZeroAddress"));
    try std.testing.expect(!isPathFormName("MinValue"));
    try std.testing.expect(hasNativeMlirTypeName("Exact"));
    try std.testing.expect(!hasNativeMlirTypeName("BasisPoints"));
}
