const std = @import("std");

// Neutral refinement registry shared by compiler, LSP, docs-sync checks, and
// standalone type rendering.

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

pub fn entryForKind(kind: Kind) ?Entry {
    for (entries) |entry| {
        if (entry.kind == kind) return entry;
    }
    return null;
}

pub fn kindForName(name: []const u8) ?Kind {
    return if (entryForName(name)) |entry| entry.kind else null;
}

pub fn nameForKind(kind: Kind) []const u8 {
    return (entryForKind(kind) orelse unreachable).name;
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

pub fn isBoundsBackedKind(kind: Kind) bool {
    return switch (kind) {
        .min_value, .max_value, .in_range, .basis_points, .non_zero => true,
        .non_zero_address, .exact, .scaled => false,
    };
}

test "refinement registry classifies runtime and compile-time-only refinements" {
    try std.testing.expect(isKnownName("MinValue"));
    try std.testing.expect(isKnownName("BasisPoints"));
    try std.testing.expect(isPathFormName("NonZeroAddress"));
    try std.testing.expect(!isPathFormName("MinValue"));
    try std.testing.expect(isBoundsBackedKind(.basis_points));
    try std.testing.expect(!isBoundsBackedKind(.scaled));
    try std.testing.expect(hasNativeMlirTypeName("Exact"));
    try std.testing.expect(!hasNativeMlirTypeName("BasisPoints"));
}

test "every refinement kind has a registry entry" {
    inline for (std.meta.fields(Kind)) |field| {
        const kind: Kind = @enumFromInt(field.value);
        try std.testing.expect(entryForKind(kind) != null);
    }
}

test "refinement registry native MLIR flags match TableGen definitions" {
    const tablegen = @embedFile("../mlir/ora/td/OraTypes.td");

    for (entries) |entry| {
        const needle = try std.fmt.allocPrint(std.testing.allocator, "Ora_Type<\"{s}\",", .{entry.name});
        defer std.testing.allocator.free(needle);

        const has_tablegen_type = std.mem.indexOf(u8, tablegen, needle) != null;
        try std.testing.expectEqual(entry.has_native_mlir_type, has_tablegen_type);
    }
}
