const std = @import("std");
const refinements = @import("ora_refinements");

const Allocator = std.mem.Allocator;

pub const Entry = struct {
    name: []const u8,
    signature: []const u8,
    documentation: []const u8,
};

pub const entries = [_]Entry{
    .{
        .name = "MinValue",
        .signature = "MinValue<T, MIN>",
        .documentation = "Refines `T` to values greater than or equal to `MIN`. Runtime guards are emitted when the verifier cannot prove the bound.",
    },
    .{
        .name = "MaxValue",
        .signature = "MaxValue<T, MAX>",
        .documentation = "Refines `T` to values less than or equal to `MAX`. Runtime guards are emitted when the verifier cannot prove the bound.",
    },
    .{
        .name = "InRange",
        .signature = "InRange<T, MIN, MAX>",
        .documentation = "Refines `T` to values between `MIN` and `MAX`, inclusive. Runtime guards are emitted when the verifier cannot prove the range.",
    },
    .{
        .name = "NonZeroAddress",
        .signature = "NonZeroAddress",
        .documentation = "Refines `address` to exclude the zero address. Trusted environment senders such as `std.msg.sender` and `std.transaction.sender` satisfy this refinement.",
    },
    .{
        .name = "NonZero",
        .signature = "NonZero<T>",
        .documentation = "Refines `T` to values that are not zero. Runtime guards are emitted when the verifier cannot prove non-zero flow.",
    },
    .{
        .name = "BasisPoints",
        .signature = "BasisPoints<T>",
        .documentation = "Refines `T` to basis-point values in the inclusive range `0..10000`.",
    },
    .{
        .name = "Exact",
        .signature = "Exact<T, VALUE>",
        .documentation = "Compile-time-only refinement for values known to equal a specific constant.",
    },
    .{
        .name = "Scaled",
        .signature = "Scaled<T, SCALE>",
        .documentation = "Compile-time-only refinement for numeric values carrying a fixed scale.",
    },
};

comptime {
    for (refinements.entries) |registry_entry| {
        if (entryForNameComptime(registry_entry.name) == null) {
            @compileError("missing LSP refinement docs for " ++ registry_entry.name);
        }
    }
    for (entries) |entry| {
        if (refinements.entryForName(entry.name) == null) {
            @compileError("LSP refinement docs entry is not backed by the refinement registry: " ++ entry.name);
        }
    }
}

pub fn entryForName(name: []const u8) ?Entry {
    for (entries) |entry| {
        if (std.mem.eql(u8, entry.name, name)) return entry;
    }
    return null;
}

pub fn documentation(name: []const u8) ?[]const u8 {
    return if (entryForName(name)) |entry| entry.documentation else null;
}

pub fn markdownAlloc(allocator: Allocator, entry: Entry) ![]u8 {
    return std.fmt.allocPrint(
        allocator,
        "```ora\n{s}\n```\n---\n{s}",
        .{ entry.signature, entry.documentation },
    );
}

fn entryForNameComptime(name: []const u8) ?Entry {
    for (entries) |entry| {
        if (std.mem.eql(u8, entry.name, name)) return entry;
    }
    return null;
}
