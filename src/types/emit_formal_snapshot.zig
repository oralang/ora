//! Emits `formal/Ora/Generated/CompilerSnapshot.lean` — the DATA-ONLY facts the
//! Lean spec checks itself against (see `formal/Ora/Sync.lean`).
//!
//! Run: `zig run src/types/emit_formal_snapshot.zig > formal/Ora/Generated/CompilerSnapshot.lean`
//! (wrapped by `scripts/check-formal-sync.sh`). Co-located with the pure type
//! modules so it imports them same-dir (Zig 0.16 forbids `../` module escapes).

const std = @import("std");
const builtin = @import("builtin.zig");
const semantic = @import("semantic.zig");
const region_assign = @import("region_assign.zig");
const refinements = @import("ora_refinements");

const header =
    \\/-
    \\GENERATED — DATA ONLY.  Do NOT edit by hand and do NOT add any `theorem`,
    \\`lemma`, `example`, `axiom`, `sorry`, `instance`, `macro`, attribute, or extra
    \\`import` to this file. It contains only `def … := <literal>` facts emitted from
    \\the compiler. The TRUSTED checks live in `Ora/Sync.lean`.
    \\
    \\Regenerate with `scripts/check-formal-sync.sh`. Source: src/types/builtin.zig,
    \\src/types/semantic.zig, src/types/region_assign.zig, src/refinements/root.zig.
    \\-/
    \\
    \\namespace Ora.Generated
    \\
    \\
;

fn emitStringList(out: anytype, comptime name: []const u8, comptime T: type) !void {
    try out.print("def {s} : List String :=\n  [", .{name});
    inline for (@typeInfo(T).@"enum".fields, 0..) |f, i| {
        if (i != 0) try out.writeAll(", ");
        try out.print("\"{s}\"", .{f.name});
    }
    try out.writeAll("]\n\n");
}

const RefinementNameFilter = enum {
    all,
    runtime_guard,
    compile_time_only,
    native_mlir,
    path_form,
    bounds_backed,
};

fn refinementNameIncluded(entry: refinements.Entry, filter: RefinementNameFilter) bool {
    return switch (filter) {
        .all => true,
        .runtime_guard => entry.has_runtime_guard,
        .compile_time_only => entry.compile_time_only,
        .native_mlir => entry.has_native_mlir_type,
        .path_form => entry.path_form,
        .bounds_backed => refinements.isBoundsBackedKind(entry.kind),
    };
}

fn emitRefinementNameList(out: anytype, comptime name: []const u8, filter: RefinementNameFilter) !void {
    try out.print("def {s} : List String :=\n  [", .{name});
    var first = true;
    for (refinements.entries) |entry| {
        if (!refinementNameIncluded(entry, filter)) continue;
        if (!first) try out.writeAll(", ");
        first = false;
        try out.print("\"{s}\"", .{entry.name});
    }
    try out.writeAll("]\n\n");
}

const Signedness = enum { unsigned, signed };

/// Writes the comma-separated bit widths of the builtin integer types of the
/// given signedness — i.e. the body of `compilerUIntWidths` / `compilerSIntWidths`.
/// Fails CLOSED: a malformed Integer spec (missing signedness or width) is an
/// error, never a silent skip — per the no-default / no-silent-skip rule.
fn writeIntegerWidths(out: anytype, sign: Signedness) !void {
    var first = true;
    for (builtin.builtin_types) |spec| {
        if (spec.category != .Integer) continue;
        const is_signed = if (spec.signed) |s| s else return error.IntegerSpecMissingSignedness;
        const bits = if (spec.bit_width) |b| b else return error.IntegerSpecMissingBitWidth;
        const matches = switch (sign) {
            .unsigned => !is_signed,
            .signed => is_signed,
        };
        if (!matches) continue;
        if (!first) try out.writeAll(", ");
        first = false;
        try out.print("{d}", .{bits});
    }
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var out_buffer: [1 << 16]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &out_buffer);
    const out = &stdout_writer.interface;

    try out.writeAll(header);

    // Spellable scalar builtins (BuiltinTypeId), in enum order.
    try emitStringList(out, "compilerBuiltinTypeIds", builtin.BuiltinTypeId);

    // Integer widths from the builtin spec table (fail-closed on bad specs).
    try out.writeAll("def compilerUIntWidths : List Nat := [");
    try writeIntegerWidths(out, .unsigned);
    try out.writeAll("]\n");

    try out.writeAll("def compilerSIntWidths : List Nat := [");
    try writeIntegerWidths(out, .signed);
    try out.writeAll("]\n\n");

    // Fixed-bytes bounds.
    try out.print("def compilerFixedBytesMin : Nat := {d}\n", .{builtin.fixed_bytes_min_len});
    try out.print("def compilerFixedBytesMax : Nat := {d}\n\n", .{builtin.fixed_bytes_max_len});

    // TypeKind universe + regions, in enum order.
    try emitStringList(out, "compilerTypeKinds", semantic.TypeKind);
    try emitStringList(out, "compilerRegions", semantic.Region);

    // Closed refinement registry, emitted as self-describing per-property name
    // lists (not a positional tuple): membership in each list fully determines
    // each refinement's classification, and `decide` checks every list.
    try emitRefinementNameList(out, "compilerRefinementNames", .all);
    try emitRefinementNameList(out, "compilerRuntimeGuardRefinementNames", .runtime_guard);
    try emitRefinementNameList(out, "compilerCompileTimeOnlyRefinementNames", .compile_time_only);
    try emitRefinementNameList(out, "compilerNativeMlirRefinementNames", .native_mlir);
    try emitRefinementNameList(out, "compilerPathFormRefinementNames", .path_form);
    try emitRefinementNameList(out, "compilerBoundsBackedRefinementNames", .bounds_backed);

    // Region assignability table, (from, to, regionAssignable from to).
    try out.writeAll("def compilerRegionTable : List (String × String × Bool) :=\n  [");
    {
        var first = true;
        inline for (@typeInfo(semantic.Region).@"enum".fields) |fa| {
            const a = @field(semantic.Region, fa.name);
            inline for (@typeInfo(semantic.Region).@"enum".fields) |fb| {
                const b = @field(semantic.Region, fb.name);
                if (!first) try out.writeAll(",\n   ");
                first = false;
                const v = region_assign.regionAssignable(a, b);
                try out.print("(\"{s}\", \"{s}\", {s})", .{ fa.name, fb.name, if (v) "true" else "false" });
            }
        }
    }
    try out.writeAll("]\n\nend Ora.Generated\n");

    try out.flush();
}
