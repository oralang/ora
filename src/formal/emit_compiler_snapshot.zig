//! Emits `formal/Ora/Generated/CompilerSnapshot.lean` — the DATA-ONLY facts the
//! Lean spec checks itself against (see `formal/Ora/Sync.lean`).
//!
//! Wrapped by `scripts/check-formal-sync.sh`. The emitter is rooted under
//! `src/formal/` and imports compiler facts through `src/formal.zig`, so formal
//! generators stay organized without using parent-directory module escapes.

const std = @import("std");
const formal = @import("ora_formal");

const builtin = formal.builtin;
const refinements = formal.refinement_registry;
const region_assign = formal.region_assign;
const semantic = formal.semantic;
const obligation = formal.obligation;

const header =
    \\/-
    \\GENERATED — DATA ONLY.  Do NOT edit by hand and do NOT add any `theorem`,
    \\`lemma`, `example`, `axiom`, `sorry`, `instance`, `macro`, attribute, or extra
    \\`import` to this file. It contains only `def … := <literal>` facts emitted from
    \\the compiler. The TRUSTED checks live in `Ora/Sync.lean`.
    \\
    \\Regenerate with `scripts/check-formal-sync.sh`. Source:
    \\src/formal/emit_compiler_snapshot.zig, src/types/builtin.zig,
    \\src/types/semantic.zig, src/types/region_assign.zig, src/refinements/root.zig,
    \\src/formal/obligation.zig.
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

/// Emits the complete integer registration owned by `builtin_types`.
/// Fails closed: every Integer row must name its signedness and width, so the
/// formal snapshot cannot silently lose a malformed compiler registration.
fn emitIntegerTypeRegistrations(out: anytype) !void {
    try out.writeAll("def compilerIntegerTypeRegistrations : List (String × Bool × Nat × Nat) :=\n  [");
    var first = true;
    for (builtin.builtin_types) |spec| {
        if (spec.category != .Integer) continue;
        const is_signed = if (spec.signed) |s| s else return error.IntegerSpecMissingSignedness;
        const bits = if (spec.bit_width) |b| b else return error.IntegerSpecMissingBitWidth;
        if (!first) try out.writeAll(", ");
        first = false;
        try out.print(
            "(\"{s}\", {s}, {d}, {d})",
            .{ spec.source_name, if (is_signed) "true" else "false", bits, spec.comptime_type_id },
        );
    }
    try out.writeAll("]\n\n");
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var out_buffer: [1 << 16]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &out_buffer);
    const out = &stdout_writer.interface;

    try out.writeAll(header);

    // Spellable scalar builtins (BuiltinTypeId), in enum order.
    try emitStringList(out, "compilerBuiltinTypeIds", builtin.BuiltinTypeId);

    try out.writeAll("def compilerBuiltinTypeComptimeIds : List (String × Nat) :=\n  [");
    for (builtin.builtin_types, 0..) |spec, index| {
        if (index != 0) try out.writeAll(", ");
        try out.print("(\"{s}\", {d})", .{ spec.source_name, spec.comptime_type_id });
    }
    try out.writeAll("]\n\n");

    // One authoritative integer registration table: spelling, signedness,
    // width, and compiler type ID. Width projections are derived in Lean.
    try emitIntegerTypeRegistrations(out);

    // Fixed-bytes bounds.
    try out.print("def compilerFixedBytesMin : Nat := {d}\n", .{builtin.fixed_bytes_min_len});
    try out.print("def compilerFixedBytesMax : Nat := {d}\n\n", .{builtin.fixed_bytes_max_len});

    // TypeKind universe + regions, in enum order.
    try emitStringList(out, "compilerTypeKinds", semantic.TypeKind);
    try emitStringList(out, "compilerRegions", semantic.Region);

    // Resource obligation operation/property enums, in compiler enum order.
    try emitStringList(out, "compilerResourceOperations", obligation.ResourceOperation);
    try emitStringList(out, "compilerResourceProperties", obligation.ResourceProperty);

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
