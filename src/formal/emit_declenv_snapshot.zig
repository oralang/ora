//! Emits `formal/Ora/Generated/DeclEnvSnapshot.lean` — DATA-ONLY declaration rows
//! for a CURATED matrix of nominal declarations (a small hand-picked set, NOT the
//! whole compiler universe). The trusted checks live in `Ora/SyncDeclEnv.lean`.
//!
//! First slice: the nominal-KIND correspondence — for each curated declaration we
//! build the compiler's nominal `semantic.Type` and emit its `TypeKind` tag, so
//! the Lean `Decl` kinds are proven to map onto the right compiler kinds (and a
//! rename like `resource_domain` is caught). Richer rows (fields/variants/member
//! spellings) can layer on later.
//!
//! Wrapped by `scripts/check-formal-declenv-sync.sh`.

const std = @import("std");
const formal = @import("ora_formal");

const Type = formal.Type;

const header =
    \\/-
    \\GENERATED — DATA ONLY.  Do NOT edit by hand and do NOT add any `theorem`,
    \\`lemma`, `example`, `axiom`, `sorry`, `instance`, `macro`, attribute, or extra
    \\`import` to this file. It contains only `def … := <literal>` declaration rows
    \\emitted from the compiler. The TRUSTED checks live in `Ora/SyncDeclEnv.lean`.
    \\
    \\Regenerate with `scripts/check-formal-declenv-sync.sh`. Source:
    \\src/formal/emit_declenv_snapshot.zig, src/types/semantic.zig.
    \\-/
    \\
    \\namespace Ora.Generated
    \\
    \\
;

/// Source spelling of a primitive type (mirrors Lean `PrimTy.spelling`).
fn primSpelling(ty: Type) ?[]const u8 {
    return switch (ty) {
        .integer => |it| it.spelling,
        .bool => "bool",
        .address => "address",
        .string => "string",
        .bytes => "bytes",
        .void => "void",
        .fixed_bytes => |fb| fb.spelling,
        else => null,
    };
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var out_buffer: [1 << 16]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &out_buffer);
    const out = &stdout_writer.interface;

    try out.writeAll(header);

    // The curated matrix, built as the compiler's nominal `Type` values.
    const u256_ty: Type = .{ .integer = .{ .bits = 256, .signed = false, .spelling = "u256" } };
    const bytes32_ty: Type = .{ .fixed_bytes = .{ .len = 32, .spelling = "bytes32" } };
    const slice_ty: Type = .{ .slice = .{ .element_type = &u256_ty } };
    const Row = struct { name: []const u8, ty: Type };
    const rows = [_]Row{
        .{ .name = "Point", .ty = .{ .struct_ = .{ .name = "Point" } } },
        .{ .name = "Color", .ty = .{ .enum_ = .{ .name = "Color" } } },
        .{ .name = "Flags", .ty = .{ .bitfield = .{ .name = "Flags" } } },
        .{ .name = "Vault", .ty = .{ .contract = .{ .name = "Vault" } } },
        // resources with primitive / bytesN / composite carriers — exercises all
        // three carrier-spelling paths (some integer, some bytesN, none composite).
        .{ .name = "Token", .ty = .{ .resource_domain = .{ .name = "Token", .carrier_type = &u256_ty } } },
        .{ .name = "Digest", .ty = .{ .resource_domain = .{ .name = "Digest", .carrier_type = &bytes32_ty } } },
        .{ .name = "Buffer", .ty = .{ .resource_domain = .{ .name = "Buffer", .carrier_type = &slice_ty } } },
    };

    // (declaration name, compiler TypeKind tag of its nominal type).
    try out.writeAll("def compilerDeclKinds : List (String × String) :=\n  [");
    for (rows, 0..) |row, i| {
        if (i != 0) try out.writeAll(",\n   ");
        try out.print("(\"{s}\", \"{s}\")", .{ row.name, @tagName(row.ty) });
    }
    try out.writeAll("]\n\n");

    // Per-member, compiler-DERIVED: each resource's carrier kind + spelling, read
    // from its `resource_domain` carrier_type (NOT curated — straight off the Type).
    // Spelling is a genuine `Option`: `none` for composites (no magic sentinel) —
    // the always-present KIND is the universal discriminator.
    try out.writeAll("def compilerResourceCarriers : List (String × String × Option String) :=\n  [");
    {
        var first = true;
        for (rows) |row| switch (row.ty) {
            .resource_domain => |rd| {
                const carrier = rd.carrier_type.*;
                if (!first) try out.writeAll(",\n   ");
                first = false;
                try out.print("(\"{s}\", \"{s}\", ", .{ row.name, @tagName(carrier) });
                if (primSpelling(carrier)) |s| {
                    try out.print("some \"{s}\")", .{s});
                } else {
                    try out.writeAll("none)");
                }
            },
            else => {},
        };
    }
    try out.writeAll("]\n\nend Ora.Generated\n");

    try out.flush();
}
