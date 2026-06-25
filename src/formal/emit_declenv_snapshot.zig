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

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var out_buffer: [1 << 16]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &out_buffer);
    const out = &stdout_writer.interface;

    try out.writeAll(header);

    // The curated matrix, built as the compiler's nominal `Type` values.
    const u256_ty: Type = .{ .integer = .{ .bits = 256, .signed = false, .spelling = "u256" } };
    const Row = struct { name: []const u8, ty: Type };
    const rows = [_]Row{
        .{ .name = "Point", .ty = .{ .struct_ = .{ .name = "Point" } } },
        .{ .name = "Color", .ty = .{ .enum_ = .{ .name = "Color" } } },
        .{ .name = "Flags", .ty = .{ .bitfield = .{ .name = "Flags" } } },
        .{ .name = "Vault", .ty = .{ .contract = .{ .name = "Vault" } } },
        .{ .name = "Token", .ty = .{ .resource_domain = .{ .name = "Token", .carrier_type = &u256_ty } } },
    };

    // (declaration name, compiler TypeKind tag of its nominal type).
    try out.writeAll("def compilerDeclKinds : List (String × String) :=\n  [");
    for (rows, 0..) |row, i| {
        if (i != 0) try out.writeAll(",\n   ");
        try out.print("(\"{s}\", \"{s}\")", .{ row.name, @tagName(row.ty) });
    }
    try out.writeAll("]\n\nend Ora.Generated\n");

    try out.flush();
}
