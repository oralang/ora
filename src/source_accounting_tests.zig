const accounting = @import("formal/shared/source_accounting.zig");
const gate = @import("formal/kernel/source_accounting_gate.zig");
const from_syntax = @import("formal/source_accounting_from_syntax.zig");
const from_sema = @import("formal/source_accounting_from_sema.zig");
const from_comptime = @import("formal/source_accounting_from_comptime.zig");
const from_mlir = @import("formal/source_accounting_from_mlir.zig");
const from_z3 = @import("formal/source_accounting_from_z3.zig");
const artifact_catalog = @import("formal/shared/artifact_catalog.zig");

comptime {
    _ = accounting;
    _ = gate;
    _ = from_syntax;
    _ = from_sema;
    _ = from_comptime;
    _ = from_mlir;
    _ = from_z3;
    _ = artifact_catalog;
}
