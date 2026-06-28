//! Emits `formal/Ora/Generated/DispatcherStrategySnapshot.lean` — data-only
//! dispatcher strategy facts from Sinora's release planner.

const std = @import("std");
const sinora = @import("sinora");

const switch_routing = sinora.switch_routing;

const header =
    \\/-
    \\GENERATED — DATA ONLY. Do NOT edit by hand and do NOT add any `theorem`,
    \\`lemma`, `example`, `axiom`, `sorry`, `instance`, `macro`, attribute, or extra
    \\`import` to this file. It contains only `def … := <literal>` dispatcher
    \\strategy rows emitted from Sinora. The TRUSTED checks live in
    \\`Ora/DispatcherStrategySync.lean`.
    \\
    \\Regenerate with `scripts/check-formal-sync.sh`. Source:
    \\src/formal/emit_dispatcher_strategy_snapshot.zig,
    \\sinora/src/switch_routing.zig.
    \\-/
    \\
    \\namespace Ora.Generated
    \\
    \\
;

fn emitStrategyRows(out: anytype) !void {
    try out.writeAll("def compilerDispatcherStrategyRows : List (String × Bool × Bool) :=\n  [");
    for (switch_routing.dispatcher_strategy_facts, 0..) |fact, i| {
        if (i != 0) try out.writeAll(",\n   ");
        try out.print("(\"{s}\", {s}, {s})", .{
            fact.name,
            if (fact.requires_exact_selector_validation) "true" else "false",
            if (fact.uses_compressed_index) "true" else "false",
        });
    }
    try out.writeAll("]\n\n");
}

fn emitDensePlanKinds(out: anytype) !void {
    try out.writeAll("def compilerDensePlanKinds : List String :=\n  [");
    inline for (@typeInfo(switch_routing.DensePlanKind).@"enum".fields, 0..) |field, i| {
        if (i != 0) try out.writeAll(", ");
        const kind = @field(switch_routing.DensePlanKind, field.name);
        try out.print("\"{s}\"", .{kind.jsonName()});
    }
    try out.writeAll("]\n\n");
}

fn emitNatList(out: anytype, comptime name: []const u8, values: anytype) !void {
    try out.print("def {s} : List Nat := [", .{name});
    for (values, 0..) |value, i| {
        if (i != 0) try out.writeAll(", ");
        try out.print("{d}", .{value});
    }
    try out.writeAll("]\n\n");
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var out_buffer: [8192]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &out_buffer);
    const out = &stdout_writer.interface;

    try out.writeAll(header);
    try emitStrategyRows(out);
    try emitDensePlanKinds(out);
    try emitNatList(out, "compilerSparseBucketBits", switch_routing.sparse_bucket_bits);
    try emitNatList(out, "compilerSparseBucketShifts", switch_routing.sparse_bucket_shifts);
    try out.print("def compilerDenseMaxTableSlots : Nat := {d}\n", .{switch_routing.dense_max_table_slots});
    try out.print("def compilerMinSelectorCheckSavingX1000 : Nat := {d}\n\n", .{switch_routing.min_selector_check_saving_x1000});
    try out.writeAll("end Ora.Generated\n");

    try out.flush();
}
