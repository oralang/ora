//! Emits `formal/Ora/Generated/SinoraBackendSnapshot.lean` — data-only facts
//! from Sinora's pass manager and release backend.

const std = @import("std");
const sinora = @import("sinora");

const passes = sinora.passes;
const release_backend = sinora.release_generic_backend;

const header =
    \\/-
    \\GENERATED — DATA ONLY. Do NOT edit by hand and do NOT add any `theorem`,
    \\`lemma`, `example`, `axiom`, `sorry`, `instance`, `macro`, attribute, or extra
    \\`import` to this file. It contains only `def … := <literal>` Sinora backend
    \\facts emitted from the compiler workspace. The TRUSTED checks live in
    \\`Ora/SinoraBackendSync.lean`.
    \\
    \\Regenerate with `scripts/check-formal-sync.sh`. Source:
    \\src/formal/emit_sinora_backend_snapshot.zig,
    \\sinora/src/passes.zig, sinora/src/release_generic_backend.zig.
    \\-/
    \\
    \\namespace Ora.Generated
    \\
    \\
;

fn emitOptimizationPassRows(out: anytype) !void {
    try out.writeAll("def compilerSinoraOptimizationPassRows : List (String × String) :=\n  [");
    for (passes.optimization_pass_facts, 0..) |fact, index| {
        if (index != 0) try out.writeAll(",\n   ");
        try out.print("(\"{s}\", \"{c}\")", .{ fact.name, fact.cli_code });
    }
    try out.writeAll("]\n\n");
}

fn emitReleasePipelineStages(out: anytype) !void {
    try out.writeAll("def compilerSinoraReleasePipelineStages : List String :=\n  [");
    for (release_backend.release_pipeline_stages, 0..) |stage, index| {
        if (index != 0) try out.writeAll(", ");
        try out.print("\"{s}\"", .{release_backend.releasePipelineStageName(stage)});
    }
    try out.writeAll("]\n\n");
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var out_buffer: [8192]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &out_buffer);
    const out = &stdout_writer.interface;

    try out.writeAll(header);
    try emitOptimizationPassRows(out);
    try out.print("def compilerSinoraOptimizationPipelineRunsFinalLegalizer : Bool := {s}\n\n", .{
        if (passes.optimization_pipeline_runs_final_legalizer) "true" else "false",
    });
    try emitReleasePipelineStages(out);
    try out.print("def compilerSinoraReleaseSplitsCriticalEdges : Bool := {s}\n", .{
        if (release_backend.release_pipeline_splits_critical_edges) "true" else "false",
    });
    try out.print("def compilerSinoraReleaseUsesEffectfulScheduler : Bool := {s}\n", .{
        if (release_backend.release_pipeline_uses_effectful_scheduler) "true" else "false",
    });
    try out.print("def compilerSinoraReleaseSupportsSourceMaps : Bool := {s}\n\n", .{
        if (release_backend.release_pipeline_supports_source_maps) "true" else "false",
    });
    try out.writeAll("end Ora.Generated\n");

    try out.flush();
}
