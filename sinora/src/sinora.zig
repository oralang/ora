const std = @import("std");

pub const diagnostics = @import("diagnostics.zig");
pub const ir = @import("ir.zig");
pub const parser = @import("parser.zig");
pub const render = @import("render.zig");
pub const ops = @import("ops.zig");
pub const effects = @import("effects.zig");
pub const analyses = @import("analyses.zig");
pub const passes = @import("passes.zig");
pub const ssa_transform = @import("ssa_transform.zig");
pub const optimizations = @import("optimizations.zig");
pub const switch_routing = @import("switch_routing.zig");
pub const metrics = @import("metrics.zig");
pub const bytecode = @import("asm.zig");
pub const legality = @import("legality.zig");
pub const debug_codegen = @import("debug_codegen.zig");
pub const release_schedule = @import("release_schedule.zig");
pub const release_op_graph = @import("release_op_graph.zig");
pub const release_code_to_asm = @import("release_code_to_asm.zig");
pub const release_memory_layout = @import("release_memory_layout.zig");
pub const release_generic_backend = @import("release_generic_backend.zig");

pub const DiagnosticBag = diagnostics.Bag;
pub const Program = ir.Program;

pub fn parse(allocator: std.mem.Allocator, source: []const u8, bag: *DiagnosticBag) !Program {
    return parser.parse(allocator, source, bag);
}

pub fn validate(allocator: std.mem.Allocator, program: Program, bag: *DiagnosticBag) !void {
    return legality.validate(allocator, program, bag);
}

test {
    std.testing.refAllDecls(@This());
}
