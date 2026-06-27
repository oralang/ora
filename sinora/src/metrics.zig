//! Deterministic Sinora release metrics.
//!
//! These counters describe the backend shape that produced a bytecode artifact.
//! They intentionally avoid wall-clock timings so snapshots are stable across
//! machines. Runtime/deploy gas remain the job of Ora's conformance harness.

const std = @import("std");

const ir = @import("ir.zig");
const release_code_to_asm = @import("release_code_to_asm.zig");
const release_op_graph = @import("release_op_graph.zig");
const release_schedule = @import("release_schedule.zig");

pub const ReleaseMode = enum {
    runtime,
    deployment,

    fn jsonName(self: ReleaseMode) []const u8 {
        return switch (self) {
            .runtime => "runtime",
            .deployment => "deployment",
        };
    }
};

pub const StackOpStats = struct {
    blocks: usize = 0,
    stack_ops: usize = 0,
    swap: usize = 0,
    dup: usize = 0,
    pop: usize = 0,
    op: usize = 0,
    call_ret_push: usize = 0,
    exchange: usize = 0,
    store: usize = 0,
    load: usize = 0,

    pub fn fromScheduledBlocks(blocks: []const release_op_graph.ScheduledBlock) StackOpStats {
        var result = StackOpStats{ .blocks = blocks.len };
        for (blocks) |block| {
            for (block.ops) |stack_op| {
                result.stack_ops += 1;
                switch (stack_op) {
                    .swap => result.swap += 1,
                    .dup => result.dup += 1,
                    .pop => result.pop += 1,
                    .op => result.op += 1,
                    .call_ret_push => result.call_ret_push += 1,
                    .exchange => result.exchange += 1,
                    .store => result.store += 1,
                    .load => result.load += 1,
                }
            }
        }
        return result;
    }
};

pub const LayoutStats = struct {
    spill_slots: usize = 0,
    static_alloc_slots: usize = 0,
    static_allocs_needing_zero: usize = 0,
    has_switch_store: bool = false,
    has_dyn_free_pointer: bool = false,
    dyn_free_pointer_store_slot: ?u32 = null,
    dyn_free_pointer_start_value: ?u32 = null,

    pub fn fromLayout(layout: release_code_to_asm.MemoryLayout) LayoutStats {
        var zeroing: usize = 0;
        for (layout.static_alloc_start) |entry| {
            if (entry.needs_zeroing) zeroing += 1;
        }
        return .{
            .spill_slots = layout.alloc_start.len,
            .static_alloc_slots = layout.static_alloc_start.len,
            .static_allocs_needing_zero = zeroing,
            .has_switch_store = layout.switch_store != null,
            .has_dyn_free_pointer = layout.dyn_free_pointer != null,
            .dyn_free_pointer_store_slot = if (layout.dyn_free_pointer) |fp| fp.store_slot else null,
            .dyn_free_pointer_start_value = if (layout.dyn_free_pointer) |fp| fp.start_value else null,
        };
    }
};

pub const ReleaseMetrics = struct {
    mode: ReleaseMode,
    bytecode_bytes: usize,
    source_map_entries: usize = 0,
    input_ir: ir.Stats,
    commoned_ir: ir.Stats,
    normalized_ir: ir.Stats,
    schedule: StackOpStats,
    init_layout: ?LayoutStats = null,
    runtime_layout: ?LayoutStats = null,
};

pub fn writeReleaseMetricsJson(writer: anytype, metrics: ReleaseMetrics) !void {
    try writer.writeAll("{\n");
    try writer.print("  \"mode\": \"{s}\",\n", .{metrics.mode.jsonName()});
    try writer.print("  \"bytecode_bytes\": {},\n", .{metrics.bytecode_bytes});
    try writer.print("  \"source_map_entries\": {},\n", .{metrics.source_map_entries});
    try writeIrStatsField(writer, "input_ir", metrics.input_ir, true);
    try writeIrStatsField(writer, "commoned_ir", metrics.commoned_ir, true);
    try writeIrStatsField(writer, "normalized_ir", metrics.normalized_ir, true);
    try writeStackOpStatsField(writer, "schedule", metrics.schedule, true);
    try writeLayoutStatsField(writer, "init_layout", metrics.init_layout, true);
    try writeLayoutStatsField(writer, "runtime_layout", metrics.runtime_layout, false);
    try writer.writeAll("}\n");
}

fn writeIrStatsField(writer: anytype, name: []const u8, stats: ir.Stats, trailing_comma: bool) !void {
    try writer.print("  \"{s}\": {{", .{name});
    try writer.print("\"functions\": {}, ", .{stats.functions});
    try writer.print("\"data_segments\": {}, ", .{stats.data_segments});
    try writer.print("\"data_bytes\": {}, ", .{stats.data_bytes});
    try writer.print("\"blocks\": {}, ", .{stats.blocks});
    try writer.print("\"instructions\": {}, ", .{stats.instructions});
    try writer.print("\"terminators\": {}, ", .{stats.terminators});
    try writer.print("\"switches\": {}, ", .{stats.switches});
    try writer.print("\"switch_cases\": {}", .{stats.switch_cases});
    try writer.print("}}{s}\n", .{if (trailing_comma) "," else ""});
}

fn writeStackOpStatsField(writer: anytype, name: []const u8, stats: StackOpStats, trailing_comma: bool) !void {
    try writer.print("  \"{s}\": {{", .{name});
    try writer.print("\"blocks\": {}, ", .{stats.blocks});
    try writer.print("\"stack_ops\": {}, ", .{stats.stack_ops});
    try writer.print("\"swap\": {}, ", .{stats.swap});
    try writer.print("\"dup\": {}, ", .{stats.dup});
    try writer.print("\"pop\": {}, ", .{stats.pop});
    try writer.print("\"op\": {}, ", .{stats.op});
    try writer.print("\"call_ret_push\": {}, ", .{stats.call_ret_push});
    try writer.print("\"exchange\": {}, ", .{stats.exchange});
    try writer.print("\"store\": {}, ", .{stats.store});
    try writer.print("\"load\": {}", .{stats.load});
    try writer.print("}}{s}\n", .{if (trailing_comma) "," else ""});
}

fn writeLayoutStatsField(writer: anytype, name: []const u8, maybe_stats: ?LayoutStats, trailing_comma: bool) !void {
    try writer.print("  \"{s}\": ", .{name});
    if (maybe_stats) |stats| {
        try writer.writeAll("{");
        try writer.print("\"spill_slots\": {}, ", .{stats.spill_slots});
        try writer.print("\"static_alloc_slots\": {}, ", .{stats.static_alloc_slots});
        try writer.print("\"static_allocs_needing_zero\": {}, ", .{stats.static_allocs_needing_zero});
        try writer.print("\"has_switch_store\": {}, ", .{stats.has_switch_store});
        try writer.print("\"has_dyn_free_pointer\": {}, ", .{stats.has_dyn_free_pointer});
        try writer.writeAll("\"dyn_free_pointer_store_slot\": ");
        try writeOptionalU32(writer, stats.dyn_free_pointer_store_slot);
        try writer.writeAll(", \"dyn_free_pointer_start_value\": ");
        try writeOptionalU32(writer, stats.dyn_free_pointer_start_value);
        try writer.writeAll("}");
    } else {
        try writer.writeAll("null");
    }
    try writer.print("{s}\n", .{if (trailing_comma) "," else ""});
}

fn writeOptionalU32(writer: anytype, value: ?u32) !void {
    if (value) |n| {
        try writer.print("{}", .{n});
    } else {
        try writer.writeAll("null");
    }
}

test "release metrics json writes deterministic shape counters" {
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();

    try writeReleaseMetricsJson(&out.writer, .{
        .mode = .runtime,
        .bytecode_bytes = 12,
        .input_ir = .{ .functions = 1, .blocks = 1 },
        .commoned_ir = .{ .functions = 1, .blocks = 1 },
        .normalized_ir = .{ .functions = 1, .blocks = 2, .switches = 1, .switch_cases = 3 },
        .schedule = .{ .blocks = 2, .stack_ops = 4, .op = 2, .store = 1, .load = 1 },
        .runtime_layout = .{ .spill_slots = 1, .has_switch_store = true },
    });

    const text = out.written();
    try std.testing.expect(std.mem.indexOf(u8, text, "\"mode\": \"runtime\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "\"bytecode_bytes\": 12") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "\"switch_cases\": 3") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "\"runtime_layout\": {") != null);
}
