//! Generic release backend orchestration ported from Plank.
//!
//! This wires critical-edge normalization, effect-aware op-graph scheduling,
//! Plank-style static memory layout, and generic code-to-asm emission into one
//! release bytecode path. Some public helper names still say "Simple" because
//! they predate the effect-aware scheduler port; they now call the effectful
//! graph builder.

const std = @import("std");

const diagnostics = @import("diagnostics.zig");
const effects = @import("effects.zig");
const evm_asm = @import("asm.zig");
const ir = @import("ir.zig");
const metrics = @import("metrics.zig");
const optimizations = @import("optimizations.zig");
const parser = @import("parser.zig");
const release_code_to_asm = @import("release_code_to_asm.zig");
const release_critical_edges = @import("release_critical_edges.zig");
const release_memory_layout = @import("release_memory_layout.zig");
const release_op_graph = @import("release_op_graph.zig");
const release_schedule = @import("release_schedule.zig");
const switch_routing = @import("switch_routing.zig");

pub const GenericBackendError = error{
    MissingSchedule,
};

pub const ReleasePipelineStage = enum {
    literal_commoning,
    short_circuit_branch_threading,
    critical_edge_splitting,
    effect_analysis,
    effectful_scheduling,
    memory_layout,
    code_to_asm,
};

pub const release_pipeline_stages = [_]ReleasePipelineStage{
    .literal_commoning,
    .short_circuit_branch_threading,
    .critical_edge_splitting,
    .effect_analysis,
    .effectful_scheduling,
    .memory_layout,
    .code_to_asm,
};

pub const release_pipeline_splits_critical_edges = true;
pub const release_pipeline_uses_effectful_scheduler = true;
pub const release_pipeline_supports_source_maps = true;

pub fn releasePipelineStageName(stage: ReleasePipelineStage) []const u8 {
    return switch (stage) {
        .literal_commoning => "literal_commoning",
        .short_circuit_branch_threading => "short_circuit_branch_threading",
        .critical_edge_splitting => "critical_edge_splitting",
        .effect_analysis => "effect_analysis",
        .effectful_scheduling => "effectful_scheduling",
        .memory_layout => "memory_layout",
        .code_to_asm => "code_to_asm",
    };
}

pub const ScheduledProgram = struct {
    allocator: std.mem.Allocator,
    blocks: []release_op_graph.ScheduledBlock,
    next_alloc_id: release_schedule.AllocId,

    pub fn deinit(self: *ScheduledProgram) void {
        for (self.blocks) |block| block.deinit(self.allocator);
        self.allocator.free(self.blocks);
        self.* = undefined;
    }
};

const PreparedReleaseProgram = struct {
    commoned: ir.Program,
    normalized: ir.Program,

    fn deinit(self: *PreparedReleaseProgram) void {
        self.normalized.deinit();
        self.commoned.deinit();
        self.* = undefined;
    }
};

fn prepareReleaseProgram(allocator: std.mem.Allocator, program: ir.Program) !PreparedReleaseProgram {
    var commoned = try optimizations.literalCommoning(allocator, program);
    errdefer commoned.deinit();

    var threaded = try optimizations.shortCircuitBranchThreading(allocator, commoned);
    errdefer threaded.deinit();
    commoned.deinit();
    commoned = threaded;

    var normalized = try release_critical_edges.split(allocator, commoned);
    errdefer normalized.deinit();

    return .{
        .commoned = commoned,
        .normalized = normalized,
    };
}

pub fn scheduleProgramSimple(
    allocator: std.mem.Allocator,
    program: ir.Program,
    config: release_schedule.ScheduleConfig,
    initial_alloc_id: release_schedule.AllocId,
) !ScheduledProgram {
    var blocks: std.ArrayList(release_op_graph.ScheduledBlock) = .empty;
    errdefer {
        for (blocks.items) |block| block.deinit(allocator);
        blocks.deinit(allocator);
    }
    try blocks.ensureTotalCapacity(allocator, program.stats().blocks);

    var layout_cache = release_op_graph.LayoutCache.init(allocator, program);
    defer layout_cache.deinit();
    var function_effects = try effects.analyzeFunctions(allocator, program);
    defer function_effects.deinit();

    var next_alloc_id = initial_alloc_id;
    for (program.functions) |function| {
        for (function.blocks) |block| {
            const scheduled = try release_op_graph.scheduleBlockEffectful(
                allocator,
                program,
                function.name,
                block.name,
                config,
                next_alloc_id,
                &layout_cache,
                &function_effects,
            );
            next_alloc_id = scheduled.next_alloc_id;
            blocks.appendAssumeCapacity(scheduled);
        }
    }

    return .{
        .allocator = allocator,
        .blocks = try blocks.toOwnedSlice(allocator),
        .next_alloc_id = next_alloc_id,
    };
}

pub fn emitSimple(
    allocator: std.mem.Allocator,
    program: ir.Program,
    entry_function_name: []const u8,
    layout: release_code_to_asm.MemoryLayout,
) ![]const u8 {
    var prepared = try prepareReleaseProgram(allocator, program);
    defer prepared.deinit();
    const normalized = prepared.normalized;

    var scheduled = try scheduleProgramSimple(
        allocator,
        normalized,
        release_schedule.ScheduleConfig.pre_amsterdam,
        0,
    );
    defer scheduled.deinit();

    var generated_layout = try release_memory_layout.generateSimple(allocator, normalized, entry_function_name, scheduled.blocks);
    defer generated_layout.deinit();
    const emit_layout = mergeLayout(layout, generated_layout.layout);

    const block_schedules = try allocator.alloc(release_code_to_asm.BlockSchedule, scheduled.blocks.len);
    defer allocator.free(block_schedules);
    for (scheduled.blocks, block_schedules) |scheduled_block, *block_schedule| {
        block_schedule.* = .{
            .function_name = scheduled_block.function_name,
            .block_name = scheduled_block.block_name,
            .ops = scheduled_block.ops,
        };
    }

    return release_code_to_asm.emitFromEntry(
        allocator,
        normalized,
        entry_function_name,
        block_schedules,
        emit_layout,
    );
}

pub fn emitSimpleWithSourceMap(
    allocator: std.mem.Allocator,
    program: ir.Program,
    entry_function_name: []const u8,
    layout: release_code_to_asm.MemoryLayout,
    source_indices: *const release_code_to_asm.SourceIndexMap,
) !release_code_to_asm.EmitResult {
    var prepared = try prepareReleaseProgram(allocator, program);
    defer prepared.deinit();
    const normalized = prepared.normalized;

    var scheduled = try scheduleProgramSimple(
        allocator,
        normalized,
        release_schedule.ScheduleConfig.pre_amsterdam,
        0,
    );
    defer scheduled.deinit();

    var generated_layout = try release_memory_layout.generateSimple(allocator, normalized, entry_function_name, scheduled.blocks);
    defer generated_layout.deinit();
    const emit_layout = mergeLayout(layout, generated_layout.layout);

    const block_schedules = try allocator.alloc(release_code_to_asm.BlockSchedule, scheduled.blocks.len);
    defer allocator.free(block_schedules);
    for (scheduled.blocks, block_schedules) |scheduled_block, *block_schedule| {
        block_schedule.* = .{
            .function_name = scheduled_block.function_name,
            .block_name = scheduled_block.block_name,
            .ops = scheduled_block.ops,
        };
    }

    return release_code_to_asm.emitFromEntryWithSourceMap(
        allocator,
        normalized,
        entry_function_name,
        block_schedules,
        emit_layout,
        source_indices,
    );
}

pub fn emitDeploymentSimple(
    allocator: std.mem.Allocator,
    program: ir.Program,
) ![]const u8 {
    var prepared = try prepareReleaseProgram(allocator, program);
    defer prepared.deinit();
    const normalized = prepared.normalized;

    var scheduled = try scheduleProgramSimple(
        allocator,
        normalized,
        release_schedule.ScheduleConfig.pre_amsterdam,
        0,
    );
    defer scheduled.deinit();

    var init_layout = try release_memory_layout.generateSimple(allocator, normalized, "init", scheduled.blocks);
    defer init_layout.deinit();

    const runtime_function_name: ?[]const u8 = if (findFunction(normalized, "main") != null) "main" else null;
    var runtime_layout: ?release_memory_layout.OwnedLayout = if (runtime_function_name) |name|
        try release_memory_layout.generateSimple(allocator, normalized, name, scheduled.blocks)
    else
        null;
    defer if (runtime_layout) |*layout| layout.deinit();

    const block_schedules = try allocator.alloc(release_code_to_asm.BlockSchedule, scheduled.blocks.len);
    defer allocator.free(block_schedules);
    for (scheduled.blocks, block_schedules) |scheduled_block, *block_schedule| {
        block_schedule.* = .{
            .function_name = scheduled_block.function_name,
            .block_name = scheduled_block.block_name,
            .ops = scheduled_block.ops,
        };
    }

    return release_code_to_asm.emitDeployment(
        allocator,
        normalized,
        "init",
        runtime_function_name,
        block_schedules,
        init_layout.layout,
        if (runtime_layout) |layout| layout.layout else .{},
    );
}

pub fn emitDeploymentWithSourceMap(
    allocator: std.mem.Allocator,
    program: ir.Program,
    source_indices: *const release_code_to_asm.SourceIndexMap,
) !release_code_to_asm.EmitResult {
    var prepared = try prepareReleaseProgram(allocator, program);
    defer prepared.deinit();
    const normalized = prepared.normalized;

    var scheduled = try scheduleProgramSimple(
        allocator,
        normalized,
        release_schedule.ScheduleConfig.pre_amsterdam,
        0,
    );
    defer scheduled.deinit();

    var init_layout = try release_memory_layout.generateSimple(allocator, normalized, "init", scheduled.blocks);
    defer init_layout.deinit();

    const runtime_function_name: ?[]const u8 = if (findFunction(normalized, "main") != null) "main" else null;
    var runtime_layout: ?release_memory_layout.OwnedLayout = if (runtime_function_name) |name|
        try release_memory_layout.generateSimple(allocator, normalized, name, scheduled.blocks)
    else
        null;
    defer if (runtime_layout) |*layout| layout.deinit();

    const block_schedules = try allocator.alloc(release_code_to_asm.BlockSchedule, scheduled.blocks.len);
    defer allocator.free(block_schedules);
    for (scheduled.blocks, block_schedules) |scheduled_block, *block_schedule| {
        block_schedule.* = .{
            .function_name = scheduled_block.function_name,
            .block_name = scheduled_block.block_name,
            .ops = scheduled_block.ops,
        };
    }

    return release_code_to_asm.emitDeploymentWithSourceMap(
        allocator,
        normalized,
        "init",
        runtime_function_name,
        block_schedules,
        init_layout.layout,
        if (runtime_layout) |layout| layout.layout else .{},
        source_indices,
    );
}

/// Emit generic release bytecode for a whole program, choosing the deployment
/// path when an `init` function is present and the runtime-only path otherwise.
/// This is the single entry point behind the release CLI aliases.
pub fn emitRelease(allocator: std.mem.Allocator, program: ir.Program) ![]const u8 {
    if (findFunction(program, "init") != null) {
        return emitDeploymentSimple(allocator, program);
    }
    return emitSimple(allocator, program, "main", .{});
}

pub fn emitReleaseWithSourceMap(allocator: std.mem.Allocator, program: ir.Program) !release_code_to_asm.EmitResult {
    var source_indices = try release_code_to_asm.SourceIndexMap.fromProgram(allocator, program);
    defer source_indices.deinit();
    if (findFunction(program, "init") != null) {
        return emitDeploymentWithSourceMap(allocator, program, &source_indices);
    }
    return emitSimpleWithSourceMap(allocator, program, "main", .{}, &source_indices);
}

/// Re-run the deterministic release planning stages and summarize their shape.
/// This deliberately does not time stages; timing belongs in a separate profiler
/// because wall-clock numbers are too noisy for checked snapshots.
pub fn collectReleaseMetrics(
    allocator: std.mem.Allocator,
    program: ir.Program,
    bytecode_bytes: usize,
    source_map_entries: usize,
) !metrics.ReleaseMetrics {
    const input_stats = program.stats();
    var prepared = try prepareReleaseProgram(allocator, program);
    defer prepared.deinit();

    var scheduled = try scheduleProgramSimple(
        allocator,
        prepared.normalized,
        release_schedule.ScheduleConfig.pre_amsterdam,
        0,
    );
    defer scheduled.deinit();

    const schedule_stats = metrics.StackOpStats.fromScheduledBlocks(scheduled.blocks);
    const switch_routing_stats = metrics.SwitchRoutingStats.fromProgram(prepared.normalized);

    if (findFunction(prepared.normalized, "init") != null) {
        var init_layout = try release_memory_layout.generateSimple(allocator, prepared.normalized, "init", scheduled.blocks);
        defer init_layout.deinit();

        const runtime_function_name: ?[]const u8 = if (findFunction(prepared.normalized, "main") != null) "main" else null;
        var runtime_layout: ?release_memory_layout.OwnedLayout = if (runtime_function_name) |name|
            try release_memory_layout.generateSimple(allocator, prepared.normalized, name, scheduled.blocks)
        else
            null;
        defer if (runtime_layout) |*layout| layout.deinit();

        return .{
            .mode = .deployment,
            .bytecode_bytes = bytecode_bytes,
            .source_map_entries = source_map_entries,
            .input_ir = input_stats,
            .commoned_ir = prepared.commoned.stats(),
            .normalized_ir = prepared.normalized.stats(),
            .switch_routing = switch_routing_stats,
            .schedule = schedule_stats,
            .init_layout = metrics.LayoutStats.fromLayout(init_layout.layout),
            .runtime_layout = if (runtime_layout) |layout| metrics.LayoutStats.fromLayout(layout.layout) else null,
        };
    }

    var runtime_layout = try release_memory_layout.generateSimple(allocator, prepared.normalized, "main", scheduled.blocks);
    defer runtime_layout.deinit();

    return .{
        .mode = .runtime,
        .bytecode_bytes = bytecode_bytes,
        .source_map_entries = source_map_entries,
        .input_ir = input_stats,
        .commoned_ir = prepared.commoned.stats(),
        .normalized_ir = prepared.normalized.stats(),
        .switch_routing = switch_routing_stats,
        .schedule = schedule_stats,
        .runtime_layout = metrics.LayoutStats.fromLayout(runtime_layout.layout),
    };
}

fn mergeLayout(
    override: release_code_to_asm.MemoryLayout,
    generated: release_code_to_asm.MemoryLayout,
) release_code_to_asm.MemoryLayout {
    return .{
        .alloc_start = if (override.alloc_start.len != 0) override.alloc_start else generated.alloc_start,
        .static_alloc_start = if (override.static_alloc_start.len != 0) override.static_alloc_start else generated.static_alloc_start,
        .switch_store = override.switch_store orelse generated.switch_store,
        .switch_table_store = override.switch_table_store orelse generated.switch_table_store,
        .dyn_free_pointer = override.dyn_free_pointer orelse generated.dyn_free_pointer,
    };
}

fn findFunction(program: ir.Program, name: []const u8) ?ir.Function {
    for (program.functions) |function| {
        if (std.mem.eql(u8, function.name, name)) return function;
    }
    return null;
}

fn parseTestProgram(source: []const u8) !ir.Program {
    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator, source, &bag);
    errdefer program.deinit();
    try std.testing.expectEqual(@as(usize, 0), bag.items.items.len);
    return program;
}

test "generic backend emits straight-line parsed program" {
    var program = try parseTestProgram(
        \\fn main:
        \\    entry {
        \\        ptr = const 0x00
        \\        len = const 0x20
        \\        value = const 0x2a
        \\        mstore256 ptr value
        \\        return ptr len
        \\    }
    );
    defer program.deinit();

    const bytes = try emitSimple(std.testing.allocator, program, "main", .{});
    defer std.testing.allocator.free(bytes);

    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.MSTORE) != null);
    try std.testing.expectEqual(evm_asm.op.RETURN, bytes[bytes.len - 1]);
}

test "generic backend accepts inline numeric value operands" {
    var program = try parseTestProgram(
        \\fn main:
        \\    entry {
        \\        sum = add 0x1 0x2
        \\        mstore256 0 sum
        \\        return 0 0x20
        \\    }
    );
    defer program.deinit();

    const bytes = try emitSimple(std.testing.allocator, program, "main", .{});
    defer std.testing.allocator.free(bytes);

    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.MSTORE) != null);
    try std.testing.expectEqual(evm_asm.op.RETURN, bytes[bytes.len - 1]);
}

test "generic backend source map skips parser synthetic inline constants" {
    var program = try parseTestProgram(
        \\fn main:
        \\    entry {
        \\        sum = add 0x1 0x2
        \\        mstore256 0 sum
        \\        return 0 0x20
        \\    }
    );
    defer program.deinit();

    var result = try emitReleaseWithSourceMap(std.testing.allocator, program);
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 3), result.source_map.len);
    try std.testing.expectEqual(@as(u32, 0), result.source_map[0].idx);
    try std.testing.expectEqual(@as(u32, 1), result.source_map[1].idx);
    try std.testing.expectEqual(@as(u32, 2), result.source_map[2].idx);
}

test "generic backend commoning emits one push for repeated large literal" {
    var program = try parseTestProgram(
        \\fn main:
        \\    entry {
        \\        a = large_const 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
        \\        b = large_const 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
        \\        sum = xor a b
        \\        mstore256 0 sum
        \\        return 0 0x20
        \\    }
    );
    defer program.deinit();

    const bytes = try emitRelease(std.testing.allocator, program);
    defer std.testing.allocator.free(bytes);

    try std.testing.expectEqual(@as(usize, 1), countByte(bytes, evm_asm.op.PUSH1 + 31));
}

test "generic backend emits branch parsed program" {
    var program = try parseTestProgram(
        \\fn main:
        \\    entry {
        \\        cond = const 0x01
        \\        => cond ? @success : @failure
        \\    }
        \\
        \\    success {
        \\        stop
        \\    }
        \\
        \\    failure {
        \\        invalid
        \\    }
    );
    defer program.deinit();

    const bytes = try emitSimple(std.testing.allocator, program, "main", .{});
    defer std.testing.allocator.free(bytes);

    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.JUMPI) != null);
    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.JUMP) != null);
    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.STOP) != null);
    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.INVALID) != null);
}

test "generic backend emits deployment with branch live-through runtime" {
    var program = try parseTestProgram(
        \\fn init:
        \\    entry {
        \\        start = runtime_start_offset
        \\        len = runtime_length
        \\        ptr = malloc len
        \\        codecopy ptr start len
        \\        return ptr len
        \\    }
        \\
        \\fn main:
        \\    entry {
        \\        ptr = const 0x00
        \\        len = const 0x20
        \\        cond = const 0x01
        \\        => cond ? @success : @failure
        \\    }
        \\
        \\    success {
        \\        value = const 0x2a
        \\        mstore256 ptr value
        \\        return ptr len
        \\    }
        \\
        \\    failure {
        \\        revert ptr ptr
        \\    }
    );
    defer program.deinit();

    const bytes = try emitDeploymentSimple(std.testing.allocator, program);
    defer std.testing.allocator.free(bytes);

    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.JUMPI) != null);
    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.MSTORE) != null);
    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.REVERT) != null);
    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.RETURN) != null);
}

test "generic backend keeps unused icall operands out of shared revert layout" {
    var program = try parseTestProgram(
        \\fn run:
        \\    entry unused_a unused_b used {
        \\        slot = const 0x00
        \\        sstore slot used
        \\        iret
        \\    }
        \\
        \\fn main:
        \\    entry {
        \\        entry_cond = const 0x01
        \\        => entry_cond ? @bb0 : @abi_revert
        \\    }
        \\
        \\    bb0 {
        \\        carried = const 0x11
        \\        cond0 = const 0x01
        \\        => cond0 ? @bb1 : @abi_revert
        \\    }
        \\
        \\    bb1 {
        \\        late = const 0x22
        \\        cond1 = const 0x01
        \\        => cond1 ? @run_exec : @abi_revert
        \\    }
        \\
        \\    run_exec {
        \\        value = const 0x33
        \\        icall @run carried late value
        \\        zero = const 0x00
        \\        return zero zero
        \\    }
        \\
        \\    abi_revert {
        \\        zero_0 = const 0x00
        \\        revert zero_0 zero_0
        \\    }
    );
    defer program.deinit();

    const bytes = try emitSimple(std.testing.allocator, program, "main", .{});
    defer std.testing.allocator.free(bytes);

    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.JUMPI) != null);
    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.SSTORE) != null);
    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.REVERT) != null);
    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.RETURN) != null);
}

test "generic backend emits internal call parsed program" {
    var program = try parseTestProgram(
        \\fn callee:
        \\    entry input -> output {
        \\        one = const 0x01
        \\        output = add input one
        \\        iret
        \\    }
        \\
        \\fn main:
        \\    entry {
        \\        arg = const 0x29
        \\        value = icall @callee arg
        \\        stop
        \\    }
    );
    defer program.deinit();

    const bytes = try emitSimple(std.testing.allocator, program, "main", .{});
    defer std.testing.allocator.free(bytes);

    try std.testing.expectEqual(@as(usize, 3), countByte(bytes, evm_asm.op.JUMPDEST));
    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.JUMP) != null);
    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.STOP) != null);
}

test "generic backend generates switch scratch layout" {
    var program = try parseTestProgram(
        \\fn main:
        \\    entry {
        \\        selector = const 0x01
        \\        switch selector {
        \\        0x01 => @one
        \\        default => @other
        \\        }
        \\    }
        \\
        \\    one {
        \\        stop
        \\    }
        \\
        \\    other {
        \\        invalid
        \\    }
    );
    defer program.deinit();

    const bytes = try emitSimple(std.testing.allocator, program, "main", .{});
    defer std.testing.allocator.free(bytes);

    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.MSTORE) != null);
    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.MLOAD) != null);
    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.JUMPI) != null);
}

test "generic backend keeps small switches linear despite perfect dense windows" {
    var program = try parseTestProgram(
        \\fn main:
        \\    entry {
        \\        selector = const 0x21
        \\        switch selector {
        \\        0x00000001 => @one
        \\        0x00000011 => @two
        \\        0x00000021 => @three
        \\        0x00000031 => @four
        \\        default => @other
        \\        }
        \\    }
        \\
        \\    one {
        \\        stop
        \\    }
        \\
        \\    two {
        \\        stop
        \\    }
        \\
        \\    three {
        \\        stop
        \\    }
        \\
        \\    four {
        \\        stop
        \\    }
        \\
        \\    other {
        \\        invalid
        \\    }
    );
    defer program.deinit();

    const plan = switch (program.functions[0].blocks[0].terminator) {
        .switch_ => |switch_term| switch_routing.choosePlan(switch_term),
        else => return error.TestUnexpectedResult,
    };
    try std.testing.expect(plan == .linear);

    const bytes = try emitRelease(std.testing.allocator, program);
    defer std.testing.allocator.free(bytes);

    try std.testing.expectEqual(@as(usize, 4), countByte(bytes, evm_asm.op.JUMPI));
}

test "release pipeline fact table records mandatory backend stages" {
    try std.testing.expectEqual(@as(usize, 7), release_pipeline_stages.len);
    try std.testing.expectEqualStrings("literal_commoning", releasePipelineStageName(release_pipeline_stages[0]));
    try std.testing.expectEqualStrings("critical_edge_splitting", releasePipelineStageName(release_pipeline_stages[2]));
    try std.testing.expectEqualStrings("effectful_scheduling", releasePipelineStageName(release_pipeline_stages[4]));
    try std.testing.expect(release_pipeline_splits_critical_edges);
    try std.testing.expect(release_pipeline_uses_effectful_scheduler);
    try std.testing.expect(release_pipeline_supports_source_maps);
}

test "generic backend lowers sparse switch routing" {
    var source: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer source.deinit();

    try source.writer.writeAll(
        \\fn main:
        \\    entry {
        \\        selector = const 0x101
        \\        switch selector {
        \\
    );
    for (0..260) |index| {
        try source.writer.print("        0x{x} => @hit\n", .{index});
    }
    try source.writer.writeAll(
        \\        default => @other
        \\        }
        \\    }
        \\
        \\    hit {
        \\        stop
        \\    }
        \\
        \\    other {
        \\        invalid
        \\    }
    );

    var program = try parseTestProgram(source.written());
    defer program.deinit();

    const plan = switch (program.functions[0].blocks[0].terminator) {
        .switch_ => |switch_term| switch_routing.choosePlan(switch_term),
        else => return error.TestUnexpectedResult,
    };
    try std.testing.expect(plan == .sparse);

    const bytes = try emitRelease(std.testing.allocator, program);
    defer std.testing.allocator.free(bytes);

    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.CODECOPY) != null);
    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.AND) != null);
    try std.testing.expect(countByte(bytes, evm_asm.op.JUMPI) >= 260);
}

test "generic backend preserves switch case order in linear chains" {
    // The frontend emits dispatcher cases in priority order (state-mutating
    // first); the linear chain must check them in exactly that order, since
    // position in the chain is the gas the ordering pass is buying.
    var program = try parseTestProgram(
        \\fn main:
        \\    entry {
        \\        selector = const 0x21
        \\        switch selector {
        \\        0xaabb0001 => @one
        \\        0xaabb0002 => @two
        \\        0xaabb0003 => @three
        \\        0xaabb0004 => @four
        \\        default => @other
        \\        }
        \\    }
        \\
        \\    one {
        \\        stop
        \\    }
        \\
        \\    two {
        \\        stop
        \\    }
        \\
        \\    three {
        \\        stop
        \\    }
        \\
        \\    four {
        \\        stop
        \\    }
        \\
        \\    other {
        \\        invalid
        \\    }
    );
    defer program.deinit();

    const bytes = try emitRelease(std.testing.allocator, program);
    defer std.testing.allocator.free(bytes);

    var last_pos: usize = 0;
    for ([_]u32{ 0xaabb0001, 0xaabb0002, 0xaabb0003, 0xaabb0004 }) |selector| {
        var needle: [4]u8 = undefined;
        std.mem.writeInt(u32, &needle, selector, .big);
        const pos = std.mem.indexOfPos(u8, bytes, last_pos, &needle) orelse return error.TestUnexpectedResult;
        last_pos = pos + needle.len;
    }
}

test "generic backend lowers dense bit-window switch routing" {
    var source: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer source.deinit();

    try source.writer.writeAll(
        \\fn main:
        \\    entry {
        \\        selector = const 0x21
        \\        switch selector {
        \\
    );
    // Stride 0x10 keeps bits [4,9) collision-free while the value span
    // (0x131) exceeds the dense table cap, so only a bit-window plan exists.
    for (0..20) |index| {
        try source.writer.print("        0x{x} => @hit\n", .{index * 0x10 + 1});
    }
    try source.writer.writeAll(
        \\        default => @other
        \\        }
        \\    }
        \\
        \\    hit {
        \\        stop
        \\    }
        \\
        \\    other {
        \\        invalid
        \\    }
    );

    var program = try parseTestProgram(source.written());
    defer program.deinit();

    const plan = switch (program.functions[0].blocks[0].terminator) {
        .switch_ => |switch_term| switch_routing.choosePlan(switch_term),
        else => return error.TestUnexpectedResult,
    };
    try std.testing.expect(plan == .dense);
    try std.testing.expectEqual(switch_routing.DensePlanKind.bit_window, plan.dense.kind);

    const bytes = try emitRelease(std.testing.allocator, program);
    defer std.testing.allocator.free(bytes);

    // Mask-bounded index: no bounds check, so exactly one exact-check JUMPI
    // per landing slot and none for range guarding.
    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.CODECOPY) != null);
    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.AND) != null);
    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.SHR) != null);
    try std.testing.expectEqual(@as(usize, 20), countByte(bytes, evm_asm.op.JUMPI));
}

test "generic backend generates spill layout for deep stack schedule" {
    var program = try parseTestProgram(
        \\fn main:
        \\    entry {
        \\        v0 = const 0x00
        \\        v1 = const 0x01
        \\        v2 = const 0x02
        \\        v3 = const 0x03
        \\        v4 = const 0x04
        \\        v5 = const 0x05
        \\        v6 = const 0x06
        \\        v7 = const 0x07
        \\        v8 = const 0x08
        \\        v9 = const 0x09
        \\        v10 = const 0x0a
        \\        v11 = const 0x0b
        \\        v12 = const 0x0c
        \\        v13 = const 0x0d
        \\        v14 = const 0x0e
        \\        v15 = const 0x0f
        \\        v16 = const 0x10
        \\        v17 = const 0x11
        \\        sum = add v0 v17
        \\        stop
        \\    }
    );
    defer program.deinit();

    var scheduled = try scheduleProgramSimple(
        std.testing.allocator,
        program,
        .{ .max_swap_depth = 3, .max_dup_depth = 2, .max_exchange_range = 3 },
        0,
    );
    defer scheduled.deinit();

    var generated_layout = try release_memory_layout.generateSimple(std.testing.allocator, program, "main", scheduled.blocks);
    defer generated_layout.deinit();
    try std.testing.expect(generated_layout.alloc_start.len > 0);

    const block_schedules = try std.testing.allocator.alloc(release_code_to_asm.BlockSchedule, scheduled.blocks.len);
    defer std.testing.allocator.free(block_schedules);
    for (scheduled.blocks, block_schedules) |scheduled_block, *block_schedule| {
        block_schedule.* = .{
            .function_name = scheduled_block.function_name,
            .block_name = scheduled_block.block_name,
            .ops = scheduled_block.ops,
        };
    }

    const bytes = try release_code_to_asm.emitFromEntry(std.testing.allocator, program, "main", block_schedules, generated_layout.layout);
    defer std.testing.allocator.free(bytes);

    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.MSTORE) != null);
    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.MLOAD) != null);
}

test "generic backend emits dynamic allocation" {
    var program = try parseTestProgram(
        \\fn main:
        \\    entry {
        \\        len = const 0x20
        \\        ptr = malloc len
        \\        return ptr len
        \\    }
    );
    defer program.deinit();

    const bytes = try emitSimple(std.testing.allocator, program, "main", .{});
    defer std.testing.allocator.free(bytes);

    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.MLOAD) != null);
    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.MSTORE) != null);
    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.CALLDATACOPY) != null);
    try std.testing.expectEqual(evm_asm.op.RETURN, bytes[bytes.len - 1]);
}

test "generic backend emits static allocation" {
    var program = try parseTestProgram(
        \\fn main:
        \\    entry {
        \\        ptr = salloc 0x20
        \\        len = const 0x20
        \\        return ptr len
        \\    }
    );
    defer program.deinit();

    const bytes = try emitSimple(std.testing.allocator, program, "main", .{});
    defer std.testing.allocator.free(bytes);

    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.CALLDATASIZE) != null);
    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.CALLDATACOPY) != null);
    try std.testing.expectEqual(evm_asm.op.RETURN, bytes[bytes.len - 1]);
}

test "generic backend emits plank wide partial memory store mask" {
    var program = try parseTestProgram(
        \\fn main:
        \\    entry {
        \\        ptr = const 0x00
        \\        value = const 0xff
        \\        mstore240 ptr value
        \\        stop
        \\    }
    );
    defer program.deinit();

    const bytes = try emitSimple(std.testing.allocator, program, "main", .{});
    defer std.testing.allocator.free(bytes);

    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.AND) != null);
    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.MSTORE) != null);
    try std.testing.expectEqual(evm_asm.op.STOP, bytes[bytes.len - 1]);
}

test "generic backend emits deployment runtime data offset" {
    var program = try parseTestProgram(
        \\fn init:
        \\    entry {
        \\        start = runtime_start_offset
        \\        len = runtime_length
        \\        ptr = malloc len
        \\        codecopy ptr start len
        \\        return ptr len
        \\    }
        \\
        \\fn main:
        \\    entry {
        \\        len = const 0x4
        \\        ptr = malloc len
        \\        off = data_offset .blob
        \\        codecopy ptr off len
        \\        return ptr len
        \\    }
        \\
        \\data blob 0x11223344
    );
    defer program.deinit();

    const bytes = try emitDeploymentSimple(std.testing.allocator, program);
    defer std.testing.allocator.free(bytes);

    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.CODECOPY) != null);
    try std.testing.expect(std.mem.endsWith(u8, bytes, &.{ 0x11, 0x22, 0x33, 0x44 }));
}

test "generic backend emits deployment init runtime offsets" {
    var program = try parseTestProgram(
        \\fn init:
        \\    entry {
        \\        start = runtime_start_offset
        \\        end = init_end_offset
        \\        len = runtime_length
        \\        ptr = malloc len
        \\        codecopy ptr start len
        \\        return ptr len
        \\    }
        \\
        \\fn main:
        \\    entry {
        \\        stop
        \\    }
    );
    defer program.deinit();

    const bytes = try emitDeploymentSimple(std.testing.allocator, program);
    defer std.testing.allocator.free(bytes);

    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.CODECOPY) != null);
    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.RETURN) != null);
    try std.testing.expectEqual(evm_asm.op.STOP, bytes[bytes.len - 1]);
}

fn countByte(bytes: []const u8, needle: u8) usize {
    var count: usize = 0;
    for (bytes) |byte| {
        if (byte == needle) count += 1;
    }
    return count;
}
