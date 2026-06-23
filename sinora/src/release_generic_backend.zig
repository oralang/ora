//! Generic release backend orchestration under migration from Plank.
//!
//! This wires the current simple op-graph scheduler into the generic
//! code-to-asm emitter. It is intentionally separate from release_codegen.zig's
//! legacy selector-shape parity scaffold.

const std = @import("std");

const diagnostics = @import("diagnostics.zig");
const evm_asm = @import("asm.zig");
const ir = @import("ir.zig");
const parser = @import("parser.zig");
const release_code_to_asm = @import("release_code_to_asm.zig");
const release_critical_edges = @import("release_critical_edges.zig");
const release_memory_layout = @import("release_memory_layout.zig");
const release_op_graph = @import("release_op_graph.zig");
const release_schedule = @import("release_schedule.zig");

pub const GenericBackendError = error{
    MissingSchedule,
};

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

    var layout_cache = release_op_graph.LayoutCache.init(allocator, program);
    defer layout_cache.deinit();

    var next_alloc_id = initial_alloc_id;
    for (program.functions) |function| {
        for (function.blocks) |block| {
            const scheduled = try release_op_graph.scheduleBlockSimple(
                allocator,
                program,
                function.name,
                block.name,
                config,
                next_alloc_id,
                &layout_cache,
            );
            next_alloc_id = scheduled.next_alloc_id;
            try blocks.append(allocator, scheduled);
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
    var normalized = try release_critical_edges.split(allocator, program);
    defer normalized.deinit();

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

pub fn emitDeploymentSimple(
    allocator: std.mem.Allocator,
    program: ir.Program,
) ![]const u8 {
    var normalized = try release_critical_edges.split(allocator, program);
    defer normalized.deinit();

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

/// Emit generic release bytecode for a whole program, choosing the deployment
/// path when an `init` function is present and the runtime-only path otherwise.
/// This is the single entry point shared by the `emit-release-generic` CLI
/// command and the `--release-generic` oracle so the two never drift.
pub fn emitRelease(allocator: std.mem.Allocator, program: ir.Program) ![]const u8 {
    if (findFunction(program, "init") != null) {
        return emitDeploymentSimple(allocator, program);
    }
    return emitSimple(allocator, program, "main", .{});
}

fn mergeLayout(
    override: release_code_to_asm.MemoryLayout,
    generated: release_code_to_asm.MemoryLayout,
) release_code_to_asm.MemoryLayout {
    return .{
        .alloc_start = if (override.alloc_start.len != 0) override.alloc_start else generated.alloc_start,
        .static_alloc_start = if (override.static_alloc_start.len != 0) override.static_alloc_start else generated.static_alloc_start,
        .switch_store = override.switch_store orelse generated.switch_store,
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
