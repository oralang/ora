//! Zig port of Plank's generic release stack primitives and op scheduler.
//!
//! The graph builder and greedy final-stack shuffler live in
//! `release_op_graph.zig`; this file owns the EVM stack model, emitted stack
//! operations, spills, unspills, and single-operation scheduling. Do not add Ora
//! ABI/dispatcher concepts here.

const std = @import("std");

pub const ValueId = u32;
pub const AllocId = u32;
pub const OpId = u32;

pub const ScheduleConfig = struct {
    max_swap_depth: u8 = 16,
    max_dup_depth: u8 = 15,
    max_exchange_range: u8 = 16,
    exchange_cost: u8 = 9,

    pub const pre_amsterdam: ScheduleConfig = maxSwapNoExchange(16);

    pub fn maxSwapNoExchange(max_swap_depth: u8) ScheduleConfig {
        std.debug.assert(max_swap_depth > 0);
        return .{
            .max_swap_depth = max_swap_depth,
            .max_dup_depth = max_swap_depth - 1,
            .max_exchange_range = max_swap_depth,
            .exchange_cost = 9,
        };
    }
};

pub const StackOp = union(enum) {
    swap: u8,
    dup: u8,
    pop,
    op: OpId,
    call_ret_push: OpId,
    exchange: struct { n: u8, m: u8 },
    store: AllocId,
    load: AllocId,

    pub fn isValid(self: StackOp, config: ScheduleConfig) bool {
        return switch (self) {
            .swap => |depth| depth <= config.max_swap_depth,
            .dup => |depth| depth <= config.max_dup_depth,
            .exchange => |exchange| @as(u16, exchange.n) + @as(u16, exchange.m) <= config.max_exchange_range,
            .pop, .op, .call_ret_push, .store, .load => true,
        };
    }
};

pub const OpShape = struct {
    id: OpId,
    inputs_fifo: []const ValueId,
    outputs_fifo: []const ValueId = &.{},
    flippable: bool = false,
};

pub const EvmStack = struct {
    allocator: std.mem.Allocator,
    // The first element is the EVM top-of-stack. This matches Plank's scheduler
    // model and keeps depth calculations direct: DUP/SWAP depth is just an
    // index into this slice. Push/pop are therefore front operations; changing
    // that representation would touch every shuffler invariant, so local
    // cleanups keep the top-at-zero convention.
    items: std.ArrayList(ValueId) = .empty,

    pub fn init(allocator: std.mem.Allocator) EvmStack {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *EvmStack) void {
        self.items.deinit(self.allocator);
    }

    pub fn len(self: EvmStack) usize {
        return self.items.items.len;
    }

    pub fn isEmpty(self: EvmStack) bool {
        return self.items.items.len == 0;
    }

    pub fn push(self: *EvmStack, value: ValueId) !void {
        try self.items.insert(self.allocator, 0, value);
    }

    pub fn pop(self: *EvmStack) ?ValueId {
        if (self.items.items.len == 0) return null;
        return self.items.orderedRemove(0);
    }

    pub fn top(self: EvmStack) ?ValueId {
        if (self.items.items.len == 0) return null;
        return self.items.items[0];
    }

    pub fn fifo(self: EvmStack) []const ValueId {
        return self.items.items;
    }

    pub fn duplicate(self: *EvmStack, depth: u16) !void {
        if (depth >= self.items.items.len) return error.StackDepthOutOfRange;
        try self.push(self.items.items[depth]);
    }

    pub fn swapWithTop(self: *EvmStack, depth: u16) !void {
        try self.exchange(0, depth);
    }

    pub fn exchange(self: *EvmStack, n: u16, m: u16) !void {
        if (n >= self.items.items.len or m >= self.items.items.len) return error.StackDepthOutOfRange;
        std.mem.swap(ValueId, &self.items.items[n], &self.items.items[m]);
    }

    pub fn findFirst(self: EvmStack, target: ValueId) ?u16 {
        for (self.items.items, 0..) |value, index| {
            if (value == target) return @intCast(index);
        }
        return null;
    }
};

pub const TrackedStack = struct {
    allocator: std.mem.Allocator,
    next_alloc_id: AllocId,
    ops: std.ArrayList(StackOp) = .empty,
    spilled: std.ArrayList(Spill) = .empty,
    stack: EvmStack,

    const Spill = struct {
        value: ValueId,
        alloc: AllocId,
    };

    pub fn init(allocator: std.mem.Allocator, next_alloc_id: AllocId) TrackedStack {
        return .{
            .allocator = allocator,
            .next_alloc_id = next_alloc_id,
            .stack = EvmStack.init(allocator),
        };
    }

    pub fn deinit(self: *TrackedStack) void {
        self.ops.deinit(self.allocator);
        self.spilled.deinit(self.allocator);
        self.stack.deinit();
    }

    pub fn pushInput(self: *TrackedStack, value: ValueId) !void {
        try self.stack.push(value);
    }

    pub fn pop(self: *TrackedStack) !void {
        _ = self.stack.pop() orelse return error.StackUnderflow;
        try self.ops.append(self.allocator, .pop);
    }

    pub fn swap(self: *TrackedStack, depth: u8) !void {
        if (depth == 0) return error.StackDepthOutOfRange;
        try self.stack.swapWithTop(depth);
        try self.ops.append(self.allocator, .{ .swap = depth });
    }

    pub fn dup(self: *TrackedStack, depth: u8) !void {
        try self.stack.duplicate(depth);
        try self.ops.append(self.allocator, .{ .dup = depth });
    }

    pub fn spillTop(self: *TrackedStack) !AllocId {
        const target = self.stack.pop() orelse return error.StackUnderflow;
        const alloc = self.next_alloc_id;
        self.next_alloc_id += 1;
        try self.ops.append(self.allocator, .{ .store = alloc });
        try self.spilled.append(self.allocator, .{
            .value = target,
            .alloc = alloc,
        });
        return alloc;
    }

    pub fn getSpilled(self: TrackedStack, target: ValueId) ?AllocId {
        var index = self.spilled.items.len;
        while (index > 0) {
            index -= 1;
            const spill = self.spilled.items[index];
            if (spill.value == target) return spill.alloc;
        }
        return null;
    }

    pub fn unspill(self: *TrackedStack, target: ValueId) !void {
        const alloc = self.getSpilled(target) orelse return error.ValueNotSpilled;
        try self.stack.push(target);
        try self.ops.append(self.allocator, .{ .load = alloc });
    }
};

pub fn loadInputForUse(
    allocator: std.mem.Allocator,
    stack: *TrackedStack,
    config: ScheduleConfig,
    input: ValueId,
) !void {
    if (stack.stack.findFirst(input)) |depth| {
        if (depth <= config.max_dup_depth) {
            try stack.dup(@intCast(depth));
            return;
        }

        if (stack.getSpilled(input)) |alloc| {
            try stack.stack.push(input);
            try stack.ops.append(stack.allocator, .{ .load = alloc });
            return;
        }

        const delta_to_max = depth - config.max_dup_depth;
        const in_the_way = try allocator.alloc(ValueId, delta_to_max);
        defer allocator.free(in_the_way);
        @memcpy(in_the_way, stack.stack.fifo()[0..delta_to_max]);

        var spill_count: usize = 0;
        while (spill_count < delta_to_max) : (spill_count += 1) {
            const top = stack.stack.top() orelse return error.StackUnderflow;
            if (stack.getSpilled(top) != null) {
                try stack.pop();
            } else {
                _ = try stack.spillTop();
            }
        }

        try stack.dup(config.max_dup_depth);
        _ = try stack.spillTop();

        var restore_index = in_the_way.len;
        while (restore_index > 0) {
            restore_index -= 1;
            try stack.unspill(in_the_way[restore_index]);
        }

        try stack.unspill(input);
        return;
    }

    if (stack.getSpilled(input)) |alloc| {
        try stack.stack.push(input);
        try stack.ops.append(stack.allocator, .{ .load = alloc });
        return;
    }

    return error.ValueNotOnStack;
}

pub fn scheduleOp(
    allocator: std.mem.Allocator,
    stack: *TrackedStack,
    config: ScheduleConfig,
    op: OpShape,
) !void {
    var input_index = op.inputs_fifo.len;
    while (input_index > 0) {
        input_index -= 1;
        try loadInputForUse(allocator, stack, config, op.inputs_fifo[input_index]);
    }

    try completeOp(stack, op);
}

pub fn completeOp(stack: *TrackedStack, op: OpShape) !void {
    var flipping = false;
    for (op.inputs_fifo, 0..) |target, index| {
        const actual = stack.stack.pop() orelse return error.StackUnderflow;
        const correct = if (op.flippable and index == 0 and op.inputs_fifo.len >= 2 and actual == op.inputs_fifo[1]) blk: {
            flipping = true;
            break :blk true;
        } else if (op.flippable and flipping and index == 1)
            actual == op.inputs_fifo[0]
        else
            actual == target;
        if (!correct) return error.IncorrectSchedule;
    }

    try stack.ops.append(stack.allocator, .{ .op = op.id });

    var output_index = op.outputs_fifo.len;
    while (output_index > 0) {
        output_index -= 1;
        try stack.stack.push(op.outputs_fifo[output_index]);
    }
}

fn expectStackOps(actual: []const StackOp, expected: []const StackOp) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |expected_op, actual_op| {
        try std.testing.expectEqual(expected_op, actual_op);
    }
}

test "tracked stack emits dup for reachable input" {
    var stack = TrackedStack.init(std.testing.allocator, 0);
    defer stack.deinit();
    try stack.pushInput(0);
    try stack.pushInput(1);
    try stack.pushInput(2);

    try loadInputForUse(std.testing.allocator, &stack, .{}, 0);

    try expectStackOps(stack.ops.items, &.{.{ .dup = 2 }});
    try std.testing.expectEqualSlices(ValueId, &.{ 0, 2, 1, 0 }, stack.stack.fifo());
}

test "tracked stack spills values blocking a deep input" {
    var stack = TrackedStack.init(std.testing.allocator, 0);
    defer stack.deinit();
    try stack.pushInput(0);
    try stack.pushInput(1);
    try stack.pushInput(2);
    try stack.pushInput(3);
    try stack.pushInput(4);

    try loadInputForUse(std.testing.allocator, &stack, .{
        .max_swap_depth = 3,
        .max_dup_depth = 2,
        .max_exchange_range = 3,
    }, 0);

    try expectStackOps(stack.ops.items, &.{
        .{ .store = 0 },
        .{ .store = 1 },
        .{ .dup = 2 },
        .{ .store = 2 },
        .{ .load = 1 },
        .{ .load = 0 },
        .{ .load = 2 },
    });
    try std.testing.expectEqualSlices(ValueId, &.{ 0, 4, 3, 2, 1, 0 }, stack.stack.fifo());
}

test "scheduler loads inputs and completes non-commutative operation" {
    var stack = TrackedStack.init(std.testing.allocator, 0);
    defer stack.deinit();
    try stack.pushInput(0);
    try stack.pushInput(1);

    try scheduleOp(std.testing.allocator, &stack, .{}, .{
        .id = 7,
        .inputs_fifo = &.{ 0, 1 },
        .outputs_fifo = &.{2},
    });

    try expectStackOps(stack.ops.items, &.{
        .{ .dup = 0 },
        .{ .dup = 2 },
        .{ .op = 7 },
    });
    try std.testing.expectEqualSlices(ValueId, &.{ 2, 1, 0 }, stack.stack.fifo());
}

test "scheduler permits flipped operands for flippable operation" {
    var stack = TrackedStack.init(std.testing.allocator, 0);
    defer stack.deinit();
    try stack.pushInput(0);
    try stack.pushInput(1);

    try completeOp(&stack, .{
        .id = 9,
        .inputs_fifo = &.{ 0, 1 },
        .outputs_fifo = &.{2},
        .flippable = true,
    });

    try expectStackOps(stack.ops.items, &.{.{ .op = 9 }});
    try std.testing.expectEqualSlices(ValueId, &.{2}, stack.stack.fifo());
}

test "scheduler spills deep input before operation use" {
    var stack = TrackedStack.init(std.testing.allocator, 0);
    defer stack.deinit();
    try stack.pushInput(0);
    try stack.pushInput(1);
    try stack.pushInput(2);
    try stack.pushInput(3);
    try stack.pushInput(4);

    try scheduleOp(std.testing.allocator, &stack, .{
        .max_swap_depth = 3,
        .max_dup_depth = 2,
        .max_exchange_range = 3,
    }, .{
        .id = 11,
        .inputs_fifo = &.{0},
        .outputs_fifo = &.{5},
    });

    try expectStackOps(stack.ops.items, &.{
        .{ .store = 0 },
        .{ .store = 1 },
        .{ .dup = 2 },
        .{ .store = 2 },
        .{ .load = 1 },
        .{ .load = 0 },
        .{ .load = 2 },
        .{ .op = 11 },
    });
    try std.testing.expectEqualSlices(ValueId, &.{ 5, 4, 3, 2, 1, 0 }, stack.stack.fifo());
}
