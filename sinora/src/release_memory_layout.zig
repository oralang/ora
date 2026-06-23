//! Partial Zig port of Plank's `sir_static_memory_allocator::BumpAllocateAll`.
//!
//! Rust source of truth:
//! - vendor/plank/plankc/sir/crates/static-memory-allocator/src/bump_allocate_all.rs

const std = @import("std");

const ir = @import("ir.zig");
const release_code_to_asm = @import("release_code_to_asm.zig");
const release_op_graph = @import("release_op_graph.zig");
const release_schedule = @import("release_schedule.zig");

const word_size: u32 = 0x20;

pub const OwnedLayout = struct {
    allocator: std.mem.Allocator,
    alloc_start: []release_code_to_asm.AllocAddress,
    static_alloc_start: []release_code_to_asm.StaticAllocAddress,
    layout: release_code_to_asm.MemoryLayout,

    pub fn deinit(self: *OwnedLayout) void {
        self.allocator.free(self.alloc_start);
        self.allocator.free(self.static_alloc_start);
        self.* = undefined;
    }
};

pub fn generateSimple(
    allocator: std.mem.Allocator,
    program: ir.Program,
    entry_function_name: []const u8,
    schedules: []const release_op_graph.ScheduledBlock,
) !OwnedLayout {
    var collector = Collector{
        .allocator = allocator,
    };
    defer collector.deinit();

    var function_worklist: std.ArrayList([]const u8) = .empty;
    defer function_worklist.deinit(allocator);
    var seen_functions: std.ArrayList([]const u8) = .empty;
    defer seen_functions.deinit(allocator);
    var block_worklist: std.ArrayList(BlockRef) = .empty;
    defer block_worklist.deinit(allocator);
    var seen_blocks: std.ArrayList(BlockRef) = .empty;
    defer seen_blocks.deinit(allocator);

    try enqueueFunction(allocator, &seen_functions, &function_worklist, entry_function_name);
    while (function_worklist.items.len != 0) {
        const function_name = function_worklist.orderedRemove(function_worklist.items.len - 1);
        const function = findFunction(program, function_name) orelse return error.MissingFunction;
        if (function.blocks.len == 0) return error.MissingBlock;
        try enqueueBlock(allocator, &seen_blocks, &block_worklist, .{
            .function_name = function.name,
            .block_name = function.blocks[0].name,
        });

        while (block_worklist.items.len != 0) {
            const block_ref = block_worklist.orderedRemove(block_worklist.items.len - 1);
            const block_function = findFunction(program, block_ref.function_name) orelse return error.MissingFunction;
            const block = findBlock(block_function, block_ref.block_name) orelse return error.MissingBlock;

            for (block.instructions, 0..) |instruction, instruction_index| {
                const op_id = try release_code_to_asm.operationId(
                    program,
                    block_function.name,
                    block.name,
                    instruction_index,
                );
                try collector.collectInstruction(op_id, instruction);
                try collectInternalCallTarget(
                    allocator,
                    &seen_functions,
                    &function_worklist,
                    instruction,
                );
            }

            const schedule = findSchedule(schedules, block_function.name, block.name) orelse return error.MissingSchedule;
            for (schedule.ops) |stack_op| {
                try collector.collectStackOp(stack_op);
            }

            if (block.terminator == .switch_ and collector.switch_store == null) {
                collector.switch_store = collector.alloc(word_size);
            }

            try enqueueSuccessors(
                allocator,
                &seen_blocks,
                &block_worklist,
                block_function.name,
                block.terminator,
            );
        }
    }

    const alloc_start = try collector.alloc_start.toOwnedSlice(allocator);
    errdefer allocator.free(alloc_start);
    const static_alloc_start = try collector.static_alloc_start.toOwnedSlice(allocator);
    return .{
        .allocator = allocator,
        .alloc_start = alloc_start,
        .static_alloc_start = static_alloc_start,
        .layout = .{
            .alloc_start = alloc_start,
            .static_alloc_start = static_alloc_start,
            .switch_store = collector.switch_store,
            .dyn_free_pointer = if (collector.dyn_free_pointer_store) |store_slot| .{
                .store_slot = store_slot,
                .start_value = collector.next_free,
            } else null,
        },
    };
}

const BlockRef = struct {
    function_name: []const u8,
    block_name: []const u8,
};

const Collector = struct {
    allocator: std.mem.Allocator,
    next_free: u32 = 0,
    alloc_start: std.ArrayList(release_code_to_asm.AllocAddress) = .empty,
    static_alloc_start: std.ArrayList(release_code_to_asm.StaticAllocAddress) = .empty,
    dyn_free_pointer_store: ?u32 = null,
    switch_store: ?u32 = null,

    fn deinit(self: *Collector) void {
        self.alloc_start.deinit(self.allocator);
        self.static_alloc_start.deinit(self.allocator);
    }

    fn collectInstruction(self: *Collector, op_id: release_schedule.OpId, instruction: ir.Instruction) !void {
        if ((std.mem.eql(u8, instruction.mnemonic, "malloc") or
            std.mem.eql(u8, instruction.mnemonic, "mallocany") or
            std.mem.eql(u8, instruction.mnemonic, "freeptr")) and
            self.dyn_free_pointer_store == null)
        {
            self.dyn_free_pointer_store = self.alloc(word_size);
        }

        if (std.mem.eql(u8, instruction.mnemonic, "salloc") or
            std.mem.eql(u8, instruction.mnemonic, "sallocany"))
        {
            if (instruction.operands.len != 1) return error.UnsupportedSir;
            const bytes = parseU32(instruction.operands[0]) orelse return error.UnsupportedSir;
            try self.allocStaticInstruction(
                op_id,
                bytes,
                std.mem.eql(u8, instruction.mnemonic, "salloc"),
            );
        }
    }

    fn collectStackOp(self: *Collector, stack_op: release_schedule.StackOp) !void {
        switch (stack_op) {
            .store => |alloc_id| try self.allocStatic(alloc_id, word_size),
            .load => |alloc_id| {
                if (!self.hasAlloc(alloc_id)) return error.LoadFromUnallocatedSpill;
            },
            .swap, .dup, .pop, .op, .call_ret_push, .exchange => {},
        }
    }

    fn allocStatic(self: *Collector, alloc_id: release_schedule.AllocId, bytes: u32) !void {
        if (self.hasAlloc(alloc_id)) return;
        try self.alloc_start.append(self.allocator, .{
            .alloc = alloc_id,
            .address = self.alloc(bytes),
        });
    }

    fn allocStaticInstruction(
        self: *Collector,
        op_id: release_schedule.OpId,
        bytes: u32,
        needs_zeroing: bool,
    ) !void {
        if (self.hasStaticAlloc(op_id)) return;
        try self.static_alloc_start.append(self.allocator, .{
            .op = op_id,
            .address = self.alloc(bytes),
            .needs_zeroing = needs_zeroing,
        });
    }

    fn hasAlloc(self: Collector, alloc_id: release_schedule.AllocId) bool {
        for (self.alloc_start.items) |entry| {
            if (entry.alloc == alloc_id) return true;
        }
        return false;
    }

    fn hasStaticAlloc(self: Collector, op_id: release_schedule.OpId) bool {
        for (self.static_alloc_start.items) |entry| {
            if (entry.op == op_id) return true;
        }
        return false;
    }

    fn alloc(self: *Collector, bytes: u32) u32 {
        const address = self.next_free;
        self.next_free = std.math.add(u32, self.next_free, bytes) catch unreachable;
        return address;
    }
};

fn enqueueFunction(
    allocator: std.mem.Allocator,
    seen_functions: *std.ArrayList([]const u8),
    function_worklist: *std.ArrayList([]const u8),
    function_name: []const u8,
) !void {
    for (seen_functions.items) |seen| {
        if (std.mem.eql(u8, seen, function_name)) return;
    }
    try seen_functions.append(allocator, function_name);
    try function_worklist.append(allocator, function_name);
}

fn collectInternalCallTarget(
    allocator: std.mem.Allocator,
    seen_functions: *std.ArrayList([]const u8),
    function_worklist: *std.ArrayList([]const u8),
    instruction: ir.Instruction,
) !void {
    if (!std.mem.eql(u8, instruction.mnemonic, "icall")) return;
    if (instruction.operands.len == 0 or instruction.operands[0].len <= 1 or instruction.operands[0][0] != '@') {
        return error.UnsupportedSir;
    }
    try enqueueFunction(allocator, seen_functions, function_worklist, instruction.operands[0][1..]);
}

fn enqueueSuccessors(
    allocator: std.mem.Allocator,
    seen_blocks: *std.ArrayList(BlockRef),
    block_worklist: *std.ArrayList(BlockRef),
    function_name: []const u8,
    terminator: ir.Terminator,
) !void {
    switch (terminator) {
        .jump => |target| try enqueueBlock(allocator, seen_blocks, block_worklist, .{
            .function_name = function_name,
            .block_name = target,
        }),
        .branch => |branch| {
            try enqueueBlock(allocator, seen_blocks, block_worklist, .{
                .function_name = function_name,
                .block_name = branch.zero_target,
            });
            try enqueueBlock(allocator, seen_blocks, block_worklist, .{
                .function_name = function_name,
                .block_name = branch.non_zero_target,
            });
        },
        .switch_ => |switch_term| {
            for (switch_term.cases) |case| {
                try enqueueBlock(allocator, seen_blocks, block_worklist, .{
                    .function_name = function_name,
                    .block_name = case.target,
                });
            }
            if (switch_term.default_target.len != 0) {
                try enqueueBlock(allocator, seen_blocks, block_worklist, .{
                    .function_name = function_name,
                    .block_name = switch_term.default_target,
                });
            }
        },
        .return_, .revert, .stop, .invalid, .selfdestruct, .iret => {},
    }
}

fn enqueueBlock(
    allocator: std.mem.Allocator,
    seen_blocks: *std.ArrayList(BlockRef),
    block_worklist: *std.ArrayList(BlockRef),
    block_ref: BlockRef,
) !void {
    for (seen_blocks.items) |seen| {
        if (std.mem.eql(u8, seen.function_name, block_ref.function_name) and
            std.mem.eql(u8, seen.block_name, block_ref.block_name))
        {
            return;
        }
    }
    try seen_blocks.append(allocator, block_ref);
    try block_worklist.append(allocator, block_ref);
}

fn findSchedule(
    schedules: []const release_op_graph.ScheduledBlock,
    function_name: []const u8,
    block_name: []const u8,
) ?release_op_graph.ScheduledBlock {
    for (schedules) |schedule| {
        if (std.mem.eql(u8, schedule.function_name, function_name) and
            std.mem.eql(u8, schedule.block_name, block_name))
        {
            return schedule;
        }
    }
    return null;
}

fn findFunction(program: ir.Program, name: []const u8) ?ir.Function {
    for (program.functions) |function| {
        if (std.mem.eql(u8, function.name, name)) return function;
    }
    return null;
}

fn findBlock(function: ir.Function, name: []const u8) ?ir.Block {
    for (function.blocks) |block| {
        if (std.mem.eql(u8, block.name, name)) return block;
    }
    return null;
}

fn parseU32(text: []const u8) ?u32 {
    if (std.mem.startsWith(u8, text, "0x")) {
        return std.fmt.parseUnsigned(u32, text[2..], 16) catch null;
    }
    return std.fmt.parseUnsigned(u32, text, 10) catch null;
}

test "memory layout allocates scheduler spills" {
    var program = try parseProgram(
        \\fn main:
        \\    entry {
        \\        stop
        \\    }
    );
    defer program.deinit();

    const ops = [_]release_schedule.StackOp{
        .{ .store = 0 },
        .{ .load = 0 },
    };
    const schedules = [_]release_op_graph.ScheduledBlock{.{
        .function_name = "main",
        .block_name = "entry",
        .ops = &ops,
        .next_alloc_id = 1,
    }};

    var layout = try generateSimple(std.testing.allocator, program, "main", &schedules);
    defer layout.deinit();

    try std.testing.expectEqual(@as(usize, 1), layout.alloc_start.len);
    try std.testing.expectEqual(@as(release_schedule.AllocId, 0), layout.alloc_start[0].alloc);
    try std.testing.expectEqual(@as(u32, 0), layout.alloc_start[0].address);
}

test "memory layout allocates switch scratch slot" {
    var program = try parseProgram(
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

    const schedules = [_]release_op_graph.ScheduledBlock{
        .{ .function_name = "main", .block_name = "entry", .ops = &.{}, .next_alloc_id = 0 },
        .{ .function_name = "main", .block_name = "one", .ops = &.{}, .next_alloc_id = 0 },
        .{ .function_name = "main", .block_name = "other", .ops = &.{}, .next_alloc_id = 0 },
    };
    var layout = try generateSimple(std.testing.allocator, program, "main", &schedules);
    defer layout.deinit();

    try std.testing.expectEqual(@as(?u32, 0), layout.layout.switch_store);
}

test "memory layout allocates reachable static allocation" {
    var program = try parseProgram(
        \\fn main:
        \\    entry {
        \\        ptr = salloc 0x40
        \\        stop
        \\    }
    );
    defer program.deinit();

    const schedules = [_]release_op_graph.ScheduledBlock{
        .{ .function_name = "main", .block_name = "entry", .ops = &.{}, .next_alloc_id = 0 },
    };
    var layout = try generateSimple(std.testing.allocator, program, "main", &schedules);
    defer layout.deinit();

    const op_id = try release_code_to_asm.operationId(program, "main", "entry", 0);
    try std.testing.expectEqual(@as(usize, 1), layout.static_alloc_start.len);
    try std.testing.expectEqual(op_id, layout.static_alloc_start[0].op);
    try std.testing.expectEqual(@as(u32, 0), layout.static_alloc_start[0].address);
    try std.testing.expect(layout.static_alloc_start[0].needs_zeroing);
}

test "memory layout ignores unreachable dynamic allocation" {
    var program = try parseProgram(
        \\fn main:
        \\    entry {
        \\        stop
        \\    }
        \\
        \\    dead {
        \\        len = const 0x20
        \\        ptr = malloc len
        \\        stop
        \\    }
    );
    defer program.deinit();

    const schedules = [_]release_op_graph.ScheduledBlock{
        .{ .function_name = "main", .block_name = "entry", .ops = &.{}, .next_alloc_id = 0 },
    };
    var layout = try generateSimple(std.testing.allocator, program, "main", &schedules);
    defer layout.deinit();

    try std.testing.expectEqual(@as(?release_code_to_asm.FreePointer, null), layout.layout.dyn_free_pointer);
}

fn parseProgram(source: []const u8) !ir.Program {
    const diagnostics = @import("diagnostics.zig");
    const parser = @import("parser.zig");
    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator, source, &bag);
    errdefer program.deinit();
    try std.testing.expectEqual(@as(usize, 0), bag.items.items.len);
    return program;
}
