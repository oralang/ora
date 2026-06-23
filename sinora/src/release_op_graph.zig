//! Partial Zig port of Plank's release stack-scheduling op graph.
//!
//! Rust source of truth:
//! - vendor/plank/plankc/sir/crates/stack-scheduling/src/op_graph/mod.rs
//! - vendor/plank/plankc/sir/crates/stack-scheduling/src/op_graph/builder.rs
//! - vendor/plank/plankc/sir/crates/stack-scheduling/src/state.rs

const std = @import("std");
const diagnostics = @import("diagnostics.zig");
const ir = @import("ir.zig");
const ops = @import("ops.zig");
const parser = @import("parser.zig");
const release_code_to_asm = @import("release_code_to_asm.zig");
const release_schedule = @import("release_schedule.zig");

pub const OpNodeId = u32;
pub const ValueNodeId = release_schedule.ValueId;
pub const OperationId = release_schedule.OpId;
pub const BitsetWord = u8;

pub const ScheduleGraphError = error{
    IncompleteGraphSchedule,
    ShuffleLimitExceeded,
    MissingLocal,
    MissingFunction,
    MissingBlock,
    MissingInstruction,
    UnsupportedSir,
};

const LivenessError = ScheduleGraphError || std.mem.Allocator.Error;

pub const OpNodeKind = union(enum) {
    flippable: OperationId,
    ret_dest_push: OperationId,
    normal: OperationId,

    pub fn operationId(self: OpNodeKind) OperationId {
        return switch (self) {
            .flippable => |id| id,
            .ret_dest_push => |id| id,
            .normal => |id| id,
        };
    }
};

const StoredOpView = struct {
    inputs_outputs_start: usize,
    input_count: u32,
    kind: OpNodeKind,
};

const ValueState = struct {
    source: ?OpNodeId,
    consumers: std.ArrayList(BitsetWord),

    fn init(allocator: std.mem.Allocator, source: ?OpNodeId, estimated_words: usize) !ValueState {
        var consumers: std.ArrayList(BitsetWord) = .empty;
        try consumers.ensureTotalCapacity(allocator, estimated_words);
        return .{ .source = source, .consumers = consumers };
    }

    fn deinit(self: *ValueState, allocator: std.mem.Allocator) void {
        self.consumers.deinit(allocator);
    }
};

pub const OpSet = struct {
    words: []const BitsetWord,
    total_ops: u32,

    pub fn init(words: []const BitsetWord, total_ops: u32) OpSet {
        std.debug.assert(words.len == bitWordsPerSet(total_ops));
        return .{ .words = words, .total_ops = total_ops };
    }

    pub fn countMembers(self: OpSet) u32 {
        var count: u32 = 0;
        for (self.words) |word| count += @popCount(word);
        return count;
    }

    pub fn intersectCount(self: OpSet, other: OpSet) u32 {
        std.debug.assert(self.total_ops == other.total_ops);
        var count: u32 = 0;
        for (self.words, other.words) |lhs, rhs| count += @popCount(lhs & rhs);
        return count;
    }

    pub fn isSuper(self: OpSet, other: OpSet) bool {
        std.debug.assert(self.total_ops == other.total_ops);
        for (self.words, other.words) |super_word, sub_word| {
            if ((super_word & sub_word) != sub_word) return false;
        }
        return true;
    }

    pub fn contains(self: OpSet, op: OpNodeId) bool {
        std.debug.assert(op < self.total_ops);
        const word_index: usize = @intCast(op / @bitSizeOf(BitsetWord));
        const word_shift: u3 = @intCast(op % @bitSizeOf(BitsetWord));
        return (self.words[word_index] & (@as(BitsetWord, 1) << word_shift)) != 0;
    }
};

pub const OpSetMut = struct {
    words: []BitsetWord,
    total_ops: u32,

    pub fn init(words: []BitsetWord, total_ops: u32) OpSetMut {
        std.debug.assert(words.len == bitWordsPerSet(total_ops));
        return .{ .words = words, .total_ops = total_ops };
    }

    pub fn asConst(self: OpSetMut) OpSet {
        return .{ .words = self.words, .total_ops = self.total_ops };
    }

    pub fn contains(self: OpSetMut, op: OpNodeId) bool {
        return self.asConst().contains(op);
    }

    pub fn clear(self: *OpSetMut) void {
        @memset(self.words, 0);
    }

    pub fn add(self: *OpSetMut, op: OpNodeId) void {
        std.debug.assert(op < self.total_ops);
        const word_index: usize = @intCast(op / @bitSizeOf(BitsetWord));
        const word_shift: u3 = @intCast(op % @bitSizeOf(BitsetWord));
        self.words[word_index] |= @as(BitsetWord, 1) << word_shift;
    }

    pub fn first(self: OpSetMut) ?OpNodeId {
        var op: OpNodeId = 0;
        while (op < self.total_ops) : (op += 1) {
            if (self.contains(op)) return op;
        }
        return null;
    }
};

pub const OpGraph = struct {
    allocator: std.mem.Allocator,
    total_ops_value: u32,
    total_values_value: u32,
    inputs_end: ValueNodeId,
    end_stack_fifo_start: usize,
    input_values: []ValueNodeId,
    values_arena: []ValueNodeId,
    operations: []StoredOpView,
    bit_sets_arena: []BitsetWord,

    pub fn deinit(self: *OpGraph) void {
        self.allocator.free(self.input_values);
        self.allocator.free(self.values_arena);
        self.allocator.free(self.operations);
        self.allocator.free(self.bit_sets_arena);
        self.* = undefined;
    }

    pub fn totalOps(self: OpGraph) u32 {
        return self.total_ops_value;
    }

    pub fn totalValues(self: OpGraph) u32 {
        return self.total_values_value;
    }

    pub fn inputValuesFifo(self: OpGraph) []const ValueNodeId {
        return self.input_values;
    }

    pub fn outputValuesFifo(self: OpGraph) []const ValueNodeId {
        return self.values_arena[self.end_stack_fifo_start..];
    }

    pub fn isInput(self: OpGraph, id: ValueNodeId) bool {
        return id < self.inputs_end;
    }

    pub fn wordsPerSet(self: OpGraph) usize {
        return bitWordsPerSet(self.total_ops_value);
    }

    pub fn usesRemaining(self: OpGraph, completed: OpSet, value: ValueNodeId) u32 {
        const consumers = self.getConsumers(value);
        return consumers.countMembers() - consumers.intersectCount(completed);
    }

    pub fn getPredecessors(self: OpGraph, id: OpNodeId) OpSet {
        std.debug.assert(id < self.total_ops_value);
        return self.getBitSet(@intCast(id));
    }

    pub fn getConsumers(self: OpGraph, id: ValueNodeId) OpSet {
        std.debug.assert(id < self.total_values_value);
        return self.getBitSet(@as(usize, self.total_ops_value) + @as(usize, id));
    }

    pub fn getOp(self: OpGraph, id: OpNodeId) OpView {
        std.debug.assert(id < self.total_ops_value);
        const stored = self.operations[id];
        const op_values_end = if (id + 1 < self.total_ops_value)
            self.operations[id + 1].inputs_outputs_start
        else
            self.end_stack_fifo_start;
        const op_values = self.values_arena[stored.inputs_outputs_start..op_values_end];
        const input_count: usize = @intCast(stored.input_count);
        return .{
            .inputs_fifo = op_values[0..input_count],
            .outputs_fifo = op_values[input_count..],
            .predecessors = self.getPredecessors(id),
            .kind = stored.kind,
        };
    }

    fn getBitSet(self: OpGraph, bit_set_index: usize) OpSet {
        const set_words = self.wordsPerSet();
        const start = bit_set_index * set_words;
        return OpSet.init(self.bit_sets_arena[start..][0..set_words], self.total_ops_value);
    }
};

pub const OpView = struct {
    inputs_fifo: []const ValueNodeId,
    outputs_fifo: []const ValueNodeId,
    predecessors: OpSet,
    kind: OpNodeKind,
};

const LocalValue = struct {
    name: []const u8,
    value: ValueNodeId,
};

const LayoutMember = union(enum) {
    return_dest,
    input_output: u32,
    local: []const u8,
};

const BlockLayout = struct {
    block_name: []const u8,
    input: []const LayoutMember,
    output: []const LayoutMember,
};

const FunctionLayouts = struct {
    allocator: std.mem.Allocator,
    blocks: []BlockLayout,

    fn deinit(self: *FunctionLayouts) void {
        for (self.blocks) |block| {
            self.allocator.free(block.input);
            self.allocator.free(block.output);
        }
        self.allocator.free(self.blocks);
        self.* = undefined;
    }

    fn forBlock(self: FunctionLayouts, block_name: []const u8) ?BlockLayout {
        for (self.blocks) |block| {
            if (std.mem.eql(u8, block.block_name, block_name)) return block;
        }
        return null;
    }
};

/// Per-emit memoization of `buildFunctionLayouts`. Scheduling a function walks
/// every block, and each block previously rebuilt the whole function's layouts
/// (including the liveness fixpoint and, for icalls, every callee's liveness),
/// giving O(blocks^3)+ behavior on large functions (kitchen_sink ~84s). Caching
/// one layout set per function makes scheduling linear in blocks. Each
/// `FunctionLayouts` is heap-allocated so returned pointers stay stable as more
/// functions are cached.
pub const LayoutCache = struct {
    allocator: std.mem.Allocator,
    program: ir.Program,
    names: std.ArrayList([]const u8) = .empty,
    layouts: std.ArrayList(*FunctionLayouts) = .empty,

    pub fn init(allocator: std.mem.Allocator, program: ir.Program) LayoutCache {
        return .{ .allocator = allocator, .program = program };
    }

    pub fn deinit(self: *LayoutCache) void {
        for (self.layouts.items) |layouts| {
            layouts.deinit();
            self.allocator.destroy(layouts);
        }
        self.layouts.deinit(self.allocator);
        self.names.deinit(self.allocator);
        self.* = undefined;
    }

    fn getOrBuild(self: *LayoutCache, function_name: []const u8) !*const FunctionLayouts {
        for (self.names.items, self.layouts.items) |name, layouts| {
            if (std.mem.eql(u8, name, function_name)) return layouts;
        }
        const function = findFunction(self.program, function_name) orelse return ScheduleGraphError.MissingFunction;
        try self.names.ensureUnusedCapacity(self.allocator, 1);
        try self.layouts.ensureUnusedCapacity(self.allocator, 1);
        const layouts = try self.allocator.create(FunctionLayouts);
        errdefer self.allocator.destroy(layouts);
        layouts.* = try buildFunctionLayouts(self.allocator, self.program, function);
        self.names.appendAssumeCapacity(function_name);
        self.layouts.appendAssumeCapacity(layouts);
        return layouts;
    }
};

pub fn buildBlockGraphSimple(
    allocator: std.mem.Allocator,
    program: ir.Program,
    function_name: []const u8,
    block_name: []const u8,
    cache: ?*LayoutCache,
) !OpGraph {
    const function = findFunction(program, function_name) orelse return ScheduleGraphError.MissingFunction;
    const block = findBlock(function, block_name) orelse return ScheduleGraphError.MissingBlock;
    var owned_layouts: ?FunctionLayouts = null;
    defer if (owned_layouts) |*owned| owned.deinit();
    const layouts: *const FunctionLayouts = if (cache) |layout_cache|
        try layout_cache.getOrBuild(function_name)
    else owned_blk: {
        owned_layouts = try buildFunctionLayouts(allocator, program, function);
        break :owned_blk &owned_layouts.?;
    };
    const block_layout = layouts.forBlock(block.name) orelse return ScheduleGraphError.MissingBlock;
    const estimated_ops = @max(block.instructions.len + 1, 1);
    const estimated_values = estimated_ops * 2 + block_layout.input.len + block_layout.output.len + 1;
    var builder = try OpGraphBuilder.init(allocator, estimated_ops, estimated_values);
    errdefer builder.deinit();

    var locals: std.ArrayList(LocalValue) = .empty;
    defer locals.deinit(allocator);

    var return_dest_value: ?ValueNodeId = null;
    for (block_layout.input) |member| {
        const value = try builder.pushInputValue();
        switch (member) {
            .return_dest => return_dest_value = value,
            .input_output => |position| {
                if (position >= block.inputs.len) return ScheduleGraphError.UnsupportedSir;
                try putLocal(allocator, &locals, block.inputs[position], value);
            },
            .local => |name| try putLocal(allocator, &locals, name, value),
        }
    }
    builder.endInputsBeginOps();

    var previous: ?OpNodeId = null;
    for (block.instructions, 0..) |instruction, instruction_index| {
        const op_id = try operationId(program, function_name, block_name, instruction_index);
        if (std.mem.eql(u8, instruction.mnemonic, "icall")) {
            if (instruction.operands.len == 0) return ScheduleGraphError.UnsupportedSir;
            const callee_name = try operandFunctionName(instruction.operands[0]);
            const callee = findFunction(program, callee_name) orelse return ScheduleGraphError.MissingFunction;
            const callee_entry = functionEntryBlock(callee) orelse return ScheduleGraphError.MissingBlock;
            var owned_callee_layouts: ?FunctionLayouts = null;
            defer if (owned_callee_layouts) |*owned| owned.deinit();
            const callee_entry_layout = if (cache) |layout_cache|
                ((try layout_cache.getOrBuild(callee_name)).forBlock(callee_entry.name) orelse return ScheduleGraphError.MissingBlock)
            else callee_blk: {
                owned_callee_layouts = try buildFunctionLayouts(allocator, program, callee);
                break :callee_blk (owned_callee_layouts.?.forBlock(callee_entry.name) orelse return ScheduleGraphError.MissingBlock);
            };

            var call_return_dest: ?ValueNodeId = null;
            if (layoutContains(callee_entry_layout.input, .return_dest)) {
                var ret_dest_push = try builder.beginOp(.{ .ret_dest_push = op_id });
                call_return_dest = try ret_dest_push.addOutput();
            }

            var op_builder = try builder.beginOp(if (isFlippable(instruction.mnemonic))
                .{ .flippable = op_id }
            else
                .{ .normal = op_id });
            if (previous) |predecessor| try op_builder.addPredecessor(predecessor);
            previous = op_builder.id();

            for (callee_entry_layout.input) |member| {
                switch (member) {
                    .return_dest => try op_builder.addInput(call_return_dest orelse return ScheduleGraphError.UnsupportedSir),
                    .input_output => |position| {
                        const operand_index = @as(usize, position) + 1;
                        if (operand_index >= instruction.operands.len) return ScheduleGraphError.UnsupportedSir;
                        try op_builder.addInput(try localValue(locals.items, instruction.operands[operand_index]));
                    },
                    .local => return ScheduleGraphError.UnsupportedSir,
                }
            }
            for (instruction.results) |result| {
                const value = try op_builder.addOutput();
                try putLocal(allocator, &locals, result, value);
            }
            continue;
        }

        var op_builder = try builder.beginOp(if (isFlippable(instruction.mnemonic))
            .{ .flippable = op_id }
        else
            .{ .normal = op_id });
        if (previous) |predecessor| try op_builder.addPredecessor(predecessor);
        previous = op_builder.id();

        for (instruction.operands) |operand| {
            if (isNonValueOperand(instruction.mnemonic, operand)) continue;
            try op_builder.addInput(try localValue(locals.items, operand));
        }

        for (instruction.results) |result| {
            const value = try op_builder.addOutput();
            try putLocal(allocator, &locals, result, value);
        }
    }

    builder.endOpsBeginEndStack();
    switch (block.terminator) {
        .branch => |branch| try builder.pushEndStackValue(try localValue(locals.items, branch.condition)),
        .switch_ => |switch_term| try builder.pushEndStackValue(try localValue(locals.items, switch_term.selector)),
        .iret => {
            try builder.pushEndStackValue(return_dest_value orelse return ScheduleGraphError.UnsupportedSir);
        },
        .return_ => |ret| {
            try builder.pushEndStackValue(try localValue(locals.items, ret.ptr));
            try builder.pushEndStackValue(try localValue(locals.items, ret.len));
        },
        .revert => |revert| {
            try builder.pushEndStackValue(try localValue(locals.items, revert.ptr));
            try builder.pushEndStackValue(try localValue(locals.items, revert.len));
        },
        .selfdestruct => |beneficiary| try builder.pushEndStackValue(try localValue(locals.items, beneficiary)),
        .jump, .stop, .invalid => {},
    }
    for (block_layout.output) |member| {
        try builder.pushEndStackValue(try layoutMemberValue(member, block, locals.items, return_dest_value));
    }

    return builder.finish();
}

pub fn scheduleBlockSimple(
    allocator: std.mem.Allocator,
    program: ir.Program,
    function_name: []const u8,
    block_name: []const u8,
    config: release_schedule.ScheduleConfig,
    next_alloc_id: release_schedule.AllocId,
    cache: ?*LayoutCache,
) !ScheduledBlock {
    const function = findFunction(program, function_name) orelse return ScheduleGraphError.MissingFunction;
    const block = findBlock(function, block_name) orelse return ScheduleGraphError.MissingBlock;
    var graph = try buildBlockGraphSimple(allocator, program, function_name, block_name, cache);
    defer graph.deinit();

    var stack = release_schedule.TrackedStack.init(allocator, next_alloc_id);
    defer stack.deinit();
    try pushGraphInputs(&stack, graph);
    try scheduleGraphOperationsIntoStackMode(
        allocator,
        &stack,
        config,
        graph,
        if (isLastOpTerminatingControl(block.terminator)) .terminal_use else .exact,
    );
    return .{
        .function_name = function_name,
        .block_name = block_name,
        .ops = try stack.ops.toOwnedSlice(allocator),
        .next_alloc_id = stack.next_alloc_id,
    };
}

/// Dump, per (function, block): the bundled input layout, the output layout
/// (the greedy-shuffle target), and the scheduled stack ops. Used to localize
/// remaining byte mismatches against Plank's disassembly without guessing.
pub fn traceProgram(
    writer: anytype,
    allocator: std.mem.Allocator,
    program: ir.Program,
    config: release_schedule.ScheduleConfig,
) !void {
    var layout_cache = LayoutCache.init(allocator, program);
    defer layout_cache.deinit();
    const cache: ?*LayoutCache = &layout_cache;
    for (program.functions) |function| {
        const layouts = try layout_cache.getOrBuild(function.name);
        try writer.print("fn {s}:\n", .{function.name});
        var next_alloc_id: release_schedule.AllocId = 0;
        for (function.blocks) |block| {
            const layout = layouts.forBlock(block.name) orelse continue;
            try writer.print("  block {s}\n    in :", .{block.name});
            try traceLayout(writer, layout.input);
            try writer.writeAll("\n    out:");
            try traceLayout(writer, layout.output);
            const scheduled = try scheduleBlockSimple(allocator, program, function.name, block.name, config, next_alloc_id, cache);
            defer scheduled.deinit(allocator);
            next_alloc_id = scheduled.next_alloc_id;
            try writer.writeAll("\n    ops:");
            for (scheduled.ops) |stack_op| {
                try writer.writeByte(' ');
                try traceStackOp(writer, stack_op);
            }
            try writer.writeByte('\n');
        }
    }
}

fn traceLayout(writer: anytype, members: []const LayoutMember) !void {
    for (members) |member| {
        try writer.writeByte(' ');
        switch (member) {
            .return_dest => try writer.writeAll("ret_dest"),
            .input_output => |position| try writer.print("io{d}", .{position}),
            .local => |name| try writer.print("{s}", .{name}),
        }
    }
}

fn traceStackOp(writer: anytype, stack_op: release_schedule.StackOp) !void {
    switch (stack_op) {
        .swap => |depth| try writer.print("swap{d}", .{depth}),
        .dup => |depth| try writer.print("dup{d}", .{depth}),
        .pop => try writer.writeAll("pop"),
        .op => |id| try writer.print("op#{d}", .{id}),
        .call_ret_push => |id| try writer.print("callret#{d}", .{id}),
        .exchange => |exchange| try writer.print("exch({d},{d})", .{ exchange.n, exchange.m }),
        .store => |alloc_id| try writer.print("store@{d}", .{alloc_id}),
        .load => |alloc_id| try writer.print("load@{d}", .{alloc_id}),
    }
}

pub const ScheduledBlock = struct {
    function_name: []const u8,
    block_name: []const u8,
    ops: []const release_schedule.StackOp,
    next_alloc_id: release_schedule.AllocId,

    pub fn deinit(self: ScheduledBlock, allocator: std.mem.Allocator) void {
        allocator.free(self.ops);
    }
};

pub const OpGraphBuilder = struct {
    allocator: std.mem.Allocator,
    op_predecessors: std.ArrayList(std.ArrayList(BitsetWord)) = .empty,
    operations: std.ArrayList(StoredOpView) = .empty,
    values: std.ArrayList(ValueState) = .empty,
    values_arena: std.ArrayList(ValueNodeId) = .empty,
    estimated_ops: usize,
    inputs_end: ?ValueNodeId = null,
    end_stack_fifo_start: ?usize = null,

    pub fn init(
        allocator: std.mem.Allocator,
        estimated_ops: usize,
        estimated_values: usize,
    ) !OpGraphBuilder {
        var self = OpGraphBuilder{
            .allocator = allocator,
            .estimated_ops = estimated_ops,
        };
        errdefer self.deinit();

        try self.op_predecessors.ensureTotalCapacity(allocator, estimated_ops);
        try self.operations.ensureTotalCapacity(allocator, estimated_ops);
        try self.values.ensureTotalCapacity(allocator, estimated_values);
        try self.values_arena.ensureTotalCapacity(allocator, estimated_ops * 4);

        return self;
    }

    pub fn deinit(self: *OpGraphBuilder) void {
        for (self.op_predecessors.items) |*predecessors| predecessors.deinit(self.allocator);
        self.op_predecessors.deinit(self.allocator);
        self.operations.deinit(self.allocator);
        for (self.values.items) |*value| value.deinit(self.allocator);
        self.values.deinit(self.allocator);
        self.values_arena.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn pushInputValue(self: *OpGraphBuilder) !ValueNodeId {
        std.debug.assert(self.inputs_end == null);
        const value: ValueNodeId = @intCast(self.values.items.len);
        try self.values.append(self.allocator, try ValueState.init(self.allocator, null, self.estimatedWords()));
        return value;
    }

    pub fn endInputsBeginOps(self: *OpGraphBuilder) void {
        std.debug.assert(self.inputs_end == null);
        self.inputs_end = @intCast(self.values.items.len);
    }

    pub fn beginOp(self: *OpGraphBuilder, kind: OpNodeKind) !OpBuilder {
        std.debug.assert(self.inputs_end != null);
        std.debug.assert(self.end_stack_fifo_start == null);
        const op: OpNodeId = @intCast(self.operations.items.len);
        try self.operations.append(self.allocator, .{
            .inputs_outputs_start = self.values_arena.items.len,
            .input_count = 0,
            .kind = kind,
        });
        var predecessors: std.ArrayList(BitsetWord) = .empty;
        try predecessors.ensureTotalCapacity(self.allocator, self.estimatedWords());
        try self.op_predecessors.append(self.allocator, predecessors);
        return .{ .graph = self, .op = op };
    }

    pub fn endOpsBeginEndStack(self: *OpGraphBuilder) void {
        std.debug.assert(self.inputs_end != null);
        std.debug.assert(self.end_stack_fifo_start == null);
        self.end_stack_fifo_start = self.values_arena.items.len;
    }

    pub fn pushEndStackValue(self: *OpGraphBuilder, value: ValueNodeId) !void {
        std.debug.assert(self.end_stack_fifo_start != null);
        std.debug.assert(value < self.values.items.len);
        try self.values_arena.append(self.allocator, value);
    }

    pub fn finish(self: *OpGraphBuilder) !OpGraph {
        if (self.inputs_end == null) self.endInputsBeginOps();
        const inputs_end = self.inputs_end.?;
        const end_stack_fifo_start = self.end_stack_fifo_start orelse self.values_arena.items.len;
        const total_ops: u32 = @intCast(self.operations.items.len);
        const total_values: u32 = @intCast(self.values.items.len);
        const set_words = bitWordsPerSet(total_ops);
        const bit_set_count = @as(usize, total_ops) + @as(usize, total_values);
        const bit_sets_arena = try self.allocator.alloc(BitsetWord, set_words * bit_set_count);
        errdefer self.allocator.free(bit_sets_arena);
        @memset(bit_sets_arena, 0);

        var offset: usize = 0;
        for (self.op_predecessors.items) |predecessors| {
            copyBitset(bit_sets_arena[offset..][0..set_words], predecessors.items);
            offset += set_words;
        }
        for (self.values.items) |value| {
            copyBitset(bit_sets_arena[offset..][0..set_words], value.consumers.items);
            offset += set_words;
        }

        const input_values = try self.allocator.alloc(ValueNodeId, @intCast(inputs_end));
        errdefer self.allocator.free(input_values);
        for (input_values, 0..) |*value, index| value.* = @intCast(index);

        const values_arena = try self.values_arena.toOwnedSlice(self.allocator);
        errdefer self.allocator.free(values_arena);
        const operations = try self.operations.toOwnedSlice(self.allocator);
        errdefer self.allocator.free(operations);

        for (self.op_predecessors.items) |*predecessors| predecessors.deinit(self.allocator);
        self.op_predecessors.clearAndFree(self.allocator);
        for (self.values.items) |*value| value.deinit(self.allocator);
        self.values.clearAndFree(self.allocator);

        return .{
            .allocator = self.allocator,
            .total_ops_value = total_ops,
            .total_values_value = total_values,
            .inputs_end = inputs_end,
            .end_stack_fifo_start = end_stack_fifo_start,
            .input_values = input_values,
            .values_arena = values_arena,
            .operations = operations,
            .bit_sets_arena = bit_sets_arena,
        };
    }

    fn estimatedWords(self: OpGraphBuilder) usize {
        return bitWordsPerSet(@intCast(@max(self.operations.items.len, self.estimated_ops)));
    }
};

pub const OpBuilder = struct {
    graph: *OpGraphBuilder,
    op: OpNodeId,

    pub fn id(self: OpBuilder) OpNodeId {
        return self.op;
    }

    pub fn addPredecessor(self: *OpBuilder, predecessor: OpNodeId) !void {
        std.debug.assert(predecessor < self.op);
        try addToSet(self.graph.allocator, &self.graph.op_predecessors.items[self.op], predecessor);
    }

    pub fn addInput(self: *OpBuilder, value: ValueNodeId) !void {
        std.debug.assert(value < self.graph.values.items.len);
        try self.graph.values_arena.append(self.graph.allocator, value);
        self.graph.operations.items[self.op].input_count += 1;

        const value_state = &self.graph.values.items[value];
        try addToSet(self.graph.allocator, &value_state.consumers, self.op);
        if (value_state.source) |source| {
            try addToSet(self.graph.allocator, &self.graph.op_predecessors.items[self.op], source);
        }
    }

    pub fn addOutput(self: *OpBuilder) !ValueNodeId {
        const value: ValueNodeId = @intCast(self.graph.values.items.len);
        try self.graph.values.append(self.graph.allocator, try ValueState.init(
            self.graph.allocator,
            self.op,
            self.graph.estimatedWords(),
        ));
        try self.graph.values_arena.append(self.graph.allocator, value);
        return value;
    }
};

pub fn collectNextCompletableInto(graph: OpGraph, complete: OpSet, out: *OpSetMut) void {
    var op: OpNodeId = 0;
    while (op < graph.totalOps()) : (op += 1) {
        if (!complete.contains(op) and complete.isSuper(graph.getPredecessors(op))) {
            out.add(op);
        }
    }
}

pub fn pushGraphInputs(stack: *release_schedule.TrackedStack, graph: OpGraph) !void {
    const inputs = graph.inputValuesFifo();
    var index = inputs.len;
    while (index > 0) {
        index -= 1;
        try stack.pushInput(inputs[index]);
    }
}

pub fn scheduleGraphOperationsIntoStack(
    allocator: std.mem.Allocator,
    stack: *release_schedule.TrackedStack,
    config: release_schedule.ScheduleConfig,
    graph: OpGraph,
) !void {
    try scheduleGraphOperationsIntoStackMode(allocator, stack, config, graph, .exact);
}

const FinalStackMode = enum {
    exact,
    terminal_use,
};

fn scheduleGraphOperationsIntoStackMode(
    allocator: std.mem.Allocator,
    stack: *release_schedule.TrackedStack,
    config: release_schedule.ScheduleConfig,
    graph: OpGraph,
    final_stack_mode: FinalStackMode,
) !void {
    const set_words = graph.wordsPerSet();
    const complete_words = try allocator.alloc(BitsetWord, set_words);
    defer allocator.free(complete_words);
    const completable_words = try allocator.alloc(BitsetWord, set_words);
    defer allocator.free(completable_words);
    @memset(complete_words, 0);
    @memset(completable_words, 0);

    var complete = OpSetMut.init(complete_words, graph.totalOps());
    var completable = OpSetMut.init(completable_words, graph.totalOps());

    var completed_count: u32 = 0;
    while (completed_count < graph.totalOps()) {
        completable.clear();
        collectNextCompletableInto(graph, complete.asConst(), &completable);
        const op = completable.first() orelse return ScheduleGraphError.IncompleteGraphSchedule;

        try scheduleGraphOp(allocator, stack, config, graph, op);
        complete.add(op);
        completed_count += 1;
    }

    switch (final_stack_mode) {
        .exact => try shuffleStackToTarget(config, stack, graph.outputValuesFifo()),
        .terminal_use => try loadOutputValuesForTerminalUse(allocator, stack, config, graph.outputValuesFifo()),
    }
}

fn loadOutputValuesForTerminalUse(
    allocator: std.mem.Allocator,
    stack: *release_schedule.TrackedStack,
    config: release_schedule.ScheduleConfig,
    target: []const ValueNodeId,
) !void {
    var index = target.len;
    while (index > 0) {
        index -= 1;
        try release_schedule.loadInputForUse(allocator, stack, config, target[index]);
    }
}

pub fn shuffleStackToTarget(
    config: release_schedule.ScheduleConfig,
    stack: *release_schedule.TrackedStack,
    target: []const ValueNodeId,
) !void {
    var shuffler = GreedyShuffler{
        .current = stack,
        .target = target,
        .max_swap_depth = config.max_swap_depth,
        .max_dup_depth = config.max_dup_depth,
    };
    try shuffler.run();
}

const shuffle_limit = 100_000;

const GreedyShuffler = struct {
    complete_at_bottom: usize = 0,
    current: *release_schedule.TrackedStack,
    target: []const ValueNodeId,
    max_swap_depth: usize,
    max_dup_depth: usize,

    fn run(self: *GreedyShuffler) !void {
        self.updateCompleteAtBottom();
        try self.shrink();
        try self.grow();
        try self.cleanupUnneededTop();
    }

    fn cleanupUnneededTop(self: *GreedyShuffler) !void {
        while (self.currentLen() > self.target.len) {
            try self.current.pop();
        }
    }

    fn shrink(self: *GreedyShuffler) !void {
        var limit: u32 = 0;
        const can_access_length = self.max_swap_depth + 1;
        while (can_access_length < self.incompleteCurrentLen()) {
            limit += 1;
            if (limit > shuffle_limit) return ScheduleGraphError.ShuffleLimitExceeded;

            const stepped = try self.popUnneeded() or
                try self.swapToCorrectPosition() or
                try self.popExtra() or
                try self.swapAndPopExtra() or
                try self.popDuplicate();
            if (!stepped) {
                _ = try self.current.spillTop();
            }
            self.updateCompleteAtBottom();
        }
    }

    fn grow(self: *GreedyShuffler) !void {
        var limit: u32 = 0;
        while (self.complete_at_bottom < self.target.len) {
            limit += 1;
            if (limit > shuffle_limit) return ScheduleGraphError.ShuffleLimitExceeded;

            const current_incomplete = self.currentLen() > self.complete_at_bottom;
            const stepped = if (current_incomplete)
                try self.popUnneeded() or
                    try self.swapToCorrectPosition() or
                    try self.exchangeViaTop() or
                    try self.popExtra()
            else
                false;

            if (!stepped) {
                if (self.canPush()) {
                    const pushed = try self.unspillUnavailableHorizon() or
                        try self.dupNeeded() or
                        try self.unspillNeeded();
                    std.debug.assert(pushed);
                } else {
                    _ = try self.current.spillTop();
                }
            }
            self.updateCompleteAtBottom();
        }
    }

    fn updateCompleteAtBottom(self: *GreedyShuffler) void {
        var bottom = self.complete_at_bottom;
        while (bottom < self.currentLen() and bottom < self.target.len) : (bottom += 1) {
            const current_value = self.currentAtBottom(bottom);
            const target_value = self.targetAtBottom(bottom);
            if (current_value != target_value) break;

            if (containsValue(prefixToBottomExclusive(self.target, bottom), target_value)) {
                if (!containsValue(prefixToBottomExclusive(self.currentFifo(), bottom), target_value)) break;
            }

            self.complete_at_bottom += 1;
        }
    }

    fn popUnneeded(self: *GreedyShuffler) !bool {
        if (self.currentLen() == 0) return false;
        if (self.isUnneeded(self.currentAtTop(0))) {
            try self.current.pop();
            return true;
        }
        return false;
    }

    fn swapToCorrectPosition(self: *GreedyShuffler) !bool {
        if (self.currentLen() <= self.complete_at_bottom) return false;

        const top = self.currentAtTop(0);
        const max_search_depth = @min(self.max_swap_depth, @min(self.toCurrentTop(self.complete_at_bottom), self.currentLen() - 1));
        const start_bottom = self.currentBottomFromTop(max_search_depth);

        var bottom = start_bottom;
        while (bottom < self.currentLen() and bottom < self.target.len) : (bottom += 1) {
            const current_value = self.currentAtBottom(bottom);
            const target_value = self.targetAtBottom(bottom);
            if (current_value != top and target_value == top) {
                try self.swap(self.toCurrentTop(bottom));
                return true;
            }
        }

        return false;
    }

    fn popExtra(self: *GreedyShuffler) !bool {
        if (self.currentLen() == 0) return false;
        const top = self.currentAtTop(0);
        if (self.isExtra(top)) {
            try self.current.pop();
            return true;
        }
        return false;
    }

    fn swapAndPopExtra(self: *GreedyShuffler) !bool {
        if (self.currentLen() == 0) return false;

        const max_search_depth = @min(self.max_swap_depth, @min(self.toCurrentTop(self.complete_at_bottom), self.currentLen() - 1));
        const start_bottom = self.currentBottomFromTop(max_search_depth);
        var bottom = start_bottom;
        while (bottom < self.currentLen()) : (bottom += 1) {
            if (self.isExtra(self.currentAtBottom(bottom))) {
                try self.swap(self.toCurrentTop(bottom));
                try self.current.pop();
                return true;
            }
        }

        return false;
    }

    fn popDuplicate(self: *GreedyShuffler) !bool {
        if (self.currentLen() == 0) return false;
        const top = self.currentAtTop(0);
        if (self.isDuplicate(top)) {
            try self.current.pop();
            return true;
        }
        return false;
    }

    fn exchangeViaTop(self: *GreedyShuffler) !bool {
        if (self.currentLen() == 0) return false;

        const max_top = @min(self.max_swap_depth, @min(self.toCurrentTop(self.complete_at_bottom), self.currentLen() - 1));
        const start_bottom = self.currentBottomFromTop(max_top);

        var dest_bottom = start_bottom;
        while (dest_bottom < self.currentLen() and dest_bottom < self.target.len) : (dest_bottom += 1) {
            const current_value = self.currentAtBottom(dest_bottom);
            const target_value = self.targetAtBottom(dest_bottom);
            if (current_value == target_value) continue;

            var src_bottom = start_bottom;
            while (src_bottom < self.currentLen() and src_bottom < self.target.len) : (src_bottom += 1) {
                const src_value = self.currentAtBottom(src_bottom);
                const target_at_src = self.targetAtBottom(src_bottom);
                if (src_value != target_at_src and src_value == target_value) {
                    try self.swap(self.toCurrentTop(src_bottom));
                    try self.swap(self.toCurrentTop(dest_bottom));
                    return true;
                }
            }
        }

        return false;
    }

    fn canPush(self: GreedyShuffler) bool {
        if (self.currentLen() <= self.max_swap_depth) return true;

        const horizon_bottom = self.currentBottomFromTop(self.max_swap_depth);
        if (horizon_bottom >= self.target.len) return false;

        const value = self.targetAtBottom(horizon_bottom);
        const current_value = self.currentAtBottom(horizon_bottom);
        if (current_value != value) return false;

        if (containsValue(prefixToBottomExclusive(self.target, horizon_bottom), value)) {
            return containsValue(prefixToBottomExclusive(self.currentFifo(), horizon_bottom), value) or
                self.current.getSpilled(value) != null;
        }

        return true;
    }

    fn unspillUnavailableHorizon(self: *GreedyShuffler) !bool {
        if (self.currentLen() < self.max_swap_depth or self.max_swap_depth == 0) return false;

        const horizon_bottom = self.currentBottomFromTop(self.max_swap_depth - 1);
        if (horizon_bottom >= self.target.len) return false;

        const target_value = self.targetAtBottom(horizon_bottom);
        const current_value = self.currentAtBottom(horizon_bottom);
        if (target_value != current_value and !containsValue(prefixToTopExclusive(self.currentFifo(), self.max_swap_depth), target_value)) {
            try self.current.unspill(target_value);
            return true;
        }

        return false;
    }

    fn dupNeeded(self: *GreedyShuffler) !bool {
        if (self.currentLen() == 0) return false;

        const max_dup_depth = @min(self.max_dup_depth, self.currentLen() - 1);
        const search_bottom = self.currentBottomFromTop(max_dup_depth);

        var bottom = search_bottom;
        while (bottom < self.currentLen() and bottom < self.target.len) : (bottom += 1) {
            const target_value = self.targetAtBottom(bottom);
            const required_copies = countValue(prefixToBottomInclusive(self.target, search_bottom), target_value);

            var available_copies: usize = 0;
            var dup_index: ?usize = null;
            const max_top = self.toCurrentTop(search_bottom);
            var top: usize = 0;
            while (top <= max_top) : (top += 1) {
                if (self.currentAtTop(top) == target_value) {
                    available_copies += 1;
                    if (dup_index == null) dup_index = top;
                }
            }

            if (dup_index) |depth| {
                if (available_copies < required_copies) {
                    try self.dup(depth);
                    return true;
                }
            }
        }

        return false;
    }

    fn unspillNeeded(self: *GreedyShuffler) !bool {
        const max_dup_depth_exclusive = @min(self.max_dup_depth + 1, self.currentLen());
        const needed = prefixToBottomInclusive(self.target, self.complete_at_bottom);
        var index = needed.len;
        while (index > 0) {
            index -= 1;
            const value = needed[index];
            if (!containsValue(prefixToTopExclusive(self.currentFifo(), max_dup_depth_exclusive), value)) {
                try self.current.unspill(value);
                return true;
            }
        }
        return false;
    }

    fn isUnneeded(self: GreedyShuffler, value: ValueNodeId) bool {
        if (self.target.len == self.complete_at_bottom) return true;
        return !containsValue(prefixToBottomInclusive(self.target, self.complete_at_bottom), value);
    }

    fn isExtra(self: GreedyShuffler, value: ValueNodeId) bool {
        const target_count = countValue(prefixToBottomInclusive(self.target, self.complete_at_bottom), value);
        const current_count = countValue(prefixToBottomInclusive(self.currentFifo(), self.complete_at_bottom), value);
        return current_count > target_count;
    }

    fn isDuplicate(self: GreedyShuffler, value: ValueNodeId) bool {
        return countValue(prefixToBottomInclusive(self.currentFifo(), self.complete_at_bottom), value) >= 2;
    }

    fn swap(self: *GreedyShuffler, depth: usize) !void {
        if (depth == 0) return;
        std.debug.assert(depth <= self.max_swap_depth);
        try self.current.swap(@intCast(depth));
    }

    fn dup(self: *GreedyShuffler, depth: usize) !void {
        std.debug.assert(depth <= self.max_dup_depth);
        try self.current.dup(@intCast(depth));
    }

    fn incompleteCurrentLen(self: GreedyShuffler) usize {
        if (self.currentLen() <= self.complete_at_bottom) return 0;
        return self.currentLen() - self.complete_at_bottom;
    }

    fn currentFifo(self: GreedyShuffler) []const ValueNodeId {
        return self.current.stack.fifo();
    }

    fn currentLen(self: GreedyShuffler) usize {
        return self.current.stack.len();
    }

    fn currentAtTop(self: GreedyShuffler, top: usize) ValueNodeId {
        return self.currentFifo()[top];
    }

    fn currentAtBottom(self: GreedyShuffler, bottom: usize) ValueNodeId {
        return self.currentFifo()[self.toCurrentTop(bottom)];
    }

    fn targetAtBottom(self: GreedyShuffler, bottom: usize) ValueNodeId {
        return self.target[topFromBottom(self.target.len, bottom)];
    }

    fn toCurrentTop(self: GreedyShuffler, bottom: usize) usize {
        return topFromBottom(self.currentLen(), bottom);
    }

    fn currentBottomFromTop(self: GreedyShuffler, top: usize) usize {
        return bottomFromTop(self.currentLen(), top);
    }
};

fn expectShuffle(
    config: release_schedule.ScheduleConfig,
    start_top_first: []const ValueNodeId,
    target_top_first: []const ValueNodeId,
    expected_ops: []const release_schedule.StackOp,
) !void {
    var stack = release_schedule.TrackedStack.init(std.testing.allocator, 0);
    defer stack.deinit();
    // Plank seeds by pushing the start stack in reverse so the first element
    // becomes the top; EvmStack.push prepends, matching that orientation.
    var index = start_top_first.len;
    while (index > 0) {
        index -= 1;
        try stack.pushInput(start_top_first[index]);
    }

    try shuffleStackToTarget(config, &stack, target_top_first);

    try std.testing.expectEqualSlices(ValueNodeId, target_top_first, stack.stack.fifo());
    expectStackOpsEqual(stack.ops.items, expected_ops) catch |err| {
        std.debug.print("shuffle ops mismatch\n  start ={any}\n  target={any}\n  got   ={any}\n  want  ={any}\n", .{
            start_top_first, target_top_first, stack.ops.items, expected_ops,
        });
        return err;
    };
}

fn expectStackOpsEqual(actual: []const release_schedule.StackOp, expected: []const release_schedule.StackOp) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (actual, expected) |got, want| {
        try std.testing.expect(std.meta.eql(got, want));
    }
}

test "greedy shuffler matches plank oracle: noop" {
    try expectShuffle(.{}, &.{ 1, 2, 3 }, &.{ 1, 2, 3 }, &.{});
}

test "greedy shuffler matches plank oracle: pops_unneeded" {
    try expectShuffle(release_schedule.ScheduleConfig.maxSwapNoExchange(2), &.{ 4, 2, 3, 1 }, &.{ 1, 2, 3 }, &.{ .pop, .{ .swap = 1 }, .{ .swap = 2 } });
}

test "greedy shuffler matches plank oracle: swaps_top_to_correct_position" {
    try expectShuffle(release_schedule.ScheduleConfig.maxSwapNoExchange(2), &.{ 1, 9, 3, 4 }, &.{ 3, 1, 4, 3 }, &.{ .{ .swap = 1 }, .pop, .{ .swap = 1 }, .{ .swap = 2 }, .{ .swap = 1 }, .{ .store = 0 }, .{ .dup = 1 }, .{ .load = 0 }, .{ .swap = 1 } });
}

test "greedy shuffler matches plank oracle: pops_extra_top_value_single" {
    try expectShuffle(release_schedule.ScheduleConfig.maxSwapNoExchange(2), &.{ 1, 1, 2, 3 }, &.{ 1, 2, 3, 2 }, &.{ .pop, .{ .swap = 1 }, .{ .swap = 2 }, .{ .swap = 1 }, .{ .store = 0 }, .{ .dup = 1 }, .{ .load = 0 } });
}

test "greedy shuffler matches plank oracle: swaps_and_pops_extra_value" {
    try expectShuffle(release_schedule.ScheduleConfig.maxSwapNoExchange(2), &.{ 2, 1, 1, 3 }, &.{ 2, 1, 3, 2 }, &.{ .{ .swap = 2 }, .pop, .{ .swap = 1 }, .{ .swap = 2 }, .{ .swap = 1 }, .{ .store = 0 }, .{ .dup = 1 }, .{ .load = 0 }, .{ .swap = 1 } });
}

test "greedy shuffler matches plank oracle: pops_duplicate_top_value" {
    try expectShuffle(release_schedule.ScheduleConfig.maxSwapNoExchange(2), &.{ 1, 1, 2, 4 }, &.{ 1, 1, 4, 2 }, &.{ .pop, .{ .swap = 1 }, .{ .swap = 2 }, .{ .swap = 1 }, .{ .dup = 0 } });
}

test "greedy shuffler matches plank oracle: spills_when_no_shrink_step_applies" {
    try expectShuffle(release_schedule.ScheduleConfig.maxSwapNoExchange(2), &.{ 1, 2, 3, 4 }, &.{ 1, 2, 4, 3 }, &.{ .{ .store = 0 }, .{ .swap = 1 }, .{ .swap = 2 }, .{ .swap = 1 }, .{ .load = 0 } });
}

test "greedy shuffler matches plank oracle: repeatedly_pops_extra_top_values" {
    try expectShuffle(release_schedule.ScheduleConfig.maxSwapNoExchange(2), &.{ 1, 1, 1, 2, 3 }, &.{ 1, 2, 3, 2, 3 }, &.{ .pop, .pop, .{ .store = 0 }, .{ .dup = 1 }, .{ .dup = 1 }, .{ .load = 0 } });
}

test "greedy shuffler matches plank oracle: repeatedly_swaps_and_pops_extra_values" {
    try expectShuffle(release_schedule.ScheduleConfig.maxSwapNoExchange(2), &.{ 2, 1, 1, 3, 3 }, &.{ 2, 1, 3, 2, 2 }, &.{ .{ .swap = 2 }, .pop, .{ .swap = 2 }, .pop, .{ .swap = 2 }, .{ .store = 0 }, .{ .dup = 1 }, .{ .swap = 1 }, .{ .dup = 1 }, .{ .swap = 1 }, .{ .load = 0 }, .{ .swap = 2 } });
}

test "greedy shuffler matches plank oracle: simple_swap_only" {
    try expectShuffle(.{}, &.{ 5, 1, 2, 3, 4 }, &.{ 1, 2, 3, 4, 5 }, &.{ .{ .swap = 4 }, .{ .swap = 3 }, .{ .swap = 2 }, .{ .swap = 1 } });
}

test "greedy shuffler matches plank oracle: needs_unspill" {
    try expectShuffle(release_schedule.ScheduleConfig.maxSwapNoExchange(3), &.{ 1, 2, 3, 4, 5, 6 }, &.{ 1, 6, 3, 4, 5, 6 }, &.{ .{ .swap = 1 }, .pop, .{ .store = 0 }, .{ .store = 1 }, .{ .dup = 2 }, .{ .load = 1 }, .{ .swap = 1 }, .{ .load = 0 } });
}

test "greedy shuffler matches plank oracle: current_is_already_correct_prefix" {
    try expectShuffle(release_schedule.ScheduleConfig.maxSwapNoExchange(2), &.{ 1, 0 }, &.{0}, &.{.pop});
}

test "greedy shuffler matches plank oracle: correct_after_swap_but_trash_top" {
    try expectShuffle(.{}, &.{ 1, 3, 2 }, &.{ 1, 2 }, &.{ .{ .swap = 1 }, .pop });
}

test "greedy shuffler matches plank oracle: pop_lower2" {
    try expectShuffle(release_schedule.ScheduleConfig.maxSwapNoExchange(1), &.{ 0, 1, 2 }, &.{0}, &.{ .{ .swap = 1 }, .pop, .{ .swap = 1 }, .pop });
}

test "greedy shuffler matches plank oracle: unspill_horizon_before_dup_top1" {
    try expectShuffle(release_schedule.ScheduleConfig.maxSwapNoExchange(1), &.{ 0, 1 }, &.{ 1, 1, 0, 1 }, &.{ .{ .store = 0 }, .{ .dup = 0 }, .{ .load = 0 }, .{ .swap = 1 }, .{ .dup = 0 } });
}

test "greedy shuffler matches plank oracle: unspill_horizon_before_dup_top2" {
    try expectShuffle(release_schedule.ScheduleConfig.maxSwapNoExchange(1), &.{ 0, 1 }, &.{ 0, 0, 1, 1, 0 }, &.{ .{ .swap = 1 }, .{ .store = 0 }, .{ .dup = 0 }, .{ .load = 0 }, .{ .swap = 1 }, .{ .load = 0 }, .{ .swap = 1 }, .{ .dup = 0 } });
}

test "greedy shuffler matches plank oracle: intricate_spill_dup_swap" {
    try expectShuffle(release_schedule.ScheduleConfig.maxSwapNoExchange(3), &.{ 10, 17, 2 }, &.{ 10, 2, 2, 10, 17, 17 }, &.{ .{ .dup = 1 }, .{ .swap = 3 }, .{ .dup = 1 }, .{ .dup = 1 }, .{ .swap = 1 } });
}

fn scheduleGraphOp(
    allocator: std.mem.Allocator,
    stack: *release_schedule.TrackedStack,
    config: release_schedule.ScheduleConfig,
    graph: OpGraph,
    op: OpNodeId,
) !void {
    const view = graph.getOp(op);
    var input_index = view.inputs_fifo.len;
    while (input_index > 0) {
        input_index -= 1;
        try release_schedule.loadInputForUse(allocator, stack, config, view.inputs_fifo[input_index]);
    }

    try completeGraphOp(stack, view);
}

fn completeGraphOp(stack: *release_schedule.TrackedStack, view: OpView) !void {
    const flippable = switch (view.kind) {
        .flippable => true,
        .normal, .ret_dest_push => false,
    };

    var flipping = false;
    for (view.inputs_fifo, 0..) |target, index| {
        const actual = stack.stack.pop() orelse return error.StackUnderflow;
        const correct = if (flippable and index == 0 and view.inputs_fifo.len >= 2 and actual == view.inputs_fifo[1]) blk: {
            flipping = true;
            break :blk true;
        } else if (flippable and flipping and index == 1)
            actual == view.inputs_fifo[0]
        else
            actual == target;
        if (!correct) return error.IncorrectSchedule;
    }

    switch (view.kind) {
        .ret_dest_push => |op| try stack.ops.append(stack.allocator, .{ .call_ret_push = op }),
        .flippable, .normal => try stack.ops.append(stack.allocator, .{ .op = view.kind.operationId() }),
    }

    var output_index = view.outputs_fifo.len;
    while (output_index > 0) {
        output_index -= 1;
        try stack.stack.push(view.outputs_fifo[output_index]);
    }
}

pub fn bitWordsPerSet(total_ops: u32) usize {
    return std.math.divCeil(usize, total_ops, @bitSizeOf(BitsetWord)) catch unreachable;
}

fn addToSet(allocator: std.mem.Allocator, set: *std.ArrayList(BitsetWord), id: OpNodeId) !void {
    const word_index: usize = @intCast(id / @bitSizeOf(BitsetWord));
    const word_shift: u3 = @intCast(id % @bitSizeOf(BitsetWord));
    if (word_index >= set.items.len) {
        try set.appendNTimes(allocator, 0, word_index + 1 - set.items.len);
    }
    set.items[word_index] |= @as(BitsetWord, 1) << word_shift;
}

fn copyBitset(dst: []BitsetWord, src: []const BitsetWord) void {
    std.debug.assert(src.len <= dst.len);
    if (src.len > 0) @memcpy(dst[0..src.len], src);
}

fn topFromBottom(len: usize, bottom: usize) usize {
    std.debug.assert(bottom < len);
    return len - bottom - 1;
}

fn bottomFromTop(len: usize, top: usize) usize {
    std.debug.assert(top < len);
    return len - top - 1;
}

fn prefixToBottomInclusive(values: []const ValueNodeId, bottom: usize) []const ValueNodeId {
    if (values.len == 0) return values[0..0];
    if (bottom >= values.len) return values;
    const top = topFromBottom(values.len, bottom);
    return values[0 .. top + 1];
}

fn prefixToBottomExclusive(values: []const ValueNodeId, bottom: usize) []const ValueNodeId {
    if (values.len == 0) return values[0..0];
    if (bottom >= values.len) return values;
    const top = topFromBottom(values.len, bottom);
    return values[0..top];
}

fn prefixToTopExclusive(values: []const ValueNodeId, top: usize) []const ValueNodeId {
    return values[0..@min(values.len, top)];
}

fn containsValue(values: []const ValueNodeId, value: ValueNodeId) bool {
    for (values) |candidate| {
        if (candidate == value) return true;
    }
    return false;
}

fn countValue(values: []const ValueNodeId, value: ValueNodeId) usize {
    var count: usize = 0;
    for (values) |candidate| {
        if (candidate == value) count += 1;
    }
    return count;
}

pub fn operationId(
    program: ir.Program,
    function_name: []const u8,
    block_name: []const u8,
    instruction_index: usize,
) !OperationId {
    var op_id: OperationId = 0;
    for (program.functions) |function| {
        for (function.blocks) |block| {
            for (block.instructions, 0..) |_, index| {
                if (std.mem.eql(u8, function.name, function_name) and
                    std.mem.eql(u8, block.name, block_name) and
                    index == instruction_index)
                {
                    return op_id;
                }
                op_id += 1;
            }
        }
    }
    return ScheduleGraphError.MissingInstruction;
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

// Bundle CFG in/out layouts the way Plank's `ControlFlowGraphInOutBundling`
// does: every edge `bb -> succ` forces `out(bb)` and `in(succ)` to share one
// layout group. The group's layout is the union of the live-in members of every
// block bundled into it, so a value a sibling successor needs is still placed on
// this block's incoming stack (and popped at entry when this block doesn't use
// it). Computed as a union-find over CFG edges, which yields the same partition
// as Plank's RPO worklist independent of traversal order.
//
// Node numbering: `in(i) = i`, `out(i) = block_count + i`.
fn unionFind(parent: []usize, node: usize) usize {
    var root = node;
    while (parent[root] != root) root = parent[root];
    var current = node;
    while (parent[current] != root) {
        const next = parent[current];
        parent[current] = root;
        current = next;
    }
    return root;
}

fn unionMerge(parent: []usize, a: usize, b: usize) void {
    const ra = unionFind(parent, a);
    const rb = unionFind(parent, b);
    if (ra == rb) return;
    // Merge toward the lower root for stable, traversal-independent roots.
    if (ra < rb) parent[rb] = ra else parent[ra] = rb;
}

fn buildFunctionLayouts(allocator: std.mem.Allocator, program: ir.Program, function: ir.Function) !FunctionLayouts {
    const block_count = function.blocks.len;
    const facts = try allocator.alloc(BlockLiveFacts, block_count);
    defer {
        for (facts) |*fact| fact.deinit(allocator);
        allocator.free(facts);
    }
    for (function.blocks, facts) |block, *fact| {
        fact.* = try collectBlockLiveFacts(allocator, block);
    }

    // Match Plank's `local_liveness`: EVERY icall operand is a use (`op.inputs()`
    // returns all call args), so an argument the callee ignores is still kept
    // live across blocks and cleaned up after the call. Callee-layout filtering
    // (which args to actually pass) is a SCHEDULING concern handled separately in
    // `buildBlockGraphSimple`'s icall loop, NOT a liveness concern. Using the
    // callee-filtered liveness here dropped callee-ignored args early, giving a
    // shallower stack and a divergent cleanup shuffle (e.g. memory_primitive bb0
    // SWAP8 vs Plank SWAP6) — the root of the `delta=-2` mismatch cluster.
    // (`.all_textual` requires asm.zig's `sortedRawLabels` to tolerate
    // declared-but-unreferenced labels; see that function.)
    try solveLiveness(allocator, program, function, facts, .all_textual);

    const node_count = block_count * 2;
    const parent = try allocator.alloc(usize, node_count);
    defer allocator.free(parent);
    for (parent, 0..) |*p, i| p.* = i;

    var successor_storage: [max_successors][]const u8 = undefined;
    for (function.blocks, 0..) |block, block_index| {
        for (successors(block, &successor_storage)) |target| {
            const target_index = blockIndex(function, target) orelse continue;
            // out(block_index) ~ in(target_index)
            unionMerge(parent, block_count + block_index, target_index);
        }
    }

    // One member set per union-find root (indexed by node id for simplicity).
    const group_members = try allocator.alloc(std.ArrayList(LayoutMember), node_count);
    defer {
        for (group_members) |*members| members.deinit(allocator);
        allocator.free(group_members);
    }
    for (group_members) |*members| members.* = .empty;

    const internal_function = isInternalFunctionName(function.name);
    for (function.blocks, facts, 0..) |block, fact, block_index| {
        const in_root = unionFind(parent, block_index);
        if (internal_function) {
            try appendLayoutMember(allocator, &group_members[in_root], .return_dest);
        }
        for (fact.live_in.items) |name| {
            try appendLayoutMember(allocator, &group_members[in_root], layoutMemberForInput(block, name));
        }
        if (block.terminator == .iret) {
            const out_root = unionFind(parent, block_count + block_index);
            for (block.outputs, 0..) |_, output_index| {
                try appendLayoutMember(allocator, &group_members[out_root], .{ .input_output = @intCast(output_index) });
            }
        }
    }

    for (group_members) |*members| sortLayoutMembers(function, members.items);

    const block_layouts = try allocator.alloc(BlockLayout, block_count);
    var initialized_count: usize = 0;
    errdefer {
        for (block_layouts[0..initialized_count]) |layout| {
            allocator.free(layout.input);
            allocator.free(layout.output);
        }
        allocator.free(block_layouts);
    }
    for (function.blocks, block_layouts, 0..) |block, *layout, block_index| {
        const in_root = unionFind(parent, block_index);
        const out_root = unionFind(parent, block_count + block_index);
        const input = try allocator.dupe(LayoutMember, group_members[in_root].items);
        errdefer allocator.free(input);
        const output = try allocator.dupe(LayoutMember, group_members[out_root].items);
        layout.* = .{
            .block_name = block.name,
            .input = input,
            .output = output,
        };
        initialized_count += 1;
    }

    return .{
        .allocator = allocator,
        .blocks = block_layouts,
    };
}

const BlockLiveFacts = struct {
    live_in: std.ArrayList([]const u8) = .empty,
    live_out: std.ArrayList([]const u8) = .empty,

    fn deinit(self: *BlockLiveFacts, allocator: std.mem.Allocator) void {
        self.live_out.deinit(allocator);
        self.live_in.deinit(allocator);
    }
};

fn collectBlockLiveFacts(allocator: std.mem.Allocator, block: ir.Block) !BlockLiveFacts {
    _ = allocator;
    _ = block;
    return .{};
}

const LivenessOperandMode = enum {
    all_textual,
    layout_aware_icall,
};

fn solveLiveness(
    allocator: std.mem.Allocator,
    program: ?ir.Program,
    function: ir.Function,
    facts: []BlockLiveFacts,
    operand_mode: LivenessOperandMode,
) LivenessError!void {
    var changed = true;
    while (changed) {
        changed = false;
        var block_offset = function.blocks.len;
        while (block_offset > 0) {
            block_offset -= 1;
            const block = function.blocks[block_offset];

            var next_live_in: std.ArrayList([]const u8) = .empty;
            defer next_live_in.deinit(allocator);
            try computeBlockEntryLiveness(allocator, program, block, facts[block_offset].live_out.items, &next_live_in, operand_mode);
            if (!sameNameSet(facts[block_offset].live_in.items, next_live_in.items)) {
                facts[block_offset].live_in.clearRetainingCapacity();
                try facts[block_offset].live_in.appendSlice(allocator, next_live_in.items);
                changed = true;
            }

            for (function.blocks, facts, 0..) |predecessor, *predecessor_facts, predecessor_index| {
                if (predecessor_index == block_offset) continue;
                if (!blockTargets(predecessor, block.name)) continue;
                for (next_live_in.items) |name| {
                    const propagated = try predecessorOutputForSuccessorInput(predecessor, block, name);
                    if (!containsName(predecessor_facts.live_out.items, propagated)) {
                        try predecessor_facts.live_out.append(allocator, propagated);
                        sortNames(predecessor_facts.live_out.items);
                        changed = true;
                    }
                }
            }
        }
    }
}

const max_successors = 128;

fn successors(block: ir.Block, storage: *[max_successors][]const u8) []const []const u8 {
    var count: usize = 0;
    switch (block.terminator) {
        .jump => |target| {
            storage[count] = target;
            count += 1;
        },
        .branch => |branch| {
            storage[count] = branch.zero_target;
            count += 1;
            storage[count] = branch.non_zero_target;
            count += 1;
        },
        .switch_ => |switch_term| {
            for (switch_term.cases) |case| {
                if (count >= storage.len) break;
                storage[count] = case.target;
                count += 1;
            }
            if (switch_term.default_target.len != 0 and count < storage.len) {
                storage[count] = switch_term.default_target;
                count += 1;
            }
        },
        .return_, .revert, .stop, .invalid, .selfdestruct, .iret => {},
    }
    return storage[0..count];
}

fn blockIndex(function: ir.Function, block_name: []const u8) ?usize {
    for (function.blocks, 0..) |block, index| {
        if (std.mem.eql(u8, block.name, block_name)) return index;
    }
    return null;
}

fn computeBlockEntryLiveness(
    allocator: std.mem.Allocator,
    program: ?ir.Program,
    block: ir.Block,
    live_out: []const []const u8,
    live_in: *std.ArrayList([]const u8),
    operand_mode: LivenessOperandMode,
) LivenessError!void {
    for (live_out) |name| try appendName(allocator, live_in, name);

    switch (block.terminator) {
        .branch => |branch| try appendName(allocator, live_in, branch.condition),
        .switch_ => |switch_term| try appendName(allocator, live_in, switch_term.selector),
        .iret => {
            for (block.outputs) |output| try appendName(allocator, live_in, output);
        },
        .return_ => |ret| {
            try appendName(allocator, live_in, ret.ptr);
            try appendName(allocator, live_in, ret.len);
        },
        .revert => |revert| {
            try appendName(allocator, live_in, revert.ptr);
            try appendName(allocator, live_in, revert.len);
        },
        .selfdestruct => |beneficiary| try appendName(allocator, live_in, beneficiary),
        .jump, .stop, .invalid => {},
    }

    var instruction_index = block.instructions.len;
    while (instruction_index > 0) {
        instruction_index -= 1;
        const instruction = block.instructions[instruction_index];
        for (instruction.results) |result| {
            removeName(live_in, result);
        }
        try appendInstructionLiveOperands(allocator, program, live_in, instruction, operand_mode);
    }
    sortNames(live_in.items);
}

fn appendInstructionLiveOperands(
    allocator: std.mem.Allocator,
    program: ?ir.Program,
    live_in: *std.ArrayList([]const u8),
    instruction: ir.Instruction,
    operand_mode: LivenessOperandMode,
) LivenessError!void {
    if (operand_mode == .layout_aware_icall and std.mem.eql(u8, instruction.mnemonic, "icall")) {
        const concrete_program = program orelse return ScheduleGraphError.UnsupportedSir;
        const positions = try calleeEntryLiveInputPositions(allocator, concrete_program, instruction);
        defer allocator.free(positions);
        for (positions) |position| {
            const operand_index = @as(usize, position) + 1;
            if (operand_index >= instruction.operands.len) return ScheduleGraphError.UnsupportedSir;
            try appendName(allocator, live_in, instruction.operands[operand_index]);
        }
        return;
    }

    for (instruction.operands) |operand| {
        if (isNonValueOperand(instruction.mnemonic, operand)) continue;
        try appendName(allocator, live_in, operand);
    }
}

fn calleeEntryLiveInputPositions(
    allocator: std.mem.Allocator,
    program: ir.Program,
    instruction: ir.Instruction,
) LivenessError![]const u32 {
    if (instruction.operands.len == 0) return ScheduleGraphError.UnsupportedSir;
    const callee_name = try operandFunctionName(instruction.operands[0]);
    const callee = findFunction(program, callee_name) orelse return ScheduleGraphError.MissingFunction;
    const entry_index = functionEntryBlockIndex(callee) orelse return ScheduleGraphError.MissingBlock;
    const entry = callee.blocks[entry_index];

    const facts = try allocator.alloc(BlockLiveFacts, callee.blocks.len);
    defer {
        for (facts) |*fact| fact.deinit(allocator);
        allocator.free(facts);
    }
    for (callee.blocks, facts) |block, *fact| {
        fact.* = try collectBlockLiveFacts(allocator, block);
    }
    try solveLiveness(allocator, null, callee, facts, .all_textual);

    var positions: std.ArrayList(u32) = .empty;
    errdefer positions.deinit(allocator);
    for (facts[entry_index].live_in.items) |name| {
        for (entry.inputs, 0..) |input, position| {
            if (std.mem.eql(u8, input, name)) {
                try positions.append(allocator, @intCast(position));
                break;
            }
        }
    }
    sortInt(u32, positions.items);
    return positions.toOwnedSlice(allocator);
}

fn predecessorOutputForSuccessorInput(
    predecessor: ir.Block,
    successor: ir.Block,
    name: []const u8,
) ![]const u8 {
    for (successor.inputs, 0..) |input, position| {
        if (!std.mem.eql(u8, input, name)) continue;
        if (position >= predecessor.outputs.len) return ScheduleGraphError.UnsupportedSir;
        return predecessor.outputs[position];
    }
    return name;
}

fn blockTargets(block: ir.Block, target_name: []const u8) bool {
    var storage: [max_successors][]const u8 = undefined;
    for (successors(block, &storage)) |target| {
        if (std.mem.eql(u8, target, target_name)) return true;
    }
    return false;
}

fn blockHasSuccessors(block: ir.Block) bool {
    var storage: [max_successors][]const u8 = undefined;
    return successors(block, &storage).len != 0;
}

fn layoutMemberForInput(block: ir.Block, name: []const u8) LayoutMember {
    for (block.inputs, 0..) |input, index| {
        if (std.mem.eql(u8, input, name)) return .{ .input_output = @intCast(index) };
    }
    return .{ .local = name };
}

fn layoutMemberForOutput(block: ir.Block, name: []const u8) LayoutMember {
    for (block.outputs, 0..) |output, index| {
        if (std.mem.eql(u8, output, name)) return .{ .input_output = @intCast(index) };
    }
    return .{ .local = name };
}

fn layoutMemberValue(
    member: LayoutMember,
    block: ir.Block,
    locals: []const LocalValue,
    return_dest_value: ?ValueNodeId,
) !ValueNodeId {
    return switch (member) {
        .return_dest => return_dest_value orelse ScheduleGraphError.UnsupportedSir,
        .input_output => |position| blk: {
            if (position >= block.outputs.len) return ScheduleGraphError.UnsupportedSir;
            break :blk try localValue(locals, block.outputs[position]);
        },
        .local => |name| try localValue(locals, name),
    };
}

fn appendLayoutMember(
    allocator: std.mem.Allocator,
    members: *std.ArrayList(LayoutMember),
    member: LayoutMember,
) !void {
    for (members.items) |entry| {
        if (sameLayoutMember(entry, member)) return;
    }
    try members.append(allocator, member);
}

fn layoutContains(members: []const LayoutMember, member: LayoutMember) bool {
    for (members) |entry| {
        if (sameLayoutMember(entry, member)) return true;
    }
    return false;
}

fn sameLayoutMember(lhs: LayoutMember, rhs: LayoutMember) bool {
    return switch (lhs) {
        .return_dest => rhs == .return_dest,
        .input_output => |lhs_position| rhs == .input_output and rhs.input_output == lhs_position,
        .local => |lhs_name| rhs == .local and std.mem.eql(u8, lhs_name, rhs.local),
    };
}

fn sortLayoutMembers(function: ir.Function, members: []LayoutMember) void {
    std.mem.sort(LayoutMember, members, function, struct {
        fn less(context: ir.Function, lhs: LayoutMember, rhs: LayoutMember) bool {
            const lhs_rank = layoutMemberRank(lhs);
            const rhs_rank = layoutMemberRank(rhs);
            if (lhs_rank != rhs_rank) return lhs_rank < rhs_rank;
            return switch (lhs) {
                .return_dest => false,
                .input_output => |lhs_position| lhs_position < rhs.input_output,
                .local => |lhs_name| blk: {
                    const lhs_order = localFirstOccurrence(context, lhs_name);
                    const rhs_order = localFirstOccurrence(context, rhs.local);
                    if (lhs_order != rhs_order) break :blk lhs_order < rhs_order;
                    break :blk std.mem.order(u8, lhs_name, rhs.local) == .lt;
                },
            };
        }
    }.less);
}

// Rank a local by its DEFINITION order, matching Plank's `LocalId` (assigned at
// SSA construction). A value is defined by being a block input (phi-like) or an
// instruction result; block-output declarations and operand uses are NOT
// definitions. The previous version also scanned outputs/operands, so a value
// re-declared in an `entry ... -> vN` output list ranked before a value defined
// earlier in the body, producing a layout order that diverged from Plank's
// `LocalId` sort and a different greedy-shuffle target (loop blocks especially).
fn localFirstOccurrence(function: ir.Function, target: []const u8) usize {
    var index: usize = 0;
    for (function.blocks) |block| {
        for (block.inputs) |name| {
            if (localOccurrence(&index, name, target)) |found| return found;
        }
        for (block.instructions) |instruction| {
            for (instruction.results) |name| {
                if (localOccurrence(&index, name, target)) |found| return found;
            }
        }
    }
    return std.math.maxInt(usize);
}

fn localOccurrence(index: *usize, name: []const u8, target: []const u8) ?usize {
    defer index.* += 1;
    if (std.mem.eql(u8, name, target)) return index.*;
    return null;
}

fn layoutMemberRank(member: LayoutMember) u8 {
    return switch (member) {
        .return_dest => 0,
        .input_output => 1,
        .local => 2,
    };
}

fn appendName(
    allocator: std.mem.Allocator,
    names: *std.ArrayList([]const u8),
    name: []const u8,
) !void {
    if (containsName(names.items, name)) return;
    try names.append(allocator, name);
}

fn containsName(names: []const []const u8, name: []const u8) bool {
    for (names) |entry| {
        if (std.mem.eql(u8, entry, name)) return true;
    }
    return false;
}

fn removeName(names: *std.ArrayList([]const u8), name: []const u8) void {
    for (names.items, 0..) |entry, index| {
        if (std.mem.eql(u8, entry, name)) {
            _ = names.orderedRemove(index);
            return;
        }
    }
}

fn sortNames(names: [][]const u8) void {
    std.mem.sort([]const u8, names, {}, struct {
        fn less(_: void, lhs: []const u8, rhs: []const u8) bool {
            return std.mem.order(u8, lhs, rhs) == .lt;
        }
    }.less);
}

fn sortInt(comptime T: type, values: []T) void {
    std.mem.sort(T, values, {}, struct {
        fn less(_: void, lhs: T, rhs: T) bool {
            return lhs < rhs;
        }
    }.less);
}

fn sameNameSet(lhs: []const []const u8, rhs: []const []const u8) bool {
    if (lhs.len != rhs.len) return false;
    for (lhs, rhs) |lhs_name, rhs_name| {
        if (!std.mem.eql(u8, lhs_name, rhs_name)) return false;
    }
    return true;
}

fn putLocal(
    allocator: std.mem.Allocator,
    locals: *std.ArrayList(LocalValue),
    name: []const u8,
    value: ValueNodeId,
) !void {
    for (locals.items) |entry| {
        if (std.mem.eql(u8, entry.name, name)) return ScheduleGraphError.UnsupportedSir;
    }
    try locals.append(allocator, .{ .name = name, .value = value });
}

fn localValue(locals: []const LocalValue, name: []const u8) !ValueNodeId {
    for (locals) |entry| {
        if (std.mem.eql(u8, entry.name, name)) return entry.value;
    }
    return ScheduleGraphError.MissingLocal;
}

fn operandFunctionName(operand: []const u8) ![]const u8 {
    if (operand.len < 2 or operand[0] != '@') return ScheduleGraphError.UnsupportedSir;
    return operand[1..];
}

fn functionEntryBlock(function: ir.Function) ?ir.Block {
    return findBlock(function, "entry") orelse if (function.blocks.len != 0) function.blocks[0] else null;
}

fn functionEntryBlockIndex(function: ir.Function) ?usize {
    return blockIndex(function, "entry") orelse if (function.blocks.len != 0) 0 else null;
}

fn isInternalFunctionName(name: []const u8) bool {
    return !std.mem.eql(u8, name, "init") and !std.mem.eql(u8, name, "main");
}

fn isLastOpTerminatingControl(terminator: ir.Terminator) bool {
    return switch (terminator) {
        .return_, .revert, .stop, .invalid, .selfdestruct => true,
        .jump, .branch, .switch_, .iret => false,
    };
}

fn isNonValueOperand(mnemonic: []const u8, operand: []const u8) bool {
    if (std.mem.eql(u8, mnemonic, "const") or std.mem.eql(u8, mnemonic, "large_const")) return true;
    if (std.mem.eql(u8, mnemonic, "salloc") or std.mem.eql(u8, mnemonic, "sallocany")) return true;
    if (std.mem.eql(u8, mnemonic, "data_offset")) return true;
    if (operand.len > 0 and (operand[0] == '@' or operand[0] == '.')) return true;
    return false;
}

fn isFlippable(mnemonic: []const u8) bool {
    const spec = ops.lookup(mnemonic) orelse return false;
    return switch (spec) {
        .fixed => std.mem.eql(u8, mnemonic, "add") or
            std.mem.eql(u8, mnemonic, "mul") or
            std.mem.eql(u8, mnemonic, "addmod") or
            std.mem.eql(u8, mnemonic, "mulmod") or
            std.mem.eql(u8, mnemonic, "lt") or
            std.mem.eql(u8, mnemonic, "gt") or
            std.mem.eql(u8, mnemonic, "slt") or
            std.mem.eql(u8, mnemonic, "sgt") or
            std.mem.eql(u8, mnemonic, "eq") or
            std.mem.eql(u8, mnemonic, "and") or
            std.mem.eql(u8, mnemonic, "or") or
            std.mem.eql(u8, mnemonic, "xor"),
        .memory_load, .memory_store, .internal_call => false,
    };
}

fn expectOpSetMembers(set: OpSet, expected: []const OpNodeId) !void {
    var op: OpNodeId = 0;
    var expected_index: usize = 0;
    while (op < set.total_ops) : (op += 1) {
        if (!set.contains(op)) continue;
        try std.testing.expect(expected_index < expected.len);
        try std.testing.expectEqual(expected[expected_index], op);
        expected_index += 1;
    }
    try std.testing.expectEqual(expected.len, expected_index);
}

test "op graph records value producers and consumers" {
    var builder = try OpGraphBuilder.init(std.testing.allocator, 2, 4);
    defer builder.deinit();

    const input0 = try builder.pushInputValue();
    const input1 = try builder.pushInputValue();
    builder.endInputsBeginOps();

    var first = try builder.beginOp(.{ .normal = 10 });
    try first.addInput(input0);
    const first_output = try first.addOutput();

    var second = try builder.beginOp(.{ .normal = 11 });
    try second.addInput(first_output);
    try second.addInput(input1);
    const second_output = try second.addOutput();

    builder.endOpsBeginEndStack();
    try builder.pushEndStackValue(second_output);

    var graph = try builder.finish();
    defer graph.deinit();

    try std.testing.expectEqual(@as(u32, 2), graph.totalOps());
    try std.testing.expectEqual(@as(u32, 4), graph.totalValues());
    try std.testing.expectEqualSlices(ValueNodeId, &.{ input0, input1 }, graph.inputValuesFifo());
    try std.testing.expectEqualSlices(ValueNodeId, &.{second_output}, graph.outputValuesFifo());

    const first_view = graph.getOp(0);
    try std.testing.expectEqual(OpNodeKind{ .normal = 10 }, first_view.kind);
    try std.testing.expectEqualSlices(ValueNodeId, &.{input0}, first_view.inputs_fifo);
    try std.testing.expectEqualSlices(ValueNodeId, &.{first_output}, first_view.outputs_fifo);

    try expectOpSetMembers(graph.getConsumers(input0), &.{0});
    try expectOpSetMembers(graph.getConsumers(first_output), &.{1});
    try expectOpSetMembers(graph.getPredecessors(1), &.{0});
}

test "op graph collects next completable operations" {
    var builder = try OpGraphBuilder.init(std.testing.allocator, 2, 3);
    defer builder.deinit();

    const input = try builder.pushInputValue();
    builder.endInputsBeginOps();

    var first = try builder.beginOp(.{ .normal = 20 });
    try first.addInput(input);
    const first_output = try first.addOutput();

    var second = try builder.beginOp(.{ .normal = 21 });
    try second.addInput(first_output);
    _ = try second.addOutput();

    builder.endOpsBeginEndStack();

    var graph = try builder.finish();
    defer graph.deinit();

    const set_words = graph.wordsPerSet();
    const complete_words = try std.testing.allocator.alloc(BitsetWord, set_words);
    defer std.testing.allocator.free(complete_words);
    const completable_words = try std.testing.allocator.alloc(BitsetWord, set_words);
    defer std.testing.allocator.free(completable_words);
    @memset(complete_words, 0);
    @memset(completable_words, 0);

    var complete = OpSetMut.init(complete_words, graph.totalOps());
    var completable = OpSetMut.init(completable_words, graph.totalOps());

    collectNextCompletableInto(graph, complete.asConst(), &completable);
    try expectOpSetMembers(completable.asConst(), &.{0});

    complete.add(0);
    completable.clear();
    collectNextCompletableInto(graph, complete.asConst(), &completable);
    try expectOpSetMembers(completable.asConst(), &.{1});
}

test "op graph scheduler emits operations in dependency order" {
    var builder = try OpGraphBuilder.init(std.testing.allocator, 2, 4);
    defer builder.deinit();

    const input0 = try builder.pushInputValue();
    const input1 = try builder.pushInputValue();
    builder.endInputsBeginOps();

    var first = try builder.beginOp(.{ .normal = 30 });
    try first.addInput(input0);
    try first.addInput(input1);
    const first_output = try first.addOutput();

    var second = try builder.beginOp(.{ .normal = 31 });
    try second.addInput(first_output);
    const second_output = try second.addOutput();

    builder.endOpsBeginEndStack();
    try builder.pushEndStackValue(second_output);

    var graph = try builder.finish();
    defer graph.deinit();

    var stack = release_schedule.TrackedStack.init(std.testing.allocator, 0);
    defer stack.deinit();
    try pushGraphInputs(&stack, graph);

    try scheduleGraphOperationsIntoStack(std.testing.allocator, &stack, .{}, graph);

    try std.testing.expectEqualSlices(release_schedule.StackOp, &.{
        .{ .dup = 1 },
        .{ .dup = 1 },
        .{ .op = 30 },
        .{ .dup = 0 },
        .{ .op = 31 },
        .{ .swap = 3 },
        .pop,
        .pop,
        .pop,
    }, stack.ops.items);
    try std.testing.expectEqualSlices(ValueNodeId, &.{second_output}, stack.stack.fifo());
}

test "op graph scheduler emits call return destination push" {
    var builder = try OpGraphBuilder.init(std.testing.allocator, 1, 1);
    defer builder.deinit();

    _ = try builder.pushInputValue();
    builder.endInputsBeginOps();

    var ret_dest = try builder.beginOp(.{ .ret_dest_push = 41 });
    const ret_dest_value = try ret_dest.addOutput();

    builder.endOpsBeginEndStack();
    try builder.pushEndStackValue(ret_dest_value);

    var graph = try builder.finish();
    defer graph.deinit();

    var stack = release_schedule.TrackedStack.init(std.testing.allocator, 0);
    defer stack.deinit();
    try pushGraphInputs(&stack, graph);

    try scheduleGraphOperationsIntoStack(std.testing.allocator, &stack, .{}, graph);

    try std.testing.expectEqualSlices(release_schedule.StackOp, &.{
        .{ .call_ret_push = 41 },
        .{ .swap = 1 },
        .pop,
    }, stack.ops.items);
    try std.testing.expectEqualSlices(ValueNodeId, &.{ret_dest_value}, stack.stack.fifo());
}

test "greedy shuffler pops unneeded values to match target" {
    var stack = release_schedule.TrackedStack.init(std.testing.allocator, 0);
    defer stack.deinit();
    try stack.pushInput(0);
    try stack.pushInput(1);
    try stack.pushInput(2);

    try shuffleStackToTarget(.{}, &stack, &.{1});

    try std.testing.expectEqualSlices(release_schedule.StackOp, &.{
        .pop,
        .{ .swap = 1 },
        .pop,
    }, stack.ops.items);
    try std.testing.expectEqualSlices(ValueNodeId, &.{1}, stack.stack.fifo());
}

test "greedy shuffler duplicates needed values" {
    var stack = release_schedule.TrackedStack.init(std.testing.allocator, 0);
    defer stack.deinit();
    try stack.pushInput(0);
    try stack.pushInput(1);

    try shuffleStackToTarget(.{}, &stack, &.{ 1, 1 });

    try std.testing.expectEqualSlices(release_schedule.StackOp, &.{
        .{ .swap = 1 },
        .pop,
        .{ .dup = 0 },
    }, stack.ops.items);
    try std.testing.expectEqualSlices(ValueNodeId, &.{ 1, 1 }, stack.stack.fifo());
}

test "simple block graph schedules parsed straight-line SIR" {
    var program = try parseTestProgram(
        \\fn main:
        \\    entry {
        \\        a = const 0x02
        \\        b = const 0x03
        \\        c = add a b
        \\        stop
        \\    }
    );
    defer program.deinit();

    var scheduled = try scheduleBlockSimple(std.testing.allocator, program, "main", "entry", .{}, 0, null);
    defer scheduled.deinit(std.testing.allocator);

    try std.testing.expectEqualSlices(release_schedule.StackOp, &.{
        .{ .op = try operationId(program, "main", "entry", 0) },
        .{ .op = try operationId(program, "main", "entry", 1) },
        .{ .dup = 0 },
        .{ .dup = 2 },
        .{ .op = try operationId(program, "main", "entry", 2) },
    }, scheduled.ops);
}

test "simple block graph feeds generic code-to-asm" {
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

    var scheduled = try scheduleBlockSimple(std.testing.allocator, program, "main", "entry", .{}, 0, null);
    defer scheduled.deinit(std.testing.allocator);

    const schedules = [_]release_code_to_asm.BlockSchedule{.{
        .function_name = scheduled.function_name,
        .block_name = scheduled.block_name,
        .ops = scheduled.ops,
    }};
    const bytes = try release_code_to_asm.emitFromEntry(std.testing.allocator, program, "main", &schedules, .{});
    defer std.testing.allocator.free(bytes);

    try std.testing.expectEqual(evmOp("return"), bytes[bytes.len - 1]);
    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evmOp("mstore256")) != null);
}

test "simple block graph schedules icall return destination" {
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

    var main_schedule = try scheduleBlockSimple(std.testing.allocator, program, "main", "entry", .{}, 0, null);
    defer main_schedule.deinit(std.testing.allocator);
    var callee_schedule = try scheduleBlockSimple(std.testing.allocator, program, "callee", "entry", .{}, main_schedule.next_alloc_id, null);
    defer callee_schedule.deinit(std.testing.allocator);

    try std.testing.expectEqual(release_schedule.StackOp{ .call_ret_push = try operationId(program, "main", "entry", 1) }, main_schedule.ops[1]);
    try expectContainsStackOp(callee_schedule.ops, .{ .op = try operationId(program, "callee", "entry", 1) });
}

test "simple block graph takes icall inputs from callee entry layout" {
    var program = try parseTestProgram(
        \\fn ident:
        \\    entry x y -> x {
        \\        iret
        \\    }
        \\
        \\fn main:
        \\    entry {
        \\        a = const 0x01
        \\        b = const 0x02
        \\        out = icall @ident a b
        \\        stop
        \\    }
    );
    defer program.deinit();

    var graph = try buildBlockGraphSimple(std.testing.allocator, program, "main", "entry", null);
    defer graph.deinit();

    const icall = graph.getOp(3);
    try std.testing.expectEqual(OpNodeKind{ .normal = try operationId(program, "main", "entry", 2) }, icall.kind);
    try std.testing.expectEqual(@as(usize, 2), icall.inputs_fifo.len);
}

fn parseTestProgram(source: []const u8) !ir.Program {
    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator, source, &bag);
    errdefer program.deinit();
    try std.testing.expectEqual(@as(usize, 0), bag.items.items.len);
    return program;
}

fn evmOp(mnemonic: []const u8) u8 {
    const evm_asm = @import("asm.zig");
    if (std.mem.eql(u8, mnemonic, "return")) return evm_asm.op.RETURN;
    if (std.mem.eql(u8, mnemonic, "mstore256")) return evm_asm.op.MSTORE;
    unreachable;
}

fn expectContainsStackOp(haystack: []const release_schedule.StackOp, needle: release_schedule.StackOp) !void {
    for (haystack) |candidate| {
        if (std.meta.eql(candidate, needle)) return;
    }
    return error.TestExpectedEqual;
}
