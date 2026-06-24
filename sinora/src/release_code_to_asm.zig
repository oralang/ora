//! Zig port of Plank's generic release `code_to_asm` emitter.
//!
//! This module consumes scheduled stack ops and emits EVM bytecode for generic
//! SIR blocks. It is separate from the old selector-shape scaffold and is the
//! bytecode emission half of the owned release backend.

const std = @import("std");

const evm_asm = @import("asm.zig");
const ir = @import("ir.zig");
const ops = @import("ops.zig");
const parser = @import("parser.zig");
const diagnostics = @import("diagnostics.zig");
const release_schedule = @import("release_schedule.zig");

pub const CodeToAsmError = error{
    UnsupportedSir,
    FunctionNotFound,
    BlockNotFound,
    MissingBlockSchedule,
    MissingInstruction,
    MissingLabel,
    MissingReturnDestination,
};

pub const AllocAddress = struct {
    alloc: release_schedule.AllocId,
    address: u32,
};

pub const StaticAllocAddress = struct {
    op: release_schedule.OpId,
    address: u32,
    needs_zeroing: bool,
};

pub const MemoryLayout = struct {
    alloc_start: []const AllocAddress = &.{},
    static_alloc_start: []const StaticAllocAddress = &.{},
    switch_store: ?u32 = null,
    dyn_free_pointer: ?FreePointer = null,
};

pub const FreePointer = struct {
    store_slot: u32,
    start_value: u32,
};

pub const BlockSchedule = struct {
    function_name: []const u8,
    block_name: []const u8,
    ops: []const release_schedule.StackOp,
};

pub const SourceMapEntry = struct {
    idx: u32,
    pc: u32,
};

const SourceMapMark = struct {
    label: evm_asm.Label,
    idx: u32,
};

pub const SourceIndexMap = struct {
    allocator: std.mem.Allocator,
    op_indices: []const OpSourceIndex,
    control_indices: []const ControlSourceIndex,

    const OpSourceIndex = struct {
        function_name: []const u8,
        block_name: []const u8,
        instruction_index: usize,
        idx: u32,
    };

    const ControlSourceIndex = struct {
        function_name: []const u8,
        block_name: []const u8,
        idx: u32,
    };

    pub fn deinit(self: *SourceIndexMap) void {
        self.allocator.free(self.control_indices);
        self.allocator.free(self.op_indices);
        self.* = undefined;
    }

    pub fn fromProgram(allocator: std.mem.Allocator, program: ir.Program) !SourceIndexMap {
        var builder = SourceIndexBuilder.init(allocator, program);
        defer builder.deinit();
        try builder.visitEntry("init");
        try builder.visitEntry("main");
        return .{
            .allocator = allocator,
            .op_indices = try builder.op_indices.toOwnedSlice(allocator),
            .control_indices = try builder.control_indices.toOwnedSlice(allocator),
        };
    }

    fn opIndex(self: SourceIndexMap, function_name: []const u8, block_name: []const u8, instruction_index: usize) ?u32 {
        for (self.op_indices) |entry| {
            if (entry.instruction_index == instruction_index and
                std.mem.eql(u8, entry.function_name, function_name) and
                std.mem.eql(u8, entry.block_name, block_name))
            {
                return entry.idx;
            }
        }
        return null;
    }

    fn controlIndex(self: SourceIndexMap, function_name: []const u8, block_name: []const u8) ?u32 {
        for (self.control_indices) |entry| {
            if (std.mem.eql(u8, entry.function_name, function_name) and
                std.mem.eql(u8, entry.block_name, block_name))
            {
                return entry.idx;
            }
        }
        return null;
    }
};

pub const EmitResult = struct {
    allocator: std.mem.Allocator,
    bytes: []const u8,
    source_map: []const SourceMapEntry,
    runtime_start_pc: u32,

    pub fn deinit(self: *EmitResult) void {
        self.allocator.free(self.source_map);
        self.allocator.free(self.bytes);
        self.* = undefined;
    }
};

const SourceIndexBuilder = struct {
    allocator: std.mem.Allocator,
    program: ir.Program,
    op_indices: std.ArrayList(SourceIndexMap.OpSourceIndex) = .empty,
    control_indices: std.ArrayList(SourceIndexMap.ControlSourceIndex) = .empty,
    worklist: std.ArrayList(BlockRef) = .empty,
    visited: std.ArrayList(BlockRef) = .empty,
    next_idx: u32 = 0,

    fn init(allocator: std.mem.Allocator, program: ir.Program) SourceIndexBuilder {
        return .{
            .allocator = allocator,
            .program = program,
        };
    }

    fn deinit(self: *SourceIndexBuilder) void {
        self.visited.deinit(self.allocator);
        self.worklist.deinit(self.allocator);
        self.control_indices.deinit(self.allocator);
        self.op_indices.deinit(self.allocator);
    }

    fn visitEntry(self: *SourceIndexBuilder, function_name: []const u8) !void {
        const function = findFunction(self.program, function_name) orelse return;
        if (function.blocks.len == 0) return;
        try self.enqueue(function.name, function.blocks[0].name);

        while (self.worklist.pop()) |block_ref| {
            if (self.hasVisited(block_ref.function_name, block_ref.block_name)) continue;
            try self.visited.append(self.allocator, block_ref);

            const function_for_block = findFunction(self.program, block_ref.function_name) orelse return CodeToAsmError.FunctionNotFound;
            const block = findBlock(function_for_block, block_ref.block_name) orelse return CodeToAsmError.BlockNotFound;

            var callees: std.ArrayList([]const u8) = .empty;
            defer callees.deinit(self.allocator);
            for (block.instructions, 0..) |instruction, instruction_index| {
                try self.op_indices.append(self.allocator, .{
                    .function_name = function_for_block.name,
                    .block_name = block.name,
                    .instruction_index = instruction_index,
                    .idx = self.takeIndex(),
                });
                if (std.mem.eql(u8, instruction.mnemonic, "icall")) {
                    if (instruction.operands.len != 0 and instruction.operands[0].len > 1 and instruction.operands[0][0] == '@') {
                        try callees.append(self.allocator, instruction.operands[0][1..]);
                    }
                }
            }

            try self.control_indices.append(self.allocator, .{
                .function_name = function_for_block.name,
                .block_name = block.name,
                .idx = self.takeIndex(),
            });

            for (callees.items) |callee_name| {
                const callee = findFunction(self.program, callee_name) orelse continue;
                if (callee.blocks.len != 0) try self.enqueue(callee.name, callee.blocks[0].name);
            }

            var successors = ir.successors(&block);
            while (successors.next()) |target| {
                try self.enqueue(function_for_block.name, target);
            }
        }
    }

    fn takeIndex(self: *SourceIndexBuilder) u32 {
        const result = self.next_idx;
        self.next_idx += 1;
        return result;
    }

    fn enqueue(self: *SourceIndexBuilder, function_name: []const u8, block_name: []const u8) !void {
        try self.worklist.append(self.allocator, .{ .function_name = function_name, .block_name = block_name });
    }

    fn hasVisited(self: SourceIndexBuilder, function_name: []const u8, block_name: []const u8) bool {
        for (self.visited.items) |entry| {
            if (std.mem.eql(u8, entry.function_name, function_name) and
                std.mem.eql(u8, entry.block_name, block_name))
            {
                return true;
            }
        }
        return false;
    }
};

const ReferenceMode = enum {
    direct,
    runcode_delta,
};

const EmitContext = struct {
    reference_mode: ReferenceMode,
    allow_initcode_introspection: bool,
};

const BlockLabel = struct {
    function_name: []const u8,
    block_name: []const u8,
    label: evm_asm.Label,
};

const PendingBlock = struct {
    function_name: []const u8,
    block_name: []const u8,
};

const VisitedBlock = struct {
    function_name: []const u8,
    block_name: []const u8,
};

const OperationEntry = struct {
    function_index: usize,
    block_index: usize,
    instruction_index: usize,
};

const IcallReturnMark = struct {
    op: release_schedule.OpId,
    label: evm_asm.Label,
};

const BlockRef = struct {
    function_name: []const u8,
    block_name: []const u8,
};

pub fn emitFromEntry(
    allocator: std.mem.Allocator,
    program: ir.Program,
    entry_function_name: []const u8,
    schedules: []const BlockSchedule,
    layout: MemoryLayout,
) ![]const u8 {
    var emitter = GenericEmitter.init(allocator, program, schedules, layout, null);
    defer emitter.deinit();
    try emitter.prepare();
    try emitter.markRuncodeStart();
    const entry = findFunction(program, entry_function_name) orelse return CodeToAsmError.FunctionNotFound;
    try emitter.emitFromFunction(entry, .{
        .reference_mode = .direct,
        .allow_initcode_introspection = true,
    });
    try emitter.emitUsedData(null);
    try emitter.markInitcodeEnd();
    return emitter.bytecode.toOwnedSlice();
}

pub fn emitFromEntryWithSourceMap(
    allocator: std.mem.Allocator,
    program: ir.Program,
    entry_function_name: []const u8,
    schedules: []const BlockSchedule,
    layout: MemoryLayout,
    source_indices: *const SourceIndexMap,
) !EmitResult {
    var emitter = GenericEmitter.init(allocator, program, schedules, layout, source_indices);
    defer emitter.deinit();
    try emitter.prepare();
    try emitter.markRuncodeStart();
    const entry = findFunction(program, entry_function_name) orelse return CodeToAsmError.FunctionNotFound;
    try emitter.emitFromFunction(entry, .{
        .reference_mode = .direct,
        .allow_initcode_introspection = true,
    });
    try emitter.emitUsedData(null);
    try emitter.markInitcodeEnd();
    return emitter.finishWithSourceMap();
}

pub fn emitDeployment(
    allocator: std.mem.Allocator,
    program: ir.Program,
    init_function_name: []const u8,
    runtime_function_name: ?[]const u8,
    schedules: []const BlockSchedule,
    init_layout: MemoryLayout,
    runtime_layout: MemoryLayout,
) ![]const u8 {
    var runtime_data = try collectRuntimeData(allocator, program, runtime_function_name);
    defer runtime_data.deinit(allocator);

    var emitter = GenericEmitter.init(allocator, program, schedules, init_layout, null);
    defer emitter.deinit();
    try emitter.prepare();

    const init_entry = findFunction(program, init_function_name) orelse return CodeToAsmError.FunctionNotFound;
    emitter.layout = init_layout;
    try emitter.emitFromFunction(init_entry, .{
        .reference_mode = .direct,
        .allow_initcode_introspection = true,
    });
    try emitter.emitUsedData(runtime_data.items);

    try emitter.markRuncodeStart();
    if (runtime_function_name) |runtime_name| {
        const runtime_entry = findFunction(program, runtime_name) orelse return CodeToAsmError.FunctionNotFound;
        emitter.layout = runtime_layout;
        try emitter.emitFromFunction(runtime_entry, .{
            .reference_mode = .runcode_delta,
            .allow_initcode_introspection = false,
        });
        try emitter.emitSelectedData(runtime_data.items);
    }
    try emitter.markInitcodeEnd();
    return emitter.bytecode.toOwnedSlice();
}

pub fn emitDeploymentWithSourceMap(
    allocator: std.mem.Allocator,
    program: ir.Program,
    init_function_name: []const u8,
    runtime_function_name: ?[]const u8,
    schedules: []const BlockSchedule,
    init_layout: MemoryLayout,
    runtime_layout: MemoryLayout,
    source_indices: *const SourceIndexMap,
) !EmitResult {
    var runtime_data = try collectRuntimeData(allocator, program, runtime_function_name);
    defer runtime_data.deinit(allocator);

    var emitter = GenericEmitter.init(allocator, program, schedules, init_layout, source_indices);
    defer emitter.deinit();
    try emitter.prepare();

    const init_entry = findFunction(program, init_function_name) orelse return CodeToAsmError.FunctionNotFound;
    emitter.layout = init_layout;
    try emitter.emitFromFunction(init_entry, .{
        .reference_mode = .direct,
        .allow_initcode_introspection = true,
    });
    try emitter.emitUsedData(runtime_data.items);

    try emitter.markRuncodeStart();
    if (runtime_function_name) |runtime_name| {
        const runtime_entry = findFunction(program, runtime_name) orelse return CodeToAsmError.FunctionNotFound;
        emitter.layout = runtime_layout;
        try emitter.emitFromFunction(runtime_entry, .{
            .reference_mode = .runcode_delta,
            .allow_initcode_introspection = false,
        });
        try emitter.emitSelectedData(runtime_data.items);
    }
    try emitter.markInitcodeEnd();
    return emitter.finishWithSourceMap();
}

pub fn operationId(
    program: ir.Program,
    function_name: []const u8,
    block_name: []const u8,
    instruction_index: usize,
) !release_schedule.OpId {
    var op_id: release_schedule.OpId = 0;
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
    return CodeToAsmError.MissingInstruction;
}

const GenericEmitter = struct {
    allocator: std.mem.Allocator,
    program: ir.Program,
    schedules: []const BlockSchedule,
    layout: MemoryLayout,
    bytecode: evm_asm.LabelBytecode,
    block_labels: std.ArrayList(BlockLabel) = .empty,
    data_labels: std.ArrayList(evm_asm.Label) = .empty,
    used_data: std.ArrayList(bool) = .empty,
    pending_blocks: std.ArrayList(PendingBlock) = .empty,
    visited_blocks: std.ArrayList(VisitedBlock) = .empty,
    operations: std.ArrayList(OperationEntry) = .empty,
    icall_return_marks: std.ArrayList(IcallReturnMark) = .empty,
    source_indices: ?*const SourceIndexMap = null,
    source_map_marks: std.ArrayList(SourceMapMark) = .empty,
    runcode_start: ?evm_asm.Label = null,
    initcode_end: ?evm_asm.Label = null,

    fn init(
        allocator: std.mem.Allocator,
        program: ir.Program,
        schedules: []const BlockSchedule,
        layout: MemoryLayout,
        source_indices: ?*const SourceIndexMap,
    ) GenericEmitter {
        return .{
            .allocator = allocator,
            .program = program,
            .schedules = schedules,
            .layout = layout,
            .source_indices = source_indices,
            .bytecode = evm_asm.LabelBytecode.init(allocator),
        };
    }

    fn deinit(self: *GenericEmitter) void {
        self.source_map_marks.deinit(self.allocator);
        self.icall_return_marks.deinit(self.allocator);
        self.operations.deinit(self.allocator);
        self.visited_blocks.deinit(self.allocator);
        self.pending_blocks.deinit(self.allocator);
        self.used_data.deinit(self.allocator);
        self.data_labels.deinit(self.allocator);
        self.block_labels.deinit(self.allocator);
        self.bytecode.deinit();
    }

    fn prepare(self: *GenericEmitter) !void {
        const stats = self.program.stats();
        const reserve_hint = evm_asm.estimateReserveHint(stats);
        try self.bytecode.reserve(
            reserve_hint.byte_capacity,
            reserve_hint.label_capacity,
            reserve_hint.patch_capacity,
        );
        try self.block_labels.ensureTotalCapacity(self.allocator, stats.blocks);
        try self.data_labels.ensureTotalCapacity(self.allocator, stats.data_segments);
        try self.used_data.ensureTotalCapacity(self.allocator, stats.data_segments);
        try self.operations.ensureTotalCapacity(self.allocator, stats.instructions);
        try self.source_map_marks.ensureTotalCapacity(self.allocator, stats.instructions + stats.terminators);
        try self.pending_blocks.ensureTotalCapacity(self.allocator, @max(stats.blocks, 1));
        try self.visited_blocks.ensureTotalCapacity(self.allocator, @max(stats.blocks, 1));

        self.runcode_start = try self.bytecode.newLabel();
        self.initcode_end = try self.bytecode.newLabel();
        for (self.program.data_segments) |_| {
            const label = try self.bytecode.newLabel();
            self.data_labels.appendAssumeCapacity(label);
            self.used_data.appendAssumeCapacity(false);
        }

        for (self.program.functions, 0..) |function, function_index| {
            for (function.blocks, 0..) |block, block_index| {
                const label = try self.bytecode.newLabel();
                self.block_labels.appendAssumeCapacity(.{
                    .function_name = function.name,
                    .block_name = block.name,
                    .label = label,
                });
                for (block.instructions, 0..) |_, instruction_index| {
                    self.operations.appendAssumeCapacity(.{
                        .function_index = function_index,
                        .block_index = block_index,
                        .instruction_index = instruction_index,
                    });
                }
            }
        }
    }

    fn markRuncodeStart(self: *GenericEmitter) !void {
        try self.bytecode.mark(self.runcode_start orelse return CodeToAsmError.MissingLabel);
    }

    fn markInitcodeEnd(self: *GenericEmitter) !void {
        try self.bytecode.mark(self.initcode_end orelse return CodeToAsmError.MissingLabel);
    }

    fn finishWithSourceMap(self: *GenericEmitter) !EmitResult {
        var finalized = try self.bytecode.toOwnedFinalized();
        errdefer finalized.deinit();

        const entries = try self.allocator.alloc(SourceMapEntry, self.source_map_marks.items.len);
        errdefer self.allocator.free(entries);
        for (self.source_map_marks.items, entries) |mark, *entry| {
            entry.* = .{
                .idx = mark.idx,
                .pc = evm_asm.LabelBytecode.labelOffset(finalized, mark.label) orelse return CodeToAsmError.MissingLabel,
            };
        }

        const runtime_start_pc = evm_asm.LabelBytecode.labelOffset(
            finalized,
            self.runcode_start orelse return CodeToAsmError.MissingLabel,
        ) orelse return CodeToAsmError.MissingLabel;
        const bytes = finalized.bytes;
        self.allocator.free(finalized.label_offsets);
        finalized.bytes = &.{};
        finalized.label_offsets = &.{};
        return .{
            .allocator = self.allocator,
            .bytes = bytes,
            .source_map = entries,
            .runtime_start_pc = runtime_start_pc,
        };
    }

    fn pushSourceMapMark(self: *GenericEmitter, idx: u32) !void {
        const label = try self.bytecode.newLabel();
        try self.bytecode.mark(label);
        try self.source_map_marks.append(self.allocator, .{ .label = label, .idx = idx });
    }

    fn pushOpSourceMapMark(self: *GenericEmitter, op_id: release_schedule.OpId) !void {
        const source_indices = self.source_indices orelse return;
        if (op_id >= self.operations.items.len) return CodeToAsmError.MissingInstruction;
        const entry = self.operations.items[op_id];
        const function = self.program.functions[entry.function_index];
        const block = function.blocks[entry.block_index];
        const idx = source_indices.opIndex(function.name, block.name, entry.instruction_index) orelse return;
        try self.pushSourceMapMark(idx);
    }

    fn pushControlSourceMapMark(self: *GenericEmitter, function: ir.Function, block: ir.Block) !void {
        const source_indices = self.source_indices orelse return;
        const idx = source_indices.controlIndex(function.name, block.name) orelse return;
        try self.pushSourceMapMark(idx);
    }

    fn emitFromFunction(self: *GenericEmitter, entry: ir.Function, context: EmitContext) !void {
        self.visited_blocks.clearRetainingCapacity();
        self.pending_blocks.clearRetainingCapacity();
        self.icall_return_marks.clearRetainingCapacity();

        if (self.layout.dyn_free_pointer) |free_pointer| {
            try self.bytecode.pushU32(free_pointer.start_value);
            try self.bytecode.pushU32(free_pointer.store_slot);
            try self.bytecode.pushOp(evm_asm.op.MSTORE);
        }

        if (entry.blocks.len == 0) return;
        try self.enqueueBlock(entry.name, entry.blocks[0].name);

        while (self.pending_blocks.pop()) |pending| {
            const function = findFunction(self.program, pending.function_name) orelse return CodeToAsmError.FunctionNotFound;
            const block = findBlock(function, pending.block_name) orelse return CodeToAsmError.BlockNotFound;

            const label = self.blockLabel(pending.function_name, pending.block_name) orelse return CodeToAsmError.MissingLabel;
            try self.bytecode.markJumpDest(label);

            const schedule = self.blockSchedule(pending.function_name, pending.block_name) orelse return CodeToAsmError.MissingBlockSchedule;
            for (schedule.ops) |stack_op| {
                try self.emitStackOp(stack_op, context);
            }

            try self.emitControl(function, block, context);
            try self.enqueueSuccessors(function, block);
        }
    }

    fn emitStackOp(self: *GenericEmitter, stack_op: release_schedule.StackOp, context: EmitContext) !void {
        switch (stack_op) {
            .swap => |depth| {
                if (depth == 0) return CodeToAsmError.UnsupportedSir;
                try self.bytecode.pushOp(evm_asm.op.SWAP1 + depth - 1);
            },
            .dup => |depth| try self.bytecode.pushOp(evm_asm.op.DUP1 + depth),
            .pop => try self.bytecode.pushOp(evm_asm.op.POP),
            .exchange => return CodeToAsmError.UnsupportedSir,
            .store => |alloc| {
                try self.bytecode.pushU32(try self.allocAddress(alloc));
                try self.bytecode.pushOp(evm_asm.op.MSTORE);
            },
            .load => |alloc| {
                try self.bytecode.pushU32(try self.allocAddress(alloc));
                try self.bytecode.pushOp(evm_asm.op.MLOAD);
            },
            .call_ret_push => |op_id| {
                const label = try self.bytecode.newLabel();
                try self.icall_return_marks.append(self.allocator, .{ .op = op_id, .label = label });
                try self.pushLabelRef(label, context);
            },
            .op => |op_id| try self.emitOperation(op_id, context),
        }
    }

    fn emitOperation(self: *GenericEmitter, op_id: release_schedule.OpId, context: EmitContext) !void {
        try self.pushOpSourceMapMark(op_id);
        const instruction = self.instructionById(op_id) orelse return CodeToAsmError.MissingInstruction;
        const spec = ops.lookup(instruction.mnemonic) orelse return CodeToAsmError.UnsupportedSir;
        switch (spec) {
            .fixed => |fixed| try self.emitFixedOperation(op_id, instruction, fixed, context),
            .memory_load => |memory| try self.emitMemoryLoad(memory.bits),
            .memory_store => |memory| try self.emitMemoryStore(memory.bits),
            .internal_call => try self.emitInternalCall(op_id, instruction, context),
        }
    }

    fn emitFixedOperation(self: *GenericEmitter, op_id: release_schedule.OpId, instruction: ir.Instruction, fixed: ops.Fixed, context: EmitContext) !void {
        if (std.mem.eql(u8, instruction.mnemonic, "const") or std.mem.eql(u8, instruction.mnemonic, "large_const")) {
            try self.bytecode.pushU256(parseU256(instruction.operands[0]) orelse return CodeToAsmError.UnsupportedSir);
            return;
        }

        if (std.mem.eql(u8, instruction.mnemonic, "copy") or std.mem.eql(u8, instruction.mnemonic, "noop")) {
            return;
        }

        if (std.mem.eql(u8, instruction.mnemonic, "malloc")) {
            if (instruction.operands.len != 1 or instruction.results.len != 1) return CodeToAsmError.UnsupportedSir;
            try self.emitDynamicAllocZeroed();
            return;
        }

        if (std.mem.eql(u8, instruction.mnemonic, "mallocany")) {
            if (instruction.operands.len != 1 or instruction.results.len != 1) return CodeToAsmError.UnsupportedSir;
            try self.emitDynamicAllocAnyBytes();
            return;
        }

        if (std.mem.eql(u8, instruction.mnemonic, "freeptr")) {
            if (instruction.operands.len != 0 or instruction.results.len != 1) return CodeToAsmError.UnsupportedSir;
            try self.emitAcquireFreePointer();
            return;
        }

        if (std.mem.eql(u8, instruction.mnemonic, "salloc") or std.mem.eql(u8, instruction.mnemonic, "sallocany")) {
            if (instruction.operands.len != 1 or instruction.results.len != 1) return CodeToAsmError.UnsupportedSir;
            try self.emitStaticAlloc(op_id, instruction);
            return;
        }

        if (std.mem.eql(u8, instruction.mnemonic, "data_offset")) {
            try self.emitDataOffset(instruction, context);
            return;
        }

        if (std.mem.eql(u8, instruction.mnemonic, "runtime_start_offset") or
            std.mem.eql(u8, instruction.mnemonic, "init_end_offset") or
            std.mem.eql(u8, instruction.mnemonic, "runtime_length"))
        {
            try self.emitDeploymentOffsetIntrinsic(instruction, context);
            return;
        }

        if (fixed.extra != .none) return CodeToAsmError.UnsupportedSir;
        const opcode = evm_asm.evmOpcode(instruction.mnemonic) orelse return CodeToAsmError.UnsupportedSir;
        try self.bytecode.pushOp(opcode);
    }

    fn emitDynamicAllocZeroed(self: *GenericEmitter) !void {
        const free_pointer = self.layout.dyn_free_pointer orelse return CodeToAsmError.UnsupportedSir;
        try self.bytecode.pushU32(free_pointer.store_slot);
        try self.bytecode.pushOp(evm_asm.op.MLOAD);
        try self.bytecode.pushOp(evm_asm.op.DUP1 + 1);
        try self.bytecode.pushOp(evm_asm.op.DUP1 + 1);
        try self.bytecode.pushOp(evm_asm.op.ADD);
        try self.bytecode.pushU32(free_pointer.store_slot);
        try self.bytecode.pushOp(evm_asm.op.MSTORE);
        try self.bytecode.pushOp(evm_asm.op.SWAP1);
        try self.bytecode.pushOp(evm_asm.op.CALLDATASIZE);
        try self.bytecode.pushOp(evm_asm.op.DUP1 + 2);
        try self.bytecode.pushOp(evm_asm.op.CALLDATACOPY);
    }

    fn emitDynamicAllocAnyBytes(self: *GenericEmitter) !void {
        const free_pointer = self.layout.dyn_free_pointer orelse return CodeToAsmError.UnsupportedSir;
        try self.bytecode.pushU32(free_pointer.store_slot);
        try self.bytecode.pushOp(evm_asm.op.MLOAD);
        try self.bytecode.pushOp(evm_asm.op.SWAP1);
        try self.bytecode.pushOp(evm_asm.op.DUP1 + 1);
        try self.bytecode.pushOp(evm_asm.op.ADD);
        try self.bytecode.pushU32(free_pointer.store_slot);
        try self.bytecode.pushOp(evm_asm.op.MSTORE);
    }

    fn emitAcquireFreePointer(self: *GenericEmitter) !void {
        const free_pointer = self.layout.dyn_free_pointer orelse return CodeToAsmError.UnsupportedSir;
        try self.bytecode.pushU32(free_pointer.store_slot);
        try self.bytecode.pushOp(evm_asm.op.MLOAD);
    }

    fn emitStaticAlloc(self: *GenericEmitter, op_id: release_schedule.OpId, instruction: ir.Instruction) !void {
        const static_alloc = self.staticAllocAddress(op_id) orelse return CodeToAsmError.UnsupportedSir;
        try self.bytecode.pushU32(static_alloc.address);
        if (static_alloc.needs_zeroing) {
            try self.bytecode.pushU32(parseU32(instruction.operands[0]) orelse return CodeToAsmError.UnsupportedSir);
            try self.bytecode.pushOp(evm_asm.op.CALLDATASIZE);
            try self.bytecode.pushOp(evm_asm.op.DUP1 + 2);
            try self.bytecode.pushOp(evm_asm.op.CALLDATACOPY);
        }
    }

    fn emitDataOffset(self: *GenericEmitter, instruction: ir.Instruction, context: EmitContext) !void {
        if (instruction.operands.len != 1 or instruction.results.len != 1) return CodeToAsmError.UnsupportedSir;
        const data_index = findDataSegment(self.program, instruction.operands[0]) orelse return CodeToAsmError.UnsupportedSir;
        self.used_data.items[data_index] = true;
        try self.pushLabelRef(self.data_labels.items[data_index], context);
    }

    fn emitDeploymentOffsetIntrinsic(self: *GenericEmitter, instruction: ir.Instruction, context: EmitContext) !void {
        if (!context.allow_initcode_introspection) return CodeToAsmError.UnsupportedSir;
        if (instruction.operands.len != 0 or instruction.results.len != 1) return CodeToAsmError.UnsupportedSir;

        if (std.mem.eql(u8, instruction.mnemonic, "runtime_start_offset")) {
            try self.bytecode.pushLabelRef(self.runcode_start orelse return CodeToAsmError.MissingLabel);
        } else if (std.mem.eql(u8, instruction.mnemonic, "init_end_offset")) {
            try self.bytecode.pushLabelRef(self.initcode_end orelse return CodeToAsmError.MissingLabel);
        } else if (std.mem.eql(u8, instruction.mnemonic, "runtime_length")) {
            try self.bytecode.pushLabelDelta(
                self.runcode_start orelse return CodeToAsmError.MissingLabel,
                self.initcode_end orelse return CodeToAsmError.MissingLabel,
            );
        } else {
            return CodeToAsmError.UnsupportedSir;
        }
    }

    fn emitMemoryLoad(self: *GenericEmitter, bits: u16) !void {
        try self.bytecode.pushOp(evm_asm.op.MLOAD);
        if (bits != 256) {
            try self.bytecode.pushU32(256 - bits);
            try self.bytecode.pushOp(evm_asm.op.SHR);
        }
    }

    fn emitMemoryStore(self: *GenericEmitter, bits: u16) !void {
        switch (bits) {
            8 => try self.bytecode.pushOp(evm_asm.op.MSTORE8),
            256 => try self.bytecode.pushOp(evm_asm.op.MSTORE),
            else => {
                try self.bytecode.pushOp(evm_asm.op.SWAP1);
                try self.bytecode.pushU32(256 - bits);
                try self.bytecode.pushOp(evm_asm.op.SHL);
                try self.bytecode.pushOp(evm_asm.op.DUP1 + 1);
                try self.bytecode.pushOp(evm_asm.op.MLOAD);

                if (bits >= 224) {
                    const preserved_bits: u8 = @intCast(256 - bits);
                    const preserved_word_mask = (@as(u256, 1) << preserved_bits) - 1;
                    try self.bytecode.pushU256(preserved_word_mask);
                    try self.bytecode.pushOp(evm_asm.op.AND);
                } else {
                    try self.bytecode.pushU32(bits);
                    try self.bytecode.pushOp(evm_asm.op.SHL);
                    try self.bytecode.pushU32(bits);
                    try self.bytecode.pushOp(evm_asm.op.SHR);
                }

                try self.bytecode.pushOp(evm_asm.op.XOR);
                try self.bytecode.pushOp(evm_asm.op.SWAP1);
                try self.bytecode.pushOp(evm_asm.op.MSTORE);
            },
        }
    }

    fn emitInternalCall(self: *GenericEmitter, op_id: release_schedule.OpId, instruction: ir.Instruction, context: EmitContext) !void {
        if (instruction.operands.len == 0 or instruction.operands[0].len <= 1 or instruction.operands[0][0] != '@') {
            return CodeToAsmError.UnsupportedSir;
        }
        const callee_name = instruction.operands[0][1..];
        const callee = findFunction(self.program, callee_name) orelse return CodeToAsmError.FunctionNotFound;
        if (callee.blocks.len == 0) return CodeToAsmError.BlockNotFound;
        try self.enqueueBlock(callee.name, callee.blocks[0].name);

        const return_label = self.takeReturnMark(op_id) orelse return CodeToAsmError.MissingReturnDestination;
        const callee_label = self.blockLabel(callee.name, callee.blocks[0].name) orelse return CodeToAsmError.MissingLabel;
        try self.pushLabelRef(callee_label, context);
        try self.bytecode.pushOp(evm_asm.op.JUMP);
        try self.bytecode.markJumpDest(return_label);
    }

    fn emitControl(self: *GenericEmitter, function: ir.Function, block: ir.Block, context: EmitContext) !void {
        try self.pushControlSourceMapMark(function, block);
        switch (block.terminator) {
            .jump => |target| {
                try self.pushBlockLabel(function.name, target, context);
                try self.bytecode.pushOp(evm_asm.op.JUMP);
            },
            .branch => |branch| {
                try self.pushBlockLabel(function.name, branch.non_zero_target, context);
                try self.bytecode.pushOp(evm_asm.op.JUMPI);
                try self.pushBlockLabel(function.name, branch.zero_target, context);
                try self.bytecode.pushOp(evm_asm.op.JUMP);
            },
            .switch_ => |switch_term| {
                const switch_store = self.layout.switch_store orelse return CodeToAsmError.UnsupportedSir;
                try self.bytecode.pushU32(switch_store);
                try self.bytecode.pushOp(evm_asm.op.MSTORE);

                for (switch_term.cases) |case| {
                    try self.bytecode.pushU32(switch_store);
                    try self.bytecode.pushOp(evm_asm.op.MLOAD);
                    try self.bytecode.pushU256(parseU256(case.value) orelse return CodeToAsmError.UnsupportedSir);
                    try self.bytecode.pushOp(evm_asm.op.EQ);
                    try self.pushBlockLabel(function.name, case.target, context);
                    try self.bytecode.pushOp(evm_asm.op.JUMPI);
                }

                if (switch_term.default_target.len != 0) {
                    try self.pushBlockLabel(function.name, switch_term.default_target, context);
                    try self.bytecode.pushOp(evm_asm.op.JUMP);
                }
            },
            .iret => try self.bytecode.pushOp(evm_asm.op.JUMP),
            .return_ => try self.bytecode.pushOp(evm_asm.op.RETURN),
            .revert => try self.bytecode.pushOp(evm_asm.op.REVERT),
            .stop => try self.bytecode.pushOp(evm_asm.op.STOP),
            .invalid => try self.bytecode.pushOp(evm_asm.op.INVALID),
            .selfdestruct => try self.bytecode.pushOp(evm_asm.op.SELFDESTRUCT),
        }
    }

    fn enqueueSuccessors(self: *GenericEmitter, function: ir.Function, block: ir.Block) !void {
        switch (block.terminator) {
            .jump => |target| try self.enqueueBlock(function.name, target),
            .branch => |branch| {
                try self.enqueueBlock(function.name, branch.zero_target);
                try self.enqueueBlock(function.name, branch.non_zero_target);
            },
            .switch_ => |switch_term| {
                for (switch_term.cases) |case| try self.enqueueBlock(function.name, case.target);
                if (switch_term.default_target.len != 0) try self.enqueueBlock(function.name, switch_term.default_target);
            },
            .return_, .revert, .stop, .invalid, .selfdestruct, .iret => {},
        }
    }

    fn enqueueBlock(self: *GenericEmitter, function_name: []const u8, block_name: []const u8) !void {
        if (self.hasVisitedBlock(function_name, block_name)) return;
        try self.visited_blocks.append(self.allocator, .{
            .function_name = function_name,
            .block_name = block_name,
        });
        try self.pending_blocks.append(self.allocator, .{ .function_name = function_name, .block_name = block_name });
    }

    fn hasVisitedBlock(self: GenericEmitter, function_name: []const u8, block_name: []const u8) bool {
        for (self.visited_blocks.items) |visited| {
            if (std.mem.eql(u8, visited.function_name, function_name) and std.mem.eql(u8, visited.block_name, block_name)) return true;
        }
        return false;
    }

    fn pushBlockLabel(self: *GenericEmitter, function_name: []const u8, block_name: []const u8, context: EmitContext) !void {
        const label = self.blockLabel(function_name, block_name) orelse return CodeToAsmError.MissingLabel;
        try self.pushLabelRef(label, context);
    }

    fn pushLabelRef(self: *GenericEmitter, label: evm_asm.Label, context: EmitContext) !void {
        switch (context.reference_mode) {
            .direct => try self.bytecode.pushLabelRef(label),
            .runcode_delta => try self.bytecode.pushLabelDelta(
                self.runcode_start orelse return CodeToAsmError.MissingLabel,
                label,
            ),
        }
    }

    fn emitUsedData(self: *GenericEmitter, exclude_data: ?[]const usize) !void {
        for (self.used_data.items, 0..) |used, index| {
            if (!used or containsIndex(exclude_data, index)) continue;
            try self.emitDataSegment(index);
        }
    }

    fn emitSelectedData(self: *GenericEmitter, data_indices: []const usize) !void {
        for (data_indices) |index| {
            try self.emitDataSegment(index);
        }
    }

    fn emitDataSegment(self: *GenericEmitter, index: usize) !void {
        try self.bytecode.mark(self.data_labels.items[index]);
        try self.bytecode.pushData(self.program.data_segments[index].bytes);
    }

    fn blockLabel(self: GenericEmitter, function_name: []const u8, block_name: []const u8) ?evm_asm.Label {
        for (self.block_labels.items) |entry| {
            if (std.mem.eql(u8, entry.function_name, function_name) and std.mem.eql(u8, entry.block_name, block_name)) return entry.label;
        }
        return null;
    }

    fn blockSchedule(self: GenericEmitter, function_name: []const u8, block_name: []const u8) ?BlockSchedule {
        for (self.schedules) |schedule| {
            if (std.mem.eql(u8, schedule.function_name, function_name) and std.mem.eql(u8, schedule.block_name, block_name)) return schedule;
        }
        return null;
    }

    fn instructionById(self: GenericEmitter, op_id: release_schedule.OpId) ?ir.Instruction {
        if (op_id >= self.operations.items.len) return null;
        const entry = self.operations.items[op_id];
        return self.program.functions[entry.function_index].blocks[entry.block_index].instructions[entry.instruction_index];
    }

    fn takeReturnMark(self: *GenericEmitter, op_id: release_schedule.OpId) ?evm_asm.Label {
        for (self.icall_return_marks.items, 0..) |mark, index| {
            if (mark.op != op_id) continue;
            _ = self.icall_return_marks.swapRemove(index);
            return mark.label;
        }
        return null;
    }

    fn allocAddress(self: GenericEmitter, alloc: release_schedule.AllocId) !u32 {
        for (self.layout.alloc_start) |entry| {
            if (entry.alloc == alloc) return entry.address;
        }
        return CodeToAsmError.UnsupportedSir;
    }

    fn staticAllocAddress(self: GenericEmitter, op_id: release_schedule.OpId) ?StaticAllocAddress {
        for (self.layout.static_alloc_start) |entry| {
            if (entry.op == op_id) return entry;
        }
        return null;
    }
};

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

fn findDataSegment(program: ir.Program, name: []const u8) ?usize {
    const canonical = canonicalDataName(name);
    for (program.data_segments, 0..) |segment, index| {
        if (std.mem.eql(u8, segment.name, canonical)) return index;
    }
    return null;
}

fn canonicalDataName(name: []const u8) []const u8 {
    if (name.len > 0 and name[0] == '.') return name[1..];
    return name;
}

fn containsIndex(indices: ?[]const usize, needle: usize) bool {
    const items = indices orelse return false;
    for (items) |index| {
        if (index == needle) return true;
    }
    return false;
}

fn collectRuntimeData(
    allocator: std.mem.Allocator,
    program: ir.Program,
    runtime_function_name: ?[]const u8,
) !std.ArrayList(usize) {
    var result: std.ArrayList(usize) = .empty;
    errdefer result.deinit(allocator);

    const runtime_name = runtime_function_name orelse return result;
    const used_data = try allocator.alloc(bool, program.data_segments.len);
    defer allocator.free(used_data);
    @memset(used_data, false);

    var function_worklist: std.ArrayList([]const u8) = .empty;
    defer function_worklist.deinit(allocator);
    var seen_functions: std.ArrayList([]const u8) = .empty;
    defer seen_functions.deinit(allocator);
    var block_worklist: std.ArrayList(BlockRef) = .empty;
    defer block_worklist.deinit(allocator);
    var seen_blocks: std.ArrayList(BlockRef) = .empty;
    defer seen_blocks.deinit(allocator);

    const stats = program.stats();
    try result.ensureTotalCapacity(allocator, program.data_segments.len);
    try function_worklist.ensureTotalCapacity(allocator, @max(program.functions.len, 1));
    try seen_functions.ensureTotalCapacity(allocator, program.functions.len);
    try block_worklist.ensureTotalCapacity(allocator, @max(stats.blocks, 1));
    try seen_blocks.ensureTotalCapacity(allocator, stats.blocks);

    try enqueueFunctionName(allocator, &seen_functions, &function_worklist, runtime_name);
    while (function_worklist.pop()) |function_name| {
        const function = findFunction(program, function_name) orelse return CodeToAsmError.FunctionNotFound;
        if (function.blocks.len == 0) return CodeToAsmError.BlockNotFound;
        try enqueueBlockRef(allocator, &seen_blocks, &block_worklist, .{
            .function_name = function.name,
            .block_name = function.blocks[0].name,
        });

        while (block_worklist.pop()) |block_ref| {
            const block_function = findFunction(program, block_ref.function_name) orelse return CodeToAsmError.FunctionNotFound;
            const block = findBlock(block_function, block_ref.block_name) orelse return CodeToAsmError.BlockNotFound;

            for (block.instructions) |instruction| {
                if (std.mem.eql(u8, instruction.mnemonic, "data_offset")) {
                    if (instruction.operands.len != 1) return CodeToAsmError.UnsupportedSir;
                    const data_index = findDataSegment(program, instruction.operands[0]) orelse return CodeToAsmError.UnsupportedSir;
                    used_data[data_index] = true;
                } else if (std.mem.eql(u8, instruction.mnemonic, "icall")) {
                    if (instruction.operands.len == 0 or instruction.operands[0].len <= 1 or instruction.operands[0][0] != '@') {
                        return CodeToAsmError.UnsupportedSir;
                    }
                    try enqueueFunctionName(
                        allocator,
                        &seen_functions,
                        &function_worklist,
                        instruction.operands[0][1..],
                    );
                }
            }

            try enqueueTerminatorSuccessors(
                allocator,
                &seen_blocks,
                &block_worklist,
                block_function.name,
                block.terminator,
            );
        }
    }

    for (used_data, 0..) |used, index| {
        if (used) try result.append(allocator, index);
    }
    return result;
}

fn enqueueFunctionName(
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

fn enqueueTerminatorSuccessors(
    allocator: std.mem.Allocator,
    seen_blocks: *std.ArrayList(BlockRef),
    block_worklist: *std.ArrayList(BlockRef),
    function_name: []const u8,
    terminator: ir.Terminator,
) !void {
    switch (terminator) {
        .jump => |target| try enqueueBlockRef(allocator, seen_blocks, block_worklist, .{
            .function_name = function_name,
            .block_name = target,
        }),
        .branch => |branch| {
            try enqueueBlockRef(allocator, seen_blocks, block_worklist, .{
                .function_name = function_name,
                .block_name = branch.non_zero_target,
            });
            try enqueueBlockRef(allocator, seen_blocks, block_worklist, .{
                .function_name = function_name,
                .block_name = branch.zero_target,
            });
        },
        .switch_ => |switch_term| {
            for (switch_term.cases) |case| {
                try enqueueBlockRef(allocator, seen_blocks, block_worklist, .{
                    .function_name = function_name,
                    .block_name = case.target,
                });
            }
            if (switch_term.default_target.len != 0) {
                try enqueueBlockRef(allocator, seen_blocks, block_worklist, .{
                    .function_name = function_name,
                    .block_name = switch_term.default_target,
                });
            }
        },
        .return_, .revert, .stop, .invalid, .selfdestruct, .iret => {},
    }
}

fn enqueueBlockRef(
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

fn parseU256(text: []const u8) ?u256 {
    if (std.mem.startsWith(u8, text, "0x")) {
        return std.fmt.parseUnsigned(u256, text[2..], 16) catch null;
    }
    return std.fmt.parseUnsigned(u256, text, 10) catch null;
}

fn parseU32(text: []const u8) ?u32 {
    if (std.mem.startsWith(u8, text, "0x")) {
        return std.fmt.parseUnsigned(u32, text[2..], 16) catch null;
    }
    return std.fmt.parseUnsigned(u32, text, 10) catch null;
}

fn parseTestProgram(source: []const u8) !ir.Program {
    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator, source, &bag);
    errdefer program.deinit();
    try std.testing.expectEqual(@as(usize, 0), bag.items.items.len);
    return program;
}

fn testOperationId(program: ir.Program, function_name: []const u8, block_name: []const u8, instruction_index: usize) release_schedule.OpId {
    return operationId(program, function_name, block_name, instruction_index) catch unreachable;
}

fn countByte(bytes: []const u8, needle: u8) usize {
    var count: usize = 0;
    for (bytes) |byte| {
        if (byte == needle) count += 1;
    }
    return count;
}

test "generic code-to-asm emits scheduled straight-line stack operations" {
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

    const entry_ops = [_]release_schedule.StackOp{
        .{ .op = testOperationId(program, "main", "entry", 0) },
        .{ .op = testOperationId(program, "main", "entry", 1) },
        .{ .op = testOperationId(program, "main", "entry", 2) },
    };
    const schedules = [_]BlockSchedule{.{ .function_name = "main", .block_name = "entry", .ops = &entry_ops }};

    const bytes = try emitFromEntry(std.testing.allocator, program, "main", &schedules, .{});
    defer std.testing.allocator.free(bytes);

    try std.testing.expectEqualSlices(u8, &.{
        evm_asm.op.JUMPDEST,
        evm_asm.op.PUSH1,
        0x02,
        evm_asm.op.PUSH1,
        0x03,
        evm_asm.op.ADD,
        evm_asm.op.STOP,
    }, bytes);
}

test "generic code-to-asm emits scheduled branch control flow" {
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

    const entry_ops = [_]release_schedule.StackOp{.{ .op = testOperationId(program, "main", "entry", 0) }};
    const schedules = [_]BlockSchedule{
        .{ .function_name = "main", .block_name = "entry", .ops = &entry_ops },
        .{ .function_name = "main", .block_name = "success", .ops = &.{} },
        .{ .function_name = "main", .block_name = "failure", .ops = &.{} },
    };

    const bytes = try emitFromEntry(std.testing.allocator, program, "main", &schedules, .{});
    defer std.testing.allocator.free(bytes);

    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.JUMPI) != null);
    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.JUMP) != null);
    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.STOP) != null);
    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.INVALID) != null);
}

test "generic code-to-asm emits scheduler spill store and load" {
    var program = try parseTestProgram(
        \\fn main:
        \\    entry {
        \\        value = const 0x04
        \\        stop
        \\    }
    );
    defer program.deinit();

    const entry_ops = [_]release_schedule.StackOp{
        .{ .op = testOperationId(program, "main", "entry", 0) },
        .{ .store = 0 },
        .{ .load = 0 },
        .pop,
    };
    const schedules = [_]BlockSchedule{.{ .function_name = "main", .block_name = "entry", .ops = &entry_ops }};
    const allocs = [_]AllocAddress{.{ .alloc = 0, .address = 0x40 }};

    const bytes = try emitFromEntry(std.testing.allocator, program, "main", &schedules, .{ .alloc_start = &allocs });
    defer std.testing.allocator.free(bytes);

    try std.testing.expectEqualSlices(u8, &.{
        evm_asm.op.JUMPDEST,
        evm_asm.op.PUSH1,
        0x04,
        evm_asm.op.PUSH1,
        0x40,
        evm_asm.op.MSTORE,
        evm_asm.op.PUSH1,
        0x40,
        evm_asm.op.MLOAD,
        evm_asm.op.POP,
        evm_asm.op.STOP,
    }, bytes);
}

test "generic code-to-asm emits internal call return label pattern" {
    var program = try parseTestProgram(
        \\fn callee:
        \\    entry {
        \\        stop
        \\    }
        \\
        \\fn main:
        \\    entry {
        \\        icall @callee
        \\        stop
        \\    }
    );
    defer program.deinit();

    const call_id = testOperationId(program, "main", "entry", 0);
    const main_ops = [_]release_schedule.StackOp{
        .{ .call_ret_push = call_id },
        .{ .op = call_id },
    };
    const schedules = [_]BlockSchedule{
        .{ .function_name = "main", .block_name = "entry", .ops = &main_ops },
        .{ .function_name = "callee", .block_name = "entry", .ops = &.{} },
    };

    const bytes = try emitFromEntry(std.testing.allocator, program, "main", &schedules, .{});
    defer std.testing.allocator.free(bytes);

    try std.testing.expectEqual(@as(usize, 3), countByte(bytes, evm_asm.op.JUMPDEST));
    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.JUMP) != null);
    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.STOP) != null);
}
