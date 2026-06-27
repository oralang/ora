const std = @import("std");

const evm_asm = @import("asm.zig");
const diagnostics = @import("diagnostics.zig");
const ir = @import("ir.zig");
const ops = @import("ops.zig");
const parser = @import("parser.zig");

pub const CodegenError = error{
    UnsupportedSir,
    FunctionNotFound,
};

const word_size: u32 = 32;

// Debug bytecode uses a simple memory-backed execution model. SIR SSA values
// are stored in EVM memory slots, block parameters/results travel through a
// shared transfer buffer, and internal calls return through a per-function
// return-destination slot. This is intentionally straightforward and byte-for-
// byte compatible with Plank's debug backend; release codegen uses a different
// stack-scheduled model.
const MemoryLayout = struct {
    // One scratch word used while lowering switch terminators. The selector is
    // stored once, then each case reloads it before comparing with the case
    // constant.
    switch_store: u32,
    // Conventional bump-pointer slot. malloc/salloc load this word, return the
    // old pointer, and write back old + size.
    free_pointer: u32,
    // Shared block-argument/result buffer. A predecessor writes block.outputs()
    // here; the successor entry reads its block.inputs() from the same slots.
    transfer_start: u32,
    // Dense storage for every SIR local in every reachable/topologically-known
    // function. local_slots maps (function, value name) to an index here.
    locals_start: u32,
    // Internal-call return addresses. Each callable function gets one memory
    // word that callers fill with their continuation label before jumping.
    return_destinations_start: u32,
    // First dynamic byte after all fixed debug-codegen bookkeeping.
    dynamic_start: u32,

    fn empty() MemoryLayout {
        return .{
            .switch_store = 0,
            .free_pointer = word_size,
            .transfer_start = word_size * 2,
            .locals_start = word_size * 2,
            .return_destinations_start = word_size * 2,
            .dynamic_start = word_size * 2,
        };
    }
};

const PendingBlock = struct {
    // Worklist entry for CFG traversal. We keep the function with the block name
    // because block names are scoped to functions.
    function: ir.Function,
    block_name: []const u8,
};

// Borrowed composite key for values and blocks scoped by function. The parser
// owns the slices through Program's arena, so the maps can borrow them directly
// without allocating "function:name" strings for every lookup.
const ScopedName = struct {
    function_name: []const u8,
    name: []const u8,
};

const ScopedNameContext = struct {
    pub fn hash(_: @This(), key: ScopedName) u64 {
        var hasher = std.hash.Wyhash.init(0);
        const function_len = key.function_name.len;
        const name_len = key.name.len;
        // Include lengths before bytes so ("ab","c") and ("a","bc") cannot
        // collide by simple concatenation.
        hasher.update(std.mem.asBytes(&function_len));
        hasher.update(key.function_name);
        hasher.update(std.mem.asBytes(&name_len));
        hasher.update(key.name);
        return hasher.final();
    }

    pub fn eql(_: @This(), lhs: ScopedName, rhs: ScopedName) bool {
        return std.mem.eql(u8, lhs.function_name, rhs.function_name) and
            std.mem.eql(u8, lhs.name, rhs.name);
    }
};

fn ScopedNameMap(comptime V: type) type {
    return std.HashMap(ScopedName, V, ScopedNameContext, std.hash_map.default_max_load_percentage);
}

// DFS mark used for reachability/topological walks. `visiting` detects an
// internal-call cycle early; the debug backend currently has no call stack.
const VisitState = enum {
    visiting,
    done,
};

const FunctionVisit = struct {
    function_name: []const u8,
    state: VisitState,
};

pub fn emitFunction(
    allocator: std.mem.Allocator,
    program: ir.Program,
    function_name: ?[]const u8,
    bag: *diagnostics.Bag,
) ![]const u8 {
    // The public entry point keeps root selection simple: explicit function,
    // otherwise main, otherwise the first function. The emitter builds indexes
    // only after this root is known.
    const function = findFunction(program, function_name) orelse return CodegenError.FunctionNotFound;
    var emitter = DebugEmitter.init(allocator, program, bag);
    defer emitter.deinit();
    try emitter.emitProgram(function);
    return emitter.bytecode.toOwnedSlice();
}

fn findFunction(program: ir.Program, function_name: ?[]const u8) ?ir.Function {
    if (function_name) |name| {
        for (program.functions) |function| {
            if (std.mem.eql(u8, function.name, name)) return function;
        }
        return null;
    }
    for (program.functions) |function| {
        if (std.mem.eql(u8, function.name, "main")) return function;
    }
    return if (program.functions.len > 0) program.functions[0] else null;
}

const DebugEmitter = struct {
    allocator: std.mem.Allocator,
    program: ir.Program,
    bag: *diagnostics.Bag,
    bytecode: evm_asm.LabelBytecode,

    // Debug codegen materializes SIR locals and transfer buffers in memory.
    // Every emitted instruction may load/store locals, so these indexes are
    // kept as hash maps instead of rescanning the whole program-shaped tables.
    local_slots: ScopedNameMap(u32),
    // block_indexes gives O(1) lookup from (function, block name) to the parsed
    // block. block_labels maps the same key to the EVM jump label.
    block_indexes: ScopedNameMap(usize),
    block_labels: ScopedNameMap(evm_asm.Label),
    // Pending and translated sets drive CFG emission. A block can be enqueued
    // many times through multiple predecessors but is emitted once.
    pending_blocks: std.ArrayList(PendingBlock),
    translated_blocks: ScopedNameMap(void),
    // Data/function maps are also keyed by borrowed parser slices. They replace
    // tiny linear tables from the first Sinora version.
    data_labels: std.StringHashMap(evm_asm.Label),
    function_slots: std.StringHashMap(u32),
    functions: std.StringHashMap(ir.Function),
    // Reachability is rooted at init/main. It keeps debug codegen from assigning
    // jump labels for unreachable functions while still assigning local slots
    // for topological call targets consistently.
    reachable_functions: std.ArrayList(ir.Function),

    // Deployment mode emits init bytecode first and runtime bytecode second.
    // These labels let runtime_start_offset/init_end_offset/runtime_length be
    // patched after final byte widths are known.
    root_function_name: []const u8 = "",
    runtime_start_label: ?evm_asm.Label = null,
    init_end_label: ?evm_asm.Label = null,
    // When true, label/data offsets are emitted relative to runtime_start so
    // runtime bytecode embedded in deployment output sees local offsets.
    translating_runtime_code: bool = false,
    layout: MemoryLayout = MemoryLayout.empty(),
    next_slot: u32 = 0,

    fn init(allocator: std.mem.Allocator, program: ir.Program, bag: *diagnostics.Bag) DebugEmitter {
        return .{
            .allocator = allocator,
            .program = program,
            .bag = bag,
            .bytecode = evm_asm.LabelBytecode.init(allocator),
            .local_slots = ScopedNameMap(u32).init(allocator),
            .block_indexes = ScopedNameMap(usize).init(allocator),
            .block_labels = ScopedNameMap(evm_asm.Label).init(allocator),
            .pending_blocks = .empty,
            .translated_blocks = ScopedNameMap(void).init(allocator),
            .data_labels = std.StringHashMap(evm_asm.Label).init(allocator),
            .function_slots = std.StringHashMap(u32).init(allocator),
            .functions = std.StringHashMap(ir.Function).init(allocator),
            .reachable_functions = .empty,
        };
    }

    fn deinit(self: *DebugEmitter) void {
        self.reachable_functions.deinit(self.allocator);
        self.functions.deinit();
        self.function_slots.deinit();
        self.data_labels.deinit();
        self.translated_blocks.deinit();
        self.pending_blocks.deinit(self.allocator);
        self.block_labels.deinit();
        self.block_indexes.deinit();
        self.local_slots.deinit();
        self.bytecode.deinit();
    }

    fn emitProgram(self: *DebugEmitter, root_function: ir.Function) !void {
        // In Ora/SIR artifacts, `init` means deployment bytecode that returns
        // the runtime package. Any other selected root is emitted as standalone
        // runtime/debug bytecode.
        if (std.mem.eql(u8, root_function.name, "init")) {
            try self.emitDeployment(root_function);
            return;
        }
        try self.emitSingleRoot(root_function);
    }

    fn emitSingleRoot(self: *DebugEmitter, root_function: ir.Function) !void {
        if (root_function.blocks.len == 0) return self.failIfUnsupported();

        if (root_function.blocks[0].inputs.len != 0) {
            try self.unsupported(root_function.blocks[0].line, "debug codegen does not support root entry block inputs yet", .{});
        }

        self.root_function_name = root_function.name;
        try self.indexFunctions(root_function.line);
        // Reachability and topological collection are separate on purpose:
        // reachability decides which blocks get labels/emitted; topological
        // order decides stable function return slots and local memory slots.
        try self.reachable_functions.ensureTotalCapacity(self.allocator, self.program.functions.len);
        var visits: std.ArrayList(FunctionVisit) = .empty;
        defer visits.deinit(self.allocator);
        try visits.ensureTotalCapacity(self.allocator, self.program.functions.len);
        try self.collectReachableFunction(root_function, &visits);
        try self.prepareProgram(root_function.line);
        try self.emitFreePointerInit();

        self.translating_runtime_code = false;
        try self.emitBlocksFromEntry(root_function);
        try self.emitDataSegments();

        try self.failIfUnsupported();
    }

    fn emitDeployment(self: *DebugEmitter, init_function: ir.Function) !void {
        if (init_function.blocks.len == 0) return self.failIfUnsupported();

        try self.indexFunctions(init_function.line);
        const runtime_function = self.getFunction("main") orelse {
            try self.unsupported(init_function.line, "debug deployment codegen requires a 'main' runtime function", .{});
            return CodegenError.UnsupportedSir;
        };
        if (runtime_function.blocks.len == 0) return self.failIfUnsupported();
        if (init_function.blocks[0].inputs.len != 0) {
            try self.unsupported(init_function.blocks[0].line, "debug deployment codegen does not support init entry inputs yet", .{});
        }
        if (runtime_function.blocks[0].inputs.len != 0) {
            try self.unsupported(runtime_function.blocks[0].line, "debug deployment codegen does not support runtime entry inputs yet", .{});
        }

        self.root_function_name = init_function.name;
        // The assembler may need multiple width-solving passes before these
        // label offsets are final, so we emit symbolic references instead of
        // concrete offsets here.
        self.runtime_start_label = try self.bytecode.newLabel();
        self.init_end_label = try self.bytecode.newLabel();

        try self.reachable_functions.ensureTotalCapacity(self.allocator, self.program.functions.len);
        var visits: std.ArrayList(FunctionVisit) = .empty;
        defer visits.deinit(self.allocator);
        try visits.ensureTotalCapacity(self.allocator, self.program.functions.len);
        try self.collectReachableFunction(init_function, &visits);
        try self.collectReachableFunction(runtime_function, &visits);

        try self.prepareProgram(init_function.line);
        try self.emitFreePointerInit();
        self.translating_runtime_code = false;
        try self.emitBlocksFromEntry(init_function);

        // Runtime bytecode is appended after init code. Blocks are considered
        // un-emitted again because init/main may have identical block names but
        // must produce distinct JUMPDESTs in the final byte stream.
        const runtime_start = self.runtime_start_label orelse return CodegenError.UnsupportedSir;
        try self.bytecode.mark(runtime_start);
        try self.emitFreePointerInit();
        self.translating_runtime_code = true;
        self.translated_blocks.clearRetainingCapacity();
        self.pending_blocks.clearRetainingCapacity();
        try self.emitBlocksFromEntry(runtime_function);
        self.translating_runtime_code = false;
        try self.emitDataSegments();

        const init_end = self.init_end_label orelse return CodegenError.UnsupportedSir;
        try self.bytecode.mark(init_end);
        try self.failIfUnsupported();
    }

    fn emitBlocksFromEntry(self: *DebugEmitter, function: ir.Function) !void {
        if (function.blocks.len == 0) return;
        // Depth/worklist order matches the old implementation: push the entry,
        // pop a pending block, emit it, enqueue successors. The translated set
        // prevents duplicated bytecode for join blocks.
        try self.pending_blocks.append(self.allocator, .{
            .function = function,
            .block_name = function.blocks[0].name,
        });

        while (self.pending_blocks.pop()) |pending| {
            if (self.hasTranslatedBlock(pending.function.name, pending.block_name)) continue;
            const block = self.getBlock(pending.function, pending.block_name) orelse {
                try self.unsupported(pending.function.line, "missing pending block '{s}' in function '{s}'", .{ pending.block_name, pending.function.name });
                return CodegenError.UnsupportedSir;
            };
            try self.translated_blocks.put(.{
                .function_name = pending.function.name,
                .name = pending.block_name,
            }, {});

            const label = self.getBlockLabel(pending.function.name, block.name) orelse return CodegenError.UnsupportedSir;
            try self.bytecode.markJumpDest(label);
            // Block parameters are loaded from the transfer buffer before any
            // instruction in the block runs, implementing SIR block arguments.
            try self.emitBlockInputTransfer(pending.function, block);
            for (block.instructions) |instruction| {
                try self.emitInstruction(pending.function, instruction);
            }
            // Successors are enqueued before the branch/jump bytes are emitted.
            // That keeps traversal independent from the terminator's final
            // patched byte width.
            try self.enqueueSuccessors(pending.function, block);
            try self.emitTerminator(pending.function, block);
        }
    }

    fn indexFunctions(self: *DebugEmitter, line: u32) !void {
        try self.functions.ensureTotalCapacity(try self.hashCapacity(line, self.program.functions.len, "function index"));
        for (self.program.functions) |function| {
            // Keep the first definition, matching the old linear findFunction().
            if (!self.functions.contains(function.name)) {
                try self.functions.put(function.name, function);
            }
        }
    }

    fn getFunction(self: *DebugEmitter, name: []const u8) ?ir.Function {
        return self.functions.get(name);
    }

    fn getBlock(self: *DebugEmitter, function: ir.Function, block_name: []const u8) ?ir.Block {
        const index = self.block_indexes.get(.{
            .function_name = function.name,
            .name = block_name,
        }) orelse return null;
        return function.blocks[index];
    }

    fn collectReachableFunction(
        self: *DebugEmitter,
        function: ir.Function,
        visits: *std.ArrayList(FunctionVisit),
    ) !void {
        // Walk the internal-call graph from the selected roots. Debug codegen
        // has a single return-destination slot per function, so recursive call
        // cycles are unsupported and rejected instead of producing broken
        // continuation state.
        for (visits.items) |*visit| {
            if (!std.mem.eql(u8, visit.function_name, function.name)) continue;
            switch (visit.state) {
                .visiting => {
                    try self.unsupported(function.line, "debug codegen does not support recursive icall cycle through function '{s}' yet", .{function.name});
                    return CodegenError.UnsupportedSir;
                },
                .done => return,
            }
        }

        try visits.append(self.allocator, .{ .function_name = function.name, .state = .visiting });
        try self.reachable_functions.append(self.allocator, function);
        for (function.blocks) |block| {
            for (block.instructions) |instruction| {
                if (!std.mem.eql(u8, instruction.mnemonic, "icall")) continue;
                const callee_name = try self.requireIcallTarget(instruction);
                const callee = self.getFunction(callee_name) orelse {
                    try self.unsupported(instruction.line, "icall targets missing function '{s}'", .{callee_name});
                    return CodegenError.UnsupportedSir;
                };
                try self.collectReachableFunction(callee, visits);
            }
        }
        for (visits.items) |*visit| {
            if (std.mem.eql(u8, visit.function_name, function.name)) {
                visit.state = .done;
                return;
            }
        }
    }

    fn prepareProgram(self: *DebugEmitter, line: u32) !void {
        const stats = self.program.stats();
        // Reserve bytecode/label/patch capacity once from whole-program stats.
        // Exact byte sizes are still solved by asm.zig; this only avoids many
        // ArrayList/hash-map growth operations while walking the corpus.
        const reserve_hint = evm_asm.estimateReserveHint(stats);
        try self.bytecode.reserve(
            reserve_hint.byte_capacity,
            reserve_hint.label_capacity,
            reserve_hint.patch_capacity,
        );
        try self.data_labels.ensureTotalCapacity(try self.hashCapacity(line, stats.data_segments, "data label index"));
        try self.function_slots.ensureTotalCapacity(try self.hashCapacity(line, stats.functions, "function return-slot index"));
        try self.block_indexes.ensureTotalCapacity(try self.hashCapacity(line, stats.blocks, "block index"));
        try self.block_labels.ensureTotalCapacity(try self.hashCapacity(line, stats.blocks, "block label index"));
        try self.pending_blocks.ensureTotalCapacity(self.allocator, @max(stats.blocks, 1));
        try self.translated_blocks.ensureTotalCapacity(try self.hashCapacity(line, @max(stats.blocks, 1), "translated-block index"));
        const local_slot_capacity = self.localSlotCapacityHint(stats) catch |err| {
            if (err == error.Overflow) {
                try self.unsupported(line, "local slot index exceeds debug codegen index capacity", .{});
                return CodegenError.UnsupportedSir;
            }
            return err;
        };
        try self.local_slots.ensureTotalCapacity(try self.hashCapacity(line, local_slot_capacity, "local slot index"));

        var max_transfer_slots: usize = 0;
        var topo_functions: std.ArrayList(ir.Function) = .empty;
        defer topo_functions.deinit(self.allocator);
        try topo_functions.ensureTotalCapacity(self.allocator, stats.functions);
        var topo_visits: std.ArrayList(FunctionVisit) = .empty;
        defer topo_visits.deinit(self.allocator);
        try topo_visits.ensureTotalCapacity(self.allocator, stats.functions);

        for (self.program.functions) |function| {
            try self.collectTopoFunction(function, &topo_visits, &topo_functions);
        }

        // Data labels are code labels because data segments are appended to the
        // same byte stream and referenced by CODECOPY offsets.
        for (self.program.data_segments) |segment| {
            // Keep the first label if invalid duplicate data names reach here.
            if (!self.data_labels.contains(segment.name)) {
                try self.data_labels.put(segment.name, try self.bytecode.newLabel());
            }
        }

        // A function's return slot stores the caller continuation for icall.
        // Slots are assigned in postorder so callees receive stable addresses
        // before their callers use them.
        for (topo_functions.items) |function| {
            try self.function_slots.put(function.name, @intCast(self.function_slots.count()));
        }

        // Only reachable functions get block labels. Unreachable functions may
        // still have local slots for stable topological layout, but no bytecode.
        for (self.reachable_functions.items) |function| {
            for (function.blocks, 0..) |block, block_index| {
                const block_key: ScopedName = .{
                    .function_name = function.name,
                    .name = block.name,
                };
                // Keep first-definition behavior if invalid SIR reaches codegen.
                if (!self.block_indexes.contains(block_key)) {
                    try self.block_indexes.put(block_key, block_index);
                }
                if (!self.block_labels.contains(block_key)) {
                    try self.block_labels.put(block_key, try self.bytecode.newLabel());
                }
            }
        }

        for (topo_functions.items) |function| {
            for (function.blocks) |block| {
                // Transfer buffer size must handle both predecessor outputs and
                // successor inputs. Legal SIR has matching counts per edge, but
                // the debug backend still computes a safe maximum.
                max_transfer_slots = @max(max_transfer_slots, block.inputs.len);
                max_transfer_slots = @max(max_transfer_slots, block.outputs.len);
                for (block.inputs) |name| try self.defineLocal(function, name);
                for (block.instructions) |instruction| {
                    for (instruction.results) |result| try self.defineLocal(function, result);
                }
            }
        }

        try self.computeMemoryLayout(line, max_transfer_slots);
    }

    fn collectTopoFunction(
        self: *DebugEmitter,
        function: ir.Function,
        visits: *std.ArrayList(FunctionVisit),
        output: *std.ArrayList(ir.Function),
    ) !void {
        // Postorder over the call graph. The output is used for deterministic
        // local/return-slot allocation, not for bytecode emission order.
        for (visits.items) |*visit| {
            if (!std.mem.eql(u8, visit.function_name, function.name)) continue;
            switch (visit.state) {
                .visiting => {
                    try self.unsupported(function.line, "debug codegen does not support recursive icall cycle through function '{s}' yet", .{function.name});
                    return CodegenError.UnsupportedSir;
                },
                .done => return,
            }
        }

        try visits.append(self.allocator, .{ .function_name = function.name, .state = .visiting });
        for (function.blocks) |block| {
            for (block.instructions) |instruction| {
                if (!std.mem.eql(u8, instruction.mnemonic, "icall")) continue;
                const callee_name = try self.requireIcallTarget(instruction);
                const callee = self.getFunction(callee_name) orelse {
                    try self.unsupported(instruction.line, "icall targets missing function '{s}'", .{callee_name});
                    return CodegenError.UnsupportedSir;
                };
                try self.collectTopoFunction(callee, visits, output);
            }
        }

        for (visits.items) |*visit| {
            if (std.mem.eql(u8, visit.function_name, function.name)) {
                visit.state = .done;
                break;
            }
        }
        try output.append(self.allocator, function);
    }

    fn emitInstruction(self: *DebugEmitter, function: ir.Function, instruction: ir.Instruction) !void {
        // ops.zig classifies SIR mnemonics into fixed EVM-like operations,
        // width-parametric memory ops, and Sinora's internal-call pseudo-op.
        const spec = ops.lookup(instruction.mnemonic) orelse {
            try self.unsupported(instruction.line, "unknown opcode '{s}'", .{instruction.mnemonic});
            return;
        };

        switch (spec) {
            .fixed => |fixed| try self.emitFixedInstruction(function, instruction, fixed),
            .memory_load => |memory| try self.emitMemoryLoad(function, instruction, memory.bits),
            .memory_store => |memory| try self.emitMemoryStore(function, instruction, memory.bits),
            .internal_call => try self.emitInternalCall(function, instruction),
        }
    }

    fn emitFixedInstruction(self: *DebugEmitter, function: ir.Function, instruction: ir.Instruction, fixed: ops.Fixed) !void {
        // Some SIR mnemonics are debug-backend intrinsics rather than direct
        // one-byte EVM opcodes. Handle those before falling through to the
        // generic stack operation path.
        if (std.mem.eql(u8, instruction.mnemonic, "const") or std.mem.eql(u8, instruction.mnemonic, "large_const")) {
            const value = parseU256(instruction.operands[0]) orelse {
                try self.unsupported(instruction.line, "invalid numeric literal '{s}'", .{instruction.operands[0]});
                return;
            };
            try self.bytecode.pushU256(value);
            try self.emitLocalStore(function, instruction.results[0]);
            return;
        }

        if (std.mem.eql(u8, instruction.mnemonic, "copy")) {
            try self.emitLocalLoad(function, instruction.operands[0]);
            try self.emitLocalStore(function, instruction.results[0]);
            return;
        }

        if (std.mem.eql(u8, instruction.mnemonic, "freeptr")) {
            try self.emitFreePtrLoad();
            try self.emitLocalStore(function, instruction.results[0]);
            return;
        }

        if (std.mem.eql(u8, instruction.mnemonic, "malloc") or std.mem.eql(u8, instruction.mnemonic, "mallocany")) {
            try self.emitDynamicMemoryAlloc(function, instruction.operands[0], instruction.results[0]);
            return;
        }

        if (std.mem.eql(u8, instruction.mnemonic, "salloc") or std.mem.eql(u8, instruction.mnemonic, "sallocany")) {
            const size = parseU32(instruction.operands[0]) orelse {
                try self.unsupported(instruction.line, "invalid static allocation size '{s}'", .{instruction.operands[0]});
                return;
            };
            try self.emitStaticMemoryAlloc(function, size, instruction.results[0]);
            return;
        }

        if (std.mem.eql(u8, instruction.mnemonic, "data_offset")) {
            try self.emitDataOffset(function, instruction);
            return;
        }

        if (std.mem.eql(u8, instruction.mnemonic, "runtime_start_offset") or
            std.mem.eql(u8, instruction.mnemonic, "init_end_offset") or
            std.mem.eql(u8, instruction.mnemonic, "runtime_length"))
        {
            try self.emitDeploymentOffsetIntrinsic(function, instruction);
            return;
        }

        if (std.mem.eql(u8, instruction.mnemonic, "noop")) return;

        if (fixed.extra != .none) {
            try self.unsupported(instruction.line, "debug codegen does not support opcode '{s}' yet", .{instruction.mnemonic});
            return;
        }

        const opcode = evm_asm.evmOpcode(instruction.mnemonic) orelse {
            try self.unsupported(instruction.line, "debug codegen does not support opcode '{s}' yet", .{instruction.mnemonic});
            return;
        };
        try self.emitSimpleOperation(function, opcode, instruction.operands, instruction.results);
    }

    fn emitMemoryLoad(self: *DebugEmitter, function: ir.Function, instruction: ir.Instruction, bits: u16) !void {
        // SIR memory loads are right-aligned EVM words. Narrow loads read a full
        // MLOAD and shift away high padding bits.
        try self.emitLocalLoad(function, instruction.operands[0]);
        try self.bytecode.pushOp(evm_asm.op.MLOAD);
        try self.bytecode.pushU32(256 - bits);
        try self.bytecode.pushOp(evm_asm.op.SHR);
        try self.emitLocalStore(function, instruction.results[0]);
    }

    fn emitMemoryStore(self: *DebugEmitter, function: ir.Function, instruction: ir.Instruction, bits: u16) !void {
        // Narrow stores preserve the untouched high bits of the destination word
        // and splice the new value into the low `bits` region.
        try self.emitLocalLoad(function, instruction.operands[0]);
        try self.bytecode.pushOp(evm_asm.op.DUP1);
        try self.bytecode.pushOp(evm_asm.op.MLOAD);
        try self.bytecode.pushU32(bits);
        try self.bytecode.pushOp(evm_asm.op.SHL);
        try self.bytecode.pushU32(bits);
        try self.bytecode.pushOp(evm_asm.op.SHR);
        try self.emitLocalLoad(function, instruction.operands[1]);
        try self.bytecode.pushU32(256 - bits);
        try self.bytecode.pushOp(evm_asm.op.SHL);
        try self.bytecode.pushOp(evm_asm.op.OR);
        try self.bytecode.pushOp(evm_asm.op.SWAP1);
        try self.bytecode.pushOp(evm_asm.op.MSTORE);
    }

    fn emitTerminator(self: *DebugEmitter, function: ir.Function, block: ir.Block) !void {
        // Terminators are the only place block output transfer happens. This
        // mirrors SIR's per-block transfer model: the source block writes a
        // single output vector before choosing its successor.
        switch (block.terminator) {
            .return_ => |ret| try self.emitSimpleOperation(function, evm_asm.op.RETURN, &.{ ret.ptr, ret.len }, &.{}),
            .revert => |revert| try self.emitSimpleOperation(function, evm_asm.op.REVERT, &.{ revert.ptr, revert.len }, &.{}),
            .stop => try self.bytecode.pushOp(evm_asm.op.STOP),
            .invalid => try self.bytecode.pushOp(evm_asm.op.INVALID),
            .selfdestruct => |beneficiary| try self.emitSimpleOperation(function, evm_asm.op.SELFDESTRUCT, &.{beneficiary}, &.{}),
            .jump => |target| {
                try self.emitBlockOutputTransfer(function, block);
                try self.emitJump(function, target);
            },
            .branch => |branch| {
                try self.emitBlockOutputTransfer(function, block);
                try self.emitLocalLoad(function, branch.condition);
                try self.emitLabelRef(function.name, branch.non_zero_target);
                try self.bytecode.pushOp(evm_asm.op.JUMPI);
                try self.emitJump(function, branch.zero_target);
            },
            .switch_ => |switch_term| {
                try self.emitBlockOutputTransfer(function, block);
                try self.emitSwitch(function, switch_term);
            },
            .iret => try self.emitInternalReturn(function, block),
        }

        try self.validateSuccessorTransferCounts(function, block);
    }

    fn emitSimpleOperation(
        self: *DebugEmitter,
        function: ir.Function,
        opcode: u8,
        inputs: []const []const u8,
        outputs: []const []const u8,
    ) !void {
        // EVM operands are stack-based. SIR lists operands left-to-right, so we
        // load them in reverse order before pushing the opcode. This preserves
        // non-commutative operations like `sub a b`.
        var index = inputs.len;
        while (index > 0) {
            index -= 1;
            try self.emitLocalLoad(function, inputs[index]);
        }
        try self.bytecode.pushOp(opcode);
        for (outputs) |output| {
            try self.emitLocalStore(function, output);
        }
    }

    fn emitLocalLoad(self: *DebugEmitter, function: ir.Function, name: []const u8) !void {
        // A SIR local is a memory word in locals_start + slot*32.
        try self.bytecode.pushU32(try self.localAddr(function, name));
        try self.bytecode.pushOp(evm_asm.op.MLOAD);
    }

    fn emitLocalStore(self: *DebugEmitter, function: ir.Function, name: []const u8) !void {
        // Store the current top-of-stack into the local's memory word.
        try self.bytecode.pushU32(try self.localAddr(function, name));
        try self.bytecode.pushOp(evm_asm.op.MSTORE);
    }

    fn localAddr(self: *DebugEmitter, function: ir.Function, name: []const u8) !u32 {
        const slot = self.localSlot(function.name, name) orelse {
            try self.unsupported(0, "undefined local '{s}' in function '{s}' during codegen", .{ name, function.name });
            return CodegenError.UnsupportedSir;
        };
        return self.layout.locals_start + slot * word_size;
    }

    fn defineLocal(self: *DebugEmitter, function: ir.Function, name: []const u8) !void {
        // First definition wins. This matches the previous linear table and
        // avoids changing behavior if invalid duplicate values reach codegen.
        if (self.localSlot(function.name, name) != null) return;
        try self.local_slots.put(.{
            .function_name = function.name,
            .name = name,
        }, self.next_slot);
        self.next_slot += 1;
    }

    fn emitBlockInputTransfer(self: *DebugEmitter, function: ir.Function, block: ir.Block) !void {
        // Read transfer[i] into each block parameter. The predecessor wrote the
        // values immediately before jumping/branching here.
        for (block.inputs, 0..) |input, index| {
            try self.bytecode.pushU32(try self.transferAddr(block.line, index));
            try self.bytecode.pushOp(evm_asm.op.MLOAD);
            try self.emitLocalStore(function, input);
        }
    }

    fn emitBlockOutputTransfer(self: *DebugEmitter, function: ir.Function, block: ir.Block) !void {
        // Write each block output into transfer[i]. All outgoing edges from this
        // block share that vector in the current SIR model.
        for (block.outputs, 0..) |output, index| {
            try self.emitLocalLoad(function, output);
            try self.bytecode.pushU32(try self.transferAddr(block.line, index));
            try self.bytecode.pushOp(evm_asm.op.MSTORE);
        }
    }

    fn emitJump(self: *DebugEmitter, function: ir.Function, target: []const u8) !void {
        try self.emitLabelRef(function.name, target);
        try self.bytecode.pushOp(evm_asm.op.JUMP);
    }

    fn emitLabelRef(self: *DebugEmitter, function_name: []const u8, target: []const u8) !void {
        const label = self.getBlockLabel(function_name, target) orelse {
            try self.unsupported(0, "missing block label '{s}' in function '{s}' during codegen", .{ target, function_name });
            return CodegenError.UnsupportedSir;
        };
        try self.emitCodeOffset(label);
    }

    fn emitCodeOffset(self: *DebugEmitter, label: evm_asm.Label) !void {
        // Init code refers to absolute offsets in the whole deployment package.
        // Runtime code refers to offsets relative to runtime_start because that
        // is the bytecode that will actually live at the account address.
        if (self.translating_runtime_code) {
            const runtime_start = self.runtime_start_label orelse {
                try self.unsupported(0, "runtime code offset requires a runtime start label", .{});
                return CodegenError.UnsupportedSir;
            };
            try self.bytecode.pushLabelDelta(runtime_start, label);
        } else {
            try self.bytecode.pushLabelRef(label);
        }
    }

    fn emitSwitch(self: *DebugEmitter, function: ir.Function, switch_term: ir.SwitchTerminator) !void {
        // Store the selector once in scratch memory. Each case reloads it so the
        // generated stack shape is simple and independent of case count.
        try self.emitLocalLoad(function, switch_term.selector);
        try self.bytecode.pushU32(self.layout.switch_store);
        try self.bytecode.pushOp(evm_asm.op.MSTORE);

        for (switch_term.cases) |case| {
            const value = parseU256(case.value) orelse {
                try self.unsupported(case.line, "invalid switch case literal '{s}'", .{case.value});
                continue;
            };
            try self.bytecode.pushU32(self.layout.switch_store);
            try self.bytecode.pushOp(evm_asm.op.MLOAD);
            try self.bytecode.pushU256(value);
            try self.bytecode.pushOp(evm_asm.op.EQ);
            try self.emitLabelRef(function.name, case.target);
            try self.bytecode.pushOp(evm_asm.op.JUMPI);
        }

        try self.emitJump(function, switch_term.default_target);
    }

    fn emitInternalCall(self: *DebugEmitter, caller: ir.Function, instruction: ir.Instruction) !void {
        // Debug icall is not an EVM CALL. It is a jump into another SIR function
        // plus a manually stored continuation. Arguments and results use the
        // same transfer buffer as block parameters.
        const callee_name = try self.requireIcallTarget(instruction);
        const callee = self.getFunction(callee_name) orelse {
            try self.unsupported(instruction.line, "icall targets missing function '{s}'", .{callee_name});
            return CodegenError.UnsupportedSir;
        };
        try self.validateIcallShape(instruction, callee);

        for (instruction.operands[1..], 0..) |arg, index| {
            try self.emitLocalLoad(caller, arg);
            try self.bytecode.pushU32(try self.transferAddr(instruction.line, index));
            try self.bytecode.pushOp(evm_asm.op.MSTORE);
        }

        const return_label = try self.bytecode.newLabel();
        // The callee's iret loads this return-destination slot and jumps back
        // here. Recursion is rejected earlier because one slot per function
        // cannot represent nested activations.
        try self.emitCodeOffset(return_label);
        try self.bytecode.pushU32(try self.returnDestinationAddr(callee.name));
        try self.bytecode.pushOp(evm_asm.op.MSTORE);
        const entry_block = callee.blocks[0];
        try self.emitLabelRef(callee.name, entry_block.name);
        try self.bytecode.pushOp(evm_asm.op.JUMP);

        try self.bytecode.markJumpDest(return_label);
        for (instruction.results, 0..) |result, index| {
            try self.bytecode.pushU32(try self.transferAddr(instruction.line, index));
            try self.bytecode.pushOp(evm_asm.op.MLOAD);
            try self.emitLocalStore(caller, result);
        }
        try self.pending_blocks.append(self.allocator, .{
            .function = callee,
            .block_name = entry_block.name,
        });
    }

    fn emitInternalReturn(self: *DebugEmitter, function: ir.Function, block: ir.Block) !void {
        // iret writes the function output vector, then jumps to the continuation
        // slot most recently written by the caller.
        if (std.mem.eql(u8, function.name, self.root_function_name)) {
            try self.unsupported(block.line, "root function '{s}' cannot use iret without a caller", .{function.name});
            return;
        }
        try self.emitBlockOutputTransfer(function, block);
        try self.bytecode.pushU32(try self.returnDestinationAddr(function.name));
        try self.bytecode.pushOp(evm_asm.op.MLOAD);
        try self.bytecode.pushOp(evm_asm.op.JUMP);
    }

    fn emitDeploymentOffsetIntrinsic(self: *DebugEmitter, function: ir.Function, instruction: ir.Instruction) !void {
        // These pseudo-ops are only valid while building deployment bytecode.
        // They let SIR copy the runtime package out of the init byte stream.
        const runtime_start = self.runtime_start_label orelse {
            try self.unsupported(instruction.line, "debug codegen only supports '{s}' while emitting the init deployment package", .{instruction.mnemonic});
            return;
        };
        const init_end = self.init_end_label orelse {
            try self.unsupported(instruction.line, "debug codegen only supports '{s}' while emitting the init deployment package", .{instruction.mnemonic});
            return;
        };

        if (std.mem.eql(u8, instruction.mnemonic, "runtime_start_offset")) {
            try self.bytecode.pushLabelRef(runtime_start);
        } else if (std.mem.eql(u8, instruction.mnemonic, "init_end_offset")) {
            try self.bytecode.pushLabelRef(init_end);
        } else if (std.mem.eql(u8, instruction.mnemonic, "runtime_length")) {
            try self.bytecode.pushLabelDelta(runtime_start, init_end);
        } else {
            unreachable;
        }
        try self.emitLocalStore(function, instruction.results[0]);
    }

    fn emitDataOffset(self: *DebugEmitter, function: ir.Function, instruction: ir.Instruction) !void {
        // data_offset .name materializes a code offset for an appended data
        // segment. In runtime mode the offset is relative to runtime_start.
        if (instruction.operands.len != 1 or instruction.operands[0].len <= 1 or instruction.operands[0][0] != '.') {
            try self.unsupported(instruction.line, "data_offset expects a data reference like '.name'", .{});
            return;
        }

        const data_name = instruction.operands[0][1..];
        const data_label = self.getDataLabel(data_name) orelse {
            try self.unsupported(instruction.line, "data_offset references missing data segment '{s}'", .{data_name});
            return CodegenError.UnsupportedSir;
        };

        if (self.translating_runtime_code) {
            const runtime_start = self.runtime_start_label orelse {
                try self.unsupported(instruction.line, "runtime data_offset requires a runtime start label", .{});
                return;
            };
            try self.bytecode.pushLabelDelta(runtime_start, data_label);
        } else {
            try self.bytecode.pushLabelRef(data_label);
        }
        try self.emitLocalStore(function, instruction.results[0]);
    }

    fn emitDataSegments(self: *DebugEmitter) !void {
        // Data is appended after executable blocks and marked with normal code
        // labels so CODECOPY can reference it through the same patch solver.
        for (self.program.data_segments) |segment| {
            const label = self.getDataLabel(segment.name) orelse {
                try self.unsupported(segment.line, "missing data label for segment '{s}'", .{segment.name});
                return CodegenError.UnsupportedSir;
            };
            try self.bytecode.mark(label);
            try self.bytecode.pushData(segment.bytes);
        }
    }

    fn validateSuccessorTransferCounts(self: *DebugEmitter, function: ir.Function, block: ir.Block) !void {
        // The legalizer should already enforce this, but debug codegen checks it
        // before emitting bytecode because a mismatch would silently read/write
        // the wrong transfer slots.
        switch (block.terminator) {
            .jump => |target| try self.validateTransferCount(function, block, target),
            .branch => |branch| {
                try self.validateTransferCount(function, block, branch.non_zero_target);
                try self.validateTransferCount(function, block, branch.zero_target);
            },
            .switch_ => |switch_term| {
                for (switch_term.cases) |case| {
                    try self.validateTransferCount(function, block, case.target);
                }
                try self.validateTransferCount(function, block, switch_term.default_target);
            },
            .return_, .revert, .stop, .invalid, .selfdestruct, .iret => {},
        }
    }

    fn enqueueSuccessors(self: *DebugEmitter, function: ir.Function, block: ir.Block) !void {
        // Enqueue in the same successor order as ir.SuccessorIter/Plank debug:
        // branch zero-target before non-zero-target; switch cases then default.
        switch (block.terminator) {
            .jump => |target| try self.enqueueBlock(function, target),
            .branch => |branch| {
                try self.enqueueBlock(function, branch.zero_target);
                try self.enqueueBlock(function, branch.non_zero_target);
            },
            .switch_ => |switch_term| {
                for (switch_term.cases) |case| {
                    try self.enqueueBlock(function, case.target);
                }
                if (switch_term.default_target.len != 0) {
                    try self.enqueueBlock(function, switch_term.default_target);
                }
            },
            .return_, .revert, .stop, .invalid, .selfdestruct, .iret => {},
        }
    }

    fn enqueueBlock(self: *DebugEmitter, function: ir.Function, block_name: []const u8) !void {
        try self.pending_blocks.append(self.allocator, .{
            .function = function,
            .block_name = block_name,
        });
    }

    fn hasTranslatedBlock(self: *DebugEmitter, function_name: []const u8, block_name: []const u8) bool {
        return self.translated_blocks.contains(.{
            .function_name = function_name,
            .name = block_name,
        });
    }

    fn validateTransferCount(self: *DebugEmitter, function: ir.Function, block: ir.Block, target: []const u8) !void {
        const target_block = self.getBlock(function, target) orelse {
            try self.unsupported(block.line, "missing target block '{s}' during codegen", .{target});
            return;
        };
        if (block.outputs.len != target_block.inputs.len) {
            try self.unsupported(
                block.line,
                "block '{s}' has {d} outputs but target '{s}' expects {d} inputs",
                .{ block.name, block.outputs.len, target_block.name, target_block.inputs.len },
            );
        }
    }

    fn validateIcallShape(self: *DebugEmitter, instruction: ir.Instruction, callee: ir.Function) !void {
        // Keep shape errors grouped on this icall. If any diagnostic was added,
        // abort this path instead of trying to emit partial call bytecode.
        const diagnostics_before = self.bag.items.items.len;
        if (callee.blocks.len == 0) {
            try self.unsupported(instruction.line, "icall target function '{s}' has no blocks", .{callee.name});
            return CodegenError.UnsupportedSir;
        }
        const expected_args = callee.blocks[0].inputs.len;
        const actual_args = instruction.operands.len - 1;
        if (expected_args != actual_args) {
            try self.unsupported(
                instruction.line,
                "icall to '{s}' expects {d} arguments, got {d}",
                .{ callee.name, expected_args, actual_args },
            );
        }

        const expected_results = try self.iretOutputCount(callee);
        if (expected_results != instruction.results.len) {
            try self.unsupported(
                instruction.line,
                "icall to '{s}' expects {d} results, got {d}",
                .{ callee.name, expected_results, instruction.results.len },
            );
        }
        if (self.bag.items.items.len != diagnostics_before) return CodegenError.UnsupportedSir;
    }

    fn iretOutputCount(self: *DebugEmitter, function: ir.Function) !usize {
        // A callable function may have multiple iret blocks, but every iret must
        // return the same arity so callers know how many result slots to read.
        var result: ?usize = null;
        for (function.blocks) |block| {
            if (block.terminator != .iret) continue;
            if (result) |count| {
                if (count != block.outputs.len) {
                    try self.unsupported(block.line, "function '{s}' has inconsistent iret output counts", .{function.name});
                    return CodegenError.UnsupportedSir;
                }
            } else {
                result = block.outputs.len;
            }
        }
        return result orelse 0;
    }

    fn unsupported(self: *DebugEmitter, line: u32, comptime fmt: []const u8, args: anytype) !void {
        try self.bag.errorAt(line, 1, fmt, args);
    }

    fn failIfUnsupported(self: *DebugEmitter) !void {
        if (self.bag.hasErrors()) return CodegenError.UnsupportedSir;
    }

    fn computeMemoryLayout(self: *DebugEmitter, line: u32, max_transfer_slots: usize) !void {
        // Fixed debug memory areas are laid out once:
        //   0x00 switch scratch
        //   0x20 free pointer
        //   0x40 transfer buffer
        //   ...  local slots
        //   ...  return-destination slots
        //   ...  dynamic allocations
        const transfer_bytes = try self.checkedWordsToBytes(line, max_transfer_slots, "block transfer area");
        const locals_bytes = try self.checkedWordsToBytes(line, self.next_slot, "local area");
        const return_destination_bytes = try self.checkedWordsToBytes(line, self.function_slots.count(), "return destination area");
        const free_pointer = word_size;
        const transfer_start = try self.checkedAdd(line, free_pointer, word_size, "block transfer area");
        const locals_start = try self.checkedAdd(line, transfer_start, transfer_bytes, "local area");
        const return_destinations_start = try self.checkedAdd(line, locals_start, locals_bytes, "return destination area");
        const dynamic_start = try self.checkedAdd(line, return_destinations_start, return_destination_bytes, "dynamic allocation area");
        self.layout = .{
            .switch_store = 0,
            .free_pointer = free_pointer,
            .transfer_start = transfer_start,
            .locals_start = locals_start,
            .return_destinations_start = return_destinations_start,
            .dynamic_start = dynamic_start,
        };
    }

    fn emitFreePointerInit(self: *DebugEmitter) !void {
        try self.bytecode.pushU32(self.layout.dynamic_start);
        try self.bytecode.pushU32(self.layout.free_pointer);
        try self.bytecode.pushOp(evm_asm.op.MSTORE);
    }

    fn emitFreePtrLoad(self: *DebugEmitter) !void {
        try self.bytecode.pushU32(self.layout.free_pointer);
        try self.bytecode.pushOp(evm_asm.op.MLOAD);
    }

    fn emitDynamicMemoryAlloc(self: *DebugEmitter, function: ir.Function, size_local: []const u8, result_local: []const u8) !void {
        // Match Plank debug's bump allocation sequence. The CALLDATACOPY uses
        // zero-sized/out-of-range calldata as cheap memory initialization while
        // preserving byte parity with the Rust backend.
        try self.emitFreePtrLoad();
        try self.bytecode.pushOp(evm_asm.op.DUP1);
        try self.emitLocalLoad(function, size_local);
        try self.bytecode.pushOp(evm_asm.op.DUP1);
        try self.bytecode.pushOp(evm_asm.op.CALLDATASIZE);
        try self.bytecode.pushOp(evm_asm.op.DUP4);
        try self.bytecode.pushOp(evm_asm.op.CALLDATACOPY);
        try self.bytecode.pushOp(evm_asm.op.ADD);
        try self.bytecode.pushU32(self.layout.free_pointer);
        try self.bytecode.pushOp(evm_asm.op.MSTORE);
        try self.emitLocalStore(function, result_local);
    }

    fn emitStaticMemoryAlloc(self: *DebugEmitter, function: ir.Function, size: u32, result_local: []const u8) !void {
        // Static allocation is the same bump-pointer sequence with the size
        // already known at compile time.
        try self.emitFreePtrLoad();
        try self.bytecode.pushOp(evm_asm.op.DUP1);
        try self.bytecode.pushU32(size);
        try self.bytecode.pushOp(evm_asm.op.DUP1);
        try self.bytecode.pushOp(evm_asm.op.CALLDATASIZE);
        try self.bytecode.pushOp(evm_asm.op.DUP4);
        try self.bytecode.pushOp(evm_asm.op.CALLDATACOPY);
        try self.bytecode.pushOp(evm_asm.op.ADD);
        try self.bytecode.pushU32(self.layout.free_pointer);
        try self.bytecode.pushOp(evm_asm.op.MSTORE);
        try self.emitLocalStore(function, result_local);
    }

    fn getBlockLabel(self: *DebugEmitter, function_name: []const u8, block_name: []const u8) ?evm_asm.Label {
        return self.block_labels.get(.{
            .function_name = function_name,
            .name = block_name,
        });
    }

    fn getDataLabel(self: *DebugEmitter, data_name: []const u8) ?evm_asm.Label {
        return self.data_labels.get(data_name);
    }

    fn localSlot(self: *DebugEmitter, function_name: []const u8, local_name: []const u8) ?u32 {
        return self.local_slots.get(.{
            .function_name = function_name,
            .name = local_name,
        });
    }

    fn returnDestinationAddr(self: *DebugEmitter, function_name: []const u8) !u32 {
        if (self.function_slots.get(function_name)) |slot| {
            return self.layout.return_destinations_start + slot * word_size;
        }
        try self.unsupported(0, "missing return destination slot for function '{s}'", .{function_name});
        return CodegenError.UnsupportedSir;
    }

    fn requireIcallTarget(self: *DebugEmitter, instruction: ir.Instruction) ![]const u8 {
        // SIR text writes function references as @name. Strip the sigil only
        // after validating it, so bad text gets a precise diagnostic.
        if (instruction.operands.len == 0) {
            try self.unsupported(instruction.line, "icall is missing a function target", .{});
            return CodegenError.UnsupportedSir;
        }
        const target = instruction.operands[0];
        if (target.len <= 1 or target[0] != '@') {
            try self.unsupported(instruction.line, "icall target must be written as '@function'", .{});
            return CodegenError.UnsupportedSir;
        }
        return target[1..];
    }

    fn transferAddr(self: *DebugEmitter, line: u32, index: usize) !u32 {
        const offset = try self.checkedWordsToBytes(line, index, "block transfer slot");
        return self.checkedAdd(line, self.layout.transfer_start, offset, "block transfer slot");
    }

    fn checkedWordsToBytes(self: *DebugEmitter, line: u32, words: anytype, role: []const u8) !u32 {
        // Debug memory addresses are emitted as u32 immediates. Oversized SIR
        // therefore fails closed instead of wrapping slot arithmetic.
        const words_usize: usize = @intCast(words);
        const max_words: usize = @as(usize, std.math.maxInt(u32) / word_size);
        if (words_usize > max_words) {
            try self.unsupported(line, "{s} exceeds debug codegen address space", .{role});
            return CodegenError.UnsupportedSir;
        }
        return @as(u32, @intCast(words_usize)) * word_size;
    }

    fn checkedAdd(self: *DebugEmitter, line: u32, lhs: u32, rhs: u32, role: []const u8) !u32 {
        return std.math.add(u32, lhs, rhs) catch {
            try self.unsupported(line, "{s} exceeds debug codegen address space", .{role});
            return CodegenError.UnsupportedSir;
        };
    }

    fn hashCapacity(self: *DebugEmitter, line: u32, count: usize, role: []const u8) !u32 {
        // Zig's managed hash maps use a u32 count internally. Treat impossible
        // program sizes as unsupported SIR rather than allowing @intCast traps.
        if (count > std.math.maxInt(u32)) {
            try self.unsupported(line, "{s} exceeds debug codegen index capacity", .{role});
            return CodegenError.UnsupportedSir;
        }
        return @intCast(count);
    }

    fn localSlotCapacityHint(_: *DebugEmitter, stats: ir.Stats) !usize {
        // Upper-bound local slots before reserving the local-slot index. Inputs
        // and results are the only named locals this debug backend materializes.
        const result_slots = try std.math.mul(usize, stats.instructions, 2);
        const block_slots = try std.math.mul(usize, stats.blocks, 4);
        return std.math.add(usize, result_slots, block_slots);
    }
};

fn parseU256(text: []const u8) ?u256 {
    // SIR constants are word-sized. Negative literals are represented in the
    // EVM/SIR bit pattern by wrapping subtraction from zero.
    if (text.len == 0) return null;
    const negative = text[0] == '-';
    const unsigned = if (negative) text[1..] else text;
    if (unsigned.len == 0) return null;

    const parsed = if (std.mem.startsWith(u8, unsigned, "0x") or std.mem.startsWith(u8, unsigned, "0X"))
        std.fmt.parseUnsigned(u256, unsigned[2..], 16) catch return null
    else
        std.fmt.parseUnsigned(u256, unsigned, 10) catch return null;

    return if (negative) 0 -% parsed else parsed;
}

fn parseU32(text: []const u8) ?u32 {
    // Address/size immediates in this debug backend are intentionally capped at
    // u32 because LabelBytecode emits compact PUSH immediates for them.
    const parsed = parseU256(text) orelse return null;
    if (parsed > std.math.maxInt(u32)) return null;
    return @intCast(parsed);
}

test "debug codegen emits simple returning bytecode" {
    const source =
        \\fn main:
        \\    entry {
        \\        ptr = const 0x0
        \\        len = const 0x20
        \\        a = const 0x1
        \\        b = const 0x2
        \\        sum = add a b
        \\        mstore256 ptr sum
        \\        return ptr len
        \\    }
    ;

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator, source, &bag);
    defer program.deinit();

    const bytes = try emitFunction(std.testing.allocator, program, "main", &bag);
    defer std.testing.allocator.free(bytes);

    try std.testing.expect(!bag.hasErrors());
    try std.testing.expect(bytes.len > 0);
    try std.testing.expectEqual(evm_asm.op.RETURN, bytes[bytes.len - 1]);
}

test "debug codegen accepts inline numeric value operands" {
    const source =
        \\fn main:
        \\    entry {
        \\        sum = add 0x1 0x2
        \\        mstore256 0 sum
        \\        return 0 0x20
        \\    }
    ;

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator, source, &bag);
    defer program.deinit();

    const bytes = try emitFunction(std.testing.allocator, program, "main", &bag);
    defer std.testing.allocator.free(bytes);

    try std.testing.expect(!bag.hasErrors());
    try std.testing.expect(bytes.len > 0);
    try std.testing.expectEqual(evm_asm.op.RETURN, bytes[bytes.len - 1]);
}

test "debug codegen emits branch bytecode" {
    const source =
        \\fn main:
        \\    entry {
        \\        ptr = const 0x0
        \\        len = const 0x20
        \\        c = const 0x1
        \\        => c ? @yes : @no
        \\    }
        \\    yes {
        \\        value = const 0x2a
        \\        mstore256 ptr value
        \\        return ptr len
        \\    }
        \\    no {
        \\        revert ptr ptr
        \\    }
    ;

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator, source, &bag);
    defer program.deinit();

    const bytes = try emitFunction(std.testing.allocator, program, "main", &bag);
    defer std.testing.allocator.free(bytes);

    try std.testing.expect(!bag.hasErrors());
    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.JUMPI) != null);
}

test "debug codegen preserves non-commutative operand order" {
    const source =
        \\fn main:
        \\    entry {
        \\        ptr = const 0x0
        \\        len = const 0x20
        \\        a = const 0x9
        \\        b = const 0x4
        \\        diff = sub a b
        \\        mstore256 ptr diff
        \\        return ptr len
        \\    }
    ;

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator, source, &bag);
    defer program.deinit();

    const bytes = try emitFunction(std.testing.allocator, program, "main", &bag);
    defer std.testing.allocator.free(bytes);

    const sub_sequence = [_]u8{
        evm_asm.op.PUSH1,  0xa0,             evm_asm.op.MLOAD,
        evm_asm.op.PUSH1,  0x80,             evm_asm.op.MLOAD,
        evm_asm.op.SUB,    evm_asm.op.PUSH1, 0xc0,
        evm_asm.op.MSTORE,
    };
    try std.testing.expect(!bag.hasErrors());
    try std.testing.expect(std.mem.indexOf(u8, bytes, &sub_sequence) != null);
}

test "debug codegen emits bump allocation and free pointer load" {
    const source =
        \\fn main:
        \\    entry {
        \\        word = const 0x20
        \\        ptr = malloc word
        \\        next = freeptr
        \\        mstore256 ptr next
        \\        return ptr word
        \\    }
    ;

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator, source, &bag);
    defer program.deinit();

    const bytes = try emitFunction(std.testing.allocator, program, "main", &bag);
    defer std.testing.allocator.free(bytes);

    const init_sequence = [_]u8{
        evm_asm.op.PUSH1,  0xc0,
        evm_asm.op.PUSH1,  0x20,
        evm_asm.op.MSTORE,
    };
    const malloc_sequence = [_]u8{
        evm_asm.op.PUSH1, 0x20,                    evm_asm.op.MLOAD,
        evm_asm.op.DUP1,  evm_asm.op.PUSH1,        0x40,
        evm_asm.op.MLOAD, evm_asm.op.DUP1,         evm_asm.op.CALLDATASIZE,
        evm_asm.op.DUP4,  evm_asm.op.CALLDATACOPY, evm_asm.op.ADD,
        evm_asm.op.PUSH1, 0x20,                    evm_asm.op.MSTORE,
        evm_asm.op.PUSH1, 0x60,                    evm_asm.op.MSTORE,
    };
    try std.testing.expect(!bag.hasErrors());
    try std.testing.expect(std.mem.startsWith(u8, bytes, &init_sequence));
    try std.testing.expect(std.mem.indexOf(u8, bytes, &malloc_sequence) != null);
}

test "debug codegen emits static allocation through the same bump allocator" {
    const source =
        \\fn main:
        \\    entry {
        \\        ptr = sallocany 0x40
        \\        word = const 0x20
        \\        return ptr word
        \\    }
    ;

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator, source, &bag);
    defer program.deinit();

    const bytes = try emitFunction(std.testing.allocator, program, "main", &bag);
    defer std.testing.allocator.free(bytes);

    const salloc_sequence = [_]u8{
        evm_asm.op.PUSH1,        0x20,                    evm_asm.op.MLOAD,
        evm_asm.op.DUP1,         evm_asm.op.PUSH1,        0x40,
        evm_asm.op.DUP1,         evm_asm.op.CALLDATASIZE, evm_asm.op.DUP4,
        evm_asm.op.CALLDATACOPY, evm_asm.op.ADD,          evm_asm.op.PUSH1,
        0x20,                    evm_asm.op.MSTORE,       evm_asm.op.PUSH1,
        0x40,                    evm_asm.op.MSTORE,
    };
    try std.testing.expect(!bag.hasErrors());
    try std.testing.expect(std.mem.indexOf(u8, bytes, &salloc_sequence) != null);
}

test "debug codegen rejects static allocations outside u32 address space" {
    const source =
        \\fn main:
        \\    entry {
        \\        ptr = salloc 0x100000000
        \\        stop
        \\    }
    ;

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator, source, &bag);
    defer program.deinit();

    try std.testing.expectError(
        CodegenError.UnsupportedSir,
        emitFunction(std.testing.allocator, program, "main", &bag),
    );
    try std.testing.expect(bag.hasErrors());
}

test "debug codegen lowers internal calls and returns" {
    const source =
        \\fn add_one:
        \\    entry input -> output {
        \\        one = const 0x1
        \\        output = add input one
        \\        iret
        \\    }
        \\
        \\fn main:
        \\    entry {
        \\        ptr = const 0x0
        \\        len = const 0x20
        \\        arg = const 0x29
        \\        value = icall @add_one arg
        \\        mstore256 ptr value
        \\        return ptr len
        \\    }
    ;

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator, source, &bag);
    defer program.deinit();

    const bytes = try emitFunction(std.testing.allocator, program, "main", &bag);
    defer std.testing.allocator.free(bytes);

    try std.testing.expect(!bag.hasErrors());
    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.JUMP) != null);
    try std.testing.expect(std.mem.indexOfScalarPos(u8, bytes, 1, evm_asm.op.JUMPDEST) != null);
}

test "debug codegen rejects recursive internal calls" {
    const source =
        \\fn recurse:
        \\    entry {
        \\        icall @recurse
        \\        iret
        \\    }
        \\
        \\fn main:
        \\    entry {
        \\        icall @recurse
        \\        stop
        \\    }
    ;

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator, source, &bag);
    defer program.deinit();

    try std.testing.expectError(
        CodegenError.UnsupportedSir,
        emitFunction(std.testing.allocator, program, "main", &bag),
    );
    try std.testing.expect(bag.hasErrors());
}

test "debug codegen patches init runtime offsets in deployment mode" {
    const source =
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
    ;

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator, source, &bag);
    defer program.deinit();

    const bytes = try emitFunction(std.testing.allocator, program, "init", &bag);
    defer std.testing.allocator.free(bytes);

    try std.testing.expect(!bag.hasErrors());
    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.CODECOPY) != null);
    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.RETURN) != null);
    try std.testing.expectEqual(evm_asm.op.STOP, bytes[bytes.len - 1]);
}

test "debug codegen appends data segments and lowers data offsets" {
    const source =
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
    ;

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator, source, &bag);
    defer program.deinit();

    const bytes = try emitFunction(std.testing.allocator, program, "main", &bag);
    defer std.testing.allocator.free(bytes);

    try std.testing.expect(!bag.hasErrors());
    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.CODECOPY) != null);
    try std.testing.expect(std.mem.endsWith(u8, bytes, &.{ 0x11, 0x22, 0x33, 0x44 }));
}

test "debug deployment data offsets are runtime-relative" {
    const source =
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
        \\data blob 0xaabbccdd
    ;

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator, source, &bag);
    defer program.deinit();

    const bytes = try emitFunction(std.testing.allocator, program, "init", &bag);
    defer std.testing.allocator.free(bytes);

    try std.testing.expect(!bag.hasErrors());
    try std.testing.expect(std.mem.endsWith(u8, bytes, &.{ 0xaa, 0xbb, 0xcc, 0xdd }));
    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.CODECOPY) != null);
}

test "debug codegen supports block input transfer" {
    const source =
        \\fn main:
        \\    entry -> ptr len {
        \\        ptr = const 0x0
        \\        len = const 0x20
        \\        => @done
        \\    }
        \\    done ptr_in len_in {
        \\        return ptr_in len_in
        \\    }
    ;

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator, source, &bag);
    defer program.deinit();

    const bytes = try emitFunction(std.testing.allocator, program, "main", &bag);
    defer std.testing.allocator.free(bytes);

    try std.testing.expect(!bag.hasErrors());
    try std.testing.expect(std.mem.indexOfScalar(u8, bytes, evm_asm.op.JUMP) != null);
}

test "debug codegen rejects mismatched block transfer counts" {
    const source =
        \\fn main:
        \\    entry -> only {
        \\        only = const 0x1
        \\        => @done
        \\    }
        \\    done a b {
        \\        stop
        \\    }
    ;

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator, source, &bag);
    defer program.deinit();

    try std.testing.expectError(
        CodegenError.UnsupportedSir,
        emitFunction(std.testing.allocator, program, "main", &bag),
    );
    try std.testing.expect(bag.hasErrors());
}
