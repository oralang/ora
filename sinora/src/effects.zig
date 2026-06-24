//! EVM/SIR side-effect model used by release stack scheduling.
//!
//! This is the Zig-side port of Plank's `operation/effects.rs` plus the
//! function-summary lookup used by `build_graph_effectful`. The scheduler needs
//! this layer because EVM operations are not only dataflow: `keccak256` reads
//! memory, `sstore` writes persistent storage, `call` updates returndata, logs
//! are ordered, and allocation has observable free-pointer effects. Pure
//! dataflow alone would let the stack scheduler move those operations across
//! each other incorrectly.

const std = @import("std");

const ir = @import("ir.zig");
const ops = @import("ops.zig");

pub const Effect = struct {
    bits: u16 = 0,

    // Each pair below is one ordering channel. The "read/minor" half may move
    // past other reads in the same channel, but the "write/major" half must
    // fence both earlier reads and later reads. `release_op_graph.zig` is the
    // only place that interprets these bits as scheduling edges.
    pub const PURE: Effect = .{ .bits = 0 };
    pub const MEMORY_READ: Effect = .{ .bits = 1 << 0 };
    pub const MEMORY_WRITE: Effect = .{ .bits = 1 << 1 };
    pub const RETURNDATA_READ: Effect = .{ .bits = 1 << 2 };
    pub const RETURNDATA_WRITE: Effect = .{ .bits = 1 << 3 };
    pub const ACCOUNTS_READ: Effect = .{ .bits = 1 << 4 };
    pub const ACCOUNTS_WRITE: Effect = .{ .bits = 1 << 5 };
    pub const PERSISTENT_READ: Effect = .{ .bits = 1 << 6 };
    pub const PERSISTENT_WRITE: Effect = .{ .bits = 1 << 7 };
    pub const TRANSIENT_READ: Effect = .{ .bits = 1 << 8 };
    pub const TRANSIENT_WRITE: Effect = .{ .bits = 1 << 9 };
    pub const REVERT: Effect = .{ .bits = 1 << 10 };
    pub const TERMINATE: Effect = .{ .bits = 1 << 11 };
    pub const ALLOC_ADVANCE: Effect = .{ .bits = 1 << 12 };
    pub const ALLOC_USE_FREE: Effect = .{ .bits = 1 << 13 };
    pub const LOGS: Effect = .{ .bits = 1 << 14 };

    // External calls can mutate accounts/storage/transient storage, append
    // logs, and replace returndata. Static calls get their narrower read/write
    // shape in the mnemonic table below.
    pub const EXTCALL: Effect = unionAll(&.{
        ACCOUNTS_WRITE,
        PERSISTENT_WRITE,
        TRANSIENT_WRITE,
        LOGS,
        RETURNDATA_WRITE,
    });

    pub const MINOR: Effect = unionAll(&.{
        MEMORY_READ,
        RETURNDATA_READ,
        ACCOUNTS_READ,
        PERSISTENT_READ,
        TRANSIENT_READ,
        REVERT,
        ALLOC_ADVANCE,
    });

    // A major effect orders against both previous major effects and every minor
    // effect since that previous major. This is Plank's MINOR/MAJOR rule in a
    // compact bitset form.
    pub const MAJOR: Effect = unionAll(&.{
        MEMORY_WRITE,
        RETURNDATA_WRITE,
        ACCOUNTS_WRITE,
        PERSISTENT_WRITE,
        TRANSIENT_WRITE,
        TERMINATE,
        ALLOC_USE_FREE,
    });

    // Unknown operations/callees are modeled as "write every effectful channel"
    // instead of as pure. That preserves correctness while keeping valid
    // programs schedulable.
    pub const FULL_BARRIER: Effect = unionAll(&.{ MAJOR, LOGS });

    pub fn unionAll(comptime effects: []const Effect) Effect {
        var bits: u16 = 0;
        for (effects) |effect| bits |= effect.bits;
        return .{ .bits = bits };
    }

    pub fn unionWith(self: Effect, other: Effect) Effect {
        return .{ .bits = self.bits | other.bits };
    }

    pub fn contains(self: Effect, other: Effect) bool {
        return (self.bits & other.bits) == other.bits;
    }

    pub fn intersects(self: Effect, other: Effect) bool {
        return (self.bits & other.bits) != 0;
    }

    pub fn remove(self: *Effect, other: Effect) void {
        self.bits &= ~other.bits;
    }

    pub fn isPure(self: Effect) bool {
        return self.bits == 0;
    }

    pub fn simplify(self: Effect) Effect {
        // If an operation both reads and writes a channel, the write side is
        // sufficient for scheduling: a major effect already fences the previous
        // reads and becomes the next read barrier. The adjacent bit layout lets
        // us drop those redundant read bits without branching per channel.
        var result = self;
        const reads_and_writes = Effect{
            .bits = (result.bits & Effect.MINOR.bits) & (result.bits >> 1),
        };
        result.remove(reads_and_writes);
        return result;
    }
};

const effect_map = std.StaticStringMap(Effect).initComptime(.{
    .{ "add", Effect.PURE },
    .{ "mul", Effect.PURE },
    .{ "sub", Effect.PURE },
    .{ "div", Effect.PURE },
    .{ "sdiv", Effect.PURE },
    .{ "mod", Effect.PURE },
    .{ "smod", Effect.PURE },
    .{ "addmod", Effect.PURE },
    .{ "mulmod", Effect.PURE },
    .{ "exp", Effect.PURE },
    .{ "signextend", Effect.PURE },
    .{ "lt", Effect.PURE },
    .{ "gt", Effect.PURE },
    .{ "slt", Effect.PURE },
    .{ "sgt", Effect.PURE },
    .{ "eq", Effect.PURE },
    .{ "iszero", Effect.PURE },
    .{ "and", Effect.PURE },
    .{ "or", Effect.PURE },
    .{ "xor", Effect.PURE },
    .{ "not", Effect.PURE },
    .{ "byte", Effect.PURE },
    .{ "shl", Effect.PURE },
    .{ "shr", Effect.PURE },
    .{ "sar", Effect.PURE },

    .{ "keccak256", Effect.MEMORY_READ },

    .{ "address", Effect.PURE },
    .{ "balance", Effect.ACCOUNTS_READ },
    .{ "origin", Effect.PURE },
    .{ "caller", Effect.PURE },
    .{ "callvalue", Effect.PURE },
    .{ "calldataload", Effect.PURE },
    .{ "calldatasize", Effect.PURE },
    .{ "calldatacopy", Effect.MEMORY_WRITE },
    .{ "codesize", Effect.PURE },
    .{ "codecopy", Effect.MEMORY_WRITE },
    .{ "gasprice", Effect.PURE },
    .{ "extcodesize", Effect.ACCOUNTS_READ },
    .{ "extcodecopy", Effect.unionAll(&.{ Effect.ACCOUNTS_READ, Effect.MEMORY_WRITE }) },
    .{ "returndatasize", Effect.RETURNDATA_READ },
    .{ "returndatacopy", Effect.unionAll(&.{ Effect.RETURNDATA_READ, Effect.MEMORY_WRITE, Effect.REVERT }) },
    .{ "extcodehash", Effect.ACCOUNTS_READ },
    .{ "gas", Effect.unionAll(&.{ Effect.ACCOUNTS_WRITE, Effect.PERSISTENT_WRITE, Effect.LOGS }) },

    .{ "blockhash", Effect.PURE },
    .{ "coinbase", Effect.PURE },
    .{ "timestamp", Effect.PURE },
    .{ "number", Effect.PURE },
    .{ "difficulty", Effect.PURE },
    .{ "gaslimit", Effect.PURE },
    .{ "chainid", Effect.PURE },
    .{ "selfbalance", Effect.ACCOUNTS_READ },
    .{ "basefee", Effect.PURE },
    .{ "blobhash", Effect.PURE },
    .{ "blobbasefee", Effect.PURE },

    .{ "sload", Effect.PERSISTENT_READ },
    .{ "sstore", Effect.PERSISTENT_WRITE },
    .{ "tload", Effect.TRANSIENT_READ },
    .{ "tstore", Effect.TRANSIENT_WRITE },

    .{ "log0", Effect.unionAll(&.{ Effect.MEMORY_READ, Effect.LOGS }) },
    .{ "log1", Effect.unionAll(&.{ Effect.MEMORY_READ, Effect.LOGS }) },
    .{ "log2", Effect.unionAll(&.{ Effect.MEMORY_READ, Effect.LOGS }) },
    .{ "log3", Effect.unionAll(&.{ Effect.MEMORY_READ, Effect.LOGS }) },
    .{ "log4", Effect.unionAll(&.{ Effect.MEMORY_READ, Effect.LOGS }) },

    .{ "create", Effect.unionAll(&.{ Effect.EXTCALL, Effect.MEMORY_READ }) },
    .{ "create2", Effect.unionAll(&.{ Effect.EXTCALL, Effect.MEMORY_READ }) },
    .{ "call", Effect.unionAll(&.{ Effect.EXTCALL, Effect.MEMORY_WRITE }) },
    .{ "callcode", Effect.unionAll(&.{ Effect.EXTCALL, Effect.MEMORY_WRITE }) },
    .{ "delegatecall", Effect.unionAll(&.{ Effect.EXTCALL, Effect.MEMORY_WRITE }) },
    .{ "staticcall", Effect.unionAll(&.{
        Effect.ACCOUNTS_READ,
        Effect.PERSISTENT_READ,
        Effect.TRANSIENT_READ,
        Effect.MEMORY_WRITE,
        Effect.RETURNDATA_WRITE,
    }) },

    .{ "malloc", Effect.ALLOC_ADVANCE },
    .{ "mallocany", Effect.ALLOC_ADVANCE },
    .{ "freeptr", Effect.ALLOC_USE_FREE },
    .{ "salloc", Effect.PURE },
    .{ "sallocany", Effect.PURE },

    .{ "mcopy", Effect.MEMORY_WRITE },
    .{ "copy", Effect.PURE },
    .{ "const", Effect.PURE },
    .{ "large_const", Effect.PURE },
    .{ "data_offset", Effect.PURE },
    .{ "noop", Effect.PURE },
    .{ "runtime_start_offset", Effect.PURE },
    .{ "init_end_offset", Effect.PURE },
    .{ "runtime_length", Effect.PURE },
});

pub fn ofInstruction(instruction: ir.Instruction) ?Effect {
    // `icall` is resolved through function summaries, not through the local
    // mnemonic table. Returning null tells the caller to take that path.
    if (std.mem.eql(u8, instruction.mnemonic, "icall")) return null;

    // Most instructions are fixed opcodes, so check the static effect table
    // first. Dynamic memory aliases (`mload256`, `mstore256`, etc.) fall through
    // to the opcode classifier only when needed.
    if (effect_map.get(instruction.mnemonic)) |effect| return effect;

    return switch (ops.lookup(instruction.mnemonic) orelse return null) {
        .memory_load => Effect.MEMORY_READ,
        .memory_store => Effect.MEMORY_WRITE,
        .internal_call, .fixed => null,
    };
}

pub fn ofTerminator(terminator: ir.Terminator) Effect {
    // Sinora keeps block terminators separate from ordinary instructions, while
    // Plank models stop/revert/return as effectful operations. Keep the same
    // ordering semantics here so terminators fence the final scheduled ops.
    return switch (terminator) {
        .return_ => Effect.unionAll(&.{ Effect.TERMINATE, Effect.MEMORY_READ }),
        .stop, .selfdestruct => Effect.TERMINATE,
        .revert => Effect.unionAll(&.{ Effect.MEMORY_READ, Effect.REVERT }),
        .invalid => Effect.REVERT,
        .jump, .branch, .switch_, .iret => Effect.PURE,
    };
}

pub const FunctionEffects = struct {
    allocator: std.mem.Allocator,
    function_indices: std.StringHashMap(usize),
    values: []Effect,

    pub fn deinit(self: *FunctionEffects) void {
        self.function_indices.deinit();
        self.allocator.free(self.values);
        self.* = undefined;
    }

    pub fn effectOf(self: FunctionEffects, name: []const u8) ?Effect {
        // The release scheduler asks this once per direct `icall`. Keeping a
        // map here avoids repeating a function-name scan while preserving the
        // same fail-closed contract: absent summaries are unknown effects.
        const index = self.function_indices.get(name) orelse return null;
        return self.values[index];
    }
};

pub fn analyzeFunctions(allocator: std.mem.Allocator, program: ir.Program) !FunctionEffects {
    const values = try allocator.alloc(Effect, program.functions.len);
    errdefer allocator.free(values);

    var function_indices = std.StringHashMap(usize).init(allocator);
    errdefer function_indices.deinit();
    try function_indices.ensureTotalCapacity(hashCapacity(program.functions.len));

    const block_indices = try allocator.alloc(std.StringHashMap(usize), program.functions.len);
    var initialized_block_indices: usize = 0;
    errdefer {
        for (block_indices[0..initialized_block_indices]) |*index| index.deinit();
        allocator.free(block_indices);
    }

    // The DFS state is one flat array over all blocks. Precomputing the first
    // block index for each function keeps per-block state dense, while the name
    // maps below make successor and direct-call lookup O(1) instead of rescans.
    const block_starts = try allocator.alloc(usize, program.functions.len);
    defer allocator.free(block_starts);

    var block_count: usize = 0;
    for (program.functions, 0..) |function, index| {
        try function_indices.put(function.name, index);
        values[index] = Effect.PURE;
        block_starts[index] = block_count;
        block_count += function.blocks.len;

        block_indices[index] = std.StringHashMap(usize).init(allocator);
        initialized_block_indices += 1;
        try block_indices[index].ensureTotalCapacity(hashCapacity(function.blocks.len));
        for (function.blocks, 0..) |block, block_index| {
            try block_indices[index].put(block.name, block_index);
        }
    }

    const block_state = try allocator.alloc(BlockState, block_count);
    defer allocator.free(block_state);
    @memset(block_state, .not_visited);

    var analysis = FunctionEffectsAnalysis{
        .program = program,
        .effects = values,
        .block_state = block_state,
        .block_starts = block_starts,
        .function_indices = &function_indices,
        .block_indices = block_indices,
    };
    for (program.functions, 0..) |_, function_index| {
        _ = analysis.getFunctionEffect(function_index);
    }

    for (block_indices[0..initialized_block_indices]) |*index| index.deinit();
    allocator.free(block_indices);
    initialized_block_indices = 0;

    return .{
        .allocator = allocator,
        .function_indices = function_indices,
        .values = values,
    };
}

const BlockState = enum {
    not_visited,
    processing,
    done,
};

const FunctionEffectsAnalysis = struct {
    program: ir.Program,
    effects: []Effect,
    block_state: []BlockState,
    block_starts: []const usize,
    function_indices: *const std.StringHashMap(usize),
    block_indices: []const std.StringHashMap(usize),

    fn getFunctionEffect(self: *FunctionEffectsAnalysis, function_index: usize) Effect {
        const function = self.program.functions[function_index];
        if (function.blocks.len == 0) return Effect.PURE;
        const entry_index = self.block_starts[function_index];
        return switch (self.block_state[entry_index]) {
            .not_visited => blk: {
                // Function summaries describe effects reachable from the entry
                // block. Unreachable blocks do not constrain call scheduling.
                const effect = self.aggregateBlockEffect(function_index, 0).simplify();
                self.effects[function_index] = effect;
                break :blk effect;
            },
            // Recursive functions or CFG backedges can consume gas forever.
            // Plank models that as possible revert-like control behavior; do
            // the same so callers cannot move terminating/reverting effects
            // across the call.
            .processing => Effect.REVERT,
            .done => self.effects[function_index],
        };
    }

    fn aggregateBlockEffect(self: *FunctionEffectsAnalysis, function_index: usize, block_index: usize) Effect {
        const function = self.program.functions[function_index];
        const block = function.blocks[block_index];
        const global_index = self.block_starts[function_index] + block_index;
        self.block_state[global_index] = .processing;

        var effect = Effect.PURE;
        for (block.instructions) |instruction| {
            effect = effect.unionWith(self.instructionEffect(instruction));
        }
        effect = effect.unionWith(ofTerminator(block.terminator));

        var successor_it = ir.successors(&block);
        while (successor_it.next()) |successor_name| {
            const successor_index = self.block_indices[function_index].get(successor_name) orelse {
                effect = effect.unionWith(Effect.FULL_BARRIER);
                continue;
            };
            const successor_global = self.block_starts[function_index] + successor_index;
            switch (self.block_state[successor_global]) {
                .not_visited => effect = effect.unionWith(self.aggregateBlockEffect(function_index, successor_index)),
                .processing => effect = effect.unionWith(Effect.REVERT),
                // The completed successor's effect has already been folded
                // into the entry summary if it is reachable along this DFS.
                .done => {},
            }
        }

        self.block_state[global_index] = .done;
        return effect;
    }

    fn instructionEffect(self: *FunctionEffectsAnalysis, instruction: ir.Instruction) Effect {
        if (!std.mem.eql(u8, instruction.mnemonic, "icall")) {
            return ofInstruction(instruction) orelse Effect.FULL_BARRIER;
        }
        if (instruction.operands.len == 0) return Effect.FULL_BARRIER;
        const name = canonicalFunctionRef(instruction.operands[0]);
        const function_index = self.function_indices.get(name) orelse return Effect.FULL_BARRIER;
        return self.getFunctionEffect(function_index);
    }
};

fn canonicalFunctionRef(operand: []const u8) []const u8 {
    if (operand.len > 0 and operand[0] == '@') return operand[1..];
    return operand;
}

fn hashCapacity(count: usize) u32 {
    return @intCast(@min(count, std.math.maxInt(u32)));
}

test "operation effects follow Plank channels" {
    try std.testing.expect((ofInstruction(.{
        .results = &.{"h"},
        .mnemonic = "keccak256",
        .operands = &.{ "p", "l" },
        .line = 1,
    }) orelse unreachable).contains(Effect.MEMORY_READ));
    try std.testing.expect((ofInstruction(.{
        .results = &.{},
        .mnemonic = "sstore",
        .operands = &.{ "k", "v" },
        .line = 1,
    }) orelse unreachable).contains(Effect.PERSISTENT_WRITE));
    try std.testing.expect((ofInstruction(.{
        .results = &.{"p"},
        .mnemonic = "malloc",
        .operands = &.{"n"},
        .line = 1,
    }) orelse unreachable).contains(Effect.ALLOC_ADVANCE));
    try std.testing.expectEqual(@as(?Effect, null), ofInstruction(.{
        .results = &.{"r"},
        .mnemonic = "icall",
        .operands = &.{"@callee"},
        .line = 1,
    }));
}

test "dynamic memory operations use the opcode classifier" {
    try std.testing.expect((ofInstruction(.{
        .results = &.{"v"},
        .mnemonic = "mload256",
        .operands = &.{"p"},
        .line = 1,
    }) orelse unreachable).contains(Effect.MEMORY_READ));
    try std.testing.expect((ofInstruction(.{
        .results = &.{},
        .mnemonic = "mstore256",
        .operands = &.{ "p", "v" },
        .line = 1,
    }) orelse unreachable).contains(Effect.MEMORY_WRITE));
}

test "simplify drops minor read when matching write exists" {
    const both = Effect.unionAll(&.{ Effect.MEMORY_READ, Effect.MEMORY_WRITE, Effect.LOGS });
    const simplified = both.simplify();
    try std.testing.expect(!simplified.contains(Effect.MEMORY_READ));
    try std.testing.expect(simplified.contains(Effect.MEMORY_WRITE));
    try std.testing.expect(simplified.contains(Effect.LOGS));
}

test "function effects include internal callee summaries" {
    var program = ir.Program.init(std.testing.allocator);
    defer program.deinit();
    const arena = program.allocator();
    program.functions = try arena.dupe(ir.Function, &.{
        .{
            .name = try arena.dupe(u8, "store"),
            .blocks = try arena.dupe(ir.Block, &.{.{
                .name = try arena.dupe(u8, "entry"),
                .inputs = &.{},
                .outputs = &.{},
                .instructions = try arena.dupe(ir.Instruction, &.{.{
                    .results = &.{},
                    .mnemonic = try arena.dupe(u8, "sstore"),
                    .operands = &.{ "k", "v" },
                    .line = 1,
                }}),
                .terminator = .iret,
                .line = 1,
            }}),
            .line = 1,
        },
        .{
            .name = try arena.dupe(u8, "main"),
            .blocks = try arena.dupe(ir.Block, &.{.{
                .name = try arena.dupe(u8, "entry"),
                .inputs = &.{},
                .outputs = &.{},
                .instructions = try arena.dupe(ir.Instruction, &.{.{
                    .results = &.{},
                    .mnemonic = try arena.dupe(u8, "icall"),
                    .operands = &.{"@store"},
                    .line = 1,
                }}),
                .terminator = .stop,
                .line = 1,
            }}),
            .line = 1,
        },
    });

    var summaries = try analyzeFunctions(std.testing.allocator, program);
    defer summaries.deinit();
    try std.testing.expect(summaries.effectOf("main").?.contains(Effect.PERSISTENT_WRITE));
    try std.testing.expect(summaries.effectOf("main").?.contains(Effect.TERMINATE));
}

test "function effects treat control-flow cycles as possible revert" {
    var program = ir.Program.init(std.testing.allocator);
    defer program.deinit();
    const arena = program.allocator();
    program.functions = try arena.dupe(ir.Function, &.{.{
        .name = try arena.dupe(u8, "main"),
        .blocks = try arena.dupe(ir.Block, &.{
            .{
                .name = try arena.dupe(u8, "entry"),
                .inputs = &.{},
                .outputs = &.{},
                .instructions = &.{},
                .terminator = .{ .jump = "loop" },
                .line = 1,
            },
            .{
                .name = try arena.dupe(u8, "loop"),
                .inputs = &.{},
                .outputs = &.{},
                .instructions = &.{},
                .terminator = .{ .jump = "loop" },
                .line = 2,
            },
        }),
        .line = 1,
    }});

    var summaries = try analyzeFunctions(std.testing.allocator, program);
    defer summaries.deinit();
    try std.testing.expect(summaries.effectOf("main").?.contains(Effect.REVERT));
}
