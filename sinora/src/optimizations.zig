//! First Sinora-owned ports of Plank's small optimization passes.
//!
//! These passes intentionally preserve Plank's conservative shape:
//! copy-propagation rewrites uses but keeps the copy operation, unused-op
//! elimination replaces dead removable operations with `noop`, and the switch
//! peephole only rewrites the exact one-zero-case-with-default form.

const std = @import("std");

const effects = @import("effects.zig");
const ir = @import("ir.zig");
const ops = @import("ops.zig");

pub const OptimizeError = std.mem.Allocator.Error;

const EvalOp = enum {
    not,
    iszero,
    add,
    mul,
    sub,
    div,
    mod,
    sdiv,
    smod,
    lt,
    gt,
    slt,
    sgt,
    eq,
    and_,
    or_,
    xor,
    byte,
    shl,
    shr,
    sar,
    signextend,
    exp,
    addmod,
    mulmod,
};

const eval_op_map = std.StaticStringMap(EvalOp).initComptime(.{
    .{ "not", .not },
    .{ "iszero", .iszero },
    .{ "add", .add },
    .{ "mul", .mul },
    .{ "sub", .sub },
    .{ "div", .div },
    .{ "mod", .mod },
    .{ "sdiv", .sdiv },
    .{ "smod", .smod },
    .{ "lt", .lt },
    .{ "gt", .gt },
    .{ "slt", .slt },
    .{ "sgt", .sgt },
    .{ "eq", .eq },
    .{ "and", .and_ },
    .{ "or", .or_ },
    .{ "xor", .xor },
    .{ "byte", .byte },
    .{ "shl", .shl },
    .{ "shr", .shr },
    .{ "sar", .sar },
    .{ "signextend", .signextend },
    .{ "exp", .exp },
    .{ "addmod", .addmod },
    .{ "mulmod", .mulmod },
});

const evm_const_map = std.StaticStringMap(EvmConstKind).initComptime(.{
    .{ "address", .address },
    .{ "origin", .origin },
    .{ "caller", .caller },
    .{ "callvalue", .callvalue },
    .{ "calldatasize", .calldatasize },
    .{ "codesize", .codesize },
    .{ "gasprice", .gasprice },
    .{ "coinbase", .coinbase },
    .{ "timestamp", .timestamp },
    .{ "number", .number },
    .{ "difficulty", .difficulty },
    .{ "gaslimit", .gaslimit },
    .{ "chainid", .chainid },
    .{ "basefee", .basefee },
    .{ "blobbasefee", .blobbasefee },
    .{ "runtime_start_offset", .runtime_start_offset },
    .{ "init_end_offset", .init_end_offset },
    .{ "runtime_length", .runtime_length },
});

pub fn copyPropagation(allocator: std.mem.Allocator, source: ir.Program) OptimizeError!ir.Program {
    return rewriteProgram(allocator, source, copyPropBlock);
}

pub fn literalCommoning(allocator: std.mem.Allocator, source: ir.Program) OptimizeError!ir.Program {
    return rewriteProgram(allocator, source, literalCommoningBlock);
}

pub fn switchPeephole(allocator: std.mem.Allocator, source: ir.Program) OptimizeError!ir.Program {
    return rewriteProgram(allocator, source, switchPeepholeBlock);
}

pub fn sccp(allocator: std.mem.Allocator, source: ir.Program) OptimizeError!ir.Program {
    var state = SccpState.init(allocator);
    defer state.deinit();
    try state.analyze(source);
    return state.apply(allocator, source);
}

pub fn unusedOperationElimination(allocator: std.mem.Allocator, source: ir.Program) OptimizeError!ir.Program {
    var current = try cloneProgram(allocator, source);
    errdefer current.deinit();

    var changed = true;
    while (changed) {
        changed = false;
        var uses = try collectUses(allocator, current);
        defer uses.deinit();

        var next = ir.Program.init(allocator);
        errdefer next.deinit();
        const arena = next.allocator();
        const functions = try arena.alloc(ir.Function, current.functions.len);
        for (current.functions, functions) |function, *out_function| {
            const blocks = try arena.alloc(ir.Block, function.blocks.len);
            for (function.blocks, blocks) |block, *out_block| {
                const instructions = try arena.alloc(ir.Instruction, block.instructions.len);
                for (block.instructions, instructions) |instruction, *out_instruction| {
                    if (isRemovableWhenUnused(instruction, &uses)) {
                        out_instruction.* = .{
                            .results = &.{},
                            .mnemonic = try arena.dupe(u8, "noop"),
                            .operands = &.{},
                            .line = instruction.line,
                        };
                        changed = true;
                    } else {
                        out_instruction.* = try cloneInstruction(arena, instruction);
                    }
                }
                out_block.* = .{
                    .name = try arena.dupe(u8, block.name),
                    .inputs = try cloneStringSlice(arena, block.inputs),
                    .outputs = try cloneStringSlice(arena, block.outputs),
                    .instructions = instructions,
                    .terminator = try cloneTerminator(arena, block.terminator),
                    .line = block.line,
                };
            }
            out_function.* = .{
                .name = try arena.dupe(u8, function.name),
                .blocks = blocks,
                .line = function.line,
            };
        }
        next.functions = functions;
        next.data_segments = try cloneDataSegments(arena, current.data_segments);

        current.deinit();
        current = next;
    }

    return current;
}

pub fn defragment(allocator: std.mem.Allocator, source: ir.Program) OptimizeError!ir.Program {
    var reachable = ReachableGraph.init(allocator);
    defer reachable.deinit();
    try reachable.compute(source);

    var result = ir.Program.init(allocator);
    errdefer result.deinit();
    const arena = result.allocator();

    var data_refs = std.StringHashMap(void).init(allocator);
    defer data_refs.deinit();

    var functions: std.ArrayList(ir.Function) = .empty;
    defer functions.deinit(arena);
    try functions.ensureTotalCapacity(arena, source.functions.len);
    for (source.functions) |function| {
        if (!reachable.hasFunction(function.name)) continue;

        var blocks: std.ArrayList(ir.Block) = .empty;
        defer blocks.deinit(arena);
        try blocks.ensureTotalCapacity(arena, function.blocks.len);
        for (function.blocks) |block| {
            if (!reachable.hasBlock(function.name, block.name)) continue;
            const cloned = try defragmentBlock(arena, block, &data_refs);
            blocks.appendAssumeCapacity(cloned);
        }
        functions.appendAssumeCapacity(.{
            .name = try arena.dupe(u8, function.name),
            .blocks = try blocks.toOwnedSlice(arena),
            .line = function.line,
        });
    }

    result.functions = try functions.toOwnedSlice(arena);
    result.data_segments = try cloneReferencedDataSegments(arena, source.data_segments, &data_refs);
    return result;
}

const ReachableGraph = struct {
    allocator: std.mem.Allocator,
    functions: std.StringHashMap(void),
    blocks: ScopedNameSet,

    fn init(allocator: std.mem.Allocator) ReachableGraph {
        return .{
            .allocator = allocator,
            .functions = std.StringHashMap(void).init(allocator),
            .blocks = ScopedNameSet.init(allocator),
        };
    }

    fn deinit(self: *ReachableGraph) void {
        self.blocks.deinit();
        self.functions.deinit();
        self.* = undefined;
    }

    fn compute(self: *ReachableGraph, program: ir.Program) !void {
        // Defragmentation is the only pass here that can remove whole
        // functions/blocks/data segments. It starts from Plank's conventional
        // roots (`init`, `main`, or the first function for tiny fixtures), then
        // follows direct `icall` references and CFG successors. Unknown dynamic
        // calls do not exist in SIR; indirect control flow is represented by
        // ordinary terminators.
        var function_worklist: std.ArrayList([]const u8) = .empty;
        defer function_worklist.deinit(self.allocator);
        try function_worklist.ensureTotalCapacity(self.allocator, program.functions.len);

        if (findFunctionByName(program, "init") != null) try self.enqueueFunction(&function_worklist, "init");
        if (findFunctionByName(program, "main") != null) try self.enqueueFunction(&function_worklist, "main");
        if (function_worklist.items.len == 0 and program.functions.len != 0) {
            try self.enqueueFunction(&function_worklist, program.functions[0].name);
        }

        while (function_worklist.items.len != 0) {
            const function_name = function_worklist.orderedRemove(function_worklist.items.len - 1);
            const function = findFunctionByName(program, function_name) orelse continue;
            if (function.blocks.len == 0) continue;

            var block_worklist: std.ArrayList([]const u8) = .empty;
            defer block_worklist.deinit(self.allocator);
            try block_worklist.ensureTotalCapacity(self.allocator, function.blocks.len);
            try self.enqueueBlock(&block_worklist, function.name, function.blocks[0].name);

            while (block_worklist.items.len != 0) {
                const block_name = block_worklist.orderedRemove(block_worklist.items.len - 1);
                const block = findBlockByName(function, block_name) orelse continue;

                for (block.instructions) |instruction| {
                    if (instruction.operands.len != 0 and std.mem.eql(u8, instruction.mnemonic, "icall")) {
                        if (functionNameFromRef(instruction.operands[0])) |callee| {
                            try self.enqueueFunction(&function_worklist, callee);
                        }
                    }
                }

                var successors = ir.successors(&block);
                while (successors.next()) |successor| {
                    try self.enqueueBlock(&block_worklist, function.name, successor);
                }
            }
        }
    }

    fn enqueueFunction(self: *ReachableGraph, worklist: *std.ArrayList([]const u8), name: []const u8) !void {
        const entry = try self.functions.getOrPut(name);
        if (!entry.found_existing) try worklist.append(self.allocator, name);
    }

    fn enqueueBlock(self: *ReachableGraph, worklist: *std.ArrayList([]const u8), function_name: []const u8, block_name: []const u8) !void {
        if (try self.blocks.add(function_name, block_name)) {
            try worklist.append(self.allocator, block_name);
        }
    }

    fn hasFunction(self: ReachableGraph, name: []const u8) bool {
        return self.functions.contains(name);
    }

    fn hasBlock(self: ReachableGraph, function_name: []const u8, block_name: []const u8) bool {
        return self.blocks.contains(function_name, block_name);
    }
};

const ScopedName = struct {
    function_name: []const u8,
    name: []const u8,
};

const ScopedNameContext = struct {
    pub fn hash(_: @This(), key: ScopedName) u64 {
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(std.mem.asBytes(&key.function_name.len));
        hasher.update(key.function_name);
        hasher.update(std.mem.asBytes(&key.name.len));
        hasher.update(key.name);
        return hasher.final();
    }

    pub fn eql(_: @This(), lhs: ScopedName, rhs: ScopedName) bool {
        return std.mem.eql(u8, lhs.function_name, rhs.function_name) and std.mem.eql(u8, lhs.name, rhs.name);
    }
};

const ScopedNameSet = struct {
    map: std.HashMap(ScopedName, void, ScopedNameContext, std.hash_map.default_max_load_percentage),

    fn init(allocator: std.mem.Allocator) ScopedNameSet {
        return .{ .map = std.HashMap(ScopedName, void, ScopedNameContext, std.hash_map.default_max_load_percentage).init(allocator) };
    }

    fn deinit(self: *ScopedNameSet) void {
        self.map.deinit();
    }

    fn add(self: *ScopedNameSet, function_name: []const u8, name: []const u8) !bool {
        const entry = try self.map.getOrPut(.{ .function_name = function_name, .name = name });
        return !entry.found_existing;
    }

    fn contains(self: ScopedNameSet, function_name: []const u8, name: []const u8) bool {
        return self.map.contains(.{ .function_name = function_name, .name = name });
    }
};

fn defragmentBlock(arena: std.mem.Allocator, block: ir.Block, data_refs: *std.StringHashMap(void)) !ir.Block {
    var instructions: std.ArrayList(ir.Instruction) = .empty;
    defer instructions.deinit(arena);
    try instructions.ensureTotalCapacity(arena, block.instructions.len);
    for (block.instructions) |instruction| {
        if (std.mem.eql(u8, instruction.mnemonic, "noop")) continue;
        for (instruction.operands) |operand| {
            if (isDataRef(operand)) try data_refs.put(dataRefName(operand), {});
        }
        instructions.appendAssumeCapacity(try cloneInstruction(arena, instruction));
    }

    return .{
        .name = try arena.dupe(u8, block.name),
        .inputs = try cloneStringSlice(arena, block.inputs),
        .outputs = try cloneStringSlice(arena, block.outputs),
        .instructions = try instructions.toOwnedSlice(arena),
        .terminator = try cloneTerminator(arena, block.terminator),
        .line = block.line,
    };
}

const RewriteBlockFn = *const fn (std.mem.Allocator, ir.Block) OptimizeError!ir.Block;

const LatticeValue = union(enum) {
    unknown,
    constant: u256,
    evm_const: EvmConstKind,
    overdefined,

    fn eql(lhs: LatticeValue, rhs: LatticeValue) bool {
        return switch (lhs) {
            .unknown => rhs == .unknown,
            .overdefined => rhs == .overdefined,
            .constant => |left| switch (rhs) {
                .constant => |right| left == right,
                else => false,
            },
            .evm_const => |left| switch (rhs) {
                .evm_const => |right| left == right,
                else => false,
            },
        };
    }

    fn meet(lhs: LatticeValue, rhs: LatticeValue) LatticeValue {
        if (lhs.eql(rhs)) return lhs;
        return switch (lhs) {
            .unknown => rhs,
            .overdefined => .overdefined,
            .constant, .evm_const => switch (rhs) {
                .unknown => lhs,
                .overdefined => .overdefined,
                .constant, .evm_const => .overdefined,
            },
        };
    }
};

const EvmConstKind = enum {
    address,
    origin,
    caller,
    callvalue,
    calldatasize,
    codesize,
    gasprice,
    coinbase,
    timestamp,
    number,
    difficulty,
    gaslimit,
    chainid,
    basefee,
    blobbasefee,
    runtime_start_offset,
    init_end_offset,
    runtime_length,

    fn knownNonZeroForBranch(self: EvmConstKind) bool {
        return switch (self) {
            .address,
            .origin,
            .caller,
            .timestamp,
            .gaslimit,
            .chainid,
            => true,
            else => false,
        };
    }
};

const SccpState = struct {
    allocator: std.mem.Allocator,
    lattice: std.StringHashMap(LatticeValue),
    reachable: ScopedNameSet,

    fn init(allocator: std.mem.Allocator) SccpState {
        return .{
            .allocator = allocator,
            .lattice = std.StringHashMap(LatticeValue).init(allocator),
            .reachable = ScopedNameSet.init(allocator),
        };
    }

    fn deinit(self: *SccpState) void {
        self.reachable.deinit();
        self.lattice.deinit();
        self.* = undefined;
    }

    fn analyze(self: *SccpState, program: ir.Program) !void {
        // SCCP is deliberately sparse but not aggressive: it marks reachable
        // blocks, flows block-output values into successor inputs, and folds
        // operations only when all required operands are known constants. Any
        // unknown value meets to overdefined instead of guessing.
        for (program.functions) |function| {
            if (function.blocks.len == 0) continue;
            _ = try self.reachable.add(function.name, function.blocks[0].name);
            for (function.blocks[0].inputs) |input| {
                _ = try self.meet(input, .overdefined);
            }
        }

        var changed = true;
        while (changed) {
            changed = false;
            for (program.functions) |function| {
                for (function.blocks) |block| {
                    if (!self.reachable.contains(function.name, block.name)) continue;
                    changed = (try self.processBlock(program, function, block)) or changed;
                }
            }
        }
    }

    fn processBlock(self: *SccpState, program: ir.Program, function: ir.Function, block: ir.Block) !bool {
        var changed = false;
        for (block.instructions) |instruction| {
            changed = (try self.processInstruction(instruction)) or changed;
        }
        changed = (try self.processControl(program, function, block)) or changed;
        return changed;
    }

    fn processInstruction(self: *SccpState, instruction: ir.Instruction) !bool {
        // Any op SCCP does not understand becomes overdefined. That is the
        // safety boundary: folding is an optimization, never a semantic repair.
        if (instruction.results.len == 0) return false;

        if (instructionConstant(instruction)) |constant| {
            return self.meet(instruction.results[0], constant);
        }

        if (try self.evaluateInstruction(instruction)) |evaluated| {
            return self.meet(evaluated.output, .{ .constant = evaluated.value });
        }

        var changed = false;
        for (instruction.results) |result| {
            changed = (try self.meet(result, .overdefined)) or changed;
        }
        return changed;
    }

    fn processControl(self: *SccpState, program: ir.Program, function: ir.Function, block: ir.Block) !bool {
        var changed = false;
        switch (block.terminator) {
            .jump => |target| {
                changed = (try self.markReachable(function.name, target)) or changed;
                changed = (try self.flowOutputsTo(function, block, target)) or changed;
            },
            .branch => |branch| {
                const value = self.valueOf(branch.condition);
                const reachability = branchReachability(value);
                if (reachability.zero) {
                    changed = (try self.markReachable(function.name, branch.zero_target)) or changed;
                    changed = (try self.flowOutputsTo(function, block, branch.zero_target)) or changed;
                }
                if (reachability.non_zero) {
                    changed = (try self.markReachable(function.name, branch.non_zero_target)) or changed;
                    changed = (try self.flowOutputsTo(function, block, branch.non_zero_target)) or changed;
                }
            },
            .switch_ => |switch_term| {
                if (self.constValue(switch_term.selector)) |selector| {
                    const target = switchTargetForConst(switch_term, selector);
                    changed = (try self.markReachable(function.name, target)) or changed;
                    changed = (try self.flowOutputsTo(function, block, target)) or changed;
                } else {
                    for (switch_term.cases) |case| {
                        changed = (try self.markReachable(function.name, case.target)) or changed;
                        changed = (try self.flowOutputsTo(function, block, case.target)) or changed;
                    }
                    if (switch_term.default_target.len != 0) {
                        changed = (try self.markReachable(function.name, switch_term.default_target)) or changed;
                        changed = (try self.flowOutputsTo(function, block, switch_term.default_target)) or changed;
                    }
                }
            },
            .return_, .revert, .stop, .invalid, .selfdestruct, .iret => {},
        }
        _ = program;
        return changed;
    }

    fn flowOutputsTo(self: *SccpState, function: ir.Function, from: ir.Block, target_name: []const u8) !bool {
        const target = findBlockByName(function, target_name) orelse return false;
        var changed = false;
        const count = @min(from.outputs.len, target.inputs.len);
        for (0..count) |index| {
            changed = (try self.meet(target.inputs[index], self.valueOf(from.outputs[index]))) or changed;
        }
        return changed;
    }

    fn markReachable(self: *SccpState, function_name: []const u8, block_name: []const u8) !bool {
        return self.reachable.add(function_name, block_name);
    }

    fn meet(self: *SccpState, name: []const u8, value: LatticeValue) !bool {
        const entry = try self.lattice.getOrPut(name);
        if (!entry.found_existing) {
            entry.value_ptr.* = value;
            return value != .unknown;
        }
        const merged = entry.value_ptr.*.meet(value);
        if (entry.value_ptr.*.eql(merged)) return false;
        entry.value_ptr.* = merged;
        return true;
    }

    fn valueOf(self: SccpState, name: []const u8) LatticeValue {
        return self.lattice.get(name) orelse .unknown;
    }

    fn constValue(self: SccpState, name: []const u8) ?u256 {
        return switch (self.valueOf(name)) {
            .constant => |value| value,
            else => null,
        };
    }

    fn evaluateInstruction(self: SccpState, instruction: ir.Instruction) !?struct { output: []const u8, value: u256 } {
        if (instruction.results.len != 1) return null;
        const out = instruction.results[0];
        const op = eval_op_map.get(instruction.mnemonic) orelse return null;

        switch (op) {
            .not => {
                const value = self.constValue(instruction.operands[0]) orelse return null;
                return .{ .output = out, .value = ~value };
            },
            .iszero => {
                const value = self.constValue(instruction.operands[0]) orelse return null;
                return .{ .output = out, .value = @intFromBool(value == 0) };
            },
            else => {},
        }

        if (instruction.operands.len < 2) return null;
        const a = self.constValue(instruction.operands[0]) orelse return null;
        const b = self.constValue(instruction.operands[1]) orelse return null;

        switch (op) {
            .add => return .{ .output = out, .value = a +% b },
            .mul => return .{ .output = out, .value = a *% b },
            .sub => return .{ .output = out, .value = a -% b },
            .div => return .{ .output = out, .value = if (b == 0) 0 else a / b },
            .mod => return .{ .output = out, .value = if (b == 0) 0 else a % b },
            .sdiv => return .{ .output = out, .value = sdivU256(a, b) },
            .smod => return .{ .output = out, .value = smodU256(a, b) },
            .lt => return .{ .output = out, .value = @intFromBool(a < b) },
            .gt => return .{ .output = out, .value = @intFromBool(a > b) },
            .slt => return .{ .output = out, .value = @intFromBool(asI256(a) < asI256(b)) },
            .sgt => return .{ .output = out, .value = @intFromBool(asI256(a) > asI256(b)) },
            .eq => return .{ .output = out, .value = @intFromBool(a == b) },
            .and_ => return .{ .output = out, .value = a & b },
            .or_ => return .{ .output = out, .value = a | b },
            .xor => return .{ .output = out, .value = a ^ b },
            .byte => return .{ .output = out, .value = byteU256(a, b) },
            .shl => return .{ .output = out, .value = shlU256(a, b) },
            .shr => return .{ .output = out, .value = shrU256(a, b) },
            .sar => return .{ .output = out, .value = sarU256(a, b) },
            .signextend => return .{ .output = out, .value = signExtendU256(a, b) },
            .exp => return .{ .output = out, .value = expU256(a, b) },
            .addmod, .mulmod => {},
            .not, .iszero => unreachable,
        }

        if (instruction.operands.len >= 3) {
            const n = self.constValue(instruction.operands[2]) orelse return null;
            return switch (op) {
                .addmod => .{ .output = out, .value = addmodU256(a, b, n) },
                .mulmod => .{ .output = out, .value = mulmodU256(a, b, n) },
                else => null,
            };
        }
        return null;
    }

    fn apply(self: SccpState, allocator: std.mem.Allocator, source: ir.Program) !ir.Program {
        var result = ir.Program.init(allocator);
        errdefer result.deinit();
        const arena = result.allocator();

        const functions = try arena.alloc(ir.Function, source.functions.len);
        for (source.functions, functions) |function, *out_function| {
            const blocks = try arena.alloc(ir.Block, function.blocks.len);
            for (function.blocks, blocks) |block, *out_block| {
                out_block.* = try self.rewriteBlock(arena, function, block);
            }
            out_function.* = .{
                .name = try arena.dupe(u8, function.name),
                .blocks = blocks,
                .line = function.line,
            };
        }

        result.functions = functions;
        result.data_segments = try cloneDataSegments(arena, source.data_segments);
        return result;
    }

    fn rewriteBlock(self: SccpState, arena: std.mem.Allocator, function: ir.Function, block: ir.Block) !ir.Block {
        const reachable = self.reachable.contains(function.name, block.name);
        const instructions = try arena.alloc(ir.Instruction, block.instructions.len);
        for (block.instructions, instructions) |instruction, *out_instruction| {
            out_instruction.* = if (reachable)
                try self.rewriteInstruction(arena, instruction)
            else
                try cloneInstruction(arena, instruction);
        }
        return .{
            .name = try arena.dupe(u8, block.name),
            .inputs = try cloneStringSlice(arena, block.inputs),
            .outputs = try cloneStringSlice(arena, block.outputs),
            .instructions = instructions,
            .terminator = if (reachable) try self.rewriteTerminator(arena, block.terminator) else try cloneTerminator(arena, block.terminator),
            .line = block.line,
        };
    }

    fn rewriteInstruction(self: SccpState, arena: std.mem.Allocator, instruction: ir.Instruction) !ir.Instruction {
        if (instruction.results.len == 1) {
            if (self.constValue(instruction.results[0])) |value| {
                if (value <= std.math.maxInt(u32)) {
                    return .{
                        .results = try cloneStringSlice(arena, instruction.results),
                        .mnemonic = try arena.dupe(u8, "const"),
                        .operands = try singleOwnedStringSlice(arena, try std.fmt.allocPrint(arena, "{d}", .{value})),
                        .line = instruction.line,
                    };
                }
            }
        }
        return cloneInstruction(arena, instruction);
    }

    fn rewriteTerminator(self: SccpState, arena: std.mem.Allocator, terminator: ir.Terminator) !ir.Terminator {
        return switch (terminator) {
            .branch => |branch| blk: {
                if (self.constValue(branch.condition)) |value| {
                    break :blk .{ .jump = try arena.dupe(u8, if (value == 0) branch.zero_target else branch.non_zero_target) };
                }
                break :blk try cloneTerminator(arena, terminator);
            },
            .switch_ => |switch_term| blk: {
                if (self.constValue(switch_term.selector)) |value| {
                    break :blk .{ .jump = try arena.dupe(u8, switchTargetForConst(switch_term, value)) };
                }
                break :blk try cloneTerminator(arena, terminator);
            },
            else => try cloneTerminator(arena, terminator),
        };
    }
};

const BranchReachability = struct {
    zero: bool,
    non_zero: bool,
};

fn instructionConstant(instruction: ir.Instruction) ?LatticeValue {
    if (instruction.results.len != 1) return null;
    if ((std.mem.eql(u8, instruction.mnemonic, "const") or std.mem.eql(u8, instruction.mnemonic, "large_const")) and instruction.operands.len == 1) {
        return .{ .constant = parseU256Literal(instruction.operands[0]) orelse return null };
    }
    if (evmConstKind(instruction.mnemonic)) |kind| {
        if (instruction.operands.len == 0) return .{ .evm_const = kind };
    }
    return null;
}

fn evmConstKind(mnemonic: []const u8) ?EvmConstKind {
    return evm_const_map.get(mnemonic);
}

fn branchReachability(value: LatticeValue) BranchReachability {
    return switch (value) {
        .constant => |constant| if (constant == 0)
            .{ .zero = true, .non_zero = false }
        else
            .{ .zero = false, .non_zero = true },
        .evm_const => |kind| if (kind.knownNonZeroForBranch())
            .{ .zero = false, .non_zero = true }
        else
            .{ .zero = true, .non_zero = true },
        .unknown, .overdefined => .{ .zero = true, .non_zero = true },
    };
}

fn switchTargetForConst(switch_term: ir.SwitchTerminator, value: u256) []const u8 {
    for (switch_term.cases) |case| {
        if (parseU256Literal(case.value)) |case_value| {
            if (case_value == value) return case.target;
        }
    }
    return switch_term.default_target;
}

fn parseU256Literal(text: []const u8) ?u256 {
    if (text.len == 0) return null;
    if (text[0] == '-') {
        const magnitude = std.fmt.parseUnsigned(u256, text[1..], 0) catch return null;
        return 0 -% magnitude;
    }
    return std.fmt.parseUnsigned(u256, text, 0) catch null;
}

fn asI256(value: u256) i256 {
    return @bitCast(value);
}

fn fromI256(value: i256) u256 {
    return @bitCast(value);
}

fn sdivU256(a: u256, b: u256) u256 {
    if (b == 0) return 0;
    const sa = asI256(a);
    const sb = asI256(b);
    if (sa == std.math.minInt(i256) and sb == -1) return a;
    return fromI256(@divTrunc(sa, sb));
}

fn smodU256(a: u256, b: u256) u256 {
    if (b == 0) return 0;
    const sa = asI256(a);
    const aa = signedMagnitude(a);
    const bb = signedMagnitude(b);
    const remainder = aa % bb;
    return if (sa < 0) 0 -% remainder else remainder;
}

fn signedMagnitude(value: u256) u256 {
    return if (asI256(value) < 0) 0 -% value else value;
}

fn byteU256(index: u256, value: u256) u256 {
    if (index >= 32) return 0;
    const shift: u8 = @intCast((31 - @as(u8, @intCast(index))) * 8);
    return (value >> shift) & 0xff;
}

fn shlU256(shift: u256, value: u256) u256 {
    if (shift >= 256) return 0;
    return value << @as(u8, @intCast(shift));
}

fn shrU256(shift: u256, value: u256) u256 {
    if (shift >= 256) return 0;
    return value >> @as(u8, @intCast(shift));
}

fn sarU256(shift: u256, value: u256) u256 {
    const signed = asI256(value);
    if (shift >= 256) return if (signed < 0) std.math.maxInt(u256) else 0;
    return fromI256(signed >> @as(u8, @intCast(shift)));
}

fn signExtendU256(byte_index: u256, value: u256) u256 {
    if (byte_index >= 32) return value;
    const bit_index: u16 = (@as(u16, @intCast(byte_index)) * 8) + 7;
    const mask = if (bit_index == 255) std.math.maxInt(u256) else (@as(u256, 1) << @as(u8, @intCast(bit_index + 1))) - 1;
    const sign_bit = @as(u256, 1) << @as(u8, @intCast(bit_index));
    return if ((value & sign_bit) != 0) value | ~mask else value & mask;
}

fn expU256(base: u256, exponent: u256) u256 {
    var result: u256 = 1;
    var factor = base;
    var remaining = exponent;
    while (remaining != 0) {
        if ((remaining & 1) != 0) result *%= factor;
        remaining >>= 1;
        if (remaining != 0) factor *%= factor;
    }
    return result;
}

fn addmodU256(a: u256, b: u256, n: u256) u256 {
    if (n == 0) return 0;
    const wide = (@as(u512, a) + @as(u512, b)) % @as(u512, n);
    return @intCast(wide);
}

fn mulmodU256(a: u256, b: u256, n: u256) u256 {
    if (n == 0) return 0;
    const wide = (@as(u512, a) * @as(u512, b)) % @as(u512, n);
    return @intCast(wide);
}

fn singleOwnedStringSlice(arena: std.mem.Allocator, value: []const u8) ![]const []const u8 {
    const out = try arena.alloc([]const u8, 1);
    out[0] = value;
    return out;
}

fn rewriteProgram(allocator: std.mem.Allocator, source: ir.Program, comptime rewriteBlock: RewriteBlockFn) OptimizeError!ir.Program {
    var result = ir.Program.init(allocator);
    errdefer result.deinit();
    const arena = result.allocator();

    const functions = try arena.alloc(ir.Function, source.functions.len);
    for (source.functions, functions) |function, *out_function| {
        const blocks = try arena.alloc(ir.Block, function.blocks.len);
        for (function.blocks, blocks) |block, *out_block| {
            out_block.* = try rewriteBlock(arena, block);
        }
        out_function.* = .{
            .name = try arena.dupe(u8, function.name),
            .blocks = blocks,
            .line = function.line,
        };
    }

    result.functions = functions;
    result.data_segments = try cloneDataSegments(arena, source.data_segments);
    return result;
}

fn copyPropBlock(arena: std.mem.Allocator, block: ir.Block) OptimizeError!ir.Block {
    var copy_map = std.StringHashMap([]const u8).init(arena);

    for (block.instructions) |instruction| {
        if (!isCopyInstruction(instruction)) continue;
        const source = copy_map.get(instruction.operands[0]) orelse instruction.operands[0];
        try copy_map.put(instruction.results[0], source);
    }

    const instructions = try arena.alloc(ir.Instruction, block.instructions.len);
    for (block.instructions, instructions) |instruction, *out_instruction| {
        out_instruction.* = try rewriteInstructionOperands(arena, instruction, &copy_map);
    }

    const outputs = try arena.alloc([]const u8, block.outputs.len);
    for (block.outputs, outputs) |output, *out_output| {
        out_output.* = try arena.dupe(u8, copy_map.get(output) orelse output);
    }

    return .{
        .name = try arena.dupe(u8, block.name),
        .inputs = try cloneStringSlice(arena, block.inputs),
        .outputs = outputs,
        .instructions = instructions,
        .terminator = try rewriteTerminator(arena, block.terminator, &copy_map),
        .line = block.line,
    };
}

fn literalCommoningBlock(arena: std.mem.Allocator, block: ir.Block) OptimizeError!ir.Block {
    var canonical_by_value = std.AutoHashMap(u256, []const u8).init(arena);
    var replacements = std.StringHashMap([]const u8).init(arena);

    for (block.instructions) |instruction| {
        const value = constInstructionValue(instruction) orelse continue;
        const result = instruction.results[0];
        if (canonical_by_value.get(value)) |canonical| {
            try replacements.put(result, canonical);
        } else {
            try canonical_by_value.put(value, result);
        }
    }

    const instructions = try arena.alloc(ir.Instruction, block.instructions.len);
    for (block.instructions, instructions) |instruction, *out_instruction| {
        if (instruction.results.len == 1 and replacements.contains(instruction.results[0])) {
            out_instruction.* = .{
                .results = &.{},
                .mnemonic = try arena.dupe(u8, "noop"),
                .operands = &.{},
                .line = instruction.line,
                .synthetic = instruction.synthetic,
            };
        } else {
            out_instruction.* = try rewriteInstructionOperands(arena, instruction, &replacements);
        }
    }

    const outputs = try arena.alloc([]const u8, block.outputs.len);
    for (block.outputs, outputs) |output, *out_output| {
        out_output.* = try arena.dupe(u8, replacements.get(output) orelse output);
    }

    return .{
        .name = try arena.dupe(u8, block.name),
        .inputs = try cloneStringSlice(arena, block.inputs),
        .outputs = outputs,
        .instructions = instructions,
        .terminator = try rewriteTerminator(arena, block.terminator, &replacements),
        .line = block.line,
    };
}

fn switchPeepholeBlock(arena: std.mem.Allocator, block: ir.Block) OptimizeError!ir.Block {
    var out = try cloneBlock(arena, block);
    const switch_term = switch (block.terminator) {
        .switch_ => |switch_term| switch_term,
        else => return out,
    };
    if (switch_term.cases.len != 1) return out;
    if (!isZeroLiteral(switch_term.cases[0].value)) return out;
    out.terminator = .{ .branch = .{
        .condition = try arena.dupe(u8, switch_term.selector),
        .non_zero_target = try arena.dupe(u8, switch_term.default_target),
        .zero_target = try arena.dupe(u8, switch_term.cases[0].target),
    } };
    return out;
}

fn constInstructionValue(instruction: ir.Instruction) ?u256 {
    if (instruction.results.len != 1 or instruction.operands.len != 1) return null;
    if (!std.mem.eql(u8, instruction.mnemonic, "const") and !std.mem.eql(u8, instruction.mnemonic, "large_const")) return null;
    return parseU256Literal(instruction.operands[0]);
}

fn rewriteInstructionOperands(arena: std.mem.Allocator, instruction: ir.Instruction, copy_map: *const std.StringHashMap([]const u8)) OptimizeError!ir.Instruction {
    const operands = try arena.alloc([]const u8, instruction.operands.len);
    const spec = ops.lookup(instruction.mnemonic);
    for (instruction.operands, operands, 0..) |operand, *out_operand, operand_index| {
        const rewritten = if (operandRequiresValue(spec, operand_index, operand))
            (copy_map.get(operand) orelse operand)
        else
            operand;
        out_operand.* = try arena.dupe(u8, rewritten);
    }
    return .{
        .results = try cloneStringSlice(arena, instruction.results),
        .mnemonic = try arena.dupe(u8, instruction.mnemonic),
        .operands = operands,
        .line = instruction.line,
        .synthetic = instruction.synthetic,
    };
}

fn rewriteTerminator(arena: std.mem.Allocator, terminator: ir.Terminator, copy_map: *const std.StringHashMap([]const u8)) OptimizeError!ir.Terminator {
    return switch (terminator) {
        .jump => |target| .{ .jump = try arena.dupe(u8, target) },
        .branch => |branch| .{ .branch = .{
            .condition = try arena.dupe(u8, copy_map.get(branch.condition) orelse branch.condition),
            .non_zero_target = try arena.dupe(u8, branch.non_zero_target),
            .zero_target = try arena.dupe(u8, branch.zero_target),
        } },
        .switch_ => |switch_term| blk: {
            const cases = try arena.alloc(ir.SwitchCase, switch_term.cases.len);
            for (switch_term.cases, cases) |case, *out_case| {
                out_case.* = .{
                    .value = try arena.dupe(u8, case.value),
                    .target = try arena.dupe(u8, case.target),
                    .line = case.line,
                };
            }
            break :blk .{ .switch_ = .{
                .selector = try arena.dupe(u8, copy_map.get(switch_term.selector) orelse switch_term.selector),
                .cases = cases,
                .default_target = try arena.dupe(u8, switch_term.default_target),
            } };
        },
        .return_ => |ret| .{ .return_ = .{
            .ptr = try arena.dupe(u8, copy_map.get(ret.ptr) orelse ret.ptr),
            .len = try arena.dupe(u8, copy_map.get(ret.len) orelse ret.len),
        } },
        .revert => |revert| .{ .revert = .{
            .ptr = try arena.dupe(u8, copy_map.get(revert.ptr) orelse revert.ptr),
            .len = try arena.dupe(u8, copy_map.get(revert.len) orelse revert.len),
        } },
        .selfdestruct => |beneficiary| .{ .selfdestruct = try arena.dupe(u8, copy_map.get(beneficiary) orelse beneficiary) },
        .stop => .stop,
        .invalid => .invalid,
        .iret => .iret,
    };
}

const UseSet = struct {
    map: std.StringHashMap(void),

    fn init(allocator: std.mem.Allocator) UseSet {
        return .{ .map = std.StringHashMap(void).init(allocator) };
    }

    fn deinit(self: *UseSet) void {
        self.map.deinit();
    }

    fn add(self: *UseSet, value: []const u8) !void {
        try self.map.put(value, {});
    }

    fn contains(self: *const UseSet, value: []const u8) bool {
        return self.map.contains(value);
    }
};

fn collectUses(allocator: std.mem.Allocator, program: ir.Program) !UseSet {
    var uses = UseSet.init(allocator);
    errdefer uses.deinit();
    for (program.functions) |function| {
        for (function.blocks) |block| {
            for (block.instructions) |instruction| {
                const spec = ops.lookup(instruction.mnemonic);
                for (instruction.operands, 0..) |operand, operand_index| {
                    if (operandRequiresValue(spec, operand_index, operand)) try uses.add(operand);
                }
            }
            for (block.outputs) |output| try uses.add(output);
            switch (block.terminator) {
                .branch => |branch| try uses.add(branch.condition),
                .switch_ => |switch_term| try uses.add(switch_term.selector),
                .return_ => |ret| {
                    try uses.add(ret.ptr);
                    try uses.add(ret.len);
                },
                .revert => |revert| {
                    try uses.add(revert.ptr);
                    try uses.add(revert.len);
                },
                .selfdestruct => |beneficiary| try uses.add(beneficiary),
                .jump, .stop, .invalid, .iret => {},
            }
        }
    }
    return uses;
}

fn isRemovableWhenUnused(instruction: ir.Instruction, uses: *const UseSet) bool {
    if (std.mem.eql(u8, instruction.mnemonic, "noop")) return false;
    if (instruction.results.len == 0) return false;
    for (instruction.results) |result| {
        if (uses.contains(result)) return false;
    }
    const effect = effects.ofInstruction(instruction) orelse effects.Effect.FULL_BARRIER;
    // Dead values are removable only when the operation is also side-effect
    // free. Unknown effects become FULL_BARRIER, so unsupported op shapes stay
    // in the program rather than being silently deleted.
    const forbidden = effects.Effect.unionAll(&.{
        effects.Effect.MEMORY_WRITE,
        effects.Effect.RETURNDATA_WRITE,
        effects.Effect.ACCOUNTS_WRITE,
        effects.Effect.PERSISTENT_WRITE,
        effects.Effect.TRANSIENT_WRITE,
        effects.Effect.REVERT,
        effects.Effect.TERMINATE,
        effects.Effect.ALLOC_ADVANCE,
        effects.Effect.LOGS,
    });
    return !effect.intersects(forbidden);
}

fn isCopyInstruction(instruction: ir.Instruction) bool {
    return std.mem.eql(u8, instruction.mnemonic, "copy") and instruction.results.len == 1 and instruction.operands.len == 1;
}

fn isZeroLiteral(value: []const u8) bool {
    const parsed = std.fmt.parseUnsigned(u256, value, 0) catch return false;
    return parsed == 0;
}

fn cloneProgram(allocator: std.mem.Allocator, source: ir.Program) !ir.Program {
    var result = ir.Program.init(allocator);
    errdefer result.deinit();
    const arena = result.allocator();
    const functions = try arena.alloc(ir.Function, source.functions.len);
    for (source.functions, functions) |function, *out_function| {
        const blocks = try arena.alloc(ir.Block, function.blocks.len);
        for (function.blocks, blocks) |block, *out_block| out_block.* = try cloneBlock(arena, block);
        out_function.* = .{
            .name = try arena.dupe(u8, function.name),
            .blocks = blocks,
            .line = function.line,
        };
    }
    result.functions = functions;
    result.data_segments = try cloneDataSegments(arena, source.data_segments);
    return result;
}

fn cloneBlock(arena: std.mem.Allocator, block: ir.Block) !ir.Block {
    return .{
        .name = try arena.dupe(u8, block.name),
        .inputs = try cloneStringSlice(arena, block.inputs),
        .outputs = try cloneStringSlice(arena, block.outputs),
        .instructions = try cloneInstructions(arena, block.instructions),
        .terminator = try cloneTerminator(arena, block.terminator),
        .line = block.line,
    };
}

fn cloneInstructions(arena: std.mem.Allocator, instructions: []const ir.Instruction) ![]const ir.Instruction {
    const out = try arena.alloc(ir.Instruction, instructions.len);
    for (instructions, out) |instruction, *out_instruction| out_instruction.* = try cloneInstruction(arena, instruction);
    return out;
}

fn cloneInstruction(arena: std.mem.Allocator, instruction: ir.Instruction) !ir.Instruction {
    return .{
        .results = try cloneStringSlice(arena, instruction.results),
        .mnemonic = try arena.dupe(u8, instruction.mnemonic),
        .operands = try cloneStringSlice(arena, instruction.operands),
        .line = instruction.line,
        .synthetic = instruction.synthetic,
    };
}

fn cloneTerminator(arena: std.mem.Allocator, terminator: ir.Terminator) !ir.Terminator {
    var empty = std.StringHashMap([]const u8).init(arena);
    return rewriteTerminator(arena, terminator, &empty);
}

fn cloneDataSegments(arena: std.mem.Allocator, data_segments: []const ir.DataSegment) ![]const ir.DataSegment {
    const out = try arena.alloc(ir.DataSegment, data_segments.len);
    for (data_segments, out) |segment, *out_segment| {
        out_segment.* = .{
            .name = try arena.dupe(u8, segment.name),
            .bytes = try arena.dupe(u8, segment.bytes),
            .line = segment.line,
        };
    }
    return out;
}

fn cloneReferencedDataSegments(arena: std.mem.Allocator, data_segments: []const ir.DataSegment, refs: *const std.StringHashMap(void)) ![]const ir.DataSegment {
    var out: std.ArrayList(ir.DataSegment) = .empty;
    defer out.deinit(arena);
    for (data_segments) |segment| {
        if (!refs.contains(segment.name)) continue;
        try out.append(arena, .{
            .name = try arena.dupe(u8, segment.name),
            .bytes = try arena.dupe(u8, segment.bytes),
            .line = segment.line,
        });
    }
    return out.toOwnedSlice(arena);
}

fn cloneStringSlice(arena: std.mem.Allocator, values: []const []const u8) ![]const []const u8 {
    const out = try arena.alloc([]const u8, values.len);
    for (values, out) |value, *out_value| out_value.* = try arena.dupe(u8, value);
    return out;
}

fn operandRequiresValue(spec: ?ops.Spec, index: usize, operand: []const u8) bool {
    if (operand.len == 0) return true;
    if (isNumericLiteral(operand)) return false;

    return switch (spec orelse return !isFunctionRef(operand) and !isDataRef(operand)) {
        .fixed => |fixed| !(fixed.extra != .none and index == fixed.inputs),
        .memory_load, .memory_store => true,
        .internal_call => index != 0,
    };
}

fn isNumericLiteral(text: []const u8) bool {
    const unsigned = if (text.len > 1 and text[0] == '-') text[1..] else text;
    if (unsigned.len == 0) return false;
    if (unsigned.len > 2 and unsigned[0] == '0' and (unsigned[1] == 'x' or unsigned[1] == 'X')) {
        for (unsigned[2..]) |ch| {
            if (!std.ascii.isHex(ch)) return false;
        }
        return true;
    }
    for (unsigned) |ch| {
        if (!std.ascii.isDigit(ch)) return false;
    }
    return true;
}

fn isFunctionRef(text: []const u8) bool {
    return text.len > 1 and text[0] == '@';
}

fn isDataRef(text: []const u8) bool {
    return text.len > 1 and text[0] == '.';
}

fn dataRefName(text: []const u8) []const u8 {
    return if (isDataRef(text)) text[1..] else text;
}

fn functionNameFromRef(text: []const u8) ?[]const u8 {
    if (!isFunctionRef(text)) return null;
    return text[1..];
}

fn findFunctionByName(program: ir.Program, name: []const u8) ?ir.Function {
    for (program.functions) |function| {
        if (std.mem.eql(u8, function.name, name)) return function;
    }
    return null;
}

fn findBlockByName(function: ir.Function, name: []const u8) ?ir.Block {
    for (function.blocks) |block| {
        if (std.mem.eql(u8, block.name, name)) return block;
    }
    return null;
}

test "copy propagation rewrites uses but keeps copy ops" {
    const diagnostics = @import("diagnostics.zig");
    const parser = @import("parser.zig");

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator,
        \\fn main:
        \\    entry b -> a_out {
        \\        a = copy b
        \\        a_out = copy a
        \\        => @next
        \\    }
        \\    next a_in {
        \\        c = add a_in a_in
        \\        stop
        \\    }
    , &bag);
    defer program.deinit();

    var optimized = try copyPropagation(std.testing.allocator, program);
    defer optimized.deinit();
    try std.testing.expectEqualStrings("b", optimized.functions[0].blocks[0].outputs[0]);
    try std.testing.expectEqualStrings("b", optimized.functions[0].blocks[0].instructions[1].operands[0]);
    try std.testing.expectEqualStrings("a_in", optimized.functions[0].blocks[1].instructions[0].operands[0]);
}

test "literal commoning shares same-block inline numeric constants" {
    const diagnostics = @import("diagnostics.zig");
    const parser = @import("parser.zig");

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator,
        \\fn main:
        \\    entry {
        \\        z = add 3 3
        \\        stop
        \\    }
    , &bag);
    defer program.deinit();

    var optimized = try literalCommoning(std.testing.allocator, program);
    defer optimized.deinit();

    const instructions = optimized.functions[0].blocks[0].instructions;
    try std.testing.expectEqualStrings("const", instructions[0].mnemonic);
    try std.testing.expect(instructions[0].synthetic);
    try std.testing.expectEqualStrings("noop", instructions[1].mnemonic);
    try std.testing.expect(instructions[1].synthetic);
    try std.testing.expectEqualStrings("add", instructions[2].mnemonic);
    try std.testing.expectEqualStrings(instructions[0].results[0], instructions[2].operands[0]);
    try std.testing.expectEqualStrings(instructions[0].results[0], instructions[2].operands[1]);
}

test "literal commoning shares same-block selector and mask literals" {
    const diagnostics = @import("diagnostics.zig");
    const parser = @import("parser.zig");

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator,
        \\fn main:
        \\    entry x y {
        \\        selector_a = large_const 0x1234567800000000000000000000000000000000000000000000000000000000
        \\        selector_b = large_const 0x1234567800000000000000000000000000000000000000000000000000000000
        \\        masked_a = large_const 0xffffffffffffffffffffffffffffffffffffffff
        \\        masked_b = large_const 0xffffffffffffffffffffffffffffffffffffffff
        \\        left = xor x selector_a
        \\        right = xor y selector_b
        \\        out = and masked_b masked_a
        \\        stop
        \\    }
    , &bag);
    defer program.deinit();

    var optimized = try literalCommoning(std.testing.allocator, program);
    defer optimized.deinit();

    const instructions = optimized.functions[0].blocks[0].instructions;
    try std.testing.expectEqualStrings("noop", instructions[1].mnemonic);
    try std.testing.expectEqualStrings("noop", instructions[3].mnemonic);
    try std.testing.expectEqualStrings(instructions[0].results[0], instructions[5].operands[1]);
    try std.testing.expectEqualStrings(instructions[2].results[0], instructions[6].operands[0]);
}

test "SCCP folds constants propagates block outputs and simplifies branches" {
    const diagnostics = @import("diagnostics.zig");
    const parser = @import("parser.zig");

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator,
        \\fn init:
        \\    entry -> sum_out {
        \\        one = const 1
        \\        two = const 2
        \\        sum_out = add one two
        \\        cond = eq sum_out two
        \\        => cond ? @bad : @good
        \\    }
        \\    good sum_in {
        \\        doubled = add sum_in sum_in
        \\        stop
        \\    }
        \\    bad sum_in {
        \\        stop
        \\    }
    , &bag);
    defer program.deinit();

    var optimized = try sccp(std.testing.allocator, program);
    defer optimized.deinit();

    try std.testing.expectEqualStrings("const", optimized.functions[0].blocks[0].instructions[2].mnemonic);
    try std.testing.expectEqualStrings("3", optimized.functions[0].blocks[0].instructions[2].operands[0]);
    try std.testing.expectEqualStrings("const", optimized.functions[0].blocks[1].instructions[0].mnemonic);
    try std.testing.expectEqualStrings("6", optimized.functions[0].blocks[1].instructions[0].operands[0]);
    try std.testing.expectEqualStrings("good", optimized.functions[0].blocks[0].terminator.jump);
}

test "SCCP simplifies constant switch targets" {
    const diagnostics = @import("diagnostics.zig");
    const parser = @import("parser.zig");

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator,
        \\fn init:
        \\    entry {
        \\        selector = const 2
        \\        switch selector {
        \\        1 => @one
        \\        2 => @two
        \\        default => @fallback
        \\    }
        \\    }
        \\    one {
        \\        stop
        \\    }
        \\    two {
        \\        stop
        \\    }
        \\    fallback {
        \\        stop
        \\    }
    , &bag);
    defer program.deinit();

    var optimized = try sccp(std.testing.allocator, program);
    defer optimized.deinit();
    try std.testing.expectEqualStrings("two", optimized.functions[0].blocks[0].terminator.jump);
}

test "unused operation elimination preserves side effects and removes dead chains" {
    const diagnostics = @import("diagnostics.zig");
    const parser = @import("parser.zig");

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator,
        \\fn main:
        \\    entry {
        \\        x = const 1
        \\        y = add x x
        \\        key = const 0
        \\        sstore key y
        \\        dead = add x x
        \\        stop
        \\    }
    , &bag);
    defer program.deinit();

    var optimized = try unusedOperationElimination(std.testing.allocator, program);
    defer optimized.deinit();
    try std.testing.expectEqualStrings("const", optimized.functions[0].blocks[0].instructions[0].mnemonic);
    try std.testing.expectEqualStrings("add", optimized.functions[0].blocks[0].instructions[1].mnemonic);
    try std.testing.expectEqualStrings("sstore", optimized.functions[0].blocks[0].instructions[3].mnemonic);
    try std.testing.expectEqualStrings("noop", optimized.functions[0].blocks[0].instructions[4].mnemonic);
}

test "switch peephole lowers one zero case with default into branch" {
    const diagnostics = @import("diagnostics.zig");
    const parser = @import("parser.zig");

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator,
        \\fn main:
        \\    entry {
        \\        sel = const 0
        \\        switch sel {
        \\        0 => @zero
        \\        default => @other
        \\    }
        \\    }
        \\    zero {
        \\        stop
        \\    }
        \\    other {
        \\        stop
        \\    }
    , &bag);
    defer program.deinit();

    var optimized = try switchPeephole(std.testing.allocator, program);
    defer optimized.deinit();
    const branch = optimized.functions[0].blocks[0].terminator.branch;
    try std.testing.expectEqualStrings("sel", branch.condition);
    try std.testing.expectEqualStrings("other", branch.non_zero_target);
    try std.testing.expectEqualStrings("zero", branch.zero_target);
}

test "defragment removes noops unreachable blocks functions and data" {
    const diagnostics = @import("diagnostics.zig");
    const parser = @import("parser.zig");

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator,
        \\fn init:
        \\    entry {
        \\        value = const 1
        \\        => @live
        \\    }
        \\    live {
        \\        noop
        \\        offset = data_offset .live
        \\        result = icall @helper value
        \\        sstore value result
        \\        stop
        \\    }
        \\    dead_block {
        \\        unused = data_offset .dead
        \\        stop
        \\    }
        \\
        \\fn helper:
        \\    entry input -> output {
        \\        output = add input input
        \\        iret
        \\    }
        \\
        \\fn dead_helper:
        \\    entry {
        \\        unused = data_offset .dead
        \\        stop
        \\    }
        \\
        \\data .live 0x01
        \\data .dead 0x02
    , &bag);
    defer program.deinit();

    var optimized = try defragment(std.testing.allocator, program);
    defer optimized.deinit();

    try std.testing.expectEqual(@as(usize, 2), optimized.functions.len);
    try std.testing.expectEqualStrings("init", optimized.functions[0].name);
    try std.testing.expectEqualStrings("helper", optimized.functions[1].name);
    try std.testing.expectEqual(@as(usize, 2), optimized.functions[0].blocks.len);
    try std.testing.expectEqual(@as(usize, 3), optimized.functions[0].blocks[1].instructions.len);
    try std.testing.expectEqualStrings("data_offset", optimized.functions[0].blocks[1].instructions[0].mnemonic);
    try std.testing.expectEqual(@as(usize, 1), optimized.data_segments.len);
    try std.testing.expectEqualStrings("live", optimized.data_segments[0].name);
}
