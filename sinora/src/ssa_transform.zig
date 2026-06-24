//! Plank-style sealed-block SSA construction for Sinora text IR.
//!
//! Ora already emits block-parameter SSA today, so production can choose not to
//! run this pass. The port still lives here because the backend should own the
//! same transform surface as Plank before we decide which passes Ora enables.
//! The algorithm is the Braun-style "sealed blocks" transform used by Plank:
//! reads of a variable either resolve to the current block definition, forward
//! through a single predecessor, or create a block input plus predecessor
//! outputs as the phi-like transfer.

const std = @import("std");

const analyses = @import("analyses.zig");
const ir = @import("ir.zig");
const release_critical_edges = @import("release_critical_edges.zig");

pub const TransformError = error{
    InvalidPreSsa,
    MissingBlock,
} || std.mem.Allocator.Error;

const PhiParam = union(enum) {
    missing: []const u8,
    output: []const u8,
};

pub fn transform(allocator: std.mem.Allocator, source: ir.Program) TransformError!ir.Program {
    var split = try release_critical_edges.split(allocator, source);
    errdefer split.deinit();

    var regularized = try regularizeFunctionEntries(allocator, split);
    split.deinit();
    errdefer regularized.deinit();

    const result = try constructSsa(allocator, regularized);
    regularized.deinit();
    return result;
}

fn regularizeFunctionEntries(allocator: std.mem.Allocator, source: ir.Program) TransformError!ir.Program {
    var result = ir.Program.init(allocator);
    errdefer result.deinit();
    const arena = result.allocator();

    const functions = try arena.alloc(ir.Function, source.functions.len);
    for (source.functions, functions) |function, *out_function| {
        const entry_has_pred = try validatePreSsaFunction(allocator, function);
        const extra: usize = @intFromBool(entry_has_pred);
        const blocks = try arena.alloc(ir.Block, function.blocks.len + extra);
        var out_index: usize = 0;

        if (entry_has_pred) {
            const new_entry_name = try uniqueBlockName(arena, function, "__pre_ssa_entry");
            blocks[out_index] = .{
                .name = new_entry_name,
                .inputs = try cloneStringSlice(arena, function.blocks[0].inputs),
                .outputs = &.{},
                .instructions = &.{},
                .terminator = .{ .jump = function.blocks[0].name },
                .line = function.blocks[0].line,
            };
            out_index += 1;
        }

        for (function.blocks, 0..) |block, block_index| {
            blocks[out_index] = try cloneBlock(arena, block);
            if (entry_has_pred and block_index == 0) {
                blocks[out_index].inputs = &.{};
            }
            out_index += 1;
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

fn validatePreSsaFunction(allocator: std.mem.Allocator, function: ir.Function) TransformError!bool {
    if (function.blocks.len == 0) return false;
    var entry_has_pred = false;
    var worklist: std.ArrayList(usize) = .empty;
    defer worklist.deinit(allocator);
    const seen = try allocator.alloc(bool, function.blocks.len);
    defer allocator.free(seen);
    @memset(seen, false);
    try worklist.append(allocator, 0);
    seen[0] = true;

    while (worklist.pop()) |block_index| {
        const block = function.blocks[block_index];
        if (block_index != 0 and block.inputs.len != 0) return error.InvalidPreSsa;
        if (block.terminator != .iret and block.outputs.len != 0) return error.InvalidPreSsa;

        var successors = ir.successors(&block);
        while (successors.next()) |successor_name| {
            const successor_index = blockIndex(function, successor_name) orelse return error.MissingBlock;
            entry_has_pred = entry_has_pred or successor_index == 0;
            if (!seen[successor_index]) {
                seen[successor_index] = true;
                try worklist.append(allocator, successor_index);
            }
        }
    }
    return entry_has_pred;
}

const Transformer = struct {
    allocator: std.mem.Allocator,
    source: ir.Program,
    output_arena: std.mem.Allocator,
    predecessors: *const analyses.Predecessors,
    reachability: *const analyses.Reachability,
    unfilled_until_sealed: []u32,
    filled: []bool,
    inputs: []std.ArrayList([]const u8),
    base_inputs: []std.ArrayList([]const u8),
    outputs: []std.ArrayList(PhiParam),
    base_outputs: []std.ArrayList([]const u8),
    instructions: []std.ArrayList(ir.Instruction),
    terminators: []?ir.Terminator,
    defs: []std.StringHashMap([]const u8),
    shape: Shape,
    next_name_id: usize = 0,

    fn init(
        allocator: std.mem.Allocator,
        output_arena: std.mem.Allocator,
        source: ir.Program,
        store: *analyses.AnalysesStore,
    ) !Transformer {
        var shape = try Shape.build(allocator, source);
        errdefer shape.deinit(allocator);
        const predecessors = try store.predecessors(source);
        const reachability = try store.reachability(source);

        const unfilled = try allocator.alloc(u32, shape.block_count);
        errdefer allocator.free(unfilled);
        const filled = try allocator.alloc(bool, shape.block_count);
        errdefer allocator.free(filled);
        @memset(filled, false);

        const inputs = try allocator.alloc(std.ArrayList([]const u8), shape.block_count);
        errdefer allocator.free(inputs);
        const base_inputs = try allocator.alloc(std.ArrayList([]const u8), shape.block_count);
        errdefer allocator.free(base_inputs);
        const outputs = try allocator.alloc(std.ArrayList(PhiParam), shape.block_count);
        errdefer allocator.free(outputs);
        const base_outputs = try allocator.alloc(std.ArrayList([]const u8), shape.block_count);
        errdefer allocator.free(base_outputs);
        const instructions = try allocator.alloc(std.ArrayList(ir.Instruction), shape.block_count);
        errdefer allocator.free(instructions);
        const terminators = try allocator.alloc(?ir.Terminator, shape.block_count);
        errdefer allocator.free(terminators);
        const defs = try allocator.alloc(std.StringHashMap([]const u8), shape.block_count);
        errdefer allocator.free(defs);

        for (0..shape.block_count) |index| {
            inputs[index] = .empty;
            base_inputs[index] = .empty;
            outputs[index] = .empty;
            base_outputs[index] = .empty;
            instructions[index] = .empty;
            terminators[index] = null;
            defs[index] = std.StringHashMap([]const u8).init(allocator);
        }

        for (source.functions, 0..) |function, function_index| {
            for (function.blocks, 0..) |_, block_index| {
                const ref = analyses.BlockRef{ .function = function_index, .block = block_index };
                unfilled[shape.global(ref)] = @intCast(predecessors.of(ref).len);
            }
        }

        return .{
            .allocator = allocator,
            .source = source,
            .output_arena = output_arena,
            .predecessors = predecessors,
            .reachability = reachability,
            .unfilled_until_sealed = unfilled,
            .filled = filled,
            .inputs = inputs,
            .base_inputs = base_inputs,
            .outputs = outputs,
            .base_outputs = base_outputs,
            .instructions = instructions,
            .terminators = terminators,
            .defs = defs,
            .shape = shape,
        };
    }

    fn deinit(self: *Transformer) void {
        for (0..self.shape.block_count) |index| {
            self.inputs[index].deinit(self.allocator);
            self.base_inputs[index].deinit(self.allocator);
            self.outputs[index].deinit(self.allocator);
            self.base_outputs[index].deinit(self.allocator);
            self.instructions[index].deinit(self.allocator);
            self.defs[index].deinit();
        }
        self.allocator.free(self.inputs);
        self.allocator.free(self.base_inputs);
        self.allocator.free(self.outputs);
        self.allocator.free(self.base_outputs);
        self.allocator.free(self.instructions);
        self.allocator.free(self.terminators);
        self.allocator.free(self.defs);
        self.allocator.free(self.unfilled_until_sealed);
        self.allocator.free(self.filled);
        self.shape.deinit(self.allocator);
        self.* = undefined;
    }

    fn sealed(self: Transformer, ref: analyses.BlockRef) bool {
        return self.unfilled_until_sealed[self.shape.global(ref)] == 0;
    }

    fn isFilled(self: Transformer, ref: analyses.BlockRef) bool {
        return self.filled[self.shape.global(ref)];
    }

    fn fresh(self: *Transformer) ![]const u8 {
        const name = try std.fmt.allocPrint(self.output_arena, "__ssa{d}", .{self.next_name_id});
        self.next_name_id += 1;
        return name;
    }

    fn defineFresh(self: *Transformer, ref: analyses.BlockRef, original: []const u8) ![]const u8 {
        const renamed = try self.fresh();
        try self.defs[self.shape.global(ref)].put(original, renamed);
        return renamed;
    }

    fn readVariable(self: *Transformer, ref: analyses.BlockRef, local: []const u8) ![]const u8 {
        const global = self.shape.global(ref);
        if (self.defs[global].get(local)) |renamed| return renamed;

        if (!self.sealed(ref)) {
            const renamed = try self.defineFresh(ref, local);
            try self.inputs[global].append(self.allocator, renamed);
            for (self.predecessors.of(ref)) |pred| {
                const pred_global = self.shape.global(pred);
                const phi_param: PhiParam = if (self.isFilled(pred))
                    .{ .output = try self.readVariable(pred, local) }
                else
                    .{ .missing = local };
                try self.outputs[pred_global].append(self.allocator, phi_param);
            }
            return renamed;
        }

        const preds = self.predecessors.of(ref);
        if (preds.len == 1) {
            const renamed = try self.readVariable(preds[0], local);
            try self.defs[global].put(local, renamed);
            return renamed;
        }

        const renamed = try self.defineFresh(ref, local);
        try self.inputs[global].append(self.allocator, renamed);
        for (preds) |pred| {
            try self.outputs[self.shape.global(pred)].append(self.allocator, .{
                .output = try self.readVariable(pred, local),
            });
        }
        return renamed;
    }
};

fn constructSsa(allocator: std.mem.Allocator, source: ir.Program) TransformError!ir.Program {
    var result = ir.Program.init(allocator);
    errdefer result.deinit();
    const arena = result.allocator();

    var store = analyses.AnalysesStore.init(allocator);
    defer store.deinit();
    var transformer = try Transformer.init(allocator, arena, source, &store);
    defer transformer.deinit();
    const rpo = try store.reversePostOrder(source);

    for (rpo.global()) |block_ref| {
        const function = source.functions[block_ref.function];
        const block = function.blocks[block_ref.block];
        const global = transformer.shape.global(block_ref);

        for (block.inputs) |input| {
            const renamed = try transformer.defineFresh(block_ref, input);
            try transformer.base_inputs[global].append(allocator, renamed);
        }

        for (block.instructions) |instruction| {
            try transformer.instructions[global].append(allocator, try transformInstruction(arena, &transformer, block_ref, instruction));
        }

        for (block.outputs) |output| {
            try transformer.base_outputs[global].append(allocator, try transformer.readVariable(block_ref, output));
        }

        transformer.terminators[global] = try transformTerminator(arena, &transformer, block_ref, block.terminator);

        transformer.filled[global] = true;
        var successors = ir.successors(&block);
        while (successors.next()) |successor_name| {
            const successor_index = blockIndex(function, successor_name) orelse return error.MissingBlock;
            const successor = analyses.BlockRef{ .function = block_ref.function, .block = successor_index };
            const slot = transformer.shape.global(successor);
            if (transformer.unfilled_until_sealed[slot] > 0) transformer.unfilled_until_sealed[slot] -= 1;
        }

        for (transformer.outputs[global].items) |*output| {
            switch (output.*) {
                .missing => |missing| output.* = .{ .output = try transformer.readVariable(block_ref, missing) },
                .output => {},
            }
        }
    }

    result.functions = try emitFunctions(arena, source, &transformer);
    result.data_segments = try cloneDataSegments(arena, source.data_segments);
    return result;
}

fn emitFunctions(arena: std.mem.Allocator, source: ir.Program, transformer: *Transformer) ![]const ir.Function {
    const functions = try arena.alloc(ir.Function, source.functions.len);
    for (source.functions, functions, 0..) |function, *out_function, function_index| {
        const blocks = try arena.alloc(ir.Block, function.blocks.len);
        for (function.blocks, blocks, 0..) |block, *out_block, block_index| {
            const ref = analyses.BlockRef{ .function = function_index, .block = block_index };
            const global = transformer.shape.global(ref);
            if (!transformer.reachability.contains(ref)) {
                out_block.* = try cloneBlock(arena, block);
                continue;
            }

            out_block.* = .{
                .name = try arena.dupe(u8, block.name),
                .inputs = try emitInputs(arena, transformer, global),
                .outputs = try emitOutputs(arena, transformer, global),
                .instructions = try arena.dupe(ir.Instruction, transformer.instructions[global].items),
                .terminator = transformer.terminators[global] orelse return error.InvalidPreSsa,
                .line = block.line,
            };
        }
        out_function.* = .{
            .name = try arena.dupe(u8, function.name),
            .blocks = blocks,
            .line = function.line,
        };
    }
    return functions;
}

fn emitInputs(arena: std.mem.Allocator, transformer: *Transformer, global: usize) ![]const []const u8 {
    const total = transformer.base_inputs[global].items.len + transformer.inputs[global].items.len;
    const inputs = try arena.alloc([]const u8, total);
    @memcpy(inputs[0..transformer.base_inputs[global].items.len], transformer.base_inputs[global].items);
    @memcpy(inputs[transformer.base_inputs[global].items.len..], transformer.inputs[global].items);
    return inputs;
}

fn emitOutputs(arena: std.mem.Allocator, transformer: *Transformer, global: usize) ![]const []const u8 {
    const total = transformer.base_outputs[global].items.len + transformer.outputs[global].items.len;
    const outputs = try arena.alloc([]const u8, total);
    @memcpy(outputs[0..transformer.base_outputs[global].items.len], transformer.base_outputs[global].items);
    for (transformer.outputs[global].items, outputs[transformer.base_outputs[global].items.len..]) |param, *out| {
        out.* = switch (param) {
            .output => |value| value,
            .missing => return error.InvalidPreSsa,
        };
    }
    return outputs;
}

fn transformInstruction(
    arena: std.mem.Allocator,
    transformer: *Transformer,
    ref: analyses.BlockRef,
    instruction: ir.Instruction,
) !ir.Instruction {
    const operands = try arena.alloc([]const u8, instruction.operands.len);
    const spec = @import("ops.zig").lookup(instruction.mnemonic);
    for (instruction.operands, operands, 0..) |operand, *out_operand, operand_index| {
        out_operand.* = if (operandRequiresValue(spec, operand_index, operand))
            try transformer.readVariable(ref, operand)
        else
            try arena.dupe(u8, operand);
    }

    const results = try arena.alloc([]const u8, instruction.results.len);
    for (instruction.results, results) |result_name, *out_result| {
        out_result.* = try transformer.defineFresh(ref, result_name);
    }
    return .{
        .results = results,
        .mnemonic = try arena.dupe(u8, instruction.mnemonic),
        .operands = operands,
        .line = instruction.line,
    };
}

fn transformTerminator(arena: std.mem.Allocator, transformer: *Transformer, ref: analyses.BlockRef, terminator: ir.Terminator) !ir.Terminator {
    return switch (terminator) {
        .jump => |target| .{ .jump = try arena.dupe(u8, target) },
        .branch => |branch| .{ .branch = .{
            .condition = try transformer.readVariable(ref, branch.condition),
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
                .selector = try transformer.readVariable(ref, switch_term.selector),
                .cases = cases,
                .default_target = try arena.dupe(u8, switch_term.default_target),
            } };
        },
        .return_ => |ret| .{ .return_ = .{
            .ptr = try transformer.readVariable(ref, ret.ptr),
            .len = try transformer.readVariable(ref, ret.len),
        } },
        .revert => |revert| .{ .revert = .{
            .ptr = try transformer.readVariable(ref, revert.ptr),
            .len = try transformer.readVariable(ref, revert.len),
        } },
        .selfdestruct => |beneficiary| .{ .selfdestruct = try transformer.readVariable(ref, beneficiary) },
        .stop => .stop,
        .invalid => .invalid,
        .iret => .iret,
    };
}

const Shape = struct {
    starts: []usize,
    block_count: usize,

    fn build(allocator: std.mem.Allocator, program: ir.Program) !Shape {
        var count: usize = 0;
        const starts = try allocator.alloc(usize, program.functions.len);
        for (program.functions, 0..) |function, index| {
            starts[index] = count;
            count += function.blocks.len;
        }
        return .{ .starts = starts, .block_count = count };
    }

    fn deinit(self: *Shape, allocator: std.mem.Allocator) void {
        allocator.free(self.starts);
        self.* = undefined;
    }

    fn global(self: Shape, ref: analyses.BlockRef) usize {
        return self.starts[ref.function] + ref.block;
    }
};

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
    for (instructions, out) |instruction, *out_instruction| {
        out_instruction.* = .{
            .results = try cloneStringSlice(arena, instruction.results),
            .mnemonic = try arena.dupe(u8, instruction.mnemonic),
            .operands = try cloneStringSlice(arena, instruction.operands),
            .line = instruction.line,
        };
    }
    return out;
}

fn cloneTerminator(arena: std.mem.Allocator, terminator: ir.Terminator) !ir.Terminator {
    return switch (terminator) {
        .jump => |target| .{ .jump = try arena.dupe(u8, target) },
        .branch => |branch| .{ .branch = .{
            .condition = try arena.dupe(u8, branch.condition),
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
                .selector = try arena.dupe(u8, switch_term.selector),
                .cases = cases,
                .default_target = try arena.dupe(u8, switch_term.default_target),
            } };
        },
        .return_ => |ret| .{ .return_ = .{
            .ptr = try arena.dupe(u8, ret.ptr),
            .len = try arena.dupe(u8, ret.len),
        } },
        .revert => |revert| .{ .revert = .{
            .ptr = try arena.dupe(u8, revert.ptr),
            .len = try arena.dupe(u8, revert.len),
        } },
        .selfdestruct => |beneficiary| .{ .selfdestruct = try arena.dupe(u8, beneficiary) },
        .stop => .stop,
        .invalid => .invalid,
        .iret => .iret,
    };
}

fn cloneStringSlice(arena: std.mem.Allocator, values: []const []const u8) ![]const []const u8 {
    const out = try arena.alloc([]const u8, values.len);
    for (values, out) |value, *out_value| out_value.* = try arena.dupe(u8, value);
    return out;
}

fn uniqueBlockName(arena: std.mem.Allocator, function: ir.Function, prefix: []const u8) ![]const u8 {
    var index: usize = 0;
    while (true) : (index += 1) {
        const candidate = try std.fmt.allocPrint(arena, "{s}{d}", .{ prefix, index });
        if (blockIndex(function, candidate) == null) return candidate;
    }
}

fn blockIndex(function: ir.Function, block_name: []const u8) ?usize {
    for (function.blocks, 0..) |block, index| {
        if (std.mem.eql(u8, block.name, block_name)) return index;
    }
    return null;
}

fn operandRequiresValue(spec: ?@import("ops.zig").Spec, index: usize, operand: []const u8) bool {
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

test "SSA transform inserts block parameters at diamond joins" {
    const diagnostics = @import("diagnostics.zig");
    const parser = @import("parser.zig");

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator,
        \\fn main:
        \\    entry {
        \\        x = const 0
        \\        => x ? @left : @right
        \\    }
        \\    left {
        \\        x = const 1
        \\        => @join
        \\    }
        \\    right {
        \\        x = const 2
        \\        => @join
        \\    }
        \\    join {
        \\        y = copy x
        \\        stop
        \\    }
    , &bag);
    defer program.deinit();

    var ssa = try transform(std.testing.allocator, program);
    defer ssa.deinit();

    const join = ssa.functions[0].blocks[3];
    try std.testing.expectEqual(@as(usize, 1), join.inputs.len);
    try std.testing.expectEqual(@as(usize, 1), ssa.functions[0].blocks[1].outputs.len);
    try std.testing.expectEqual(@as(usize, 1), ssa.functions[0].blocks[2].outputs.len);
    try std.testing.expectEqualStrings(join.inputs[0], join.instructions[0].operands[0]);
}

test "SSA transform regularizes looped function entries" {
    const diagnostics = @import("diagnostics.zig");
    const parser = @import("parser.zig");

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator,
        \\fn main:
        \\    entry x {
        \\        one = const 1
        \\        x = add x one
        \\        => x ? @entry : @done
        \\    }
        \\    done {
        \\        stop
        \\    }
    , &bag);
    defer program.deinit();

    var ssa = try transform(std.testing.allocator, program);
    defer ssa.deinit();

    try std.testing.expect(std.mem.startsWith(u8, ssa.functions[0].blocks[0].name, "__pre_ssa_entry"));
    try std.testing.expectEqual(@as(usize, 1), ssa.functions[0].blocks[0].inputs.len);
    try std.testing.expectEqual(@as(usize, 1), ssa.functions[0].blocks[1].inputs.len);
}
