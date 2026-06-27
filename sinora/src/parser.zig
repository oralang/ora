//! Line-oriented parser for Plank/Sinora SIR text.
//!
//! The SIR text format is intentionally simple: functions contain named blocks,
//! blocks contain SSA instructions, and each block ends in exactly one
//! terminator. This parser keeps that model direct instead of building a token
//! tree. It owns all returned strings in the `ir.Program` arena, reports
//! recoverable syntax issues into `diagnostics.Bag`, and uses `<error>` /
//! `.invalid` placeholders only so later validation can report more than the
//! first malformed line.

const std = @import("std");

const diagnostics = @import("diagnostics.zig");
const ir = @import("ir.zig");
const ops = @import("ops.zig");

pub const ParseError = error{
    ParseFailed,
};

const FunctionBuilder = struct {
    name: []const u8,
    blocks: std.ArrayList(ir.Block),
    line: u32,
};

const BlockBuilder = struct {
    name: []const u8,
    inputs: []const []const u8,
    outputs: []const []const u8,
    instructions: std.ArrayList(ir.Instruction),
    terminator: ?ir.Terminator,
    line: u32,
};

pub fn parse(backing_allocator: std.mem.Allocator, source: []const u8, bag: *diagnostics.Bag) !ir.Program {
    var program = ir.Program.init(backing_allocator);
    errdefer program.deinit();

    // All SIR identifiers and slices returned by the parser are arena-owned by
    // the Program. That lets later passes cheaply replace the whole Program
    // after a transform without per-node frees.
    const arena = program.allocator();
    var functions: std.ArrayList(ir.Function) = .empty;
    var data_segments: std.ArrayList(ir.DataSegment) = .empty;
    var current_function: ?FunctionBuilder = null;
    var current_block: ?BlockBuilder = null;
    var saw_data_segment = false;

    var lines = std.mem.splitScalar(u8, source, '\n');
    var line_no: u32 = 0;
    while (lines.next()) |raw_line| {
        line_no += 1;
        const raw = trimRightCr(raw_line);
        const line = std.mem.trim(u8, raw, " \t");
        if (line.len == 0 or std.mem.startsWith(u8, line, "//")) continue;

        if (std.mem.startsWith(u8, line, "fn ")) {
            if (saw_data_segment) {
                try bag.errorAt(line_no, 1, "function definitions must appear before data segments", .{});
            }
            if (current_block) |*block| {
                try bag.errorAt(line_no, 1, "function starts before block '{s}' is closed", .{block.name});
                current_block = null;
            }
            if (current_function) |*function| {
                try finishFunction(arena, &functions, function);
                current_function = null;
            }
            current_function = try parseFunctionHeader(arena, line, line_no, bag);
            continue;
        }

        if (std.mem.startsWith(u8, line, "data ")) {
            saw_data_segment = true;
            if (current_block) |*block| {
                try bag.errorAt(line_no, 1, "data segment starts before block '{s}' is closed", .{block.name});
                if (block.terminator == null) block.terminator = .invalid;
                try finishBlock(arena, &current_function.?.blocks, block);
                current_block = null;
            }
            if (current_function) |*function| {
                try finishFunction(arena, &functions, function);
                current_function = null;
            }
            try data_segments.append(arena, try parseDataSegment(arena, line, line_no, bag));
            continue;
        }

        if (current_function == null) {
            try bag.errorAt(line_no, 1, "expected function header or data segment before SIR content", .{});
            continue;
        }

        if (std.mem.eql(u8, line, "}")) {
            if (current_block) |*block| {
                if (block.terminator == null) {
                    try bag.errorAt(line_no, 1, "block '{s}' has no terminator", .{block.name});
                    block.terminator = .invalid;
                }
                try finishBlock(arena, &current_function.?.blocks, block);
                current_block = null;
            } else {
                try bag.errorAt(line_no, 1, "unexpected closing brace", .{});
            }
            continue;
        }

        if (std.mem.endsWith(u8, line, "{") and !std.mem.startsWith(u8, line, "switch ")) {
            if (current_block) |*block| {
                try bag.errorAt(line_no, 1, "block '{s}' starts before previous block is closed", .{block.name});
                if (block.terminator == null) block.terminator = .invalid;
                try finishBlock(arena, &current_function.?.blocks, block);
                current_block = null;
            }
            current_block = try parseBlockHeader(arena, line, line_no, bag);
            continue;
        }

        if (current_block == null) {
            try bag.errorAt(line_no, 1, "expected block header before instruction", .{});
            continue;
        }

        if (current_block.?.terminator != null) {
            try bag.errorAt(line_no, 1, "instruction appears after block terminator", .{});
            continue;
        }

        if (std.mem.startsWith(u8, line, "switch ")) {
            current_block.?.terminator = try parseSwitch(arena, line, &lines, &line_no, bag);
        } else if (isTerminatorLine(line)) {
            current_block.?.terminator = try parseTerminator(arena, line, line_no, bag);
        } else {
            try current_block.?.instructions.append(arena, try parseInstruction(arena, line, line_no, bag));
        }
    }

    if (current_block) |*block| {
        try bag.errorAt(block.line, 1, "block '{s}' was not closed", .{block.name});
        if (block.terminator == null) block.terminator = .invalid;
        try finishBlock(arena, &current_function.?.blocks, block);
    }
    if (current_function) |*function| {
        try finishFunction(arena, &functions, function);
    }

    program.functions = try functions.toOwnedSlice(arena);
    program.data_segments = try data_segments.toOwnedSlice(arena);
    if (bag.hasErrors()) return ParseError.ParseFailed;
    return program;
}

fn trimRightCr(line: []const u8) []const u8 {
    if (line.len > 0 and line[line.len - 1] == '\r') return line[0 .. line.len - 1];
    return line;
}

fn parseFunctionHeader(arena: std.mem.Allocator, line: []const u8, line_no: u32, bag: *diagnostics.Bag) !FunctionBuilder {
    if (!std.mem.endsWith(u8, line, ":")) {
        try bag.errorAt(line_no, 1, "function header must end with ':'", .{});
        return .{ .name = try arena.dupe(u8, "<error>"), .blocks = .empty, .line = line_no };
    }
    const name_text = std.mem.trim(u8, line["fn ".len .. line.len - 1], " \t");
    if (name_text.len == 0) {
        try bag.errorAt(line_no, 1, "function header is missing a name", .{});
    }
    return .{
        .name = try arena.dupe(u8, if (name_text.len == 0) "<error>" else name_text),
        .blocks = .empty,
        .line = line_no,
    };
}

fn parseDataSegment(arena: std.mem.Allocator, line: []const u8, line_no: u32, bag: *diagnostics.Bag) !ir.DataSegment {
    const tokens = try tokenize(arena, line);
    if (tokens.len != 3) {
        try bag.errorAt(line_no, 1, "data segment must be 'data name 0x...'", .{});
        return .{
            .name = try arena.dupe(u8, "<error>"),
            .bytes = &.{},
            .line = line_no,
        };
    }

    const name = canonicalDataName(tokens[1]);
    if (name.len == 0) {
        try bag.errorAt(line_no, 1, "data segment is missing a name", .{});
    }

    return .{
        .name = try arena.dupe(u8, if (name.len == 0) "<error>" else name),
        .bytes = try parseHexBytes(arena, tokens[2], line_no, bag),
        .line = line_no,
    };
}

fn canonicalDataName(name: []const u8) []const u8 {
    if (name.len > 0 and name[0] == '.') return name[1..];
    return name;
}

fn parseHexBytes(arena: std.mem.Allocator, literal: []const u8, line_no: u32, bag: *diagnostics.Bag) ![]const u8 {
    if (literal.len < 2 or literal[0] != '0' or (literal[1] != 'x' and literal[1] != 'X')) {
        try bag.errorAt(line_no, 1, "data segment bytes must be a hex literal", .{});
        return &.{};
    }

    const hex = literal[2..];
    if (hex.len % 2 != 0) {
        try bag.errorAt(line_no, 1, "data segment hex literal must contain an even number of nibbles", .{});
        return &.{};
    }

    const bytes = try arena.alloc(u8, hex.len / 2);
    var index: usize = 0;
    while (index < bytes.len) : (index += 1) {
        const hi = hexNibble(hex[index * 2]) orelse {
            try bag.errorAt(line_no, 1, "data segment bytes contain a non-hex character", .{});
            return &.{};
        };
        const lo = hexNibble(hex[index * 2 + 1]) orelse {
            try bag.errorAt(line_no, 1, "data segment bytes contain a non-hex character", .{});
            return &.{};
        };
        bytes[index] = (hi << 4) | lo;
    }
    return bytes;
}

fn hexNibble(ch: u8) ?u8 {
    if (ch >= '0' and ch <= '9') return ch - '0';
    if (ch >= 'a' and ch <= 'f') return ch - 'a' + 10;
    if (ch >= 'A' and ch <= 'F') return ch - 'A' + 10;
    return null;
}

fn parseBlockHeader(arena: std.mem.Allocator, line: []const u8, line_no: u32, bag: *diagnostics.Bag) !BlockBuilder {
    const body = std.mem.trim(u8, line[0 .. line.len - 1], " \t");
    var split = std.mem.splitSequence(u8, body, " -> ");
    const left = std.mem.trim(u8, split.next() orelse "", " \t");
    const right = if (split.next()) |part| std.mem.trim(u8, part, " \t") else "";
    if (split.next() != null) {
        try bag.errorAt(line_no, 1, "block header has more than one output separator", .{});
    }

    var left_tokens = try tokenize(arena, left);
    if (left_tokens.len == 0) {
        try bag.errorAt(line_no, 1, "block header is missing a name", .{});
        left_tokens = &.{try arena.dupe(u8, "<error>")};
    }

    return .{
        .name = left_tokens[0],
        .inputs = left_tokens[1..],
        .outputs = try tokenize(arena, right),
        .instructions = .empty,
        .terminator = null,
        .line = line_no,
    };
}

fn parseInstruction(arena: std.mem.Allocator, line: []const u8, line_no: u32, bag: *diagnostics.Bag) !ir.Instruction {
    var results: []const []const u8 = &.{};
    var op_text = line;

    if (std.mem.indexOf(u8, line, " = ")) |eq_index| {
        results = try tokenize(arena, std.mem.trim(u8, line[0..eq_index], " \t"));
        op_text = std.mem.trim(u8, line[eq_index + 3 ..], " \t");
        if (results.len == 0) {
            try bag.errorAt(line_no, 1, "instruction assignment has no result", .{});
        }
    }

    const tokens = try tokenize(arena, op_text);
    if (tokens.len == 0) {
        try bag.errorAt(line_no, 1, "empty instruction", .{});
        return .{ .results = results, .mnemonic = try arena.dupe(u8, "<error>"), .operands = &.{}, .line = line_no };
    }

    return .{
        .results = results,
        .mnemonic = tokens[0],
        .operands = tokens[1..],
        .line = line_no,
    };
}

fn isTerminatorLine(line: []const u8) bool {
    if (line.len == 0) return false;
    return switch (line[0]) {
        '=' => std.mem.startsWith(u8, line, "=>"),
        'r' => std.mem.startsWith(u8, line, "return ") or std.mem.startsWith(u8, line, "revert "),
        's' => std.mem.eql(u8, line, "stop") or std.mem.startsWith(u8, line, "selfdestruct "),
        'i' => std.mem.eql(u8, line, "invalid") or std.mem.eql(u8, line, "iret"),
        else => false,
    };
}

fn parseTerminator(arena: std.mem.Allocator, line: []const u8, line_no: u32, bag: *diagnostics.Bag) !ir.Terminator {
    if (std.mem.startsWith(u8, line, "=>")) {
        const rest = std.mem.trim(u8, line[2..], " \t");
        if (std.mem.startsWith(u8, rest, "@")) {
            return .{ .jump = try targetName(arena, rest, line_no, bag) };
        }
        var parts = std.mem.splitSequence(u8, rest, " ? ");
        const condition = std.mem.trim(u8, parts.next() orelse "", " \t");
        const target_text = parts.next() orelse {
            try bag.errorAt(line_no, 1, "branch terminator is missing '?'", .{});
            return .invalid;
        };
        if (parts.next() != null) {
            try bag.errorAt(line_no, 1, "branch terminator has more than one '?'", .{});
            return .invalid;
        }
        var targets = std.mem.splitSequence(u8, target_text, " : ");
        const non_zero = std.mem.trim(u8, targets.next() orelse "", " \t");
        const zero = std.mem.trim(u8, targets.next() orelse "", " \t");
        if (condition.len == 0 or non_zero.len == 0 or zero.len == 0 or targets.next() != null) {
            try bag.errorAt(line_no, 1, "branch terminator must be '=> cond ? @non_zero : @zero'", .{});
            return .invalid;
        }
        return .{ .branch = .{
            .condition = try arena.dupe(u8, condition),
            .non_zero_target = try targetName(arena, non_zero, line_no, bag),
            .zero_target = try targetName(arena, zero, line_no, bag),
        } };
    }

    if (std.mem.startsWith(u8, line, "return ")) {
        const args = try tokenize(arena, line["return ".len..]);
        if (args.len != 2) {
            try bag.errorAt(line_no, 1, "return terminator expects pointer and length", .{});
            return .invalid;
        }
        return .{ .return_ = .{ .ptr = args[0], .len = args[1] } };
    }
    if (std.mem.startsWith(u8, line, "revert ")) {
        const args = try tokenize(arena, line["revert ".len..]);
        if (args.len != 2) {
            try bag.errorAt(line_no, 1, "revert terminator expects pointer and length", .{});
            return .invalid;
        }
        return .{ .revert = .{ .ptr = args[0], .len = args[1] } };
    }
    if (std.mem.eql(u8, line, "stop")) return .stop;
    if (std.mem.eql(u8, line, "invalid")) return .invalid;
    if (std.mem.startsWith(u8, line, "selfdestruct ")) {
        const args = try tokenize(arena, line["selfdestruct ".len..]);
        if (args.len != 1) {
            try bag.errorAt(line_no, 1, "selfdestruct terminator expects one beneficiary", .{});
            return .invalid;
        }
        return .{ .selfdestruct = args[0] };
    }
    if (std.mem.eql(u8, line, "iret")) return .iret;

    try bag.errorAt(line_no, 1, "unknown terminator", .{});
    return .invalid;
}

fn parseSwitch(
    arena: std.mem.Allocator,
    first_line: []const u8,
    lines: *std.mem.SplitIterator(u8, .scalar),
    line_no: *u32,
    bag: *diagnostics.Bag,
) !ir.Terminator {
    if (!std.mem.endsWith(u8, first_line, "{")) {
        try bag.errorAt(line_no.*, 1, "switch terminator must end with '{{'", .{});
        return .invalid;
    }
    const selector_text = std.mem.trim(u8, first_line["switch ".len .. first_line.len - 1], " \t");
    if (selector_text.len == 0) {
        try bag.errorAt(line_no.*, 1, "switch terminator is missing a selector", .{});
    }

    // Switch bodies are the one multiline terminator in the text format. The
    // parser consumes lines until the matching `}`, so the outer function/block
    // loop never sees individual case lines as instructions.
    var cases: std.ArrayList(ir.SwitchCase) = .empty;
    var default_target: ?[]const u8 = null;
    while (lines.next()) |raw_case_line| {
        line_no.* += 1;
        const raw = trimRightCr(raw_case_line);
        const case_line = std.mem.trim(u8, raw, " \t");
        if (case_line.len == 0) continue;
        if (std.mem.eql(u8, case_line, "}")) break;

        var split = std.mem.splitSequence(u8, case_line, " => ");
        const value = std.mem.trim(u8, split.next() orelse "", " \t");
        const target = std.mem.trim(u8, split.next() orelse "", " \t");
        if (value.len == 0 or target.len == 0 or split.next() != null) {
            try bag.errorAt(line_no.*, 1, "switch case must be 'value => @target'", .{});
            continue;
        }
        const target_name = try targetName(arena, target, line_no.*, bag);
        if (std.mem.eql(u8, value, "default")) {
            if (default_target != null) {
                try bag.errorAt(line_no.*, 1, "switch has duplicate default case", .{});
            }
            default_target = target_name;
        } else {
            try cases.append(arena, .{
                .value = try arena.dupe(u8, value),
                .target = target_name,
                .line = line_no.*,
            });
        }
    } else {
        try bag.errorAt(line_no.*, 1, "switch terminator was not closed", .{});
    }

    return .{ .switch_ = .{
        .selector = try arena.dupe(u8, selector_text),
        .cases = try cases.toOwnedSlice(arena),
        .default_target = default_target orelse blk: {
            try bag.errorAt(line_no.*, 1, "switch terminator is missing default case", .{});
            break :blk try arena.dupe(u8, "<error>");
        },
    } };
}

fn targetName(arena: std.mem.Allocator, target: []const u8, line_no: u32, bag: *diagnostics.Bag) ![]const u8 {
    if (!std.mem.startsWith(u8, target, "@") or target.len == 1) {
        try bag.errorAt(line_no, 1, "target must be written as '@name'", .{});
        return arena.dupe(u8, "<error>");
    }
    return arena.dupe(u8, target[1..]);
}

fn tokenize(arena: std.mem.Allocator, text: []const u8) ![]const []const u8 {
    var count: usize = 0;
    var counter = std.mem.tokenizeAny(u8, text, " \t");
    while (counter.next()) |_| count += 1;
    if (count == 0) return &.{};

    const tokens = try arena.alloc([]const u8, count);
    var it = std.mem.tokenizeAny(u8, text, " \t");
    var index: usize = 0;
    while (it.next()) |token| : (index += 1) {
        tokens[index] = try arena.dupe(u8, token);
    }
    return tokens;
}

fn finishBlock(arena: std.mem.Allocator, blocks: *std.ArrayList(ir.Block), block: *BlockBuilder) !void {
    try blocks.append(arena, .{
        .name = block.name,
        .inputs = block.inputs,
        .outputs = block.outputs,
        .instructions = try block.instructions.toOwnedSlice(arena),
        .terminator = block.terminator orelse .invalid,
        .line = block.line,
    });
}

fn finishFunction(arena: std.mem.Allocator, functions: *std.ArrayList(ir.Function), function: *FunctionBuilder) !void {
    const blocks = try function.blocks.toOwnedSlice(arena);
    const normalized_blocks = try normalizeInlineNumericValueOperands(arena, blocks);
    try functions.append(arena, .{
        .name = function.name,
        .blocks = normalized_blocks,
        .line = function.line,
    });
}

fn normalizeInlineNumericValueOperands(arena: std.mem.Allocator, blocks: []const ir.Block) ![]const ir.Block {
    var used_names = std.StringHashMap(void).init(arena);
    for (blocks) |block| {
        for (block.inputs) |name| try used_names.put(name, {});
        for (block.instructions) |instruction| {
            for (instruction.results) |name| try used_names.put(name, {});
        }
    }

    var synthetic_counter: usize = 0;
    const normalized = try arena.alloc(ir.Block, blocks.len);
    for (blocks, normalized) |block, *out_block| {
        var instructions: std.ArrayList(ir.Instruction) = .empty;
        try instructions.ensureTotalCapacity(arena, block.instructions.len);

        for (block.instructions) |instruction| {
            try appendInstructionWithNormalizedNumericOperands(
                arena,
                &instructions,
                &used_names,
                &synthetic_counter,
                instruction,
            );
        }

        const terminator = try normalizeTerminatorNumericOperands(
            arena,
            &instructions,
            &used_names,
            &synthetic_counter,
            block.terminator,
            block.line,
        );

        out_block.* = .{
            .name = block.name,
            .inputs = block.inputs,
            .outputs = block.outputs,
            .instructions = try instructions.toOwnedSlice(arena),
            .terminator = terminator,
            .line = block.line,
        };
    }
    return normalized;
}

fn appendInstructionWithNormalizedNumericOperands(
    arena: std.mem.Allocator,
    instructions: *std.ArrayList(ir.Instruction),
    used_names: *std.StringHashMap(void),
    synthetic_counter: *usize,
    instruction: ir.Instruction,
) !void {
    const spec = ops.lookup(instruction.mnemonic) orelse {
        try instructions.append(arena, instruction);
        return;
    };

    var needs_rewrite = false;
    for (instruction.operands, 0..) |operand, index| {
        if (operandIsInlineNumericValue(spec, index, operand)) {
            needs_rewrite = true;
            break;
        }
    }
    if (!needs_rewrite) {
        try instructions.append(arena, instruction);
        return;
    }

    const operands = try arena.alloc([]const u8, instruction.operands.len);
    @memcpy(operands, instruction.operands);
    for (operands, 0..) |*operand, index| {
        if (operandIsInlineNumericValue(spec, index, operand.*)) {
            operand.* = try appendSyntheticConst(arena, instructions, used_names, synthetic_counter, operand.*, instruction.line);
        }
    }

    try instructions.append(arena, .{
        .results = instruction.results,
        .mnemonic = instruction.mnemonic,
        .operands = operands,
        .line = instruction.line,
    });
}

fn operandIsInlineNumericValue(spec: ops.Spec, index: usize, operand: []const u8) bool {
    if (!isNumericLiteral(operand)) return false;
    return switch (spec) {
        .fixed => |fixed| index < fixed.inputs,
        .memory_load => |memory| index < memory.inputs,
        .memory_store => |memory| index < memory.inputs,
        .internal_call => index != 0,
    };
}

fn normalizeTerminatorNumericOperands(
    arena: std.mem.Allocator,
    instructions: *std.ArrayList(ir.Instruction),
    used_names: *std.StringHashMap(void),
    synthetic_counter: *usize,
    terminator: ir.Terminator,
    line: u32,
) !ir.Terminator {
    return switch (terminator) {
        .branch => |branch| .{ .branch = .{
            .condition = try normalizeTerminatorValueOperand(arena, instructions, used_names, synthetic_counter, branch.condition, line),
            .non_zero_target = branch.non_zero_target,
            .zero_target = branch.zero_target,
        } },
        .switch_ => |switch_term| .{ .switch_ = .{
            .selector = try normalizeTerminatorValueOperand(arena, instructions, used_names, synthetic_counter, switch_term.selector, line),
            .cases = switch_term.cases,
            .default_target = switch_term.default_target,
        } },
        .return_ => |ret| .{ .return_ = .{
            .ptr = try normalizeTerminatorValueOperand(arena, instructions, used_names, synthetic_counter, ret.ptr, line),
            .len = try normalizeTerminatorValueOperand(arena, instructions, used_names, synthetic_counter, ret.len, line),
        } },
        .revert => |rev| .{ .revert = .{
            .ptr = try normalizeTerminatorValueOperand(arena, instructions, used_names, synthetic_counter, rev.ptr, line),
            .len = try normalizeTerminatorValueOperand(arena, instructions, used_names, synthetic_counter, rev.len, line),
        } },
        .selfdestruct => |beneficiary| .{
            .selfdestruct = try normalizeTerminatorValueOperand(arena, instructions, used_names, synthetic_counter, beneficiary, line),
        },
        .jump, .stop, .invalid, .iret => terminator,
    };
}

fn normalizeTerminatorValueOperand(
    arena: std.mem.Allocator,
    instructions: *std.ArrayList(ir.Instruction),
    used_names: *std.StringHashMap(void),
    synthetic_counter: *usize,
    operand: []const u8,
    line: u32,
) ![]const u8 {
    if (!isNumericLiteral(operand)) return operand;
    return appendSyntheticConst(arena, instructions, used_names, synthetic_counter, operand, line);
}

fn appendSyntheticConst(
    arena: std.mem.Allocator,
    instructions: *std.ArrayList(ir.Instruction),
    used_names: *std.StringHashMap(void),
    synthetic_counter: *usize,
    literal: []const u8,
    line: u32,
) ![]const u8 {
    const name = try nextSyntheticConstName(arena, used_names, synthetic_counter);
    try instructions.append(arena, .{
        .results = try singleToken(arena, name),
        .mnemonic = try arena.dupe(u8, "const"),
        .operands = try singleToken(arena, literal),
        .line = line,
        .synthetic = true,
    });
    return name;
}

fn nextSyntheticConstName(arena: std.mem.Allocator, used_names: *std.StringHashMap(void), synthetic_counter: *usize) ![]const u8 {
    while (true) {
        const name = try std.fmt.allocPrint(arena, "__sinora_inline_const_{d}", .{synthetic_counter.*});
        synthetic_counter.* += 1;
        const entry = try used_names.getOrPut(name);
        if (!entry.found_existing) return name;
    }
}

fn singleToken(arena: std.mem.Allocator, token: []const u8) ![]const []const u8 {
    const tokens = try arena.alloc([]const u8, 1);
    tokens[0] = token;
    return tokens;
}

fn isNumericLiteral(text: []const u8) bool {
    if (isHexLiteral(text)) return true;
    const unsigned = stripOptionalMinus(text);
    if (unsigned.len == 0) return false;
    for (unsigned) |ch| {
        if (!std.ascii.isDigit(ch)) return false;
    }
    return true;
}

fn isHexLiteral(text: []const u8) bool {
    const unsigned = stripOptionalMinus(text);
    if (unsigned.len <= 2) return false;
    if (unsigned[0] != '0' or (unsigned[1] != 'x' and unsigned[1] != 'X')) return false;
    for (unsigned[2..]) |ch| {
        if (!std.ascii.isHex(ch)) return false;
    }
    return true;
}

fn stripOptionalMinus(text: []const u8) []const u8 {
    if (text.len > 1 and text[0] == '-') return text[1..];
    return text;
}

test "parse smoke SIR" {
    const source =
        \\fn init:
        \\    entry {
        \\        v0 = runtime_start_offset
        \\        v1 = runtime_length
        \\        v2 = malloc v1
        \\        return v2 v1
        \\    }
        \\
        \\fn main:
        \\    main_entry {
        \\        zero = const 0x0
        \\        switch zero {
        \\        default => @revert_error
        \\    }
        \\    }
        \\    revert_error {
        \\        revert zero zero
        \\    }
    ;

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parse(std.testing.allocator, source, &bag);
    defer program.deinit();

    try std.testing.expectEqual(@as(usize, 2), program.functions.len);
    try std.testing.expectEqual(@as(usize, 3), program.stats().blocks);
    try std.testing.expectEqual(@as(usize, 1), program.stats().switches);
}

test "parse data segments" {
    const source =
        \\fn main:
        \\    entry {
        \\        off = data_offset .blob
        \\        stop
        \\    }
        \\
        \\data blob 0x11223344
        \\data .0 0x
    ;

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parse(std.testing.allocator, source, &bag);
    defer program.deinit();

    try std.testing.expectEqual(@as(usize, 2), program.data_segments.len);
    try std.testing.expectEqualStrings("blob", program.data_segments[0].name);
    try std.testing.expectEqualSlices(u8, &.{ 0x11, 0x22, 0x33, 0x44 }, program.data_segments[0].bytes);
    try std.testing.expectEqualStrings("0", program.data_segments[1].name);
    try std.testing.expectEqual(@as(usize, 0), program.data_segments[1].bytes.len);
}

test "parse normalizes inline numeric instruction operands into synthetic consts" {
    const source =
        \\fn main:
        \\    entry {
        \\        z = add 3 3
        \\        stop
        \\    }
    ;

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parse(std.testing.allocator, source, &bag);
    defer program.deinit();

    const instructions = program.functions[0].blocks[0].instructions;
    try std.testing.expectEqual(@as(usize, 3), instructions.len);
    try std.testing.expectEqualStrings("const", instructions[0].mnemonic);
    try std.testing.expectEqualStrings("3", instructions[0].operands[0]);
    try std.testing.expectEqualStrings("const", instructions[1].mnemonic);
    try std.testing.expectEqualStrings("3", instructions[1].operands[0]);
    try std.testing.expect(!std.mem.eql(u8, instructions[0].results[0], instructions[1].results[0]));
    try std.testing.expectEqualStrings("add", instructions[2].mnemonic);
    try std.testing.expectEqualStrings(instructions[0].results[0], instructions[2].operands[0]);
    try std.testing.expectEqualStrings(instructions[1].results[0], instructions[2].operands[1]);
}

test "parse keeps numeric metadata operands unchanged" {
    const source =
        \\fn main:
        \\    entry {
        \\        x = const 3
        \\        p = salloc 32
        \\        off = data_offset .blob
        \\        stop
        \\    }
        \\
        \\data blob 0x
    ;

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parse(std.testing.allocator, source, &bag);
    defer program.deinit();

    const instructions = program.functions[0].blocks[0].instructions;
    try std.testing.expectEqual(@as(usize, 3), instructions.len);
    try std.testing.expectEqualStrings("const", instructions[0].mnemonic);
    try std.testing.expectEqualStrings("3", instructions[0].operands[0]);
    try std.testing.expectEqualStrings("salloc", instructions[1].mnemonic);
    try std.testing.expectEqualStrings("32", instructions[1].operands[0]);
    try std.testing.expectEqualStrings("data_offset", instructions[2].mnemonic);
    try std.testing.expectEqualStrings(".blob", instructions[2].operands[0]);
}

test "parse normalizes inline numeric terminator operands" {
    const source =
        \\fn main:
        \\    entry {
        \\        return 0 0
        \\    }
    ;

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parse(std.testing.allocator, source, &bag);
    defer program.deinit();

    const block = program.functions[0].blocks[0];
    try std.testing.expectEqual(@as(usize, 2), block.instructions.len);
    try std.testing.expectEqualStrings("const", block.instructions[0].mnemonic);
    try std.testing.expectEqualStrings("0", block.instructions[0].operands[0]);
    try std.testing.expectEqualStrings("const", block.instructions[1].mnemonic);
    try std.testing.expectEqualStrings("0", block.instructions[1].operands[0]);

    const ret = block.terminator.return_;
    try std.testing.expectEqualStrings(block.instructions[0].results[0], ret.ptr);
    try std.testing.expectEqualStrings(block.instructions[1].results[0], ret.len);
    try std.testing.expect(!std.mem.eql(u8, ret.ptr, ret.len));
}

test "parse normalizes inline numeric branch and switch selectors" {
    const source =
        \\fn main:
        \\    entry {
        \\        => 1 ? @switcher : @done
        \\    }
        \\    switcher {
        \\        switch 2 {
        \\        2 => @done
        \\        default => @done
        \\    }
        \\    }
        \\    done {
        \\        stop
        \\    }
    ;

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parse(std.testing.allocator, source, &bag);
    defer program.deinit();

    const entry = program.functions[0].blocks[0];
    try std.testing.expectEqual(@as(usize, 1), entry.instructions.len);
    try std.testing.expectEqualStrings("const", entry.instructions[0].mnemonic);
    try std.testing.expectEqualStrings(entry.instructions[0].results[0], entry.terminator.branch.condition);

    const switcher = program.functions[0].blocks[1];
    try std.testing.expectEqual(@as(usize, 1), switcher.instructions.len);
    try std.testing.expectEqualStrings("const", switcher.instructions[0].mnemonic);
    try std.testing.expectEqualStrings(switcher.instructions[0].results[0], switcher.terminator.switch_.selector);
}
