//! Structural SIR legality checks.
//!
//! This pass sits between the forgiving line parser and bytecode generation. It
//! is intentionally not an optimizer and not a semantic Ora checker; it only
//! verifies the SIR shape that Sinora must trust before debug/release codegen:
//! names are unique, targets exist, op arities match, and every value use has a
//! definition.

const std = @import("std");

const diagnostics = @import("diagnostics.zig");
const ir = @import("ir.zig");
const ops = @import("ops.zig");

pub const ValidateError = error{
    InvalidSir,
};

pub fn validate(allocator: std.mem.Allocator, program: ir.Program, bag: *diagnostics.Bag) !void {
    // Three global tables are enough for whole-program structural checks:
    // function names for duplicate detection, function signatures for `icall`,
    // and data segment names for `.data` references. Value names stay
    // function-local and are validated below.
    var function_names = std.StringHashMap(u32).init(allocator);
    defer function_names.deinit();
    try function_names.ensureTotalCapacity(hashCapacity(program.functions.len));

    var signatures = std.StringHashMap(FunctionSignature).init(allocator);
    defer signatures.deinit();
    try signatures.ensureTotalCapacity(hashCapacity(program.functions.len));

    var data_names = std.StringHashMap(u32).init(allocator);
    defer data_names.deinit();
    try data_names.ensureTotalCapacity(hashCapacity(program.data_segments.len));

    // Build the function signature table first so icall validation can resolve
    // forward references without depending on source order.
    for (program.functions) |function| {
        const entry = try function_names.getOrPut(function.name);
        if (entry.found_existing) {
            try bag.errorAt(function.line, 1, "duplicate function '{s}' first declared at line {d}", .{ function.name, entry.value_ptr.* });
        } else {
            entry.value_ptr.* = function.line;
            try signatures.put(function.name, try inferFunctionSignature(function, bag));
        }
    }

    for (program.data_segments) |segment| {
        if (segment.name.len == 0) {
            try bag.errorAt(segment.line, 1, "data segment is missing a name", .{});
            continue;
        }
        const entry = try data_names.getOrPut(segment.name);
        if (entry.found_existing) {
            try bag.errorAt(segment.line, 1, "duplicate data segment '{s}' first declared at line {d}", .{ segment.name, entry.value_ptr.* });
        } else {
            entry.value_ptr.* = segment.line;
        }
    }

    for (program.functions) |function| {
        try validateFunction(allocator, function, &signatures, &data_names, bag);
    }

    if (bag.hasErrors()) return ValidateError.InvalidSir;
}

const FunctionSignature = struct {
    inputs: usize,
    outputs: usize,
};

fn inferFunctionSignature(function: ir.Function, bag: *diagnostics.Bag) !FunctionSignature {
    const inputs = if (function.blocks.len > 0) function.blocks[0].inputs.len else 0;
    var outputs: usize = 0;
    var output_line: u32 = function.line;
    var saw_iret = false;

    for (function.blocks) |block| {
        if (block.terminator != .iret) continue;
        if (!saw_iret) {
            outputs = block.outputs.len;
            output_line = block.line;
            saw_iret = true;
            continue;
        }
        if (outputs != block.outputs.len) {
            try bag.errorAt(block.line, 1, "function '{s}' iret output count differs from line {d}", .{ function.name, output_line });
        }
    }

    return .{
        .inputs = inputs,
        .outputs = if (saw_iret) outputs else 0,
    };
}

fn validateFunction(
    allocator: std.mem.Allocator,
    function: ir.Function,
    signatures: *const std.StringHashMap(FunctionSignature),
    data_names: *const std.StringHashMap(u32),
    bag: *diagnostics.Bag,
) !void {
    if (function.blocks.len == 0) {
        try bag.errorAt(function.line, 1, "function '{s}' has no blocks", .{function.name});
        return;
    }

    var block_names = std.StringHashMap(u32).init(allocator);
    defer block_names.deinit();
    try block_names.ensureTotalCapacity(hashCapacity(function.blocks.len));
    for (function.blocks) |block| {
        const entry = try block_names.getOrPut(block.name);
        if (entry.found_existing) {
            try bag.errorAt(block.line, 1, "duplicate block '{s}' first declared at line {d}", .{ block.name, entry.value_ptr.* });
        } else {
            entry.value_ptr.* = block.line;
        }
    }

    for (function.blocks) |block| {
        try validateTerminatorTargets(function.name, block, &block_names, bag);
    }

    try validateValuesAndInstructions(allocator, function, signatures, data_names, bag);
}

fn validateTerminatorTargets(
    function_name: []const u8,
    block: ir.Block,
    block_names: *const std.StringHashMap(u32),
    bag: *diagnostics.Bag,
) !void {
    switch (block.terminator) {
        .jump => |target| try requireBlock(function_name, block, target, block_names, bag),
        .branch => |branch| {
            try requireBlock(function_name, block, branch.non_zero_target, block_names, bag);
            try requireBlock(function_name, block, branch.zero_target, block_names, bag);
        },
        .switch_ => |switch_term| {
            for (switch_term.cases) |case| {
                try requireBlock(function_name, block, case.target, block_names, bag);
            }
            try requireBlock(function_name, block, switch_term.default_target, block_names, bag);
        },
        .return_, .revert, .stop, .invalid, .selfdestruct, .iret => {},
    }
}

fn requireBlock(
    function_name: []const u8,
    block: ir.Block,
    target: []const u8,
    block_names: *const std.StringHashMap(u32),
    bag: *diagnostics.Bag,
) !void {
    if (!block_names.contains(target)) {
        try bag.errorAt(block.line, 1, "function '{s}' block '{s}' targets missing block '{s}'", .{ function_name, block.name, target });
    }
}

fn validateValuesAndInstructions(
    allocator: std.mem.Allocator,
    function: ir.Function,
    signatures: *const std.StringHashMap(FunctionSignature),
    data_names: *const std.StringHashMap(u32),
    bag: *diagnostics.Bag,
) !void {
    var values = std.StringHashMap(u32).init(allocator);
    defer values.deinit();
    try values.ensureTotalCapacity(hashCapacity(valueDefinitionCount(function)));

    // First pass: collect all SSA definitions in the function. Block inputs are
    // definitions too; they are the block-parameter SSA values supplied by
    // predecessor block outputs.
    for (function.blocks) |block| {
        for (block.inputs) |input| {
            try defineValue(&values, input, block.line, "block input", bag);
        }
        for (block.instructions) |instruction| {
            for (instruction.results) |result| {
                try defineValue(&values, result, instruction.line, "instruction result", bag);
            }
        }
    }

    // Second pass: validate uses. A two-pass walk lets values be used before
    // their textual definition in a later block while still rejecting undefined
    // names and non-value immediates in the right places.
    for (function.blocks) |block| {
        for (block.outputs) |output| {
            try requireValue(&values, output, block.line, "block output", bag);
        }
        for (block.instructions) |instruction| {
            // Validate opcode shape and classify operands from the same lookup.
            // Unknown opcodes return null, then operandRequiresValue keeps the
            // old conservative fallback so undefined values are still reported.
            const spec = try validateInstruction(function.name, instruction, signatures, data_names, bag);
            for (instruction.operands, 0..) |operand, index| {
                if (!operandRequiresValue(spec, index, operand)) continue;
                try requireValue(&values, operand, instruction.line, "instruction operand", bag);
            }
        }
        try validateTerminatorValues(&values, block, bag);
    }
}

fn validateInstruction(
    function_name: []const u8,
    instruction: ir.Instruction,
    signatures: *const std.StringHashMap(FunctionSignature),
    data_names: *const std.StringHashMap(u32),
    bag: *diagnostics.Bag,
) !?ops.Spec {
    const spec = ops.lookup(instruction.mnemonic) orelse {
        try bag.errorAt(instruction.line, 1, "unknown instruction opcode '{s}' in function '{s}'", .{ instruction.mnemonic, function_name });
        return null;
    };

    switch (spec) {
        .fixed => |fixed| try validateFixedInstruction(instruction, fixed, data_names, bag),
        .memory_load => |memory| {
            try requireCount(instruction.line, instruction.mnemonic, "operands", memory.inputs, instruction.operands.len, bag);
            try requireCount(instruction.line, instruction.mnemonic, "results", memory.outputs, instruction.results.len, bag);
        },
        .memory_store => |memory| {
            try requireCount(instruction.line, instruction.mnemonic, "operands", memory.inputs, instruction.operands.len, bag);
            try requireCount(instruction.line, instruction.mnemonic, "results", memory.outputs, instruction.results.len, bag);
        },
        .internal_call => try validateInternalCall(instruction, signatures, bag),
    }
    return spec;
}

fn validateFixedInstruction(
    instruction: ir.Instruction,
    spec: ops.Fixed,
    data_names: *const std.StringHashMap(u32),
    bag: *diagnostics.Bag,
) !void {
    try requireCount(instruction.line, instruction.mnemonic, "operands", spec.operandCount(), instruction.operands.len, bag);
    try requireCount(instruction.line, instruction.mnemonic, "results", spec.outputs, instruction.results.len, bag);
    if (instruction.operands.len <= spec.inputs) return;

    const extra = instruction.operands[spec.inputs];
    switch (spec.extra) {
        .none => {},
        .numeric => {
            if (!isNumericLiteral(extra)) {
                try bag.errorAt(instruction.line, 1, "instruction '{s}' expects a numeric immediate", .{instruction.mnemonic});
            }
        },
        .data_ref => {
            if (!isDataRef(extra)) {
                try bag.errorAt(instruction.line, 1, "instruction '{s}' expects a data reference like '.name'", .{instruction.mnemonic});
            } else if (!data_names.contains(dataRefName(extra))) {
                try bag.errorAt(instruction.line, 1, "instruction '{s}' references missing data segment '{s}'", .{ instruction.mnemonic, dataRefName(extra) });
            }
        },
    }
}

fn validateInternalCall(
    instruction: ir.Instruction,
    signatures: *const std.StringHashMap(FunctionSignature),
    bag: *diagnostics.Bag,
) !void {
    if (instruction.operands.len == 0) {
        try bag.errorAt(instruction.line, 1, "instruction 'icall' expects a function target", .{});
        return;
    }

    const target = instruction.operands[0];
    if (!isFunctionRef(target)) {
        try bag.errorAt(instruction.line, 1, "instruction 'icall' target must be written as '@function'", .{});
        return;
    }

    const callee_name = target[1..];
    const signature = signatures.get(callee_name) orelse {
        try bag.errorAt(instruction.line, 1, "instruction 'icall' targets missing function '{s}'", .{callee_name});
        return;
    };

    try requireCount(instruction.line, "icall", "arguments", signature.inputs, instruction.operands.len - 1, bag);
    try requireCount(instruction.line, "icall", "results", signature.outputs, instruction.results.len, bag);
}

fn requireCount(
    line: u32,
    mnemonic: []const u8,
    comptime role: []const u8,
    expected: usize,
    actual: usize,
    bag: *diagnostics.Bag,
) !void {
    if (expected != actual) {
        try bag.errorAt(line, 1, "instruction '{s}' expects {d} " ++ role ++ ", got {d}", .{ mnemonic, expected, actual });
    }
}

fn defineValue(
    values: *std.StringHashMap(u32),
    name: []const u8,
    line: u32,
    comptime role: []const u8,
    bag: *diagnostics.Bag,
) !void {
    if (!isValueName(name)) {
        try bag.errorAt(line, 1, role ++ " '{s}' is not a valid value name", .{name});
        return;
    }
    const entry = try values.getOrPut(name);
    if (entry.found_existing) {
        try bag.errorAt(line, 1, "duplicate value '{s}' first defined at line {d}", .{ name, entry.value_ptr.* });
        return;
    }
    entry.value_ptr.* = line;
}

fn validateTerminatorValues(values: *const std.StringHashMap(u32), block: ir.Block, bag: *diagnostics.Bag) !void {
    switch (block.terminator) {
        .jump, .stop, .invalid, .iret => {},
        .branch => |branch| try requireValue(values, branch.condition, block.line, "branch condition", bag),
        .switch_ => |switch_term| try requireValue(values, switch_term.selector, block.line, "switch selector", bag),
        .return_ => |ret| {
            try requireValue(values, ret.ptr, block.line, "return pointer", bag);
            try requireValue(values, ret.len, block.line, "return length", bag);
        },
        .revert => |revert| {
            try requireValue(values, revert.ptr, block.line, "revert pointer", bag);
            try requireValue(values, revert.len, block.line, "revert length", bag);
        },
        .selfdestruct => |beneficiary| try requireValue(values, beneficiary, block.line, "selfdestruct beneficiary", bag),
    }
}

fn requireValue(
    values: *const std.StringHashMap(u32),
    name: []const u8,
    line: u32,
    comptime role: []const u8,
    bag: *diagnostics.Bag,
) !void {
    if (!isValueName(name)) {
        try bag.errorAt(line, 1, role ++ " '{s}' is not a valid value name", .{name});
        return;
    }
    if (!values.contains(name)) {
        try bag.errorAt(line, 1, role ++ " references undefined value '{s}'", .{name});
    }
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

fn isValueName(name: []const u8) bool {
    if (name.len == 0) return false;
    if (name[0] == '$') {
        if (name.len == 1) return false;
        for (name[1..]) |ch| {
            if (!std.ascii.isDigit(ch)) return false;
        }
        return true;
    }
    if (!(std.ascii.isAlphabetic(name[0]) or name[0] == '_')) return false;
    for (name[1..]) |ch| {
        if (!(std.ascii.isAlphanumeric(ch) or ch == '_')) return false;
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

fn isNumericLiteral(text: []const u8) bool {
    if (isHexLiteral(text)) return true;
    const unsigned = stripOptionalMinus(text);
    if (unsigned.len == 0) return false;
    for (unsigned) |ch| {
        if (!std.ascii.isDigit(ch)) return false;
    }
    return true;
}

fn stripOptionalMinus(text: []const u8) []const u8 {
    if (text.len > 1 and text[0] == '-') return text[1..];
    return text;
}

fn isFunctionRef(text: []const u8) bool {
    return isSigilRef('@', text);
}

fn isDataRef(text: []const u8) bool {
    return isSigilRef('.', text);
}

fn dataRefName(text: []const u8) []const u8 {
    return if (isDataRef(text)) text[1..] else text;
}

fn isSigilRef(comptime sigil: u8, text: []const u8) bool {
    return text.len > 1 and text[0] == sigil;
}

fn valueDefinitionCount(function: ir.Function) usize {
    var count: usize = 0;
    for (function.blocks) |block| {
        count += block.inputs.len;
        for (block.instructions) |instruction| {
            count += instruction.results.len;
        }
    }
    return count;
}

fn hashCapacity(count: usize) u32 {
    return @intCast(@min(count, std.math.maxInt(u32)));
}

test "validator rejects missing target" {
    const parser = @import("parser.zig");
    const source =
        \\fn main:
        \\    entry {
        \\        => @missing
        \\    }
    ;

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator, source, &bag);
    defer program.deinit();

    try std.testing.expectError(ValidateError.InvalidSir, validate(std.testing.allocator, program, &bag));
    try std.testing.expect(bag.hasErrors());
}

test "validator rejects undefined values" {
    const parser = @import("parser.zig");
    const source =
        \\fn main:
        \\    entry {
        \\        revert missing missing
        \\    }
    ;

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator, source, &bag);
    defer program.deinit();

    try std.testing.expectError(ValidateError.InvalidSir, validate(std.testing.allocator, program, &bag));
    try std.testing.expect(bag.hasErrors());
}

test "validator rejects unknown opcode" {
    const parser = @import("parser.zig");
    const source =
        \\fn main:
        \\    entry {
        \\        v0 = unknown_op
        \\        stop
        \\    }
    ;

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator, source, &bag);
    defer program.deinit();

    try std.testing.expectError(ValidateError.InvalidSir, validate(std.testing.allocator, program, &bag));
    try std.testing.expect(bag.hasErrors());
}

test "validator checks opcode arity" {
    const parser = @import("parser.zig");
    const source =
        \\fn main:
        \\    entry {
        \\        v0 = const 0x0
        \\        v1 = add v0
        \\        stop
        \\    }
    ;

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator, source, &bag);
    defer program.deinit();

    try std.testing.expectError(ValidateError.InvalidSir, validate(std.testing.allocator, program, &bag));
    try std.testing.expect(bag.hasErrors());
}

test "validator checks internal call signatures" {
    const parser = @import("parser.zig");
    const source =
        \\fn callee:
        \\    entry a -> out {
        \\        out = copy a
        \\        iret
        \\    }
        \\
        \\fn main:
        \\    entry {
        \\        arg = const 0x1
        \\        r0 r1 = icall @callee arg
        \\        stop
        \\    }
    ;

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator, source, &bag);
    defer program.deinit();

    try std.testing.expectError(ValidateError.InvalidSir, validate(std.testing.allocator, program, &bag));
    try std.testing.expect(bag.hasErrors());
}

test "validator checks data references" {
    const parser = @import("parser.zig");
    const source =
        \\fn main:
        \\    entry {
        \\        off = data_offset .missing
        \\        stop
        \\    }
        \\
        \\data blob 0x1122
    ;

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator, source, &bag);
    defer program.deinit();

    try std.testing.expectError(ValidateError.InvalidSir, validate(std.testing.allocator, program, &bag));
    try std.testing.expect(bag.hasErrors());
}

test "validator rejects duplicate data segments" {
    const parser = @import("parser.zig");
    const source =
        \\fn main:
        \\    entry {
        \\        off = data_offset .blob
        \\        stop
        \\    }
        \\
        \\data blob 0x1122
        \\data blob 0x3344
    ;

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator, source, &bag);
    defer program.deinit();

    try std.testing.expectError(ValidateError.InvalidSir, validate(std.testing.allocator, program, &bag));
    try std.testing.expect(bag.hasErrors());
}
