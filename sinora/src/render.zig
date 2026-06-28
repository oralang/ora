const std = @import("std");

const ir = @import("ir.zig");

// Canonical SIR text renderer.
//
// The parser accepts a small textual IR. This module prints that IR back in one
// stable layout so tests, debug output, and CLI tools can compare programs
// without depending on original whitespace. The renderer is intentionally dumb:
// it does not validate IR, reorder blocks, resolve labels, or pretty-print based
// on terminal width. It only serializes the already-built `ir.Program`.
//
// Performance model:
// - callers that stream to stdout use `writeProgram` directly;
// - callers that need a string use `allocProgram`, which pre-reserves a close
//   size estimate before rendering;
// - the hot path uses `writeAll`/`writeByte` instead of formatter calls for
//   simple concatenation.

pub fn writeProgram(writer: anytype, program: ir.Program) !void {
    // Functions are printed first, data segments second. A single blank line
    // separates functions from data and also separates adjacent functions.
    var wrote_function = false;
    for (program.functions, 0..) |function, function_index| {
        if (function_index != 0) try writer.writeByte('\n');
        wrote_function = true;
        try writer.writeAll("fn ");
        try writer.writeAll(function.name);
        try writer.writeAll(":\n");
        for (function.blocks) |block| {
            try writeBlock(writer, block);
        }
    }
    if (program.data_segments.len != 0) {
        if (wrote_function) try writer.writeByte('\n');
        for (program.data_segments) |segment| {
            try writer.writeAll("data ");
            try writer.writeAll(segment.name);
            try writer.writeByte(' ');
            try writeHexLiteral(writer, segment.bytes);
            try writer.writeByte('\n');
        }
    }
}

pub fn allocProgram(allocator: std.mem.Allocator, program: ir.Program) ![]const u8 {
    // `initCapacity` avoids repeated reallocations for the common "render into
    // a string" path. The estimate is exact for the current renderer, but this
    // function only relies on it as a reservation hint.
    const estimated_size = estimateProgramSize(program);
    var buffer = try std.Io.Writer.Allocating.initCapacity(allocator, estimated_size);
    errdefer buffer.deinit();
    try writeProgram(&buffer.writer, program);
    std.debug.assert(buffer.written().len == estimated_size);
    return buffer.toOwnedSlice();
}

fn writeBlock(writer: anytype, block: ir.Block) !void {
    // Block header syntax:
    //
    //     name input0 input1 -> output0 {
    //
    // Empty input/output lists are omitted rather than rendered as punctuation.
    try writer.writeAll("    ");
    try writer.writeAll(block.name);
    try writeNameListPrefixed(writer, " ", block.inputs);
    try writeNameListPrefixed(writer, " -> ", block.outputs);
    try writer.writeAll(" {\n");

    for (block.instructions) |instruction| {
        try writeInstruction(writer, instruction);
    }

    try writer.writeAll("        ");
    try writeTerminator(writer, block.terminator);
    try writer.writeAll("\n    }\n");
}

fn writeInstruction(writer: anytype, instruction: ir.Instruction) !void {
    // Instruction syntax:
    //
    //     result0 result1 = mnemonic operand0 operand1
    //
    // Instructions with no results omit the ` = ` prefix entirely.
    try writer.writeAll("        ");
    if (instruction.results.len != 0) {
        try writeNameList(writer, instruction.results);
        try writer.writeAll(" = ");
    }
    try writer.writeAll(instruction.mnemonic);
    try writeNameListPrefixed(writer, " ", instruction.operands);
    try writer.writeByte('\n');
}

fn writeTerminator(writer: anytype, terminator: ir.Terminator) !void {
    // Terminators are the only control-flow syntax in rendered SIR text. Branch
    // and jump targets are printed with `@` because the parser treats them as
    // block labels, not normal value names.
    switch (terminator) {
        .jump => |target| {
            try writer.writeAll("=> @");
            try writer.writeAll(target);
        },
        .branch => |branch| {
            try writer.writeAll("=> ");
            try writer.writeAll(branch.condition);
            try writer.writeAll(" ? @");
            try writer.writeAll(branch.non_zero_target);
            try writer.writeAll(" : @");
            try writer.writeAll(branch.zero_target);
        },
        .switch_ => |switch_term| {
            try writer.writeAll("switch ");
            try writer.writeAll(switch_term.selector);
            try writer.writeAll(" {\n");
            for (switch_term.cases) |case| {
                try writer.writeAll("        ");
                try writer.writeAll(case.value);
                try writer.writeAll(" => @");
                try writer.writeAll(case.target);
                try writer.writeByte('\n');
            }
            try writer.writeAll("        default => @");
            try writer.writeAll(switch_term.default_target);
            try writer.writeByte('\n');
            try writer.writeAll("    }");
        },
        .return_ => |ret| {
            try writer.writeAll("return ");
            try writer.writeAll(ret.ptr);
            try writer.writeByte(' ');
            try writer.writeAll(ret.len);
        },
        .revert => |revert| {
            try writer.writeAll("revert ");
            try writer.writeAll(revert.ptr);
            try writer.writeByte(' ');
            try writer.writeAll(revert.len);
        },
        .stop => try writer.writeAll("stop"),
        .invalid => try writer.writeAll("invalid"),
        .selfdestruct => |beneficiary| {
            try writer.writeAll("selfdestruct ");
            try writer.writeAll(beneficiary);
        },
        .iret => try writer.writeAll("iret"),
    }
}

fn writeNameListPrefixed(writer: anytype, comptime prefix: []const u8, names: []const []const u8) !void {
    // Prefixes are conditional. This keeps headers compact:
    //
    //     entry {
    //     entry a b {
    //     entry a -> out {
    //
    // `prefix` is comptime because every caller passes a string literal. That
    // lets Zig specialize this helper and keep prefix length/data static.
    if (names.len == 0) return;
    try writer.writeAll(prefix);
    try writeNameList(writer, names);
}

fn writeNameList(writer: anytype, names: []const []const u8) !void {
    // Space-separated value or block-parameter names. The renderer never quotes
    // names; the parser/IR builder is responsible for only storing legal names.
    for (names, 0..) |name, index| {
        if (index != 0) try writer.writeByte(' ');
        try writer.writeAll(name);
    }
}

fn writeHexLiteral(writer: anytype, bytes: []const u8) !void {
    // Data segments are rendered as lowercase contiguous hex with a `0x`
    // prefix. Chunking avoids two writer calls per input byte on large blobs.
    const hex = "0123456789abcdef";
    try writer.writeAll("0x");
    var buffer: [4096]u8 = undefined;
    var out: usize = 0;
    for (bytes) |byte| {
        if (out + 2 > buffer.len) {
            try writer.writeAll(buffer[0..out]);
            out = 0;
        }
        buffer[out] = hex[byte >> 4];
        buffer[out + 1] = hex[byte & 0x0f];
        out += 2;
    }
    if (out != 0) {
        try writer.writeAll(buffer[0..out]);
    }
}

fn estimateProgramSize(program: ir.Program) usize {
    // Exact byte count for the current canonical renderer. Keeping this in sync
    // matters because `allocProgram` reserves this capacity and asserts that
    // rendering produced exactly this many bytes in debug builds.
    var total: usize = 0;
    const wrote_function = program.functions.len != 0;
    for (program.functions, 0..) |function, function_index| {
        if (function_index != 0) total += 1;
        total += "fn ".len + function.name.len + ":\n".len;
        for (function.blocks) |block| {
            total += estimateBlockSize(block);
        }
    }
    if (program.data_segments.len != 0) {
        if (wrote_function) total += 1;
        for (program.data_segments) |segment| {
            total += "data ".len + segment.name.len + " ".len + 2 + segment.bytes.len * 2 + "\n".len;
        }
    }
    return total;
}

fn estimateBlockSize(block: ir.Block) usize {
    // Mirrors `writeBlock` byte-for-byte without touching a writer.
    var total: usize = 0;
    total += "    ".len + block.name.len;
    total += estimateNameListPrefixedSize(" ", block.inputs);
    total += estimateNameListPrefixedSize(" -> ", block.outputs);
    total += " {\n".len;
    for (block.instructions) |instruction| {
        total += estimateInstructionSize(instruction);
    }
    total += "        ".len + estimateTerminatorSize(block.terminator) + "\n    }\n".len;
    return total;
}

fn estimateInstructionSize(instruction: ir.Instruction) usize {
    // Mirrors `writeInstruction`, including indentation and the conditional
    // result-list prefix.
    var total: usize = "        ".len;
    if (instruction.results.len != 0) {
        total += estimateNameListSize(instruction.results) + " = ".len;
    }
    total += instruction.mnemonic.len;
    total += estimateNameListPrefixedSize(" ", instruction.operands);
    total += "\n".len;
    return total;
}

fn estimateTerminatorSize(terminator: ir.Terminator) usize {
    // Mirrors `writeTerminator`. The switch case keeps the multi-line layout
    // explicit because switch rendering is the only nested terminator format.
    return switch (terminator) {
        .jump => |target| "=> @".len + target.len,
        .branch => |branch| "=> ".len + branch.condition.len + " ? @".len + branch.non_zero_target.len + " : @".len + branch.zero_target.len,
        .switch_ => |switch_term| blk: {
            var total: usize = "switch ".len + switch_term.selector.len + " {\n".len;
            for (switch_term.cases) |case| {
                total += "        ".len + case.value.len + " => @".len + case.target.len + "\n".len;
            }
            total += "        default => @".len + switch_term.default_target.len + "\n".len;
            total += "    }".len;
            break :blk total;
        },
        .return_ => |ret| "return ".len + ret.ptr.len + " ".len + ret.len.len,
        .revert => |revert| "revert ".len + revert.ptr.len + " ".len + revert.len.len,
        .stop => "stop".len,
        .invalid => "invalid".len,
        .selfdestruct => |beneficiary| "selfdestruct ".len + beneficiary.len,
        .iret => "iret".len,
    };
}

fn estimateNameListPrefixedSize(comptime prefix: []const u8, names: []const []const u8) usize {
    // Size twin of `writeNameListPrefixed`; the prefix is comptime for the same
    // specialization reason as the writer helper.
    if (names.len == 0) return 0;
    return prefix.len + estimateNameListSize(names);
}

fn estimateNameListSize(names: []const []const u8) usize {
    // Sum name bytes plus the single spaces inserted between adjacent names.
    if (names.len == 0) return 0;
    var total: usize = names.len - 1;
    for (names) |name| {
        total += name.len;
    }
    return total;
}

test "render normalizes parsed text" {
    const parser = @import("parser.zig");
    const diagnostics = @import("diagnostics.zig");
    const source =
        \\fn main:
        \\    entry {
        \\        zero = const 0x0
        \\        revert zero zero
        \\    }
    ;

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator, source, &bag);
    defer program.deinit();

    const rendered = try allocProgram(std.testing.allocator, program);
    defer std.testing.allocator.free(rendered);
    try std.testing.expectEqualStrings(
        \\fn main:
        \\    entry {
        \\        zero = const 0x0
        \\        revert zero zero
        \\    }
        \\
    , rendered);
}

test "render includes data segments" {
    const parser = @import("parser.zig");
    const diagnostics = @import("diagnostics.zig");
    const source =
        \\fn main:
        \\    entry {
        \\        stop
        \\    }
        \\
        \\data blob 0x11223344
    ;

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator, source, &bag);
    defer program.deinit();

    const rendered = try allocProgram(std.testing.allocator, program);
    defer std.testing.allocator.free(rendered);
    try std.testing.expectEqualStrings(
        \\fn main:
        \\    entry {
        \\        stop
        \\    }
        \\
        \\data blob 0x11223344
        \\
    , rendered);
}

test "render includes switch terminators" {
    const parser = @import("parser.zig");
    const diagnostics = @import("diagnostics.zig");
    const source =
        \\fn main:
        \\    entry {
        \\        selector = const 1
        \\        switch selector {
        \\        1 => @one
        \\        default => @other
        \\        }
        \\    }
        \\    one {
        \\        stop
        \\    }
        \\    other {
        \\        invalid
        \\    }
    ;

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator, source, &bag);
    defer program.deinit();

    const rendered = try allocProgram(std.testing.allocator, program);
    defer std.testing.allocator.free(rendered);
    try std.testing.expectEqualStrings(
        \\fn main:
        \\    entry {
        \\        selector = const 1
        \\        switch selector {
        \\        1 => @one
        \\        default => @other
        \\    }
        \\    }
        \\    one {
        \\        stop
        \\    }
        \\    other {
        \\        invalid
        \\    }
        \\
    , rendered);
}
