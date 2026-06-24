const std = @import("std");

pub const Program = struct {
    arena: std.heap.ArenaAllocator,
    functions: []const Function,
    data_segments: []const DataSegment,

    pub fn init(backing_allocator: std.mem.Allocator) Program {
        return .{
            .arena = std.heap.ArenaAllocator.init(backing_allocator),
            .functions = &.{},
            .data_segments = &.{},
        };
    }

    pub fn deinit(self: *Program) void {
        self.arena.deinit();
        self.* = undefined;
    }

    pub fn allocator(self: *Program) std.mem.Allocator {
        return self.arena.allocator();
    }

    pub fn stats(self: Program) Stats {
        var result: Stats = .{};
        result.functions = self.functions.len;
        result.data_segments = self.data_segments.len;
        for (self.data_segments) |segment| {
            result.data_bytes += segment.bytes.len;
        }
        for (self.functions) |function| {
            result.blocks += function.blocks.len;
            for (function.blocks) |block| {
                result.instructions += block.instructions.len;
                result.terminators += 1;
                if (block.terminator == .switch_) {
                    result.switches += 1;
                    result.switch_cases += block.terminator.switch_.cases.len;
                }
            }
        }
        return result;
    }
};

pub const Stats = struct {
    functions: usize = 0,
    data_segments: usize = 0,
    data_bytes: usize = 0,
    blocks: usize = 0,
    instructions: usize = 0,
    terminators: usize = 0,
    switches: usize = 0,
    switch_cases: usize = 0,
};

pub const Function = struct {
    name: []const u8,
    blocks: []const Block,
    line: u32,
};

pub const DataSegment = struct {
    name: []const u8,
    bytes: []const u8,
    line: u32,
};

pub const Block = struct {
    name: []const u8,
    inputs: []const []const u8,
    outputs: []const []const u8,
    instructions: []const Instruction,
    terminator: Terminator,
    line: u32,
};

pub const Instruction = struct {
    results: []const []const u8,
    mnemonic: []const u8,
    operands: []const []const u8,
    line: u32,
};

pub const SwitchCase = struct {
    value: []const u8,
    target: []const u8,
    line: u32,
};

pub const SwitchTerminator = struct {
    selector: []const u8,
    cases: []const SwitchCase,
    default_target: []const u8,
};

pub const Terminator = union(enum) {
    jump: []const u8,
    branch: struct {
        condition: []const u8,
        non_zero_target: []const u8,
        zero_target: []const u8,
    },
    switch_: SwitchTerminator,
    return_: struct {
        ptr: []const u8,
        len: []const u8,
    },
    revert: struct {
        ptr: []const u8,
        len: []const u8,
    },
    stop,
    invalid,
    selfdestruct: []const u8,
    iret,
};

/// Iterates a block's CFG successor block names with NO fixed-size buffer, so a
/// switch with any number of cases is enumerated without truncation. Order
/// matches Plank's successor iterator: jump target; branch zero-target then
/// non-zero-target; switch cases in order then the default.
pub const SuccessorIter = struct {
    block: Block,
    index: usize = 0,

    pub fn next(self: *SuccessorIter) ?[]const u8 {
        const i = self.index;
        switch (self.block.terminator) {
            .jump => |target| {
                if (i != 0) return null;
                self.index += 1;
                return target;
            },
            .branch => |branch| switch (i) {
                0 => {
                    self.index += 1;
                    return branch.zero_target;
                },
                1 => {
                    self.index += 1;
                    return branch.non_zero_target;
                },
                else => return null,
            },
            .switch_ => |switch_term| {
                if (i < switch_term.cases.len) {
                    self.index += 1;
                    return switch_term.cases[i].target;
                }
                if (i == switch_term.cases.len and switch_term.default_target.len != 0) {
                    self.index += 1;
                    return switch_term.default_target;
                }
                return null;
            },
            .return_, .revert, .stop, .invalid, .selfdestruct, .iret => return null,
        }
    }
};

pub fn successors(block: Block) SuccessorIter {
    return .{ .block = block };
}

test "successor iterator enumerates a large switch without truncation" {
    // Regression for the old fixed-[128] successor buffer that silently dropped
    // CFG edges past 128 targets. 200 cases + a default must all be enumerated.
    const case_count = 200;
    var cases: [case_count]SwitchCase = undefined;
    for (&cases, 0..) |*case, i| {
        // distinct, stable target name per case (index encoded in `value`)
        case.* = .{ .value = "0x0", .target = "t", .line = @intCast(i) };
    }
    const block: Block = .{
        .name = "entry",
        .inputs = &.{},
        .outputs = &.{},
        .instructions = &.{},
        .terminator = .{ .switch_ = .{ .selector = "s", .cases = &cases, .default_target = "fallback" } },
        .line = 0,
    };

    var it = successors(block);
    var seen: usize = 0;
    var last: []const u8 = "";
    while (it.next()) |target| {
        last = target;
        seen += 1;
    }
    try std.testing.expectEqual(@as(usize, case_count + 1), seen);
    try std.testing.expectEqualStrings("fallback", last);
}

test "successor iterator order: branch zero-target then non-zero-target" {
    const block: Block = .{
        .name = "b",
        .inputs = &.{},
        .outputs = &.{},
        .instructions = &.{},
        .terminator = .{ .branch = .{ .condition = "c", .non_zero_target = "nz", .zero_target = "z" } },
        .line = 0,
    };
    var it = successors(block);
    try std.testing.expectEqualStrings("z", it.next().?);
    try std.testing.expectEqualStrings("nz", it.next().?);
    try std.testing.expectEqual(@as(?[]const u8, null), it.next());
}
