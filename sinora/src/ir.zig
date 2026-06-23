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
