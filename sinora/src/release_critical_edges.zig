//! Release-only critical-edge splitting, mirroring Plank's release CLI pass.

const std = @import("std");
const ir = @import("ir.zig");

pub fn split(allocator: std.mem.Allocator, program: ir.Program) !ir.Program {
    var result = ir.Program.init(allocator);
    errdefer result.deinit();
    const arena = result.allocator();

    result.data_segments = try arena.dupe(ir.DataSegment, program.data_segments);

    const functions = try arena.alloc(ir.Function, program.functions.len);
    for (program.functions, functions) |function, *out_function| {
        out_function.* = try splitFunction(arena, function);
    }
    result.functions = functions;
    return result;
}

fn splitFunction(arena: std.mem.Allocator, function: ir.Function) !ir.Function {
    const predecessor_counts = try arena.alloc(usize, function.blocks.len);
    @memset(predecessor_counts, 0);
    for (function.blocks) |block| {
        var targets: [max_successors][]const u8 = undefined;
        for (successors(block, &targets)) |target| {
            if (blockIndex(function, target)) |index| {
                predecessor_counts[index] += 1;
            }
        }
    }

    var blocks: std.ArrayList(ir.Block) = .empty;
    var forwarding_blocks: std.ArrayList(ir.Block) = .empty;
    var split_index: usize = 0;

    for (function.blocks) |block| {
        var rewritten = block;
        rewritten.terminator = try splitTerminator(
            arena,
            function,
            block,
            predecessor_counts,
            &forwarding_blocks,
            &split_index,
        );
        try blocks.append(arena, rewritten);
    }
    try blocks.appendSlice(arena, forwarding_blocks.items);

    return .{
        .name = function.name,
        .blocks = try blocks.toOwnedSlice(arena),
        .line = function.line,
    };
}

fn splitTerminator(
    arena: std.mem.Allocator,
    function: ir.Function,
    source: ir.Block,
    predecessor_counts: []const usize,
    forwarding_blocks: *std.ArrayList(ir.Block),
    split_index: *usize,
) !ir.Terminator {
    return switch (source.terminator) {
        .branch => |branch| .{ .branch = .{
            .condition = branch.condition,
            .non_zero_target = try splitEdge(arena, function, source, branch.non_zero_target, predecessor_counts, forwarding_blocks, split_index),
            .zero_target = try splitEdge(arena, function, source, branch.zero_target, predecessor_counts, forwarding_blocks, split_index),
        } },
        .switch_ => |switch_term| blk: {
            const cases = try arena.alloc(ir.SwitchCase, switch_term.cases.len);
            for (switch_term.cases, cases) |case, *out_case| {
                out_case.* = case;
                out_case.target = try splitEdge(arena, function, source, case.target, predecessor_counts, forwarding_blocks, split_index);
            }
            const default_target = if (switch_term.default_target.len != 0)
                try splitEdge(arena, function, source, switch_term.default_target, predecessor_counts, forwarding_blocks, split_index)
            else
                switch_term.default_target;
            break :blk .{ .switch_ = .{
                .selector = switch_term.selector,
                .cases = cases,
                .default_target = default_target,
            } };
        },
        else => source.terminator,
    };
}

fn splitEdge(
    arena: std.mem.Allocator,
    function: ir.Function,
    source: ir.Block,
    target: []const u8,
    predecessor_counts: []const usize,
    forwarding_blocks: *std.ArrayList(ir.Block),
    split_index: *usize,
) ![]const u8 {
    const target_index = blockIndex(function, target) orelse return target;
    if (predecessor_counts[target_index] <= 1) return target;

    const forwarding_name = try std.fmt.allocPrint(
        arena,
        "__split_{s}_{s}_{d}",
        .{ source.name, target, split_index.* },
    );
    const names = try arena.alloc([]const u8, source.outputs.len);
    for (names, 0..) |*name, index| {
        name.* = try std.fmt.allocPrint(arena, "__split_{d}_v{d}", .{ split_index.*, index });
    }
    split_index.* += 1;

    try forwarding_blocks.append(arena, .{
        .name = forwarding_name,
        .inputs = names,
        .outputs = names,
        .instructions = &.{},
        .terminator = .{ .jump = target },
        .line = source.line,
    });
    return forwarding_name;
}

const max_successors = 128;

fn successors(block: ir.Block, storage: *[max_successors][]const u8) []const []const u8 {
    var count: usize = 0;
    switch (block.terminator) {
        .jump => |target| {
            storage[count] = target;
            count += 1;
        },
        .branch => |branch| {
            storage[count] = branch.zero_target;
            count += 1;
            storage[count] = branch.non_zero_target;
            count += 1;
        },
        .switch_ => |switch_term| {
            for (switch_term.cases) |case| {
                if (count >= storage.len) break;
                storage[count] = case.target;
                count += 1;
            }
            if (switch_term.default_target.len != 0 and count < storage.len) {
                storage[count] = switch_term.default_target;
                count += 1;
            }
        },
        .return_, .revert, .stop, .invalid, .selfdestruct, .iret => {},
    }
    return storage[0..count];
}

fn blockIndex(function: ir.Function, block_name: []const u8) ?usize {
    for (function.blocks, 0..) |block, index| {
        if (std.mem.eql(u8, block.name, block_name)) return index;
    }
    return null;
}

test "release critical edge splitting inserts forwarding block for shared target" {
    const diagnostics = @import("diagnostics.zig");
    const parser = @import("parser.zig");

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(
        std.testing.allocator,
        \\fn main:
        \\    entry {
        \\        cond = const 0x01
        \\        => cond ? @left : @join
        \\    }
        \\
        \\    left {
        \\        => @join
        \\    }
        \\
        \\    join {
        \\        stop
        \\    }
    ,
        &bag,
    );
    defer program.deinit();

    var normalized = try split(std.testing.allocator, program);
    defer normalized.deinit();

    try std.testing.expect(normalized.functions[0].blocks.len > program.functions[0].blocks.len);
}
