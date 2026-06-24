//! Release-only critical-edge splitting, mirroring Plank's release CLI pass.
//!
//! The release stack scheduler expects value transfers across CFG edges to be
//! expressible as a block-local output shuffle followed by a single successor
//! input layout. A critical edge (source has multiple successors and target has
//! multiple predecessors) makes that ambiguous, so this pass inserts an empty
//! forwarding block on only those edges. The forwarding block owns the transfer
//! values and jumps to the original target.

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
    var block_indices = std.StringHashMap(usize).init(arena);
    defer block_indices.deinit();
    try block_indices.ensureTotalCapacity(@intCast(function.blocks.len));
    for (function.blocks, 0..) |block, index| {
        block_indices.putAssumeCapacity(block.name, index);
    }

    const predecessor_counts = try arena.alloc(usize, function.blocks.len);
    @memset(predecessor_counts, 0);
    for (function.blocks) |*block| {
        var it = ir.successors(block);
        while (it.next()) |target| {
            if (block_indices.get(target)) |index| {
                predecessor_counts[index] += 1;
            }
        }
    }

    var blocks: std.ArrayList(ir.Block) = .empty;
    var forwarding_blocks: std.ArrayList(ir.Block) = .empty;
    try blocks.ensureTotalCapacity(arena, function.blocks.len);
    var split_index: usize = 0;

    for (function.blocks) |block| {
        var rewritten = block;
        rewritten.terminator = try splitTerminator(
            arena,
            block,
            &block_indices,
            predecessor_counts,
            &forwarding_blocks,
            &split_index,
        );
        blocks.appendAssumeCapacity(rewritten);
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
    source: ir.Block,
    block_indices: *const std.StringHashMap(usize),
    predecessor_counts: []const usize,
    forwarding_blocks: *std.ArrayList(ir.Block),
    split_index: *usize,
) !ir.Terminator {
    return switch (source.terminator) {
        .branch => |branch| .{ .branch = .{
            .condition = branch.condition,
            .non_zero_target = try splitEdge(arena, source, block_indices, branch.non_zero_target, predecessor_counts, forwarding_blocks, split_index),
            .zero_target = try splitEdge(arena, source, block_indices, branch.zero_target, predecessor_counts, forwarding_blocks, split_index),
        } },
        .switch_ => |switch_term| blk: {
            const cases = try arena.alloc(ir.SwitchCase, switch_term.cases.len);
            for (switch_term.cases, cases) |case, *out_case| {
                out_case.* = case;
                out_case.target = try splitEdge(arena, source, block_indices, case.target, predecessor_counts, forwarding_blocks, split_index);
            }
            const default_target = if (switch_term.default_target.len != 0)
                try splitEdge(arena, source, block_indices, switch_term.default_target, predecessor_counts, forwarding_blocks, split_index)
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
    source: ir.Block,
    block_indices: *const std.StringHashMap(usize),
    target: []const u8,
    predecessor_counts: []const usize,
    forwarding_blocks: *std.ArrayList(ir.Block),
    split_index: *usize,
) ![]const u8 {
    const target_index = block_indices.get(target) orelse return target;
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
