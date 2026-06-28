//! Pass framework for owned SIR transforms and optimizations.
//!
//! Plank's Rust pass manager has two core ideas worth preserving exactly:
//! mutating passes declare which analyses they preserve, and the runner
//! invalidates every other cached analysis immediately after the pass. This file
//! gives Sinora that contract for SSA, dominance, SCCP, DCE, and the owned pass
//! pipeline.

const std = @import("std");

const analyses = @import("analyses.zig");
const diagnostics = @import("diagnostics.zig");
const ir = @import("ir.zig");
const optimizations = @import("optimizations.zig");
const release_critical_edges = @import("release_critical_edges.zig");
const ssa_transform = @import("ssa_transform.zig");

pub const PassError = error{
    InvalidSir,
    InvalidOptimizationPass,
} || std.mem.Allocator.Error;

pub const OptimizationPass = enum {
    sccp,
    copy_propagation,
    literal_commoning,
    unused_operation_elimination,
    defragment,
    switch_peephole,

    pub fn fromChar(char: u8) ?OptimizationPass {
        return switch (char) {
            's' => .sccp,
            'c' => .copy_propagation,
            'p' => .literal_commoning,
            'u' => .unused_operation_elimination,
            'd' => .defragment,
            'l' => .switch_peephole,
            else => null,
        };
    }
};

pub const OptimizationPassFact = struct {
    pass: OptimizationPass,
    name: []const u8,
    cli_code: u8,
};

pub const optimization_pass_facts = [_]OptimizationPassFact{
    .{ .pass = .sccp, .name = "sccp", .cli_code = 's' },
    .{ .pass = .copy_propagation, .name = "copy_propagation", .cli_code = 'c' },
    .{ .pass = .literal_commoning, .name = "literal_commoning", .cli_code = 'p' },
    .{ .pass = .unused_operation_elimination, .name = "unused_operation_elimination", .cli_code = 'u' },
    .{ .pass = .defragment, .name = "defragment", .cli_code = 'd' },
    .{ .pass = .switch_peephole, .name = "switch_peephole", .cli_code = 'l' },
};

pub const optimization_pipeline_runs_final_legalizer = true;

pub const optimize_help =
    \\Optimization passes to run in order. Each character is a pass:
    \\  s = SCCP (constant propagation)
    \\  c = copy propagation
    \\  p = literal commoning
    \\  u = unused operation elimination
    \\  d = defragment
    \\  l = switch peephole
    \\Example: -O csud
;

pub fn parseOptimizationString(allocator: std.mem.Allocator, passes: []const u8) PassError![]const OptimizationPass {
    const parsed = try allocator.alloc(OptimizationPass, passes.len);
    errdefer allocator.free(parsed);
    for (passes, parsed) |char, *pass| {
        pass.* = OptimizationPass.fromChar(char) orelse return error.InvalidOptimizationPass;
    }
    return parsed;
}

pub fn runPass(
    comptime PassType: type,
    pass: *PassType,
    allocator: std.mem.Allocator,
    program: *ir.Program,
    store: *analyses.AnalysesStore,
) PassError!void {
    try pass.run(allocator, program, store);
    // A pass may return a completely new owned Program. The preservation mask is
    // the only contract the analysis cache can trust after that mutation.
    store.invalidateAllExcept(pass.preserves());
}

pub const PassManager = struct {
    allocator: std.mem.Allocator,
    program: *ir.Program,
    store: analyses.AnalysesStore,

    pub fn init(allocator: std.mem.Allocator, program: *ir.Program) PassManager {
        return .{
            .allocator = allocator,
            .program = program,
            .store = analyses.AnalysesStore.init(allocator),
        };
    }

    pub fn deinit(self: *PassManager) void {
        self.store.deinit();
        self.* = undefined;
    }

    pub fn runLegalize(self: *PassManager) PassError!void {
        var pass: LegalizerPass = .{};
        try runPass(LegalizerPass, &pass, self.allocator, self.program, &self.store);
    }

    pub fn runCriticalEdgeSplitting(self: *PassManager) PassError!void {
        var pass: CriticalEdgeSplittingPass = .{};
        try runPass(CriticalEdgeSplittingPass, &pass, self.allocator, self.program, &self.store);
    }

    pub fn runSsaTransform(self: *PassManager) PassError!void {
        var pass: SsaTransformPass = .{};
        try runPass(SsaTransformPass, &pass, self.allocator, self.program, &self.store);
    }

    pub fn runSccp(self: *PassManager) PassError!void {
        var pass: SccpPass = .{};
        try runPass(SccpPass, &pass, self.allocator, self.program, &self.store);
    }

    pub fn runCopyPropagation(self: *PassManager) PassError!void {
        var pass: CopyPropagationPass = .{};
        try runPass(CopyPropagationPass, &pass, self.allocator, self.program, &self.store);
    }

    pub fn runLiteralCommoning(self: *PassManager) PassError!void {
        var pass: LiteralCommoningPass = .{};
        try runPass(LiteralCommoningPass, &pass, self.allocator, self.program, &self.store);
    }

    pub fn runUnusedOperationElimination(self: *PassManager) PassError!void {
        var pass: UnusedOperationEliminationPass = .{};
        try runPass(UnusedOperationEliminationPass, &pass, self.allocator, self.program, &self.store);
    }

    pub fn runSwitchPeephole(self: *PassManager) PassError!void {
        var pass: SwitchPeepholePass = .{};
        try runPass(SwitchPeepholePass, &pass, self.allocator, self.program, &self.store);
    }

    pub fn runDefragmenter(self: *PassManager) PassError!void {
        var pass: DefragmenterPass = .{};
        try runPass(DefragmenterPass, &pass, self.allocator, self.program, &self.store);
    }

    fn runOptimizationPass(self: *PassManager, pass: OptimizationPass) PassError!void {
        switch (pass) {
            .sccp => try self.runSccp(),
            .copy_propagation => try self.runCopyPropagation(),
            .literal_commoning => try self.runLiteralCommoning(),
            .unused_operation_elimination => try self.runUnusedOperationElimination(),
            .defragment => try self.runDefragmenter(),
            .switch_peephole => try self.runSwitchPeephole(),
        }
    }

    pub fn runOptimizations(self: *PassManager, passes: []const OptimizationPass) PassError!void {
        for (passes) |pass| {
            try self.runOptimizationPass(pass);
        }
        // Optimizations intentionally compose loosely; the final legalizer is
        // the boundary that catches any malformed IR before codegen sees it.
        try self.runLegalize();
    }

    pub fn runOptimizationString(self: *PassManager, passes: []const u8) PassError!void {
        // Validate the full string before mutating the program. That preserves
        // `parseOptimizationString`'s all-or-nothing behavior without allocating
        // a temporary pass slice for CLI usage.
        for (passes) |char| {
            _ = OptimizationPass.fromChar(char) orelse return error.InvalidOptimizationPass;
        }
        for (passes) |char| {
            try self.runOptimizationPass(OptimizationPass.fromChar(char).?);
        }
        try self.runLegalize();
    }
};

pub const LegalizerPass = struct {
    pub fn run(
        self: *@This(),
        allocator: std.mem.Allocator,
        program: *ir.Program,
        store: *analyses.AnalysesStore,
    ) PassError!void {
        _ = self;
        _ = allocator;
        _ = store.legalizer(program.*) catch |err| switch (err) {
            error.InvalidSir => return error.InvalidSir,
            else => |other| return other,
        };
    }

    pub fn preserves(self: @This()) analyses.AnalysesMask {
        _ = self;
        return analyses.AnalysesMask.all();
    }
};

pub const CriticalEdgeSplittingPass = struct {
    pub fn run(
        self: *@This(),
        allocator: std.mem.Allocator,
        program: *ir.Program,
        store: *analyses.AnalysesStore,
    ) PassError!void {
        _ = self;
        _ = store;
        const normalized = try release_critical_edges.split(allocator, program.*);
        program.deinit();
        program.* = normalized;
    }

    pub fn preserves(self: @This()) analyses.AnalysesMask {
        _ = self;
        return analyses.AnalysesMask.empty();
    }
};

pub const SsaTransformPass = struct {
    pub fn run(
        self: *@This(),
        allocator: std.mem.Allocator,
        program: *ir.Program,
        store: *analyses.AnalysesStore,
    ) PassError!void {
        _ = self;
        _ = store;
        const transformed = ssa_transform.transform(allocator, program.*) catch |err| switch (err) {
            error.InvalidPreSsa, error.MissingBlock => return error.InvalidSir,
            else => |other| return other,
        };
        program.deinit();
        program.* = transformed;
    }

    pub fn preserves(self: @This()) analyses.AnalysesMask {
        _ = self;
        return analyses.AnalysesMask.empty();
    }
};

pub const SccpPass = struct {
    pub fn run(
        self: *@This(),
        allocator: std.mem.Allocator,
        program: *ir.Program,
        store: *analyses.AnalysesStore,
    ) PassError!void {
        _ = self;
        _ = store;
        const optimized = try optimizations.sccp(allocator, program.*);
        program.deinit();
        program.* = optimized;
    }

    pub fn preserves(self: @This()) analyses.AnalysesMask {
        _ = self;
        return analyses.AnalysesMask.empty();
    }
};

pub const CopyPropagationPass = struct {
    pub fn run(
        self: *@This(),
        allocator: std.mem.Allocator,
        program: *ir.Program,
        store: *analyses.AnalysesStore,
    ) PassError!void {
        _ = self;
        _ = store;
        const optimized = try optimizations.copyPropagation(allocator, program.*);
        program.deinit();
        program.* = optimized;
    }

    pub fn preserves(self: @This()) analyses.AnalysesMask {
        _ = self;
        return analyses.AnalysesMask.empty();
    }
};

pub const LiteralCommoningPass = struct {
    pub fn run(
        self: *@This(),
        allocator: std.mem.Allocator,
        program: *ir.Program,
        store: *analyses.AnalysesStore,
    ) PassError!void {
        _ = self;
        _ = store;
        const optimized = try optimizations.literalCommoning(allocator, program.*);
        program.deinit();
        program.* = optimized;
    }

    pub fn preserves(self: @This()) analyses.AnalysesMask {
        _ = self;
        return analyses.AnalysesMask.empty();
    }
};

pub const UnusedOperationEliminationPass = struct {
    pub fn run(
        self: *@This(),
        allocator: std.mem.Allocator,
        program: *ir.Program,
        store: *analyses.AnalysesStore,
    ) PassError!void {
        _ = self;
        _ = store;
        const optimized = try optimizations.unusedOperationElimination(allocator, program.*);
        program.deinit();
        program.* = optimized;
    }

    pub fn preserves(self: @This()) analyses.AnalysesMask {
        _ = self;
        return analyses.AnalysesMask.empty();
    }
};

pub const SwitchPeepholePass = struct {
    pub fn run(
        self: *@This(),
        allocator: std.mem.Allocator,
        program: *ir.Program,
        store: *analyses.AnalysesStore,
    ) PassError!void {
        _ = self;
        _ = store;
        const optimized = try optimizations.switchPeephole(allocator, program.*);
        program.deinit();
        program.* = optimized;
    }

    pub fn preserves(self: @This()) analyses.AnalysesMask {
        _ = self;
        return analyses.AnalysesMask.empty();
    }
};

pub const DefragmenterPass = struct {
    pub fn run(
        self: *@This(),
        allocator: std.mem.Allocator,
        program: *ir.Program,
        store: *analyses.AnalysesStore,
    ) PassError!void {
        _ = self;
        _ = store;
        const optimized = try optimizations.defragment(allocator, program.*);
        program.deinit();
        program.* = optimized;
    }

    pub fn preserves(self: @This()) analyses.AnalysesMask {
        _ = self;
        return analyses.AnalysesMask.empty();
    }
};

test "pass runner preserves analyses declared by non-mutating pass" {
    const parser = @import("parser.zig");

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator,
        \\fn main:
        \\    entry {
        \\        stop
        \\    }
    , &bag);
    defer program.deinit();

    var manager = PassManager.init(std.testing.allocator, &program);
    defer manager.deinit();

    _ = try manager.store.reachability(program);
    try std.testing.expect(manager.store.isValid(.reachability));
    try manager.runLegalize();
    try std.testing.expect(manager.store.isValid(.reachability));
}

test "critical edge splitting invalidates cached analyses" {
    const parser = @import("parser.zig");

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator,
        \\fn main:
        \\    entry {
        \\        cond = const 1
        \\        => cond ? @left : @join
        \\    }
        \\    left {
        \\        => @join
        \\    }
        \\    join {
        \\        stop
        \\    }
    , &bag);
    defer program.deinit();

    var manager = PassManager.init(std.testing.allocator, &program);
    defer manager.deinit();
    _ = try manager.store.reachability(program);
    try std.testing.expect(manager.store.isValid(.reachability));

    try manager.runCriticalEdgeSplitting();
    try std.testing.expect(!manager.store.isValid(.reachability));
    try std.testing.expect(program.functions[0].blocks.len > 3);
}

test "SSA transform pass renames duplicate locals and invalidates analyses" {
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

    var manager = PassManager.init(std.testing.allocator, &program);
    defer manager.deinit();
    _ = try manager.store.reachability(program);
    try manager.runSsaTransform();
    try std.testing.expect(!manager.store.isValid(.reachability));

    const join = program.functions[0].blocks[3];
    try std.testing.expectEqual(@as(usize, 1), join.inputs.len);
    try std.testing.expectEqualStrings(join.inputs[0], join.instructions[0].operands[0]);
}

test "optimization passes replace the program and invalidate analyses" {
    const parser = @import("parser.zig");

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator,
        \\fn main:
        \\    entry {
        \\        x = const 1
        \\        y = copy x
        \\        fold = add x x
        \\        copy_use = add y y
        \\        switch y {
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

    var manager = PassManager.init(std.testing.allocator, &program);
    defer manager.deinit();
    _ = try manager.store.reachability(program);

    try manager.runSccp();
    try std.testing.expect(!manager.store.isValid(.reachability));
    try std.testing.expectEqualStrings("const", program.functions[0].blocks[0].instructions[2].mnemonic);
    try std.testing.expectEqualStrings("2", program.functions[0].blocks[0].instructions[2].operands[0]);

    _ = try manager.store.reachability(program);
    try manager.runCopyPropagation();
    try std.testing.expect(!manager.store.isValid(.reachability));
    try std.testing.expectEqualStrings("x", program.functions[0].blocks[0].instructions[3].operands[0]);

    _ = try manager.store.reachability(program);
    try manager.runUnusedOperationElimination();
    try std.testing.expect(!manager.store.isValid(.reachability));
    try std.testing.expectEqualStrings("noop", program.functions[0].blocks[0].instructions[3].mnemonic);

    _ = try manager.store.reachability(program);
    try manager.runSwitchPeephole();
    try std.testing.expect(!manager.store.isValid(.reachability));
    try std.testing.expectEqualStrings("x", program.functions[0].blocks[0].terminator.branch.condition);
}

test "optimization string parses and runs passes in order" {
    const parser = @import("parser.zig");

    const parsed = try parseOptimizationString(std.testing.allocator, "scpudl");
    defer std.testing.allocator.free(parsed);
    try std.testing.expectEqualSlices(OptimizationPass, &.{
        .sccp,
        .copy_propagation,
        .literal_commoning,
        .unused_operation_elimination,
        .defragment,
        .switch_peephole,
    }, parsed);
    try std.testing.expectError(error.InvalidOptimizationPass, parseOptimizationString(std.testing.allocator, "x"));

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator,
        \\fn main:
        \\    entry {
        \\        x = const 1
        \\        y = copy x
        \\        switch y {
        \\        1 => @one
        \\        default => @other
        \\    }
        \\    }
        \\    one {
        \\        dead = const 42
        \\        stop
        \\    }
        \\    other {
        \\        cond = const 0
        \\        => cond ? @other_yes : @one
        \\    }
        \\    other_yes {
        \\        stop
        \\    }
    , &bag);
    defer program.deinit();

    var manager = PassManager.init(std.testing.allocator, &program);
    defer manager.deinit();
    try manager.runOptimizationString("csud");

    try std.testing.expectEqual(@as(usize, 1), program.functions.len);
    try std.testing.expectEqual(@as(usize, 2), program.functions[0].blocks.len);
    try std.testing.expectEqualStrings("one", program.functions[0].blocks[0].terminator.jump);
    try std.testing.expectEqual(@as(usize, 0), program.functions[0].blocks[0].instructions.len);
}

test "optimization pass fact table covers CLI parser vocabulary" {
    const fields = @typeInfo(OptimizationPass).@"enum".fields;
    try std.testing.expectEqual(fields.len, optimization_pass_facts.len);
    inline for (fields) |field| {
        const pass: OptimizationPass = @enumFromInt(field.value);
        var found = false;
        for (optimization_pass_facts) |fact| {
            if (fact.pass == pass) {
                try std.testing.expectEqual(pass, OptimizationPass.fromChar(fact.cli_code) orelse return error.TestUnexpectedResult);
                try std.testing.expectEqualStrings(field.name, fact.name);
                found = true;
                break;
            }
        }
        try std.testing.expect(found);
    }
}

test "defragmenter pass drops unreachable program fragments" {
    const parser = @import("parser.zig");

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator,
        \\fn init:
        \\    entry {
        \\        => @live
        \\    }
        \\    live {
        \\        stop
        \\    }
        \\    dead {
        \\        stop
        \\    }
        \\
        \\fn helper:
        \\    entry {
        \\        stop
        \\    }
    , &bag);
    defer program.deinit();

    var manager = PassManager.init(std.testing.allocator, &program);
    defer manager.deinit();
    _ = try manager.store.reachability(program);
    try manager.runDefragmenter();
    try std.testing.expect(!manager.store.isValid(.reachability));
    try std.testing.expectEqual(@as(usize, 1), program.functions.len);
    try std.testing.expectEqual(@as(usize, 2), program.functions[0].blocks.len);
}
