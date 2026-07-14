//! Lazy SIR analyses and invalidation masks.
//!
//! This is the Sinora equivalent of Plank's `passes/src/analyses/cache.rs`
//! foundation. The important contract is not the individual analyses by
//! themselves; it is that mutating passes can say exactly which facts they
//! preserve, and every other cached fact is invalidated before later passes or
//! codegen can accidentally trust stale CFG information.

const std = @import("std");

const diag_mod = @import("diagnostics.zig");
const effects = @import("effects.zig");
const ir = @import("ir.zig");
const legality = @import("legality.zig");
const ops = @import("ops.zig");

pub const BlockRef = struct {
    function: usize,
    block: usize,
};

// Keep this enum in lock-step with `AnalysesStore` fields. The enum value is
// also the bit position inside `AnalysesMask`, so adding an analysis requires
// adding the cache field, accessors, and invalidation wiring below.
pub const AnalysisKind = enum(u5) {
    program_index,
    reachability,
    predecessors,
    reverse_post_order,
    dominators,
    dominance_frontiers,
    def_use,
    local_liveness,
    allocation_liveness,
    function_effects,
    basic_block_ownership,
    cfg_in_out_bundling,
    legalizer,
};

pub const AnalysesMask = struct {
    bits: u32 = 0,

    pub fn empty() AnalysesMask {
        return .{};
    }

    pub fn all() AnalysesMask {
        return .{ .bits = (1 << analysis_count) - 1 };
    }

    pub fn only(kind: AnalysisKind) AnalysesMask {
        return .{ .bits = bit(kind) };
    }

    pub fn with(self: AnalysesMask, kind: AnalysisKind) AnalysesMask {
        return .{ .bits = self.bits | bit(kind) };
    }

    pub fn contains(self: AnalysesMask, kind: AnalysisKind) bool {
        return (self.bits & bit(kind)) != 0;
    }

    fn bit(kind: AnalysisKind) u32 {
        return @as(u32, 1) << @intFromEnum(kind);
    }
};

const analysis_count = @typeInfo(AnalysisKind).@"enum".fields.len;
comptime {
    std.debug.assert(analysis_count <= @bitSizeOf(u32));
}

pub const AnalysesStore = struct {
    allocator: std.mem.Allocator,
    program_index_cache: Cached(ProgramIndex) = .{},
    reachability_cache: Cached(Reachability) = .{},
    predecessors_cache: Cached(Predecessors) = .{},
    reverse_post_order_cache: Cached(ReversePostOrder) = .{},
    dominators_cache: Cached(Dominators) = .{},
    dominance_frontiers_cache: Cached(DominanceFrontiers) = .{},
    def_use_cache: Cached(DefUse) = .{},
    local_liveness_cache: Cached(LocalLiveness) = .{},
    allocation_liveness_cache: Cached(AllocationLiveness) = .{},
    function_effects_cache: Cached(FunctionEffects) = .{},
    basic_block_ownership_cache: Cached(BasicBlockOwnership) = .{},
    cfg_in_out_bundling_cache: Cached(CfgInOutBundling) = .{},
    legalizer_cache: Cached(Legalizer) = .{},

    pub fn init(allocator: std.mem.Allocator) AnalysesStore {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *AnalysesStore) void {
        self.program_index_cache.deinit(self.allocator);
        self.reachability_cache.deinit(self.allocator);
        self.predecessors_cache.deinit(self.allocator);
        self.reverse_post_order_cache.deinit(self.allocator);
        self.dominators_cache.deinit(self.allocator);
        self.dominance_frontiers_cache.deinit(self.allocator);
        self.def_use_cache.deinit(self.allocator);
        self.local_liveness_cache.deinit(self.allocator);
        self.allocation_liveness_cache.deinit(self.allocator);
        self.function_effects_cache.deinit(self.allocator);
        self.basic_block_ownership_cache.deinit(self.allocator);
        self.cfg_in_out_bundling_cache.deinit(self.allocator);
        self.legalizer_cache.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn invalidateAllExcept(self: *AnalysesStore, preserved: AnalysesMask) void {
        if (!preserved.contains(.program_index)) self.program_index_cache.invalidate();
        if (!preserved.contains(.reachability)) self.reachability_cache.invalidate();
        if (!preserved.contains(.predecessors)) self.predecessors_cache.invalidate();
        if (!preserved.contains(.reverse_post_order)) self.reverse_post_order_cache.invalidate();
        if (!preserved.contains(.dominators)) self.dominators_cache.invalidate();
        if (!preserved.contains(.dominance_frontiers)) self.dominance_frontiers_cache.invalidate();
        if (!preserved.contains(.def_use)) self.def_use_cache.invalidate();
        if (!preserved.contains(.local_liveness)) self.local_liveness_cache.invalidate();
        if (!preserved.contains(.allocation_liveness)) self.allocation_liveness_cache.invalidate();
        if (!preserved.contains(.function_effects)) self.function_effects_cache.invalidate();
        if (!preserved.contains(.basic_block_ownership)) self.basic_block_ownership_cache.invalidate();
        if (!preserved.contains(.cfg_in_out_bundling)) self.cfg_in_out_bundling_cache.invalidate();
        if (!preserved.contains(.legalizer)) self.legalizer_cache.invalidate();
    }

    pub fn isValid(self: AnalysesStore, kind: AnalysisKind) bool {
        return switch (kind) {
            .program_index => self.program_index_cache.valid,
            .reachability => self.reachability_cache.valid,
            .predecessors => self.predecessors_cache.valid,
            .reverse_post_order => self.reverse_post_order_cache.valid,
            .dominators => self.dominators_cache.valid,
            .dominance_frontiers => self.dominance_frontiers_cache.valid,
            .def_use => self.def_use_cache.valid,
            .local_liveness => self.local_liveness_cache.valid,
            .allocation_liveness => self.allocation_liveness_cache.valid,
            .function_effects => self.function_effects_cache.valid,
            .basic_block_ownership => self.basic_block_ownership_cache.valid,
            .cfg_in_out_bundling => self.cfg_in_out_bundling_cache.valid,
            .legalizer => self.legalizer_cache.valid,
        };
    }

    pub fn programIndex(self: *AnalysesStore, program: ir.Program) !*const ProgramIndex {
        return self.program_index_cache.get(self.allocator, program, self);
    }

    pub fn reachability(self: *AnalysesStore, program: ir.Program) !*const Reachability {
        return self.reachability_cache.get(self.allocator, program, self);
    }

    pub fn predecessors(self: *AnalysesStore, program: ir.Program) !*const Predecessors {
        return self.predecessors_cache.get(self.allocator, program, self);
    }

    pub fn reversePostOrder(self: *AnalysesStore, program: ir.Program) !*const ReversePostOrder {
        return self.reverse_post_order_cache.get(self.allocator, program, self);
    }

    pub fn dominators(self: *AnalysesStore, program: ir.Program) !*const Dominators {
        return self.dominators_cache.get(self.allocator, program, self);
    }

    pub fn dominanceFrontiers(self: *AnalysesStore, program: ir.Program) !*const DominanceFrontiers {
        return self.dominance_frontiers_cache.get(self.allocator, program, self);
    }

    pub fn defUse(self: *AnalysesStore, program: ir.Program) !*const DefUse {
        return self.def_use_cache.get(self.allocator, program, self);
    }

    pub fn localLiveness(self: *AnalysesStore, program: ir.Program) !*const LocalLiveness {
        return self.local_liveness_cache.get(self.allocator, program, self);
    }

    pub fn allocationLiveness(self: *AnalysesStore, program: ir.Program) !*const AllocationLiveness {
        return self.allocation_liveness_cache.get(self.allocator, program, self);
    }

    pub fn functionEffects(self: *AnalysesStore, program: ir.Program) !*const FunctionEffects {
        return self.function_effects_cache.get(self.allocator, program, self);
    }

    pub fn basicBlockOwnership(self: *AnalysesStore, program: ir.Program) !*const BasicBlockOwnership {
        return self.basic_block_ownership_cache.get(self.allocator, program, self);
    }

    pub fn cfgInOutBundling(self: *AnalysesStore, program: ir.Program) !*const CfgInOutBundling {
        return self.cfg_in_out_bundling_cache.get(self.allocator, program, self);
    }

    pub fn legalizer(self: *AnalysesStore, program: ir.Program) !*const Legalizer {
        return self.legalizer_cache.get(self.allocator, program, self);
    }
};

fn Cached(comptime T: type) type {
    return struct {
        value: ?T = null,
        valid: bool = false,

        const Self = @This();

        fn get(
            self: *Self,
            allocator: std.mem.Allocator,
            program: ir.Program,
            store: *AnalysesStore,
        ) !*const T {
            if (!self.valid) {
                // Stale values are destroyed lazily. Invalidating a cache is a
                // cheap bit flip, which keeps pass execution cheap even when a
                // pass invalidates almost every analysis.
                if (self.value) |*old| old.deinit(allocator);
                self.value = null;
                self.value = try T.compute(allocator, program, store);
                self.valid = true;
            }
            return &self.value.?;
        }

        fn invalidate(self: *Self) void {
            self.valid = false;
        }

        fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            if (self.value) |*value| value.deinit(allocator);
            self.* = .{};
        }
    };
}

const BlockShape = struct {
    starts: []usize,
    block_count: usize,

    // Sinora uses named blocks inside each function, but analyses need dense
    // arrays for speed. `BlockShape` gives every `(function, block)` pair one
    // stable global index without mutating the IR.
    fn build(allocator: std.mem.Allocator, program: ir.Program) !BlockShape {
        const starts = try allocator.alloc(usize, program.functions.len);
        var count: usize = 0;
        for (program.functions, 0..) |function, index| {
            starts[index] = count;
            count += function.blocks.len;
        }
        return .{ .starts = starts, .block_count = count };
    }

    fn deinit(self: *BlockShape, allocator: std.mem.Allocator) void {
        allocator.free(self.starts);
        self.* = undefined;
    }

    fn global(self: BlockShape, ref: BlockRef) usize {
        return self.starts[ref.function] + ref.block;
    }
};

pub const ProgramIndex = struct {
    functions: std.StringHashMap(usize),
    blocks: []std.StringHashMap(usize),

    pub fn compute(allocator: std.mem.Allocator, program: ir.Program, store: *AnalysesStore) !ProgramIndex {
        _ = store;

        var functions = std.StringHashMap(usize).init(allocator);
        errdefer functions.deinit();
        try functions.ensureTotalCapacity(capacityLen(program.functions.len));

        const blocks = try allocator.alloc(std.StringHashMap(usize), program.functions.len);
        var initialized_count: usize = 0;
        errdefer {
            for (blocks[0..initialized_count]) |*map| map.deinit();
            allocator.free(blocks);
        }

        for (program.functions, 0..) |function, function_index| {
            try functions.put(function.name, function_index);

            blocks[function_index] = std.StringHashMap(usize).init(allocator);
            initialized_count += 1;
            try blocks[function_index].ensureTotalCapacity(capacityLen(function.blocks.len));
            for (function.blocks, 0..) |block, block_index| {
                try blocks[function_index].put(block.name, block_index);
            }
        }

        return .{
            .functions = functions,
            .blocks = blocks,
        };
    }

    pub fn deinit(self: *ProgramIndex, allocator: std.mem.Allocator) void {
        self.functions.deinit();
        for (self.blocks) |*map| map.deinit();
        allocator.free(self.blocks);
        self.* = undefined;
    }

    pub fn functionIndex(self: ProgramIndex, name: []const u8) ?usize {
        return self.functions.get(name);
    }

    pub fn blockIndex(self: ProgramIndex, function_index: usize, name: []const u8) ?usize {
        if (function_index >= self.blocks.len) return null;
        return self.blocks[function_index].get(name);
    }
};

pub const Reachability = struct {
    shape: BlockShape,
    reachable: []bool,

    pub fn compute(allocator: std.mem.Allocator, program: ir.Program, store: *AnalysesStore) !Reachability {
        const index = try store.programIndex(program);
        var shape = try BlockShape.build(allocator, program);
        errdefer shape.deinit(allocator);
        const reachable = try allocator.alloc(bool, shape.block_count);
        errdefer allocator.free(reachable);
        @memset(reachable, false);

        var result: Reachability = .{ .shape = shape, .reachable = reachable };
        for (program.functions, 0..) |function, function_index| {
            if (function.blocks.len == 0) continue;
            result.mark(program, index, .{ .function = function_index, .block = 0 });
        }
        return result;
    }

    pub fn deinit(self: *Reachability, allocator: std.mem.Allocator) void {
        self.shape.deinit(allocator);
        allocator.free(self.reachable);
        self.* = undefined;
    }

    pub fn contains(self: Reachability, ref: BlockRef) bool {
        return self.reachable[self.shape.global(ref)];
    }

    fn mark(self: *Reachability, program: ir.Program, index: *const ProgramIndex, ref: BlockRef) void {
        const global = self.shape.global(ref);
        if (self.reachable[global]) return;
        self.reachable[global] = true;

        const function = program.functions[ref.function];
        var it = ir.successors(&function.blocks[ref.block]);
        while (it.next()) |target| {
            if (index.blockIndex(ref.function, target)) |successor| {
                self.mark(program, index, .{ .function = ref.function, .block = successor });
            }
        }
    }
};

pub const Predecessors = struct {
    shape: BlockShape,
    lists: []const []const BlockRef,

    pub fn compute(allocator: std.mem.Allocator, program: ir.Program, store: *AnalysesStore) !Predecessors {
        const index = try store.programIndex(program);
        const reachability = try store.reachability(program);
        var shape = try BlockShape.build(allocator, program);
        errdefer shape.deinit(allocator);

        var builders = try allocator.alloc(std.ArrayList(BlockRef), shape.block_count);
        defer allocator.free(builders);
        for (builders) |*builder| builder.* = .empty;
        defer for (builders) |*builder| builder.deinit(allocator);

        for (program.functions, 0..) |function, function_index| {
            for (function.blocks, 0..) |block, block_index| {
                const source: BlockRef = .{ .function = function_index, .block = block_index };
                if (!reachability.contains(source)) continue;

                var it = ir.successors(&block);
                while (it.next()) |target| {
                    const target_index = index.blockIndex(function_index, target) orelse continue;
                    const target_ref: BlockRef = .{ .function = function_index, .block = target_index };
                    try builders[shape.global(target_ref)].append(allocator, source);
                }
            }
        }

        const lists = try allocator.alloc([]const BlockRef, shape.block_count);
        @memset(lists, &.{});
        errdefer {
            for (lists) |list| allocator.free(list);
            allocator.free(lists);
        }
        for (builders, lists) |*builder, *list| {
            list.* = try builder.toOwnedSlice(allocator);
        }
        return .{ .shape = shape, .lists = lists };
    }

    pub fn deinit(self: *Predecessors, allocator: std.mem.Allocator) void {
        for (self.lists) |list| allocator.free(list);
        allocator.free(self.lists);
        self.shape.deinit(allocator);
        self.* = undefined;
    }

    pub fn of(self: Predecessors, ref: BlockRef) []const BlockRef {
        return self.lists[self.shape.global(ref)];
    }
};

pub const ReversePostOrder = struct {
    order: []const BlockRef,
    function_starts: []usize,
    function_counts: []usize,

    pub fn compute(allocator: std.mem.Allocator, program: ir.Program, store: *AnalysesStore) !ReversePostOrder {
        const index = try store.programIndex(program);
        const shape = try BlockShape.build(allocator, program);
        defer {
            var mutable_shape = shape;
            mutable_shape.deinit(allocator);
        }

        var order_builder: std.ArrayList(BlockRef) = .empty;
        defer order_builder.deinit(allocator);
        try order_builder.ensureTotalCapacity(allocator, shape.block_count);

        const function_starts = try allocator.alloc(usize, program.functions.len);
        errdefer allocator.free(function_starts);
        const function_counts = try allocator.alloc(usize, program.functions.len);
        errdefer allocator.free(function_counts);

        const visited = try allocator.alloc(bool, shape.block_count);
        defer allocator.free(visited);
        @memset(visited, false);

        // RPO is per function. This mirrors Plank's global RPO but keeps cheap
        // slices for function-local algorithms such as dominators and SSA.
        for (program.functions, 0..) |program_function, function_index| {
            function_starts[function_index] = order_builder.items.len;
            if (program_function.blocks.len != 0) {
                try dfsPostorder(allocator, program, index, shape, &order_builder, visited, .{
                    .function = function_index,
                    .block = 0,
                });
                std.mem.reverse(BlockRef, order_builder.items[function_starts[function_index]..]);
            }
            function_counts[function_index] = order_builder.items.len - function_starts[function_index];
        }

        return .{
            .order = try order_builder.toOwnedSlice(allocator),
            .function_starts = function_starts,
            .function_counts = function_counts,
        };
    }

    pub fn deinit(self: *ReversePostOrder, allocator: std.mem.Allocator) void {
        allocator.free(self.order);
        allocator.free(self.function_starts);
        allocator.free(self.function_counts);
        self.* = undefined;
    }

    pub fn global(self: ReversePostOrder) []const BlockRef {
        return self.order;
    }

    pub fn function(self: ReversePostOrder, function_index: usize) []const BlockRef {
        const start = self.function_starts[function_index];
        return self.order[start..][0..self.function_counts[function_index]];
    }
};

pub const UseKind = union(enum) {
    operation: usize,
    terminator,
    block_output: usize,
};

pub const UseLocation = struct {
    block: BlockRef,
    kind: UseKind,
};

pub const DefUse = struct {
    uses: std.StringHashMap(std.ArrayList(UseLocation)),

    pub fn compute(allocator: std.mem.Allocator, program: ir.Program, store: *AnalysesStore) !DefUse {
        _ = store;
        var result = DefUse{ .uses = std.StringHashMap(std.ArrayList(UseLocation)).init(allocator) };
        errdefer result.deinit(allocator);
        const estimated_uses = valueUseEstimate(program);
        try result.uses.ensureTotalCapacity(if (estimated_uses > std.math.maxInt(u32))
            std.math.maxInt(u32)
        else
            @intCast(estimated_uses));

        for (program.functions, 0..) |function, function_index| {
            for (function.blocks, 0..) |block, block_index| {
                const block_ref: BlockRef = .{ .function = function_index, .block = block_index };
                for (block.instructions, 0..) |instruction, instruction_index| {
                    const spec = ops.lookup(instruction.mnemonic);
                    for (instruction.operands, 0..) |operand, operand_index| {
                        if (!operandRequiresValue(spec, operand_index, operand)) continue;
                        try result.addUse(allocator, operand, .{
                            .block = block_ref,
                            .kind = .{ .operation = instruction_index },
                        });
                    }
                }

                switch (block.terminator) {
                    .jump, .stop, .invalid, .iret => {},
                    .branch => |branch| try result.addUse(allocator, branch.condition, .{
                        .block = block_ref,
                        .kind = .terminator,
                    }),
                    .switch_ => |switch_term| try result.addUse(allocator, switch_term.selector, .{
                        .block = block_ref,
                        .kind = .terminator,
                    }),
                    .return_ => |ret| {
                        try result.addUse(allocator, ret.ptr, .{ .block = block_ref, .kind = .terminator });
                        try result.addUse(allocator, ret.len, .{ .block = block_ref, .kind = .terminator });
                    },
                    .revert => |revert| {
                        try result.addUse(allocator, revert.ptr, .{ .block = block_ref, .kind = .terminator });
                        try result.addUse(allocator, revert.len, .{ .block = block_ref, .kind = .terminator });
                    },
                    .selfdestruct => |beneficiary| try result.addUse(allocator, beneficiary, .{
                        .block = block_ref,
                        .kind = .terminator,
                    }),
                }

                for (block.outputs, 0..) |output, output_index| {
                    try result.addUse(allocator, output, .{
                        .block = block_ref,
                        .kind = .{ .block_output = output_index },
                    });
                }
            }
        }

        return result;
    }

    pub fn deinit(self: *DefUse, allocator: std.mem.Allocator) void {
        var values = self.uses.valueIterator();
        while (values.next()) |locations| locations.deinit(allocator);
        self.uses.deinit();
        self.* = undefined;
    }

    pub fn usesOf(self: DefUse, value: []const u8) []const UseLocation {
        const locations = self.uses.get(value) orelse return &.{};
        return locations.items;
    }

    fn addUse(self: *DefUse, allocator: std.mem.Allocator, value: []const u8, location: UseLocation) !void {
        const entry = try self.uses.getOrPut(value);
        if (!entry.found_existing) entry.value_ptr.* = .empty;
        try entry.value_ptr.append(allocator, location);
    }
};

pub const Dominators = struct {
    shape: BlockShape,
    idoms: []?BlockRef,

    pub fn compute(allocator: std.mem.Allocator, program: ir.Program, store: *AnalysesStore) !Dominators {
        const predecessors = try store.predecessors(program);
        const rpo = try store.reversePostOrder(program);
        var shape = try BlockShape.build(allocator, program);
        errdefer shape.deinit(allocator);
        const idoms = try allocator.alloc(?BlockRef, shape.block_count);
        errdefer allocator.free(idoms);
        @memset(idoms, null);

        var result = Dominators{ .shape = shape, .idoms = idoms };
        for (program.functions, 0..) |function, function_index| {
            if (function.blocks.len == 0) continue;
            try result.computeFunction(allocator, predecessors, rpo.function(function_index), function_index);
        }
        return result;
    }

    pub fn deinit(self: *Dominators, allocator: std.mem.Allocator) void {
        self.shape.deinit(allocator);
        allocator.free(self.idoms);
        self.* = undefined;
    }

    pub fn of(self: Dominators, ref: BlockRef) ?BlockRef {
        return self.idoms[self.shape.global(ref)];
    }

    fn computeFunction(
        self: *Dominators,
        allocator: std.mem.Allocator,
        predecessors: *const Predecessors,
        rpo: []const BlockRef,
        function_index: usize,
    ) !void {
        if (rpo.len == 0) return;
        const entry = BlockRef{ .function = function_index, .block = 0 };
        self.idoms[self.shape.global(entry)] = entry;

        // Cooper-Harvey-Kennedy style iterative dominators. `rpo_pos` lets the
        // `intersect` walk climb toward the entry using dense array lookups
        // instead of comparing named blocks or repeatedly scanning RPO.
        const rpo_pos = try allocator.alloc(usize, self.shape.block_count);
        defer allocator.free(rpo_pos);
        @memset(rpo_pos, std.math.maxInt(usize));
        for (rpo, 0..) |block, position| {
            rpo_pos[self.shape.global(block)] = position;
        }

        var changed = true;
        while (changed) {
            changed = false;
            for (rpo[1..]) |block| {
                const preds = predecessors.of(block);
                var new_idom: ?BlockRef = null;
                for (preds) |pred| {
                    if (self.of(pred) == null) continue;
                    new_idom = if (new_idom) |current|
                        self.intersect(pred, current, rpo_pos)
                    else
                        pred;
                }
                const resolved = new_idom orelse continue;
                const slot = self.shape.global(block);
                if (!optionalBlockRefEql(self.idoms[slot], resolved)) {
                    self.idoms[slot] = resolved;
                    changed = true;
                }
            }
        }
    }

    fn intersect(self: Dominators, lhs: BlockRef, rhs: BlockRef, rpo_pos: []const usize) BlockRef {
        var finger1 = lhs;
        var finger2 = rhs;
        while (!blockRefEql(finger1, finger2)) {
            while (rpo_pos[self.shape.global(finger1)] > rpo_pos[self.shape.global(finger2)]) {
                finger1 = self.of(finger1).?;
            }
            while (rpo_pos[self.shape.global(finger2)] > rpo_pos[self.shape.global(finger1)]) {
                finger2 = self.of(finger2).?;
            }
        }
        return finger1;
    }
};

pub const DominanceFrontiers = struct {
    shape: BlockShape,
    sets: []std.ArrayList(BlockRef),

    pub fn compute(allocator: std.mem.Allocator, program: ir.Program, store: *AnalysesStore) !DominanceFrontiers {
        const dominators = try store.dominators(program);
        const predecessors = try store.predecessors(program);
        var shape = try BlockShape.build(allocator, program);
        errdefer shape.deinit(allocator);

        const sets = try allocator.alloc(std.ArrayList(BlockRef), shape.block_count);
        errdefer allocator.free(sets);
        for (sets) |*set| set.* = .empty;
        errdefer for (sets) |*set| set.deinit(allocator);

        var result = DominanceFrontiers{ .shape = shape, .sets = sets };
        // Standard dominance-frontier construction: for every join block, walk
        // from each predecessor up to the immediate dominator of the join and
        // record the join in each runner's frontier.
        for (program.functions, 0..) |function, function_index| {
            for (function.blocks, 0..) |_, block_index| {
                const block = BlockRef{ .function = function_index, .block = block_index };
                const preds = predecessors.of(block);
                if (preds.len < 2) continue;
                const idom = dominators.of(block) orelse continue;
                for (preds) |pred| {
                    if (dominators.of(pred) == null) continue;
                    var runner = pred;
                    while (!blockRefEql(runner, idom)) {
                        try result.add(allocator, runner, block);
                        runner = dominators.of(runner) orelse break;
                    }
                }
            }
        }
        return result;
    }

    pub fn deinit(self: *DominanceFrontiers, allocator: std.mem.Allocator) void {
        for (self.sets) |*set| set.deinit(allocator);
        allocator.free(self.sets);
        self.shape.deinit(allocator);
        self.* = undefined;
    }

    pub fn of(self: DominanceFrontiers, ref: BlockRef) []const BlockRef {
        return self.sets[self.shape.global(ref)].items;
    }

    fn add(self: *DominanceFrontiers, allocator: std.mem.Allocator, owner: BlockRef, frontier: BlockRef) !void {
        const set = &self.sets[self.shape.global(owner)];
        for (set.items) |existing| {
            if (blockRefEql(existing, frontier)) return;
        }
        try set.append(allocator, frontier);
    }
};

pub const IntervalStart = union(enum) {
    block_start,
    instruction: usize,
};

pub const IntervalEnd = union(enum) {
    instruction: usize,
    block_end,
};

pub const Interval = struct {
    start: IntervalStart,
    end: IntervalEnd,
};

pub const LocalInterval = struct {
    block: BlockRef,
    interval: Interval,
};

const LiveFacts = struct {
    live_in: std.ArrayList([]const u8) = .empty,
    live_out: std.ArrayList([]const u8) = .empty,

    fn deinit(self: *LiveFacts, allocator: std.mem.Allocator) void {
        self.live_out.deinit(allocator);
        self.live_in.deinit(allocator);
        self.* = undefined;
    }
};

pub const LocalLiveness = struct {
    shape: BlockShape,
    facts: []LiveFacts,
    intervals: std.StringHashMap(std.ArrayList(LocalInterval)),

    pub fn compute(allocator: std.mem.Allocator, program: ir.Program, store: *AnalysesStore) !LocalLiveness {
        const predecessors = try store.predecessors(program);
        const rpo = try store.reversePostOrder(program);
        var shape = try BlockShape.build(allocator, program);
        errdefer shape.deinit(allocator);

        const facts = try allocator.alloc(LiveFacts, shape.block_count);
        errdefer allocator.free(facts);
        for (facts) |*fact| fact.* = .{};
        errdefer for (facts) |*fact| fact.deinit(allocator);

        var result = LocalLiveness{
            .shape = shape,
            .facts = facts,
            .intervals = std.StringHashMap(std.ArrayList(LocalInterval)).init(allocator),
        };
        errdefer result.deinit(allocator);

        try result.computeLiveness(allocator, program, predecessors, rpo);
        try result.computeIntervals(allocator, program);
        return result;
    }

    pub fn deinit(self: *LocalLiveness, allocator: std.mem.Allocator) void {
        var values = self.intervals.valueIterator();
        while (values.next()) |intervals| intervals.deinit(allocator);
        self.intervals.deinit();
        for (self.facts) |*fact| fact.deinit(allocator);
        allocator.free(self.facts);
        self.shape.deinit(allocator);
        self.* = undefined;
    }

    pub fn liveAtEntry(self: LocalLiveness, ref: BlockRef) []const []const u8 {
        return self.facts[self.shape.global(ref)].live_in.items;
    }

    pub fn liveAtExit(self: LocalLiveness, ref: BlockRef) []const []const u8 {
        return self.facts[self.shape.global(ref)].live_out.items;
    }

    pub fn intervalsOf(self: LocalLiveness, local: []const u8) []const LocalInterval {
        const intervals = self.intervals.get(local) orelse return &.{};
        return intervals.items;
    }

    fn computeLiveness(
        self: *LocalLiveness,
        allocator: std.mem.Allocator,
        program: ir.Program,
        predecessors: *const Predecessors,
        rpo: *const ReversePostOrder,
    ) !void {
        // Fixed-point backwards dataflow. The temporary vector is reused across
        // every block/iteration because this loop is hot on dispatcher-heavy
        // corpora and the live set is usually small.
        var next_live_in: std.ArrayList([]const u8) = .empty;
        defer next_live_in.deinit(allocator);

        var changed = true;
        while (changed) {
            changed = false;
            var order_index = rpo.global().len;
            while (order_index > 0) {
                order_index -= 1;
                const block_ref = rpo.global()[order_index];
                const block = program.functions[block_ref.function].blocks[block_ref.block];
                const global = self.shape.global(block_ref);

                next_live_in.clearRetainingCapacity();
                try computeBlockEntryLiveness(allocator, block, self.facts[global].live_out.items, &next_live_in);
                if (!sameNameSet(self.facts[global].live_in.items, next_live_in.items)) {
                    self.facts[global].live_in.clearRetainingCapacity();
                    try self.facts[global].live_in.appendSlice(allocator, next_live_in.items);
                    changed = true;
                }

                // Successor block inputs are phi-like edge parameters. If a
                // successor needs input `x`, the predecessor keeps alive the
                // output occupying that input position on this specific edge.
                for (predecessors.of(block_ref)) |pred_ref| {
                    const pred = program.functions[pred_ref.function].blocks[pred_ref.block];
                    const pred_global = self.shape.global(pred_ref);
                    for (next_live_in.items) |name| {
                        const propagated = try predecessorOutputForSuccessorInput(pred, block, name);
                        if (!containsName(self.facts[pred_global].live_out.items, propagated)) {
                            try self.facts[pred_global].live_out.append(allocator, propagated);
                            sortNames(self.facts[pred_global].live_out.items);
                            changed = true;
                        }
                    }
                }
            }
        }
    }

    fn computeIntervals(self: *LocalLiveness, allocator: std.mem.Allocator, program: ir.Program) !void {
        var local_ends = std.StringHashMap(IntervalEnd).init(allocator);
        defer local_ends.deinit();

        for (program.functions, 0..) |function, function_index| {
            for (function.blocks, 0..) |block, block_index| {
                local_ends.clearRetainingCapacity();
                const block_ref = BlockRef{ .function = function_index, .block = block_index };
                const global = self.shape.global(block_ref);

                for (self.facts[global].live_out.items) |name| try local_ends.put(name, .block_end);
                try addTerminatorLiveEnds(&local_ends, block);

                var instruction_index = block.instructions.len;
                while (instruction_index > 0) {
                    instruction_index -= 1;
                    const instruction = block.instructions[instruction_index];
                    for (instruction.results) |result| {
                        if (local_ends.fetchRemove(result)) |entry| {
                            try self.addInterval(allocator, result, .{
                                .block = block_ref,
                                .interval = .{
                                    .start = .{ .instruction = instruction_index },
                                    .end = entry.value,
                                },
                            });
                        }
                    }
                    const spec = ops.lookup(instruction.mnemonic);
                    for (instruction.operands, 0..) |operand, operand_index| {
                        if (!operandRequiresValue(spec, operand_index, operand)) continue;
                        if (!local_ends.contains(operand)) {
                            try local_ends.put(operand, .{ .instruction = instruction_index });
                        }
                    }
                }

                var ends = local_ends.iterator();
                while (ends.next()) |entry| {
                    try self.addInterval(allocator, entry.key_ptr.*, .{
                        .block = block_ref,
                        .interval = .{
                            .start = .block_start,
                            .end = entry.value_ptr.*,
                        },
                    });
                }
            }
        }
    }

    fn addInterval(self: *LocalLiveness, allocator: std.mem.Allocator, local: []const u8, interval: LocalInterval) !void {
        const entry = try self.intervals.getOrPut(local);
        if (!entry.found_existing) entry.value_ptr.* = .empty;
        try entry.value_ptr.append(allocator, interval);
    }
};

pub const AllocKind = union(enum) {
    static: usize,
    dynamic: []const u8,
};

pub const AllocData = struct {
    def_block: BlockRef,
    def_instruction: usize,
    base_ptr: []const u8,
    kind: AllocKind,
    escapes: bool = false,
    intervals: std.ArrayList(LocalInterval) = .empty,
};

pub const AllocationLiveness = struct {
    allocations: std.ArrayList(AllocData),
    local_to_alloc: std.StringHashMap(usize),

    pub fn compute(allocator: std.mem.Allocator, program: ir.Program, store: *AnalysesStore) !AllocationLiveness {
        var result = AllocationLiveness{
            .allocations = .empty,
            .local_to_alloc = std.StringHashMap(usize).init(allocator),
        };
        errdefer result.deinit(allocator);

        const index = try store.programIndex(program);
        const def_use = try store.defUse(program);
        try result.discoverAllocations(allocator, program, index, def_use);
        if (result.allocations.items.len != 0) {
            const local_liveness = try store.localLiveness(program);
            try result.populateIntervals(allocator, local_liveness);
        }
        return result;
    }

    pub fn deinit(self: *AllocationLiveness, allocator: std.mem.Allocator) void {
        for (self.allocations.items) |*alloc| alloc.intervals.deinit(allocator);
        self.allocations.deinit(allocator);
        self.local_to_alloc.deinit();
        self.* = undefined;
    }

    pub fn allocationCount(self: AllocationLiveness) usize {
        return self.allocations.items.len;
    }

    pub fn allocation(self: AllocationLiveness, index: usize) *const AllocData {
        return &self.allocations.items[index];
    }

    fn discoverAllocations(
        self: *AllocationLiveness,
        allocator: std.mem.Allocator,
        program: ir.Program,
        index: *const ProgramIndex,
        def_use: *const DefUse,
    ) !void {
        for (program.functions, 0..) |function, function_index| {
            for (function.blocks, 0..) |block, block_index| {
                for (block.instructions, 0..) |instruction, instruction_index| {
                    const allocation_kind = allocationKind(instruction) orelse continue;
                    if (instruction.results.len != 1) continue;
                    const alloc_id = self.allocations.items.len;
                    try self.allocations.append(allocator, .{
                        .def_block = .{ .function = function_index, .block = block_index },
                        .def_instruction = instruction_index,
                        .base_ptr = instruction.results[0],
                        .kind = allocation_kind,
                    });
                    try self.local_to_alloc.put(instruction.results[0], alloc_id);
                }
            }
        }

        var worklist: std.ArrayList([]const u8) = .empty;
        defer worklist.deinit(allocator);
        var alloc_id: usize = 0;
        while (alloc_id < self.allocations.items.len) : (alloc_id += 1) {
            // Start from the allocation result and follow only operations that
            // derive another pointer from it. Any operation that can expose the
            // pointer as data marks the allocation as escaping, preventing stack
            // scheduling from reusing its memory lifetime unsafely.
            try self.propagatePointersAndMarkEscapes(allocator, program, index, def_use, alloc_id, self.allocations.items[alloc_id].base_ptr, &worklist);
        }
    }

    fn propagatePointersAndMarkEscapes(
        self: *AllocationLiveness,
        allocator: std.mem.Allocator,
        program: ir.Program,
        index: *const ProgramIndex,
        def_use: *const DefUse,
        alloc_id: usize,
        ptr_local: []const u8,
        worklist: *std.ArrayList([]const u8),
    ) !void {
        worklist.clearRetainingCapacity();
        try worklist.append(allocator, ptr_local);

        while (worklist.pop()) |local| {
            for (def_use.usesOf(local)) |use_location| {
                switch (use_location.kind) {
                    .terminator => continue,
                    .block_output => |position| {
                        const block = program.functions[use_location.block.function].blocks[use_location.block.block];
                        if (block.terminator == .iret) {
                            self.allocations.items[alloc_id].escapes = true;
                            continue;
                        }
                        var successors = ir.successors(&block);
                        while (successors.next()) |successor_name| {
                            const successor_index = index.blockIndex(use_location.block.function, successor_name) orelse continue;
                            const successor = program.functions[use_location.block.function].blocks[successor_index];
                            if (position >= successor.inputs.len) continue;
                            try self.linkLocalToAlloc(allocator, successor.inputs[position], alloc_id, worklist);
                        }
                    },
                    .operation => |instruction_index| {
                        const instruction = program.functions[use_location.block.function].blocks[use_location.block.block].instructions[instruction_index];
                        if (canDerivePointer(instruction.mnemonic)) {
                            for (instruction.results) |result| try self.linkLocalToAlloc(allocator, result, alloc_id, worklist);
                        }
                        self.allocations.items[alloc_id].escapes = self.allocations.items[alloc_id].escapes or operationCausesPointerEscape(instruction, local);
                    },
                }
            }
        }
    }

    fn linkLocalToAlloc(
        self: *AllocationLiveness,
        allocator: std.mem.Allocator,
        local: []const u8,
        alloc_id: usize,
        worklist: *std.ArrayList([]const u8),
    ) !void {
        const entry = try self.local_to_alloc.getOrPut(local);
        if (!entry.found_existing) {
            entry.value_ptr.* = alloc_id;
            try worklist.append(allocator, local);
            return;
        }
        if (entry.value_ptr.* != alloc_id) {
            self.allocations.items[alloc_id].escapes = true;
            self.allocations.items[entry.value_ptr.*].escapes = true;
        }
    }

    fn populateIntervals(self: *AllocationLiveness, allocator: std.mem.Allocator, local_liveness: *const LocalLiveness) !void {
        var mappings = self.local_to_alloc.iterator();
        while (mappings.next()) |entry| {
            const alloc_id = entry.value_ptr.*;
            if (self.allocations.items[alloc_id].escapes) continue;
            for (local_liveness.intervalsOf(entry.key_ptr.*)) |interval| {
                try self.allocations.items[alloc_id].intervals.append(allocator, interval);
            }
        }

        for (self.allocations.items) |*alloc| {
            if (alloc.escapes) continue;
            mergeIntervals(&alloc.intervals);
        }
    }
};

pub const FunctionEffects = struct {
    summary: effects.FunctionEffects,

    pub fn compute(allocator: std.mem.Allocator, program: ir.Program, store: *AnalysesStore) !FunctionEffects {
        _ = store;
        return .{ .summary = try effects.analyzeFunctions(allocator, program) };
    }

    pub fn deinit(self: *FunctionEffects, allocator: std.mem.Allocator) void {
        _ = allocator;
        self.summary.deinit();
        self.* = undefined;
    }

    pub fn effectOf(self: FunctionEffects, name: []const u8) ?effects.Effect {
        return self.summary.effectOf(name);
    }
};

pub const BasicBlockOwnership = struct {
    shape: BlockShape,
    owners: []?usize,

    pub fn compute(allocator: std.mem.Allocator, program: ir.Program, store: *AnalysesStore) !BasicBlockOwnership {
        const index = try store.programIndex(program);
        var shape = try BlockShape.build(allocator, program);
        errdefer shape.deinit(allocator);
        const owners = try allocator.alloc(?usize, shape.block_count);
        errdefer allocator.free(owners);
        @memset(owners, null);

        var result = BasicBlockOwnership{ .shape = shape, .owners = owners };
        for (program.functions, 0..) |function, function_index| {
            if (function.blocks.len == 0) continue;
            result.mark(program, index, .{ .function = function_index, .block = 0 }, function_index);
        }
        return result;
    }

    pub fn deinit(self: *BasicBlockOwnership, allocator: std.mem.Allocator) void {
        allocator.free(self.owners);
        self.shape.deinit(allocator);
        self.* = undefined;
    }

    pub fn ownerOf(self: BasicBlockOwnership, ref: BlockRef) ?usize {
        return self.owners[self.shape.global(ref)];
    }

    pub fn isReachable(self: BasicBlockOwnership, ref: BlockRef) bool {
        return self.ownerOf(ref) != null;
    }

    fn mark(self: *BasicBlockOwnership, program: ir.Program, index: *const ProgramIndex, ref: BlockRef, owner: usize) void {
        const slot = self.shape.global(ref);
        if (self.owners[slot] != null) return;
        self.owners[slot] = owner;

        const function = program.functions[ref.function];
        var successors = ir.successors(&function.blocks[ref.block]);
        while (successors.next()) |successor_name| {
            if (index.blockIndex(ref.function, successor_name)) |successor_index| {
                self.mark(program, index, .{ .function = ref.function, .block = successor_index }, owner);
            }
        }
    }
};

pub const CfgInOutBundling = struct {
    shape: BlockShape,
    in_group: []?u32,
    out_group: []?u32,
    total_groups: u32,

    pub fn compute(allocator: std.mem.Allocator, program: ir.Program, store: *AnalysesStore) !CfgInOutBundling {
        const index = try store.programIndex(program);
        const rpo = try store.reversePostOrder(program);
        const predecessors = try store.predecessors(program);
        var shape = try BlockShape.build(allocator, program);
        errdefer shape.deinit(allocator);

        const in_group = try allocator.alloc(?u32, shape.block_count);
        errdefer allocator.free(in_group);
        const out_group = try allocator.alloc(?u32, shape.block_count);
        errdefer allocator.free(out_group);
        @memset(in_group, null);
        @memset(out_group, null);

        var result = CfgInOutBundling{
            .shape = shape,
            .in_group = in_group,
            .out_group = out_group,
            .total_groups = 0,
        };
        var worklist: std.ArrayList(BlockRef) = .empty;
        defer worklist.deinit(allocator);

        // Plank's release memory layout wants `out(pred)` and `in(succ)` for an
        // edge to share one transfer-buffer layout. When a successor input
        // group changes, all other predecessors of that successor must be
        // revisited so their output groups converge to the same group.
        for (rpo.global()) |block_ref| {
            try worklist.append(allocator, block_ref);
            while (worklist.pop()) |current| {
                const function = program.functions[current.function];
                const block = function.blocks[current.block];
                const group_id = result.successorInGroup(index, current.function, block) orelse result.nextGroup();
                result.out_group[result.shape.global(current)] = group_id;

                var successors = ir.successors(&block);
                while (successors.next()) |successor_name| {
                    const successor_index = index.blockIndex(current.function, successor_name) orelse continue;
                    const successor = BlockRef{ .function = current.function, .block = successor_index };
                    const in_slot = result.shape.global(successor);
                    const previous = result.in_group[in_slot];
                    result.in_group[in_slot] = group_id;
                    if (previous == null or previous.? == group_id) continue;
                    for (predecessors.of(successor)) |pred| {
                        if (!blockRefEql(pred, current)) try worklist.append(allocator, pred);
                    }
                }
            }
        }

        for (program.functions, 0..) |function, function_index| {
            if (function.blocks.len == 0) continue;
            const entry = BlockRef{ .function = function_index, .block = 0 };
            const slot = result.shape.global(entry);
            if (result.in_group[slot] == null) result.in_group[slot] = result.nextGroup();
        }

        return result;
    }

    pub fn deinit(self: *CfgInOutBundling, allocator: std.mem.Allocator) void {
        allocator.free(self.in_group);
        allocator.free(self.out_group);
        self.shape.deinit(allocator);
        self.* = undefined;
    }

    pub fn getInGroup(self: CfgInOutBundling, ref: BlockRef) ?u32 {
        return self.in_group[self.shape.global(ref)];
    }

    pub fn getOutGroup(self: CfgInOutBundling, ref: BlockRef) ?u32 {
        return self.out_group[self.shape.global(ref)];
    }

    pub fn totalGroups(self: CfgInOutBundling) u32 {
        return self.total_groups;
    }

    fn successorInGroup(self: CfgInOutBundling, index: *const ProgramIndex, function_index: usize, block: ir.Block) ?u32 {
        var successors = ir.successors(&block);
        while (successors.next()) |successor_name| {
            const successor_index = index.blockIndex(function_index, successor_name) orelse continue;
            const group = self.in_group[self.shape.global(.{ .function = function_index, .block = successor_index })];
            if (group) |value| return value;
        }
        return null;
    }

    fn nextGroup(self: *CfgInOutBundling) u32 {
        const group = self.total_groups;
        self.total_groups += 1;
        return group;
    }
};

pub const Legalizer = struct {
    pub fn compute(allocator: std.mem.Allocator, program: ir.Program, store: *AnalysesStore) !Legalizer {
        _ = store;
        var bag = diag_mod.Bag.init(allocator);
        defer bag.deinit();
        legality.validate(allocator, program, &bag) catch |err| switch (err) {
            legality.ValidateError.InvalidSir => return error.InvalidSir,
            else => |other| return other,
        };
        return .{};
    }

    pub fn deinit(self: *Legalizer, allocator: std.mem.Allocator) void {
        _ = allocator;
        self.* = undefined;
    }
};

fn dfsPostorder(
    allocator: std.mem.Allocator,
    program: ir.Program,
    index: *const ProgramIndex,
    shape: BlockShape,
    out: *std.ArrayList(BlockRef),
    visited: []bool,
    ref: BlockRef,
) !void {
    const global = shape.global(ref);
    if (visited[global]) return;
    visited[global] = true;

    const function = program.functions[ref.function];
    var it = ir.successors(&function.blocks[ref.block]);
    while (it.next()) |target| {
        if (index.blockIndex(ref.function, target)) |successor| {
            try dfsPostorder(allocator, program, index, shape, out, visited, .{
                .function = ref.function,
                .block = successor,
            });
        }
    }
    try out.append(allocator, ref);
}

fn valueUseEstimate(program: ir.Program) usize {
    var count: usize = 0;
    for (program.functions) |function| {
        for (function.blocks) |block| {
            for (block.instructions) |instruction| count += instruction.operands.len;
            count += block.outputs.len + 2;
        }
    }
    return count;
}

fn capacityLen(len: usize) u32 {
    return if (len > std.math.maxInt(u32)) std.math.maxInt(u32) else @intCast(len);
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

fn blockRefEql(lhs: BlockRef, rhs: BlockRef) bool {
    return lhs.function == rhs.function and lhs.block == rhs.block;
}

fn optionalBlockRefEql(lhs: ?BlockRef, rhs: BlockRef) bool {
    return if (lhs) |value| blockRefEql(value, rhs) else false;
}

fn computeBlockEntryLiveness(
    allocator: std.mem.Allocator,
    block: ir.Block,
    live_out: []const []const u8,
    live_in: *std.ArrayList([]const u8),
) !void {
    for (live_out) |name| try appendName(allocator, live_in, name);

    switch (block.terminator) {
        .branch => |branch| try appendName(allocator, live_in, branch.condition),
        .switch_ => |switch_term| try appendName(allocator, live_in, switch_term.selector),
        .iret => {
            for (block.outputs) |output| try appendName(allocator, live_in, output);
        },
        .return_ => |ret| {
            try appendName(allocator, live_in, ret.ptr);
            try appendName(allocator, live_in, ret.len);
        },
        .revert => |revert| {
            try appendName(allocator, live_in, revert.ptr);
            try appendName(allocator, live_in, revert.len);
        },
        .selfdestruct => |beneficiary| try appendName(allocator, live_in, beneficiary),
        .jump, .stop, .invalid => {},
    }

    var instruction_index = block.instructions.len;
    while (instruction_index > 0) {
        instruction_index -= 1;
        const instruction = block.instructions[instruction_index];
        for (instruction.results) |result| removeName(live_in, result);

        const spec = ops.lookup(instruction.mnemonic);
        for (instruction.operands, 0..) |operand, operand_index| {
            if (!operandRequiresValue(spec, operand_index, operand)) continue;
            try appendName(allocator, live_in, operand);
        }
    }
    sortNames(live_in.items);
}

fn addTerminatorLiveEnds(local_ends: *std.StringHashMap(IntervalEnd), block: ir.Block) !void {
    switch (block.terminator) {
        .branch => |branch| try local_ends.put(branch.condition, .block_end),
        .switch_ => |switch_term| try local_ends.put(switch_term.selector, .block_end),
        .iret => {
            for (block.outputs) |output| try local_ends.put(output, .block_end);
        },
        .return_ => |ret| {
            try local_ends.put(ret.ptr, .block_end);
            try local_ends.put(ret.len, .block_end);
        },
        .revert => |revert| {
            try local_ends.put(revert.ptr, .block_end);
            try local_ends.put(revert.len, .block_end);
        },
        .selfdestruct => |beneficiary| try local_ends.put(beneficiary, .block_end),
        .jump, .stop, .invalid => {},
    }
}

fn predecessorOutputForSuccessorInput(predecessor: ir.Block, successor: ir.Block, name: []const u8) ![]const u8 {
    for (successor.inputs, 0..) |input, position| {
        if (!std.mem.eql(u8, input, name)) continue;
        if (position >= predecessor.outputs.len) return error.InvalidSir;
        return predecessor.outputs[position];
    }
    return name;
}

fn allocationKind(instruction: ir.Instruction) ?AllocKind {
    if (instruction.results.len != 1) return null;
    if (std.mem.eql(u8, instruction.mnemonic, "salloc") or std.mem.eql(u8, instruction.mnemonic, "sallocany")) {
        if (instruction.operands.len != 1) return null;
        return .{ .static = std.fmt.parseUnsigned(usize, instruction.operands[0], 0) catch return null };
    }
    if (std.mem.eql(u8, instruction.mnemonic, "malloc") or std.mem.eql(u8, instruction.mnemonic, "mallocany")) {
        if (instruction.operands.len != 1) return null;
        return .{ .dynamic = instruction.operands[0] };
    }
    return null;
}

fn canDerivePointer(mnemonic: []const u8) bool {
    return std.mem.eql(u8, mnemonic, "copy") or
        std.mem.eql(u8, mnemonic, "add") or
        std.mem.eql(u8, mnemonic, "mul") or
        std.mem.eql(u8, mnemonic, "sub") or
        std.mem.eql(u8, mnemonic, "div") or
        std.mem.eql(u8, mnemonic, "sdiv") or
        std.mem.eql(u8, mnemonic, "mod") or
        std.mem.eql(u8, mnemonic, "smod") or
        std.mem.eql(u8, mnemonic, "addmod") or
        std.mem.eql(u8, mnemonic, "mulmod") or
        std.mem.eql(u8, mnemonic, "exp") or
        std.mem.eql(u8, mnemonic, "signextend") or
        std.mem.eql(u8, mnemonic, "and") or
        std.mem.eql(u8, mnemonic, "or") or
        std.mem.eql(u8, mnemonic, "xor") or
        std.mem.eql(u8, mnemonic, "not") or
        std.mem.eql(u8, mnemonic, "byte") or
        std.mem.eql(u8, mnemonic, "shl") or
        std.mem.eql(u8, mnemonic, "shr") or
        std.mem.eql(u8, mnemonic, "sar");
}

fn operationCausesPointerEscape(instruction: ir.Instruction, local: []const u8) bool {
    const mnemonic = instruction.mnemonic;
    if (std.mem.eql(u8, mnemonic, "keccak256")) return operandEquals(instruction, 1, local);
    if (std.mem.eql(u8, mnemonic, "calldatacopy") or
        std.mem.eql(u8, mnemonic, "codecopy") or
        std.mem.eql(u8, mnemonic, "returndatacopy"))
    {
        return operandEquals(instruction, 1, local) or operandEquals(instruction, 2, local);
    }
    if (std.mem.eql(u8, mnemonic, "extcodecopy")) {
        return operandEquals(instruction, 0, local) or operandEquals(instruction, 2, local) or operandEquals(instruction, 3, local);
    }
    if (std.mem.eql(u8, mnemonic, "mcopy")) return operandEquals(instruction, 2, local);
    if (std.mem.startsWith(u8, mnemonic, "mstore")) return operandEquals(instruction, 1, local);
    if (std.mem.eql(u8, mnemonic, "create")) {
        return operandEquals(instruction, 0, local) or operandEquals(instruction, 2, local);
    }
    if (std.mem.eql(u8, mnemonic, "create2")) {
        return operandEquals(instruction, 0, local) or operandEquals(instruction, 2, local) or operandEquals(instruction, 3, local);
    }
    if (std.mem.eql(u8, mnemonic, "call") or std.mem.eql(u8, mnemonic, "callcode")) {
        return operandEquals(instruction, 0, local) or operandEquals(instruction, 1, local) or
            operandEquals(instruction, 2, local) or operandEquals(instruction, 4, local) or
            operandEquals(instruction, 6, local);
    }
    if (std.mem.eql(u8, mnemonic, "delegatecall") or std.mem.eql(u8, mnemonic, "staticcall")) {
        return operandEquals(instruction, 0, local) or operandEquals(instruction, 1, local) or
            operandEquals(instruction, 3, local) or operandEquals(instruction, 5, local);
    }
    if (std.mem.eql(u8, mnemonic, "icall")) return true;
    if (std.mem.eql(u8, mnemonic, "balance") or
        std.mem.eql(u8, mnemonic, "calldataload") or
        std.mem.eql(u8, mnemonic, "extcodesize") or
        std.mem.eql(u8, mnemonic, "extcodehash") or
        std.mem.eql(u8, mnemonic, "blockhash") or
        std.mem.eql(u8, mnemonic, "blobhash") or
        std.mem.eql(u8, mnemonic, "sload") or
        std.mem.eql(u8, mnemonic, "sstore") or
        std.mem.eql(u8, mnemonic, "tload") or
        std.mem.eql(u8, mnemonic, "tstore"))
    {
        return true;
    }
    return false;
}

fn operandEquals(instruction: ir.Instruction, index: usize, local: []const u8) bool {
    return index < instruction.operands.len and std.mem.eql(u8, instruction.operands[index], local);
}

fn mergeIntervals(intervals: *std.ArrayList(LocalInterval)) void {
    if (intervals.items.len <= 1) return;
    std.mem.sort(LocalInterval, intervals.items, {}, struct {
        fn less(_: void, lhs: LocalInterval, rhs: LocalInterval) bool {
            if (lhs.block.function != rhs.block.function) return lhs.block.function < rhs.block.function;
            if (lhs.block.block != rhs.block.block) return lhs.block.block < rhs.block.block;
            const start_order = compareIntervalStart(lhs.interval.start, rhs.interval.start);
            if (start_order != .eq) return start_order == .lt;
            return compareIntervalEnd(lhs.interval.end, rhs.interval.end) == .lt;
        }
    }.less);

    var dst: usize = 0;
    var src: usize = 1;
    while (src < intervals.items.len) : (src += 1) {
        if (sameBlock(intervals.items[dst].block, intervals.items[src].block) and intervalsOverlap(intervals.items[dst].interval, intervals.items[src].interval)) {
            if (compareIntervalEnd(intervals.items[dst].interval.end, intervals.items[src].interval.end) == .lt) {
                intervals.items[dst].interval.end = intervals.items[src].interval.end;
            }
        } else {
            dst += 1;
            intervals.items[dst] = intervals.items[src];
        }
    }
    intervals.shrinkRetainingCapacity(dst + 1);
}

fn sameBlock(lhs: BlockRef, rhs: BlockRef) bool {
    return lhs.function == rhs.function and lhs.block == rhs.block;
}

fn intervalsOverlap(lhs: Interval, rhs: Interval) bool {
    return switch (lhs.end) {
        .block_end => true,
        .instruction => |lhs_end| switch (rhs.start) {
            .block_start => true,
            .instruction => |rhs_start| rhs_start <= lhs_end,
        },
    };
}

fn compareIntervalStart(lhs: IntervalStart, rhs: IntervalStart) std.math.Order {
    return switch (lhs) {
        .block_start => if (rhs == .block_start) .eq else .lt,
        .instruction => |lhs_instruction| switch (rhs) {
            .block_start => .gt,
            .instruction => |rhs_instruction| std.math.order(lhs_instruction, rhs_instruction),
        },
    };
}

fn compareIntervalEnd(lhs: IntervalEnd, rhs: IntervalEnd) std.math.Order {
    return switch (lhs) {
        .block_end => if (rhs == .block_end) .eq else .gt,
        .instruction => |lhs_instruction| switch (rhs) {
            .block_end => .lt,
            .instruction => |rhs_instruction| std.math.order(lhs_instruction, rhs_instruction),
        },
    };
}

fn appendName(allocator: std.mem.Allocator, names: *std.ArrayList([]const u8), name: []const u8) !void {
    if (containsName(names.items, name)) return;
    try names.append(allocator, name);
}

fn containsName(names: []const []const u8, name: []const u8) bool {
    for (names) |entry| {
        if (std.mem.eql(u8, entry, name)) return true;
    }
    return false;
}

fn removeName(names: *std.ArrayList([]const u8), name: []const u8) void {
    for (names.items, 0..) |entry, index| {
        if (std.mem.eql(u8, entry, name)) {
            const last = names.pop().?;
            if (index < names.items.len) names.items[index] = last;
            return;
        }
    }
}

fn sortNames(names: [][]const u8) void {
    std.mem.sort([]const u8, names, {}, struct {
        fn less(_: void, lhs: []const u8, rhs: []const u8) bool {
            return std.mem.order(u8, lhs, rhs) == .lt;
        }
    }.less);
}

fn sameNameSet(lhs: []const []const u8, rhs: []const []const u8) bool {
    if (lhs.len != rhs.len) return false;
    for (lhs, rhs) |lhs_name, rhs_name| {
        if (!std.mem.eql(u8, lhs_name, rhs_name)) return false;
    }
    return true;
}

test "analysis cache computes reachability and ignores unreachable blocks" {
    const diagnostics = @import("diagnostics.zig");
    const parser = @import("parser.zig");

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator,
        \\fn main:
        \\    entry {
        \\        => @live
        \\    }
        \\    live {
        \\        stop
        \\    }
        \\    dead {
        \\        stop
        \\    }
    , &bag);
    defer program.deinit();

    var store = AnalysesStore.init(std.testing.allocator);
    defer store.deinit();
    const reachability = try store.reachability(program);
    try std.testing.expect(reachability.contains(.{ .function = 0, .block = 0 }));
    try std.testing.expect(reachability.contains(.{ .function = 0, .block = 1 }));
    try std.testing.expect(!reachability.contains(.{ .function = 0, .block = 2 }));
    try std.testing.expect(store.isValid(.reachability));
}

test "program index caches function and block name lookups" {
    const diagnostics = @import("diagnostics.zig");
    const parser = @import("parser.zig");

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator,
        \\fn main:
        \\    entry {
        \\        => @join
        \\    }
        \\    join {
        \\        stop
        \\    }
        \\
        \\fn helper:
        \\    entry {
        \\        stop
        \\    }
    , &bag);
    defer program.deinit();

    var store = AnalysesStore.init(std.testing.allocator);
    defer store.deinit();
    const index = try store.programIndex(program);

    try std.testing.expectEqual(@as(?usize, 0), index.functionIndex("main"));
    try std.testing.expectEqual(@as(?usize, 1), index.functionIndex("helper"));
    try std.testing.expectEqual(@as(?usize, 1), index.blockIndex(0, "join"));
    try std.testing.expectEqual(@as(?usize, null), index.blockIndex(1, "join"));
    try std.testing.expect(store.isValid(.program_index));
}

test "predecessors are computed only for reachable CFG edges" {
    const diagnostics = @import("diagnostics.zig");
    const parser = @import("parser.zig");

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator,
        \\fn main:
        \\    entry {
        \\        => @join
        \\    }
        \\    alt {
        \\        => @join
        \\    }
        \\    join {
        \\        stop
        \\    }
    , &bag);
    defer program.deinit();

    var store = AnalysesStore.init(std.testing.allocator);
    defer store.deinit();
    const predecessors = try store.predecessors(program);
    const join_preds = predecessors.of(.{ .function = 0, .block = 2 });
    try std.testing.expectEqual(@as(usize, 1), join_preds.len);
    try std.testing.expectEqual(@as(usize, 0), join_preds[0].block);
}

test "reverse post order is per-function and starts at entry" {
    const diagnostics = @import("diagnostics.zig");
    const parser = @import("parser.zig");

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator,
        \\fn main:
        \\    entry {
        \\        => @left
        \\    }
        \\    left {
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

    var store = AnalysesStore.init(std.testing.allocator);
    defer store.deinit();
    const rpo = try store.reversePostOrder(program);
    const main_order = rpo.function(0);
    try std.testing.expectEqual(@as(usize, 2), main_order.len);
    try std.testing.expectEqual(@as(usize, 0), main_order[0].block);
    const helper_order = rpo.function(1);
    try std.testing.expectEqual(@as(usize, 1), helper_order.len);
    try std.testing.expectEqual(@as(usize, 0), helper_order[0].block);
}

test "dominators and dominance frontiers handle diamond joins" {
    const diagnostics = @import("diagnostics.zig");
    const parser = @import("parser.zig");

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator,
        \\fn main:
        \\    entry {
        \\        cond = const 1
        \\        => cond ? @left : @right
        \\    }
        \\    left {
        \\        => @join
        \\    }
        \\    right {
        \\        => @join
        \\    }
        \\    join {
        \\        stop
        \\    }
    , &bag);
    defer program.deinit();

    var store = AnalysesStore.init(std.testing.allocator);
    defer store.deinit();
    const dominators = try store.dominators(program);
    try expectBlockRef(dominators.of(.{ .function = 0, .block = 0 }), .{ .function = 0, .block = 0 });
    try expectBlockRef(dominators.of(.{ .function = 0, .block = 1 }), .{ .function = 0, .block = 0 });
    try expectBlockRef(dominators.of(.{ .function = 0, .block = 2 }), .{ .function = 0, .block = 0 });
    try expectBlockRef(dominators.of(.{ .function = 0, .block = 3 }), .{ .function = 0, .block = 0 });

    const frontiers = try store.dominanceFrontiers(program);
    try expectFrontierBlocks(frontiers, .{ .function = 0, .block = 0 }, &.{});
    try expectFrontierBlocks(frontiers, .{ .function = 0, .block = 1 }, &.{3});
    try expectFrontierBlocks(frontiers, .{ .function = 0, .block = 2 }, &.{3});
    try expectFrontierBlocks(frontiers, .{ .function = 0, .block = 3 }, &.{});
}

test "dominance frontiers include loop headers on back edges" {
    const diagnostics = @import("diagnostics.zig");
    const parser = @import("parser.zig");

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator,
        \\fn main:
        \\    entry {
        \\        => @loop
        \\    }
        \\    loop {
        \\        cond = const 1
        \\        => cond ? @body : @exit
        \\    }
        \\    body {
        \\        => @loop
        \\    }
        \\    exit {
        \\        stop
        \\    }
    , &bag);
    defer program.deinit();

    var store = AnalysesStore.init(std.testing.allocator);
    defer store.deinit();
    const dominators = try store.dominators(program);
    try expectBlockRef(dominators.of(.{ .function = 0, .block = 1 }), .{ .function = 0, .block = 0 });
    try expectBlockRef(dominators.of(.{ .function = 0, .block = 2 }), .{ .function = 0, .block = 1 });
    try expectBlockRef(dominators.of(.{ .function = 0, .block = 3 }), .{ .function = 0, .block = 1 });

    const frontiers = try store.dominanceFrontiers(program);
    try expectFrontierBlocks(frontiers, .{ .function = 0, .block = 0 }, &.{});
    try expectFrontierBlocks(frontiers, .{ .function = 0, .block = 1 }, &.{1});
    try expectFrontierBlocks(frontiers, .{ .function = 0, .block = 2 }, &.{1});
    try expectFrontierBlocks(frontiers, .{ .function = 0, .block = 3 }, &.{});
}

test "dominators ignore unreachable predecessors" {
    const diagnostics = @import("diagnostics.zig");
    const parser = @import("parser.zig");

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator,
        \\fn main:
        \\    entry {
        \\        => @b
        \\    }
        \\    orphan {
        \\        => @d
        \\    }
        \\    b {
        \\        cond = const 1
        \\        => cond ? @c : @d
        \\    }
        \\    c {
        \\        => @d
        \\    }
        \\    d {
        \\        stop
        \\    }
    , &bag);
    defer program.deinit();

    var store = AnalysesStore.init(std.testing.allocator);
    defer store.deinit();
    const dominators = try store.dominators(program);
    try expectBlockRef(dominators.of(.{ .function = 0, .block = 0 }), .{ .function = 0, .block = 0 });
    try std.testing.expectEqual(@as(?BlockRef, null), dominators.of(.{ .function = 0, .block = 1 }));
    try expectBlockRef(dominators.of(.{ .function = 0, .block = 2 }), .{ .function = 0, .block = 0 });
    try expectBlockRef(dominators.of(.{ .function = 0, .block = 3 }), .{ .function = 0, .block = 2 });
    try expectBlockRef(dominators.of(.{ .function = 0, .block = 4 }), .{ .function = 0, .block = 2 });
}

test "def-use records value operands, control uses, and block outputs" {
    const diagnostics = @import("diagnostics.zig");
    const parser = @import("parser.zig");

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator,
        \\fn main:
        \\    entry -> b {
        \\        a = const 1
        \\        b = add a 2
        \\        ptr = data_offset .payload
        \\        => b ? @left : @right
        \\    }
        \\    left {
        \\        return ptr b
        \\    }
        \\    right {
        \\        stop
        \\    }
        \\
        \\data payload 0xdeadbeef
    , &bag);
    defer program.deinit();

    var store = AnalysesStore.init(std.testing.allocator);
    defer store.deinit();
    const def_use = try store.defUse(program);

    const a_uses = def_use.usesOf("a");
    try std.testing.expectEqual(@as(usize, 1), a_uses.len);
    try expectOperationUse(a_uses[0], .{ .function = 0, .block = 0 }, 2);

    const b_uses = def_use.usesOf("b");
    try std.testing.expectEqual(@as(usize, 3), b_uses.len);
    try expectTerminatorUse(b_uses[0], .{ .function = 0, .block = 0 });
    try expectBlockOutputUse(b_uses[1], .{ .function = 0, .block = 0 }, 0);
    try expectTerminatorUse(b_uses[2], .{ .function = 0, .block = 1 });

    const ptr_uses = def_use.usesOf("ptr");
    try std.testing.expectEqual(@as(usize, 1), ptr_uses.len);
    try expectTerminatorUse(ptr_uses[0], .{ .function = 0, .block = 1 });

    try std.testing.expectEqual(@as(usize, 0), def_use.usesOf("1").len);
    try std.testing.expectEqual(@as(usize, 0), def_use.usesOf("2").len);
    try std.testing.expectEqual(@as(usize, 0), def_use.usesOf(".payload").len);
}

test "analysis cache tracks transitive dependencies" {
    const diagnostics = @import("diagnostics.zig");
    const parser = @import("parser.zig");

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator,
        \\fn main:
        \\    entry {
        \\        cond = const 1
        \\        => cond ? @left : @right
        \\    }
        \\    left {
        \\        => @join
        \\    }
        \\    right {
        \\        => @join
        \\    }
        \\    join {
        \\        stop
        \\    }
    , &bag);
    defer program.deinit();

    var store = AnalysesStore.init(std.testing.allocator);
    defer store.deinit();
    _ = try store.dominanceFrontiers(program);
    try std.testing.expect(store.isValid(.reachability));
    try std.testing.expect(store.isValid(.predecessors));
    try std.testing.expect(store.isValid(.reverse_post_order));
    try std.testing.expect(store.isValid(.dominators));
    try std.testing.expect(store.isValid(.dominance_frontiers));
}

test "legalizer analysis caches valid programs and fails closed on invalid SIR" {
    const diagnostics = @import("diagnostics.zig");
    const parser = @import("parser.zig");

    var valid_bag = diagnostics.Bag.init(std.testing.allocator);
    defer valid_bag.deinit();
    var valid_program = try parser.parse(std.testing.allocator,
        \\fn main:
        \\    entry {
        \\        stop
        \\    }
    , &valid_bag);
    defer valid_program.deinit();

    var store = AnalysesStore.init(std.testing.allocator);
    defer store.deinit();
    _ = try store.legalizer(valid_program);
    try std.testing.expect(store.isValid(.legalizer));

    var invalid_bag = diagnostics.Bag.init(std.testing.allocator);
    defer invalid_bag.deinit();
    var invalid_program = try parser.parse(std.testing.allocator,
        \\fn main:
        \\    entry {
        \\        => @missing
        \\    }
    , &invalid_bag);
    defer invalid_program.deinit();

    store.invalidateAllExcept(.empty());
    try std.testing.expectError(error.InvalidSir, store.legalizer(invalid_program));
    try std.testing.expect(!store.isValid(.legalizer));
}

test "local liveness propagates block inputs from predecessor outputs" {
    const diagnostics = @import("diagnostics.zig");
    const parser = @import("parser.zig");

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator,
        \\fn main:
        \\    entry -> buf {
        \\        buf = salloc 32
        \\        => @use_it
        \\    }
        \\    use_it ptr {
        \\        v = mload256 ptr
        \\        stop
        \\    }
    , &bag);
    defer program.deinit();

    var store = AnalysesStore.init(std.testing.allocator);
    defer store.deinit();
    const liveness = try store.localLiveness(program);

    try expectNames(liveness.liveAtExit(.{ .function = 0, .block = 0 }), &.{"buf"});
    try expectNames(liveness.liveAtEntry(.{ .function = 0, .block = 1 }), &.{"ptr"});

    const buf_intervals = liveness.intervalsOf("buf");
    try std.testing.expectEqual(@as(usize, 1), buf_intervals.len);
    try std.testing.expectEqual(@as(usize, 0), buf_intervals[0].block.block);
    const ptr_intervals = liveness.intervalsOf("ptr");
    try std.testing.expectEqual(@as(usize, 1), ptr_intervals.len);
    try std.testing.expectEqual(@as(usize, 1), ptr_intervals[0].block.block);
}

test "allocation liveness tracks derived pointers and escaping pointer values" {
    const diagnostics = @import("diagnostics.zig");
    const parser = @import("parser.zig");

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator,
        \\fn main:
        \\    entry {
        \\        buf = salloc 64
        \\        off = const 32
        \\        derived = add buf off
        \\        v = mload256 derived
        \\        scratch = salloc 32
        \\        mstore256 scratch buf
        \\        stop
        \\    }
    , &bag);
    defer program.deinit();

    var store = AnalysesStore.init(std.testing.allocator);
    defer store.deinit();
    const allocation_liveness = try store.allocationLiveness(program);

    try std.testing.expectEqual(@as(usize, 2), allocation_liveness.allocationCount());
    try std.testing.expect(allocation_liveness.allocation(0).escapes);
    try std.testing.expect(!allocation_liveness.allocation(1).escapes);
    try std.testing.expect(allocation_liveness.allocation(1).intervals.items.len != 0);
    try std.testing.expect(store.isValid(.def_use));
    try std.testing.expect(store.isValid(.local_liveness));
    try std.testing.expect(store.isValid(.allocation_liveness));
}

test "function effects are available through the analysis cache" {
    const diagnostics = @import("diagnostics.zig");
    const parser = @import("parser.zig");

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator,
        \\fn main:
        \\    entry {
        \\        zero = const 0
        \\        icall @write zero zero
        \\        stop
        \\    }
        \\
        \\fn write:
        \\    entry key value {
        \\        sstore key value
        \\        iret
        \\    }
    , &bag);
    defer program.deinit();

    var store = AnalysesStore.init(std.testing.allocator);
    defer store.deinit();
    const summaries = try store.functionEffects(program);
    const main_effect = summaries.effectOf("main") orelse return error.TestUnexpectedResult;
    const write_effect = summaries.effectOf("write") orelse return error.TestUnexpectedResult;
    try std.testing.expect(main_effect.contains(effects.Effect.PERSISTENT_WRITE));
    try std.testing.expect(write_effect.contains(effects.Effect.PERSISTENT_WRITE));
}

test "basic block ownership marks unreachable blocks" {
    const diagnostics = @import("diagnostics.zig");
    const parser = @import("parser.zig");

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator,
        \\fn main:
        \\    entry {
        \\        => @live
        \\    }
        \\    live {
        \\        stop
        \\    }
        \\    orphan {
        \\        stop
        \\    }
        \\
        \\fn helper:
        \\    entry {
        \\        stop
        \\    }
    , &bag);
    defer program.deinit();

    var store = AnalysesStore.init(std.testing.allocator);
    defer store.deinit();
    const ownership = try store.basicBlockOwnership(program);
    try std.testing.expectEqual(@as(?usize, 0), ownership.ownerOf(.{ .function = 0, .block = 0 }));
    try std.testing.expectEqual(@as(?usize, 0), ownership.ownerOf(.{ .function = 0, .block = 1 }));
    try std.testing.expectEqual(@as(?usize, null), ownership.ownerOf(.{ .function = 0, .block = 2 }));
    try std.testing.expectEqual(@as(?usize, 1), ownership.ownerOf(.{ .function = 1, .block = 0 }));
}

test "CFG in/out bundling groups predecessor outputs with successor inputs" {
    const diagnostics = @import("diagnostics.zig");
    const parser = @import("parser.zig");

    var bag = diagnostics.Bag.init(std.testing.allocator);
    defer bag.deinit();
    var program = try parser.parse(std.testing.allocator,
        \\fn main:
        \\    entry {
        \\        cond = const 1
        \\        => cond ? @left : @right
        \\    }
        \\    left {
        \\        => @join
        \\    }
        \\    right {
        \\        => @join
        \\    }
        \\    join {
        \\        stop
        \\    }
    , &bag);
    defer program.deinit();

    var store = AnalysesStore.init(std.testing.allocator);
    defer store.deinit();
    const bundling = try store.cfgInOutBundling(program);

    const entry_out = bundling.getOutGroup(.{ .function = 0, .block = 0 }) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(entry_out, bundling.getInGroup(.{ .function = 0, .block = 1 }).?);
    try std.testing.expectEqual(entry_out, bundling.getInGroup(.{ .function = 0, .block = 2 }).?);

    const join_in = bundling.getInGroup(.{ .function = 0, .block = 3 }) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(join_in, bundling.getOutGroup(.{ .function = 0, .block = 1 }).?);
    try std.testing.expectEqual(join_in, bundling.getOutGroup(.{ .function = 0, .block = 2 }).?);
    try std.testing.expect(bundling.getInGroup(.{ .function = 0, .block = 0 }) != null);
    try std.testing.expect(bundling.totalGroups() >= 3);
}

fn expectBlockRef(actual: ?BlockRef, expected: BlockRef) !void {
    const value = actual orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(expected.function, value.function);
    try std.testing.expectEqual(expected.block, value.block);
}

fn expectFrontierBlocks(frontiers: *const DominanceFrontiers, owner: BlockRef, expected_blocks: []const usize) !void {
    const actual = frontiers.of(owner);
    try std.testing.expectEqual(expected_blocks.len, actual.len);
    for (expected_blocks, actual) |expected, value| {
        try std.testing.expectEqual(owner.function, value.function);
        try std.testing.expectEqual(expected, value.block);
    }
}

fn expectOperationUse(location: UseLocation, expected_block: BlockRef, expected_instruction: usize) !void {
    try expectUseBlock(location, expected_block);
    switch (location.kind) {
        .operation => |instruction| try std.testing.expectEqual(expected_instruction, instruction),
        else => return error.TestUnexpectedResult,
    }
}

fn expectTerminatorUse(location: UseLocation, expected_block: BlockRef) !void {
    try expectUseBlock(location, expected_block);
    switch (location.kind) {
        .terminator => {},
        else => return error.TestUnexpectedResult,
    }
}

fn expectBlockOutputUse(location: UseLocation, expected_block: BlockRef, expected_output: usize) !void {
    try expectUseBlock(location, expected_block);
    switch (location.kind) {
        .block_output => |output| try std.testing.expectEqual(expected_output, output),
        else => return error.TestUnexpectedResult,
    }
}

fn expectUseBlock(location: UseLocation, expected_block: BlockRef) !void {
    try std.testing.expectEqual(expected_block.function, location.block.function);
    try std.testing.expectEqual(expected_block.block, location.block.block);
}

fn expectNames(actual: []const []const u8, expected: []const []const u8) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |expected_name, actual_name| {
        try std.testing.expectEqualStrings(expected_name, actual_name);
    }
}
