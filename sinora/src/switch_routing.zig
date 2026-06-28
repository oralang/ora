//! Switch routing planner shared by release metrics and release codegen.
//!
//! The planner is deliberately semantic-light: it only looks at SIR switch case
//! values and proposes a routing shape. Codegen remains responsible for
//! preserving the exact-selector guard. That guard is mandatory for sparse and
//! dense routing because arbitrary external selectors can alias a known bucket
//! or table slot.

const std = @import("std");

const ir = @import("ir.zig");

pub const sparse_bucket_bits = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8 };
pub const sparse_bucket_shifts = [_]u8{ 0, 4, 8, 12, 16, 20, 24 };
pub const dense_max_table_slots: usize = 256;
pub const min_selector_check_saving_x1000: usize = 4000;

pub const StrategyKind = enum {
    linear,
    sparse,
    dense,
};

pub const StrategyFact = struct {
    kind: StrategyKind,
    name: []const u8,
    requires_exact_selector_validation: bool,
    uses_compressed_index: bool,
};

pub const dispatcher_strategy_facts = [_]StrategyFact{
    .{
        .kind = .linear,
        .name = "linear",
        .requires_exact_selector_validation = true,
        .uses_compressed_index = false,
    },
    .{
        .kind = .sparse,
        .name = "sparse",
        .requires_exact_selector_validation = true,
        .uses_compressed_index = true,
    },
    .{
        .kind = .dense,
        .name = "dense",
        .requires_exact_selector_validation = true,
        .uses_compressed_index = true,
    },
};

pub const Plan = union(enum) {
    linear,
    sparse: SparsePlan,
    dense: DensePlan,
};

pub const Stats = struct {
    switches: usize = 0,
    cases: usize = 0,
    largest_switch_cases: usize = 0,
    linear_worst_checks: usize = 0,
    linear_known_selector_avg_checks_x1000: usize = 0,
    selector_width_candidates: usize = 0,
    chosen_linear: usize = 0,
    chosen_sparse: usize = 0,
    chosen_dense: usize = 0,
    sparse_candidates: usize = 0,
    dense_candidates: usize = 0,
    best_sparse: ?SparsePlan = null,
    best_dense: ?DensePlan = null,
    linear_known_selector_checks: usize = 0,

    pub fn fromProgram(program: ir.Program) Stats {
        var result: Stats = .{};
        for (program.functions) |function| {
            for (function.blocks) |block| {
                switch (block.terminator) {
                    .switch_ => |switch_term| result.observeSwitch(switch_term),
                    else => {},
                }
            }
        }
        return result;
    }

    fn observeSwitch(self: *Stats, switch_term: ir.SwitchTerminator) void {
        self.switches += 1;
        self.cases += switch_term.cases.len;
        self.largest_switch_cases = @max(self.largest_switch_cases, switch_term.cases.len);
        if (switch_term.cases.len != 0) {
            const n = switch_term.cases.len;
            self.linear_worst_checks = @max(self.linear_worst_checks, n);
            self.linear_known_selector_checks += n * (n + 1) / 2;
            self.linear_known_selector_avg_checks_x1000 = divRound(self.linear_known_selector_checks * 1000, self.cases);
        }

        switch (choosePlan(switch_term)) {
            .linear => self.chosen_linear += 1,
            .sparse => self.chosen_sparse += 1,
            .dense => self.chosen_dense += 1,
        }

        if (switch_term.cases.len == 0 or !allCasesAreU32Selectors(switch_term.cases)) return;
        self.selector_width_candidates += 1;

        if (bestSparsePlan(switch_term.cases)) |candidate| {
            self.sparse_candidates += 1;
            if (isBetterSparsePlan(candidate, self.best_sparse)) {
                self.best_sparse = candidate;
            }
        }
        const dense_result = bestDensePlan(switch_term.cases);
        self.dense_candidates += dense_result.candidates;
        if (dense_result.best) |candidate| {
            if (isBetterDensePlan(candidate, self.best_dense)) {
                self.best_dense = candidate;
            }
        }
    }
};

pub const SparsePlan = struct {
    cases: usize = 0,
    bucket_bits: u8 = 0,
    bucket_shift: u8 = 0,
    bucket_count: usize = 0,
    used_buckets: usize = 0,
    empty_buckets: usize = 0,
    singleton_buckets: usize = 0,
    max_bucket_size: usize = 0,
    avg_bucket_size_x1000: usize = 0,
    linear_worst_checks: usize = 0,
    linear_known_selector_avg_checks_x1000: usize = 0,
    sparse_worst_bucket_checks: usize = 0,
    sparse_bucket_dispatch_avg_checks_x1000: usize = 0,
    sparse_known_selector_avg_checks_x1000: usize = 0,
    sparse_total_avg_checks_x1000: usize = 0,
};

pub const DensePlanKind = enum {
    bit_window,
    range,

    pub fn jsonName(self: DensePlanKind) []const u8 {
        return switch (self) {
            .bit_window => "bit_window",
            .range => "range",
        };
    }
};

pub const DensePlan = struct {
    kind: DensePlanKind,
    cases: usize = 0,
    table_slots: usize = 0,
    used_slots: usize = 0,
    hole_slots: usize = 0,
    load_factor_x1000: usize = 0,
    index_bits: ?u8 = null,
    index_shift: ?u8 = null,
    range_min: ?u32 = null,
    range_max: ?u32 = null,
    index_bounds_checks: usize = 0,
    runtime_selector_eq_checks: usize = 1,
    dense_dispatch_avg_checks_x1000: usize = 0,
    dense_total_avg_checks_x1000: usize = 0,
    linear_worst_checks: usize = 0,
    linear_known_selector_avg_checks_x1000: usize = 0,
};

const DensePlanResult = struct {
    candidates: usize = 0,
    best: ?DensePlan = null,
};

pub fn choosePlan(switch_term: ir.SwitchTerminator) Plan {
    if (switch_term.default_target.len == 0 or switch_term.cases.len < 4 or !allCasesAreU32Selectors(switch_term.cases)) {
        return .linear;
    }

    const linear_avg = linearAverageChecksX1000(switch_term.cases.len);

    if (bestSparsePlan(switch_term.cases)) |sparse| {
        if (savesSelectorChecks(linear_avg, sparse.sparse_total_avg_checks_x1000)) {
            return .{ .sparse = sparse };
        }
    }

    // Dense candidates are still reported in metrics, but the current lowering
    // emits a dense slot guard chain plus the exact selector guard. Without a
    // real data-table dynamic jump, that is not cheaper than the linear chain.
    const dense_result = bestDensePlan(switch_term.cases);
    if (dense_result.best) |dense| {
        if (savesSelectorChecks(linear_avg, dense.dense_total_avg_checks_x1000)) return .{ .dense = dense };
    }

    return .linear;
}

pub fn bucketIndex(selector: u32, bucket_bits: u8, bucket_shift: u8) usize {
    std.debug.assert(bucket_bits > 0 and bucket_bits <= 8);
    std.debug.assert(bucket_shift <= 24);
    const mask: u32 = (@as(u32, 1) << @as(u5, @intCast(bucket_bits))) - 1;
    return @intCast((selector >> @as(u5, @intCast(bucket_shift))) & mask);
}

pub fn denseIndex(selector: u32, dense: DensePlan) usize {
    return switch (dense.kind) {
        .bit_window => bucketIndex(selector, dense.index_bits.?, dense.index_shift.?),
        .range => @intCast(selector - dense.range_min.?),
    };
}

fn bestSparsePlan(cases: []const ir.SwitchCase) ?SparsePlan {
    if (cases.len < 4) return null;

    var best: ?SparsePlan = null;
    for (sparse_bucket_bits) |bits| {
        for (sparse_bucket_shifts) |shift| {
            const candidate = analyzeSparsePlan(cases, bits, shift);
            if (candidate.max_bucket_size >= cases.len) continue;
            if (!isSparseTableCandidateAllowed(cases.len, candidate)) continue;
            if (isBetterSparsePlan(candidate, best)) best = candidate;
        }
    }
    return best;
}

fn bestDensePlan(cases: []const ir.SwitchCase) DensePlanResult {
    if (cases.len < 4) return .{};

    var result: DensePlanResult = .{};
    if (denseRangePlan(cases)) |candidate| {
        result.candidates += 1;
        result.best = candidate;
    }

    for (sparse_bucket_bits) |bits| {
        for (sparse_bucket_shifts) |shift| {
            if (denseBitWindowPlan(cases, bits, shift)) |candidate| {
                result.candidates += 1;
                if (isBetterDensePlan(candidate, result.best)) result.best = candidate;
            }
        }
    }
    return result;
}

fn analyzeSparsePlan(cases: []const ir.SwitchCase, bucket_bits: u8, bucket_shift: u8) SparsePlan {
    std.debug.assert(bucket_bits > 0 and bucket_bits <= 8);
    std.debug.assert(bucket_shift <= 24);

    const bucket_count = @as(usize, 1) << @as(std.math.Log2Int(usize), @intCast(bucket_bits));

    var counts = [_]usize{0} ** 256;
    for (cases) |case| {
        const selector = parseU32Selector(case.value).?;
        counts[bucketIndex(selector, bucket_bits, bucket_shift)] += 1;
    }

    var used_buckets: usize = 0;
    var singleton_buckets: usize = 0;
    var max_bucket_size: usize = 0;
    var successful_scan_checks: usize = 0;
    for (counts[0..bucket_count]) |count| {
        if (count == 0) continue;
        used_buckets += 1;
        if (count == 1) singleton_buckets += 1;
        max_bucket_size = @max(max_bucket_size, count);
        successful_scan_checks += count * (count + 1) / 2;
    }

    const n = cases.len;
    const sparse_exact_avg = divRound(successful_scan_checks * 1000, n);
    return .{
        .cases = n,
        .bucket_bits = bucket_bits,
        .bucket_shift = bucket_shift,
        .bucket_count = bucket_count,
        .used_buckets = used_buckets,
        .empty_buckets = bucket_count - used_buckets,
        .singleton_buckets = singleton_buckets,
        .max_bucket_size = max_bucket_size,
        .avg_bucket_size_x1000 = divRound(n * 1000, used_buckets),
        .linear_worst_checks = n,
        .linear_known_selector_avg_checks_x1000 = linearAverageChecksX1000(n),
        .sparse_worst_bucket_checks = max_bucket_size,
        .sparse_bucket_dispatch_avg_checks_x1000 = 0,
        .sparse_known_selector_avg_checks_x1000 = sparse_exact_avg,
        .sparse_total_avg_checks_x1000 = sparse_exact_avg,
    };
}

fn denseRangePlan(cases: []const ir.SwitchCase) ?DensePlan {
    var min_selector: u32 = std.math.maxInt(u32);
    var max_selector: u32 = 0;
    for (cases) |case| {
        const selector = parseU32Selector(case.value).?;
        min_selector = @min(min_selector, selector);
        max_selector = @max(max_selector, selector);
    }

    const table_slots_u64 = @as(u64, max_selector) - @as(u64, min_selector) + 1;
    if (table_slots_u64 > dense_max_table_slots) return null;
    const table_slots: usize = @intCast(table_slots_u64);

    var occupied = [_]bool{false} ** dense_max_table_slots;
    for (cases) |case| {
        const selector = parseU32Selector(case.value).?;
        const index: usize = @intCast(selector - min_selector);
        if (occupied[index]) return null;
        occupied[index] = true;
    }

    return makeDensePlan(.{
        .kind = .range,
        .cases = cases.len,
        .table_slots = table_slots,
        .index_bounds_checks = 1,
        .range_min = min_selector,
        .range_max = max_selector,
    });
}

fn denseBitWindowPlan(cases: []const ir.SwitchCase, index_bits: u8, index_shift: u8) ?DensePlan {
    std.debug.assert(index_bits > 0 and index_bits <= 8);
    std.debug.assert(index_shift <= 24);

    const table_slots = @as(usize, 1) << @as(std.math.Log2Int(usize), @intCast(index_bits));

    var occupied = [_]bool{false} ** 256;
    for (cases) |case| {
        const selector = parseU32Selector(case.value).?;
        const index = bucketIndex(selector, index_bits, index_shift);
        if (occupied[index]) return null;
        occupied[index] = true;
    }

    return makeDensePlan(.{
        .kind = .bit_window,
        .cases = cases.len,
        .table_slots = table_slots,
        .index_bits = index_bits,
        .index_shift = index_shift,
    });
}

fn makeDensePlan(args: struct {
    kind: DensePlanKind,
    cases: usize,
    table_slots: usize,
    index_bits: ?u8 = null,
    index_shift: ?u8 = null,
    range_min: ?u32 = null,
    range_max: ?u32 = null,
    index_bounds_checks: usize = 0,
}) DensePlan {
    return .{
        .kind = args.kind,
        .cases = args.cases,
        .table_slots = args.table_slots,
        .used_slots = args.cases,
        .hole_slots = args.table_slots - args.cases,
        .load_factor_x1000 = divRound(args.cases * 1000, args.table_slots),
        .index_bits = args.index_bits,
        .index_shift = args.index_shift,
        .range_min = args.range_min,
        .range_max = args.range_max,
        .index_bounds_checks = args.index_bounds_checks,
        .dense_dispatch_avg_checks_x1000 = linearAverageChecksX1000(args.cases),
        .dense_total_avg_checks_x1000 = linearAverageChecksX1000(args.cases) + 1000,
        .linear_worst_checks = args.cases,
        .linear_known_selector_avg_checks_x1000 = linearAverageChecksX1000(args.cases),
    };
}

fn allCasesAreU32Selectors(cases: []const ir.SwitchCase) bool {
    for (cases) |case| {
        _ = parseU32Selector(case.value) orelse return false;
    }
    return true;
}

fn isBetterDensePlan(candidate: DensePlan, maybe_best: ?DensePlan) bool {
    const best = maybe_best orelse return true;
    if (candidate.table_slots != best.table_slots) return candidate.table_slots < best.table_slots;
    if (candidate.index_bounds_checks != best.index_bounds_checks) return candidate.index_bounds_checks < best.index_bounds_checks;
    if (candidate.load_factor_x1000 != best.load_factor_x1000) return candidate.load_factor_x1000 > best.load_factor_x1000;
    if (candidate.kind != best.kind) return candidate.kind == .bit_window;
    const candidate_shift = candidate.index_shift orelse std.math.maxInt(u8);
    const best_shift = best.index_shift orelse std.math.maxInt(u8);
    if (candidate_shift != best_shift) return candidate_shift < best_shift;
    const candidate_bits = candidate.index_bits orelse std.math.maxInt(u8);
    const best_bits = best.index_bits orelse std.math.maxInt(u8);
    if (candidate_bits != best_bits) return candidate_bits < best_bits;
    const candidate_min = candidate.range_min orelse std.math.maxInt(u32);
    const best_min = best.range_min orelse std.math.maxInt(u32);
    return candidate_min < best_min;
}

fn isBetterSparsePlan(candidate: SparsePlan, maybe_best: ?SparsePlan) bool {
    const best = maybe_best orelse return true;
    if (candidate.sparse_total_avg_checks_x1000 != best.sparse_total_avg_checks_x1000) {
        return candidate.sparse_total_avg_checks_x1000 < best.sparse_total_avg_checks_x1000;
    }
    if (candidate.max_bucket_size != best.max_bucket_size) return candidate.max_bucket_size < best.max_bucket_size;
    if (candidate.used_buckets != best.used_buckets) return candidate.used_buckets < best.used_buckets;
    if (candidate.bucket_count != best.bucket_count) return candidate.bucket_count < best.bucket_count;
    if (candidate.bucket_shift != best.bucket_shift) return candidate.bucket_shift < best.bucket_shift;
    return candidate.bucket_bits < best.bucket_bits;
}

pub fn parseU32Selector(text: []const u8) ?u32 {
    if (std.mem.startsWith(u8, text, "0x")) {
        return std.fmt.parseUnsigned(u32, text[2..], 16) catch null;
    }
    return std.fmt.parseUnsigned(u32, text, 10) catch null;
}

fn divRound(numerator: usize, denominator: usize) usize {
    std.debug.assert(denominator != 0);
    return (numerator + denominator / 2) / denominator;
}

fn linearAverageChecksX1000(cases: usize) usize {
    std.debug.assert(cases != 0);
    return divRound((cases * (cases + 1) / 2) * 1000, cases);
}

fn savesSelectorChecks(linear_avg_x1000: usize, candidate_avg_x1000: usize) bool {
    return candidate_avg_x1000 + min_selector_check_saving_x1000 <= linear_avg_x1000;
}

fn isSparseTableCandidateAllowed(case_count: usize, candidate: SparsePlan) bool {
    return candidate.bucket_count <= case_count and candidate.used_buckets <= @max(@as(usize, 1), case_count / 4);
}

test "switch routing metrics find eligible sparse table plans" {
    const cases = [_]ir.SwitchCase{
        .{ .value = "0", .target = "a", .line = 1 },
        .{ .value = "1", .target = "b", .line = 2 },
        .{ .value = "2", .target = "c", .line = 3 },
        .{ .value = "3", .target = "d", .line = 4 },
        .{ .value = "4", .target = "e", .line = 5 },
        .{ .value = "5", .target = "f", .line = 6 },
        .{ .value = "6", .target = "g", .line = 7 },
        .{ .value = "7", .target = "h", .line = 8 },
        .{ .value = "8", .target = "i", .line = 9 },
        .{ .value = "9", .target = "j", .line = 10 },
        .{ .value = "10", .target = "k", .line = 11 },
        .{ .value = "11", .target = "l", .line = 12 },
        .{ .value = "12", .target = "m", .line = 13 },
        .{ .value = "13", .target = "n", .line = 14 },
        .{ .value = "14", .target = "o", .line = 15 },
        .{ .value = "15", .target = "p", .line = 16 },
        .{ .value = "16", .target = "q", .line = 17 },
        .{ .value = "17", .target = "r", .line = 18 },
        .{ .value = "18", .target = "s", .line = 19 },
        .{ .value = "19", .target = "t", .line = 20 },
    };
    var stats: Stats = .{};
    stats.observeSwitch(.{ .selector = "selector", .cases = &cases, .default_target = "fallback" });

    try std.testing.expectEqual(@as(usize, 1), stats.switches);
    try std.testing.expectEqual(@as(usize, cases.len), stats.cases);
    try std.testing.expectEqual(@as(usize, 1), stats.chosen_sparse);
    try std.testing.expectEqual(@as(usize, 1), stats.sparse_candidates);
    const best = stats.best_sparse orelse return error.TestUnexpectedResult;
    try std.testing.expect(best.max_bucket_size < cases.len);
    try std.testing.expect(best.sparse_known_selector_avg_checks_x1000 < best.linear_known_selector_avg_checks_x1000);
    try std.testing.expect(best.sparse_total_avg_checks_x1000 >= best.sparse_known_selector_avg_checks_x1000);
}

test "switch routing metrics find dense range plans for compact case values" {
    const cases = [_]ir.SwitchCase{
        .{ .value = "10", .target = "a", .line = 1 },
        .{ .value = "11", .target = "b", .line = 2 },
        .{ .value = "12", .target = "c", .line = 3 },
        .{ .value = "13", .target = "d", .line = 4 },
        .{ .value = "14", .target = "e", .line = 5 },
    };
    var stats: Stats = .{};
    stats.observeSwitch(.{ .selector = "tag", .cases = &cases, .default_target = "fallback" });

    try std.testing.expect(stats.dense_candidates > 0);
    try std.testing.expectEqual(@as(usize, 1), stats.chosen_linear);
    const best = stats.best_dense orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(DensePlanKind.range, best.kind);
    try std.testing.expectEqual(@as(usize, 5), best.table_slots);
    try std.testing.expectEqual(@as(usize, 0), best.hole_slots);
    try std.testing.expectEqual(@as(?u32, 10), best.range_min);
    try std.testing.expectEqual(@as(?u32, 14), best.range_max);
}

test "switch routing metrics find dense bit-window plans for selector-shaped values" {
    const cases = [_]ir.SwitchCase{
        .{ .value = "0x00000001", .target = "a", .line = 1 },
        .{ .value = "0x00000011", .target = "b", .line = 2 },
        .{ .value = "0x00000021", .target = "c", .line = 3 },
        .{ .value = "0x00000031", .target = "d", .line = 4 },
    };
    var stats: Stats = .{};
    stats.observeSwitch(.{ .selector = "selector", .cases = &cases, .default_target = "fallback" });

    try std.testing.expect(stats.dense_candidates > 0);
    try std.testing.expectEqual(@as(usize, 1), stats.chosen_linear);
    const best = stats.best_dense orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(DensePlanKind.bit_window, best.kind);
    try std.testing.expectEqual(@as(usize, 4), best.table_slots);
    try std.testing.expectEqual(@as(usize, 0), best.hole_slots);
    try std.testing.expectEqual(@as(?u8, 2), best.index_bits);
    try std.testing.expectEqual(@as(?u8, 4), best.index_shift);
    try std.testing.expectEqual(@as(usize, 1), best.runtime_selector_eq_checks);
}

test "switch routing metrics reject dense plans with duplicate selectors" {
    const cases = [_]ir.SwitchCase{
        .{ .value = "0x1", .target = "a", .line = 1 },
        .{ .value = "0x1", .target = "b", .line = 2 },
        .{ .value = "0x2", .target = "c", .line = 3 },
        .{ .value = "0x3", .target = "d", .line = 4 },
    };
    var stats: Stats = .{};
    stats.observeSwitch(.{ .selector = "selector", .cases = &cases, .default_target = "fallback" });

    try std.testing.expectEqual(@as(usize, 0), stats.dense_candidates);
    try std.testing.expect(stats.best_dense == null);
}

test "switch routing metrics ignore non-selector-width switch cases" {
    const cases = [_]ir.SwitchCase{
        .{ .value = "0x100000000", .target = "a", .line = 1 },
        .{ .value = "0x2", .target = "b", .line = 2 },
        .{ .value = "0x3", .target = "c", .line = 3 },
        .{ .value = "0x4", .target = "d", .line = 4 },
    };
    var stats: Stats = .{};
    stats.observeSwitch(.{ .selector = "value", .cases = &cases, .default_target = "fallback" });

    try std.testing.expectEqual(@as(usize, 1), stats.switches);
    try std.testing.expectEqual(@as(usize, cases.len), stats.cases);
    try std.testing.expectEqual(@as(usize, 0), stats.selector_width_candidates);
    try std.testing.expectEqual(@as(usize, 1), stats.chosen_linear);
    try std.testing.expectEqual(@as(usize, 0), stats.sparse_candidates);
    try std.testing.expectEqual(@as(usize, 0), stats.dense_candidates);
    try std.testing.expect(stats.best_sparse == null);
    try std.testing.expect(stats.best_dense == null);
}

test "switch routing chooser keeps dense candidates linear until lowering has a real table jump" {
    const cases = [_]ir.SwitchCase{
        .{ .value = "0x00000001", .target = "a", .line = 1 },
        .{ .value = "0x00000011", .target = "b", .line = 2 },
        .{ .value = "0x00000021", .target = "c", .line = 3 },
        .{ .value = "0x00000031", .target = "d", .line = 4 },
    };
    const plan = choosePlan(.{ .selector = "selector", .cases = &cases, .default_target = "fallback" });
    try std.testing.expect(plan == .linear);
}

test "switch routing chooser uses sparse only when current lowering saves selector checks" {
    const cases = [_]ir.SwitchCase{
        .{ .value = "0", .target = "a", .line = 1 },
        .{ .value = "1", .target = "b", .line = 2 },
        .{ .value = "2", .target = "c", .line = 3 },
        .{ .value = "3", .target = "d", .line = 4 },
        .{ .value = "4", .target = "e", .line = 5 },
        .{ .value = "5", .target = "f", .line = 6 },
        .{ .value = "6", .target = "g", .line = 7 },
        .{ .value = "7", .target = "h", .line = 8 },
        .{ .value = "8", .target = "i", .line = 9 },
        .{ .value = "9", .target = "j", .line = 10 },
        .{ .value = "10", .target = "k", .line = 11 },
        .{ .value = "11", .target = "l", .line = 12 },
        .{ .value = "12", .target = "m", .line = 13 },
        .{ .value = "13", .target = "n", .line = 14 },
        .{ .value = "14", .target = "o", .line = 15 },
        .{ .value = "15", .target = "p", .line = 16 },
        .{ .value = "16", .target = "q", .line = 17 },
        .{ .value = "17", .target = "r", .line = 18 },
        .{ .value = "18", .target = "s", .line = 19 },
        .{ .value = "19", .target = "t", .line = 20 },
    };
    const plan = choosePlan(.{ .selector = "selector", .cases = &cases, .default_target = "fallback" });
    try std.testing.expect(plan == .sparse);
}

test "switch routing strategy fact table covers planner variants" {
    const plan_fields = @typeInfo(Plan).@"union".fields;
    try std.testing.expectEqual(plan_fields.len, dispatcher_strategy_facts.len);
    inline for (plan_fields) |field| {
        var found = false;
        for (dispatcher_strategy_facts) |fact| {
            if (std.mem.eql(u8, fact.name, field.name)) {
                found = true;
                break;
            }
        }
        try std.testing.expect(found);
    }
}
