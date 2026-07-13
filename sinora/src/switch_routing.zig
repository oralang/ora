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
// One-check switching margin. This constant predates the policy-score
// model, when its job was to absorb the (then-unmodeled) table dispatch
// overhead; scores now price that overhead and code bytes explicitly, so
// only a small model-error margin remains. Lean-pinned in
// Ora/Spec/DispatcherFacts.lean.
pub const min_selector_check_saving_x1000: usize = 1000;
pub const check_scale_x1000: usize = 1000;
pub const exact_selector_check_x1000: usize = 1000;

/// Dispatch-overhead costs in check-equivalents. One "check" is the exact
/// case guard as release codegen emits it (selector reload + PUSH4 + EQ +
/// JUMPI ≈ 25 gas). The jump-table machinery — index compute, scratch-word
/// zero, MUL/ADD, CODECOPY, MLOAD, JUMP — costs ≈ 70 gas ≈ 2.8 checks and
/// is paid once per routed call by both sparse and dense plans.
pub const table_dispatch_overhead_checks_x1000: usize = 2800;
/// Multiplicative plans additionally pay PUSH4 C + MUL ≈ 8 gas ≈ 0.3 checks.
pub const dense_multiplicative_extra_checks_x1000: usize = 300;
/// Deterministic budget for the multiplicative-constant search, per table
/// size. Candidates come from a fixed splitmix32 sequence, so the same
/// source always finds the same constant — byte-stable builds.
pub const multiplicative_search_budget: usize = 512;

/// Dispatch code-size policy. Every plan is scored as
/// `runtime_avg_checks_x1000 + lambda_x1000_per_byte * plan_code_bytes` and
/// the cheapest score wins. Lambda converts code bytes into check
/// equivalents: code deposit costs 200 gas/byte once, runtime cost recurs
/// per call, so lambda encodes an assumed amortization horizon
/// (lambda_x1000 = 200 / 25 / N_calls * 1000). `.gas` optimizes runtime
/// only; `.balanced` assumes ~1600 calls over the contract's life; `.size`
/// assumes ~160 and suits deploy-constrained contracts.
pub const DispatchPolicy = enum {
    gas,
    balanced,
    size,

    pub fn lambdaX1000PerByte(self: DispatchPolicy) usize {
        return switch (self) {
            .gas => 0,
            .balanced => 5,
            .size => 50,
        };
    }

    pub fn jsonName(self: DispatchPolicy) []const u8 {
        return @tagName(self);
    }
};

/// Active policy for this compilation. Sinora is a batch binary: main.zig
/// sets this exactly once from `--optimize=<profile>` before any codegen or
/// metrics run, so every choosePlan call in one invocation sees the same
/// policy and builds stay byte-deterministic per (input, profile). Tests
/// that override it must restore `.balanced` before returning.
pub var dispatch_policy: DispatchPolicy = .balanced;

pub fn parsePolicyName(name: []const u8) ?DispatchPolicy {
    inline for (@typeInfo(DispatchPolicy).@"enum".fields) |field| {
        if (std.mem.eql(u8, name, field.name)) return @field(DispatchPolicy, field.name);
    }
    return null;
}

pub const jump_table_entry_bytes: usize = 2;
pub const linear_case_code_bytes: usize = 13;
pub const dense_bit_window_preamble_code_bytes: usize = 30;
pub const dense_multiplicative_preamble_code_bytes: usize = 36;
pub const dense_used_slot_code_bytes: usize = 18;
pub const sparse_preamble_code_bytes: usize = 30;
pub const sparse_used_bucket_code_bytes: usize = 5;
pub const sparse_case_code_bytes: usize = 17;

fn planScoreX1000(runtime_avg_checks_x1000: usize, code_bytes: usize) usize {
    return runtime_avg_checks_x1000 + dispatch_policy.lambdaX1000PerByte() * code_bytes;
}

// Approximate emitted-code sizes per routing shape. These only need to be
// honest enough to rank plans; the real arbiter of regressions is the
// committed bytecode-size baseline.
fn linearCodeBytes(cases: usize) usize {
    // One exact check (selector reload + PUSH4 + EQ + PUSH2 + JUMPI) per case.
    return linear_case_code_bytes * cases;
}

fn densePlanCodeBytes(plan: DensePlan) usize {
    const preamble: usize = switch (plan.kind) {
        .bit_window => dense_bit_window_preamble_code_bytes,
        .multiplicative => dense_multiplicative_preamble_code_bytes,
    };
    // Table entries plus one landing block (JUMPDEST + exact check +
    // default jump) per used slot.
    return preamble +
        jump_table_entry_bytes * plan.table_slots +
        dense_used_slot_code_bytes * plan.used_slots;
}

fn sparsePlanCodeBytes(plan: SparsePlan) usize {
    // Dispatch preamble, table entries, per-bucket block overhead, and the
    // in-bucket exact chains (one check per case plus the default jump).
    return sparse_preamble_code_bytes +
        jump_table_entry_bytes * plan.bucket_count +
        sparse_used_bucket_code_bytes * plan.used_buckets +
        sparse_case_code_bytes * plan.cases;
}

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

pub const PlanTrace = struct {
    chosen: Plan,
    accepted_dense: ?DensePlan = null,
    accepted_sparse: ?SparsePlan = null,
};

/// One proof-relevant candidate from the planner's bounded search.
pub const ScoredPlan = struct {
    plan: Plan,
    score_x1000: usize,
};

pub const MultiplicativeCollisionWitness = struct {
    constant: u32,
    first_case: usize,
    second_case: usize,
};

pub const MultiplicativeSearchTrace = struct {
    table_slots: usize,
    selected_candidate_index: ?u32,
    rejected: []MultiplicativeCollisionWitness,
};

/// Complete per-switch planner evidence consumed by the Lean userland gate.
/// The runtime planner remains allocation-free; this trace is built only when
/// formal dispatcher facts are requested.
pub const DetailedPlanTrace = struct {
    policy: DispatchPolicy,
    preconditions_met: bool,
    linear_score_x1000: usize,
    dense_candidates: []ScoredPlan,
    multiplicative_searches: []MultiplicativeSearchTrace,
    best_dense: ?ScoredPlan,
    sparse_candidates: []ScoredPlan,
    best_sparse: ?ScoredPlan,
    chosen: Plan,

    pub fn deinit(self: *DetailedPlanTrace, allocator: std.mem.Allocator) void {
        for (self.multiplicative_searches) |search| allocator.free(search.rejected);
        allocator.free(self.multiplicative_searches);
        allocator.free(self.dense_candidates);
        allocator.free(self.sparse_candidates);
        self.* = undefined;
    }
};

fn traceMultiplicativeSearch(
    allocator: std.mem.Allocator,
    cases: []const ir.SwitchCase,
    table_slots: usize,
) !MultiplicativeSearchTrace {
    const bits: u8 = @intCast(std.math.log2_int(usize, table_slots));
    var rejected: std.ArrayList(MultiplicativeCollisionWitness) = .empty;
    defer rejected.deinit(allocator);

    var candidate_index: u32 = 0;
    while (candidate_index < multiplicative_search_budget) : (candidate_index += 1) {
        const constant = multiplicativeCandidate(candidate_index);
        var occupied = [_]?usize{null} ** dense_max_table_slots;
        var collision: ?MultiplicativeCollisionWitness = null;
        for (cases, 0..) |case, case_index| {
            const selector = parseU32Selector(case.value).?;
            const slot = multiplicativeIndex(selector, constant, bits);
            if (occupied[slot]) |first_case| {
                collision = .{
                    .constant = constant,
                    .first_case = first_case,
                    .second_case = case_index,
                };
                break;
            }
            occupied[slot] = case_index;
        }
        if (collision) |witness| {
            try rejected.append(allocator, witness);
            continue;
        }
        return .{
            .table_slots = table_slots,
            .selected_candidate_index = candidate_index,
            .rejected = try rejected.toOwnedSlice(allocator),
        };
    }
    return .{
        .table_slots = table_slots,
        .selected_candidate_index = null,
        .rejected = try rejected.toOwnedSlice(allocator),
    };
}

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
            self.linear_known_selector_avg_checks_x1000 = divRound(
                self.linear_known_selector_checks * exact_selector_check_x1000,
                self.cases,
            );
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

// The `range` kind (index = selector - min with a wrap-safe bounds check)
// was retired by the policy-score model: multiplicative perfect hashing
// finds a small pow2 table for every compact case set at lower runtime AND
// byte cost, so range had no reachable selection path left — exactly the
// advertised-but-dead-strategy shape this planner is not allowed to carry.
// Recover the emitter from git history if a producer ever needs true
// range tables.
pub const DensePlanKind = enum {
    bit_window,
    multiplicative,

    pub fn jsonName(self: DensePlanKind) []const u8 {
        return switch (self) {
            .bit_window => "bit_window",
            .multiplicative => "multiplicative",
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
    mul_constant: ?u32 = null,
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

pub fn choosePlanTrace(switch_term: ir.SwitchTerminator) PlanTrace {
    if (switch_term.default_target.len == 0 or switch_term.cases.len < 4 or !allCasesAreU32Selectors(switch_term.cases)) {
        return .{ .chosen = .linear };
    }

    const linear_score = planScoreX1000(
        linearAverageChecksX1000(switch_term.cases.len),
        linearCodeBytes(switch_term.cases.len),
    );

    // Dense first: it routes through the same jump-table machinery as sparse
    // but lands on exactly one exact-selector guard, so any qualifying dense
    // plan is at least as cheap as any qualifying sparse plan (whose buckets
    // scan >= 1 guards after identical table overhead).
    const dense_result = bestDensePlan(switch_term.cases);
    if (dense_result.best) |dense| {
        const score = planScoreX1000(dense.dense_total_avg_checks_x1000, densePlanCodeBytes(dense));
        if (savesSelectorChecks(linear_score, score)) {
            return .{
                .chosen = .{ .dense = dense },
                .accepted_dense = dense,
            };
        }
    }

    if (bestSparsePlan(switch_term.cases)) |sparse| {
        const score = planScoreX1000(sparse.sparse_total_avg_checks_x1000, sparsePlanCodeBytes(sparse));
        if (savesSelectorChecks(linear_score, score)) {
            return .{
                .chosen = .{ .sparse = sparse },
                .accepted_sparse = sparse,
            };
        }
    }

    return .{ .chosen = .linear };
}

pub fn choosePlan(switch_term: ir.SwitchTerminator) Plan {
    return choosePlanTrace(switch_term).chosen;
}

/// Enumerate the complete finite search used by `choosePlan`, including the
/// multiplicative perfect-hash search. The formal emitter projects candidate
/// counts, best scores, and collision witnesses from this trace. Lean
/// regenerates the candidate lists from selectors and policy before checking
/// those projections and the chosen plan independently.
pub fn detailedPlanTrace(
    allocator: std.mem.Allocator,
    switch_term: ir.SwitchTerminator,
) !DetailedPlanTrace {
    const preconditions_met = switch_term.default_target.len != 0 and
        switch_term.cases.len >= 4 and allCasesAreU32Selectors(switch_term.cases);
    const linear_score = if (switch_term.cases.len == 0)
        0
    else
        planScoreX1000(
            linearAverageChecksX1000(switch_term.cases.len),
            linearCodeBytes(switch_term.cases.len),
        );

    var dense_candidates: std.ArrayList(ScoredPlan) = .empty;
    defer dense_candidates.deinit(allocator);
    var multiplicative_searches: std.ArrayList(MultiplicativeSearchTrace) = .empty;
    defer {
        for (multiplicative_searches.items) |search| allocator.free(search.rejected);
        multiplicative_searches.deinit(allocator);
    }
    var sparse_candidates: std.ArrayList(ScoredPlan) = .empty;
    defer sparse_candidates.deinit(allocator);

    if (preconditions_met) {
        for (sparse_bucket_bits) |bits| {
            for (sparse_bucket_shifts) |shift| {
                if (denseBitWindowPlan(switch_term.cases, bits, shift)) |candidate| {
                    try dense_candidates.append(allocator, .{
                        .plan = .{ .dense = candidate },
                        .score_x1000 = planScoreX1000(
                            candidate.dense_total_avg_checks_x1000,
                            densePlanCodeBytes(candidate),
                        ),
                    });
                }
            }
        }

        var table_slots = try std.math.ceilPowerOfTwo(usize, @max(switch_term.cases.len, 2));
        while (table_slots <= dense_max_table_slots) : (table_slots *= 2) {
            const search = try traceMultiplicativeSearch(allocator, switch_term.cases, table_slots);
            multiplicative_searches.append(allocator, search) catch |err| {
                allocator.free(search.rejected);
                return err;
            };
            if (search.selected_candidate_index) |candidate_index| {
                const bits: u8 = @intCast(std.math.log2_int(usize, table_slots));
                const candidate = makeDensePlan(.{
                    .kind = .multiplicative,
                    .cases = switch_term.cases.len,
                    .table_slots = table_slots,
                    .index_bits = bits,
                    .index_shift = 32 - bits,
                    .mul_constant = multiplicativeCandidate(candidate_index),
                });
                try dense_candidates.append(allocator, .{
                    .plan = .{ .dense = candidate },
                    .score_x1000 = planScoreX1000(
                        candidate.dense_total_avg_checks_x1000,
                        densePlanCodeBytes(candidate),
                    ),
                });
            }
        }

        for (sparse_bucket_bits) |bits| {
            for (sparse_bucket_shifts) |shift| {
                const candidate = analyzeSparsePlan(switch_term.cases, bits, shift);
                if (candidate.max_bucket_size >= switch_term.cases.len) continue;
                try sparse_candidates.append(allocator, .{
                    .plan = .{ .sparse = candidate },
                    .score_x1000 = planScoreX1000(
                        candidate.sparse_total_avg_checks_x1000,
                        sparsePlanCodeBytes(candidate),
                    ),
                });
            }
        }
    }

    const dense_owned = try dense_candidates.toOwnedSlice(allocator);
    errdefer allocator.free(dense_owned);
    const multiplicative_owned = try multiplicative_searches.toOwnedSlice(allocator);
    errdefer {
        for (multiplicative_owned) |search| allocator.free(search.rejected);
        allocator.free(multiplicative_owned);
    }
    const sparse_owned = try sparse_candidates.toOwnedSlice(allocator);
    errdefer allocator.free(sparse_owned);

    const dense_result: DensePlanResult = if (preconditions_met) bestDensePlan(switch_term.cases) else .{};
    const best_dense: ?ScoredPlan = if (dense_result.best) |dense| .{
        .plan = .{ .dense = dense },
        .score_x1000 = planScoreX1000(dense.dense_total_avg_checks_x1000, densePlanCodeBytes(dense)),
    } else null;
    const sparse = if (preconditions_met) bestSparsePlan(switch_term.cases) else null;
    const best_sparse: ?ScoredPlan = if (sparse) |candidate| .{
        .plan = .{ .sparse = candidate },
        .score_x1000 = planScoreX1000(
            candidate.sparse_total_avg_checks_x1000,
            sparsePlanCodeBytes(candidate),
        ),
    } else null;

    return .{
        .policy = dispatch_policy,
        .preconditions_met = preconditions_met,
        .linear_score_x1000 = linear_score,
        .dense_candidates = dense_owned,
        .multiplicative_searches = multiplicative_owned,
        .best_dense = best_dense,
        .sparse_candidates = sparse_owned,
        .best_sparse = best_sparse,
        .chosen = choosePlan(switch_term),
    };
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
        .multiplicative => multiplicativeIndex(selector, dense.mul_constant.?, dense.index_bits.?),
    };
}

/// Multiply-shift hashing (Dietzfelbinger): the top `bits` bits of the low
/// 32-bit half of selector*C. On EVM this is `((sel * C) >> (32 - bits)) &
/// mask` with a plain 256-bit MUL — product bits at or above 32 land at
/// positions >= bits after the shift and are cleared by the mask, so no
/// extra truncation op is needed.
pub fn multiplicativeIndex(selector: u32, constant: u32, bits: u8) usize {
    std.debug.assert(bits > 0 and bits <= 8);
    const product = @as(u64, selector) * @as(u64, constant);
    const shift: u6 = @intCast(32 - @as(u32, bits));
    const mask: u64 = (@as(u64, 1) << @as(u6, @intCast(bits))) - 1;
    return @intCast((product >> shift) & mask);
}

/// Fixed-seed splitmix32 giving the deterministic multiplicative-constant
/// candidate sequence (forced odd — even constants lose low entropy).
pub fn multiplicativeCandidate(index: u32) u32 {
    var z = index +% 0x9E3779B9;
    z = (z ^ (z >> 16)) *% 0x85EBCA6B;
    z = (z ^ (z >> 13)) *% 0xC2B2AE35;
    z = z ^ (z >> 16);
    return z | 1;
}

fn bestSparsePlan(cases: []const ir.SwitchCase) ?SparsePlan {
    if (cases.len < 4) return null;

    var best: ?SparsePlan = null;
    for (sparse_bucket_bits) |bits| {
        for (sparse_bucket_shifts) |shift| {
            const candidate = analyzeSparsePlan(cases, bits, shift);
            // A plan whose worst bucket holds every case degenerates to the
            // linear chain plus table overhead; everything else competes on
            // the policy score (big tables pay their bytes there — no more
            // hard bucket-count cliffs).
            if (candidate.max_bucket_size >= cases.len) continue;
            if (isBetterSparsePlan(candidate, best)) best = candidate;
        }
    }
    return best;
}

fn bestDensePlan(cases: []const ir.SwitchCase) DensePlanResult {
    if (cases.len < 4) return .{};

    var result: DensePlanResult = .{};
    for (sparse_bucket_bits) |bits| {
        for (sparse_bucket_shifts) |shift| {
            if (denseBitWindowPlan(cases, bits, shift)) |candidate| {
                result.candidates += 1;
                if (isBetterDensePlan(candidate, result.best)) result.best = candidate;
            }
        }
    }

    var table_slots = std.math.ceilPowerOfTwo(usize, @max(cases.len, 2)) catch return result;
    while (table_slots <= dense_max_table_slots) : (table_slots *= 2) {
        if (denseMultiplicativePlan(cases, table_slots)) |candidate| {
            result.candidates += 1;
            if (isBetterDensePlan(candidate, result.best)) result.best = candidate;
        }
    }
    return result;
}

fn denseMultiplicativePlan(cases: []const ir.SwitchCase, table_slots: usize) ?DensePlan {
    std.debug.assert(std.math.isPowerOfTwo(table_slots) and table_slots <= dense_max_table_slots);
    if (cases.len > table_slots) return null;
    const bits: u8 = @intCast(std.math.log2_int(usize, table_slots));
    if (bits == 0 or bits > 8) return null;

    var candidate_index: u32 = 0;
    while (candidate_index < multiplicative_search_budget) : (candidate_index += 1) {
        const constant = multiplicativeCandidate(candidate_index);
        var occupied = [_]bool{false} ** dense_max_table_slots;
        var collision_free = true;
        for (cases) |case| {
            const selector = parseU32Selector(case.value).?;
            const slot = multiplicativeIndex(selector, constant, bits);
            if (occupied[slot]) {
                collision_free = false;
                break;
            }
            occupied[slot] = true;
        }
        if (collision_free) {
            return makeDensePlan(.{
                .kind = .multiplicative,
                .cases = cases.len,
                .table_slots = table_slots,
                .index_bits = bits,
                .index_shift = 32 - bits,
                .mul_constant = constant,
            });
        }
    }
    return null;
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
    const sparse_exact_avg = divRound(
        successful_scan_checks * exact_selector_check_x1000,
        n,
    );
    return .{
        .cases = n,
        .bucket_bits = bucket_bits,
        .bucket_shift = bucket_shift,
        .bucket_count = bucket_count,
        .used_buckets = used_buckets,
        .empty_buckets = bucket_count - used_buckets,
        .singleton_buckets = singleton_buckets,
        .max_bucket_size = max_bucket_size,
        .avg_bucket_size_x1000 = divRound(n * check_scale_x1000, used_buckets),
        .linear_worst_checks = n,
        .linear_known_selector_avg_checks_x1000 = linearAverageChecksX1000(n),
        .sparse_worst_bucket_checks = max_bucket_size,
        .sparse_bucket_dispatch_avg_checks_x1000 = table_dispatch_overhead_checks_x1000,
        .sparse_known_selector_avg_checks_x1000 = sparse_exact_avg,
        .sparse_total_avg_checks_x1000 = sparse_exact_avg + table_dispatch_overhead_checks_x1000,
    };
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
    mul_constant: ?u32 = null,
}) DensePlan {
    const dispatch_overhead_x1000 = table_dispatch_overhead_checks_x1000 + switch (args.kind) {
        .bit_window => 0,
        .multiplicative => dense_multiplicative_extra_checks_x1000,
    };
    return .{
        .kind = args.kind,
        .cases = args.cases,
        .table_slots = args.table_slots,
        .used_slots = args.cases,
        .hole_slots = args.table_slots - args.cases,
        .load_factor_x1000 = divRound(args.cases * check_scale_x1000, args.table_slots),
        .index_bits = args.index_bits,
        .index_shift = args.index_shift,
        .mul_constant = args.mul_constant,
        .dense_dispatch_avg_checks_x1000 = dispatch_overhead_x1000,
        // O(1) table route plus exactly one exact-selector guard in the
        // landing block; independent of case count.
        .dense_total_avg_checks_x1000 = dispatch_overhead_x1000 + exact_selector_check_x1000,
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
    const candidate_score = planScoreX1000(candidate.dense_total_avg_checks_x1000, densePlanCodeBytes(candidate));
    const best_score = planScoreX1000(best.dense_total_avg_checks_x1000, densePlanCodeBytes(best));
    if (candidate_score != best_score) return candidate_score < best_score;
    if (candidate.table_slots != best.table_slots) return candidate.table_slots < best.table_slots;
    if (candidate.load_factor_x1000 != best.load_factor_x1000) return candidate.load_factor_x1000 > best.load_factor_x1000;
    if (candidate.kind != best.kind) return candidate.kind == .bit_window;
    const candidate_shift = candidate.index_shift orelse std.math.maxInt(u8);
    const best_shift = best.index_shift orelse std.math.maxInt(u8);
    if (candidate_shift != best_shift) return candidate_shift < best_shift;
    const candidate_bits = candidate.index_bits orelse std.math.maxInt(u8);
    const best_bits = best.index_bits orelse std.math.maxInt(u8);
    if (candidate_bits != best_bits) return candidate_bits < best_bits;
    return false;
}

fn isBetterSparsePlan(candidate: SparsePlan, maybe_best: ?SparsePlan) bool {
    const best = maybe_best orelse return true;
    const candidate_score = planScoreX1000(candidate.sparse_total_avg_checks_x1000, sparsePlanCodeBytes(candidate));
    const best_score = planScoreX1000(best.sparse_total_avg_checks_x1000, sparsePlanCodeBytes(best));
    if (candidate_score != best_score) return candidate_score < best_score;
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
    return divRound(
        (cases * (cases + 1) / 2) * exact_selector_check_x1000,
        cases,
    );
}

// Switching hysteresis: a table shape must beat the linear score by at
// least this margin (in milli-check-equivalents) before we leave the
// simplest lowering. Both sides are policy scores (runtime + lambda*bytes).
fn savesSelectorChecks(linear_score_x1000: usize, candidate_score_x1000: usize) bool {
    return candidate_score_x1000 + min_selector_check_saving_x1000 <= linear_score_x1000;
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
    // Sequential 0..19 admits a collision-free bit window, and a real dense
    // route (O(1) + one exact check) beats bucket scanning — chooser is dense.
    try std.testing.expectEqual(@as(usize, 1), stats.chosen_dense);
    try std.testing.expectEqual(@as(usize, 1), stats.sparse_candidates);
    const best = stats.best_sparse orelse return error.TestUnexpectedResult;
    try std.testing.expect(best.max_bucket_size < cases.len);
    try std.testing.expect(best.sparse_known_selector_avg_checks_x1000 < best.linear_known_selector_avg_checks_x1000);
    try std.testing.expect(best.sparse_total_avg_checks_x1000 >= best.sparse_known_selector_avg_checks_x1000);
    try std.testing.expectEqual(table_dispatch_overhead_checks_x1000, best.sparse_bucket_dispatch_avg_checks_x1000);
}

test "switch routing metrics keep small compact switches linear" {
    // Five compact tag values admit dense candidates (bit windows over the
    // low bits), but a 5-case linear chain out-scores any table shape under
    // every policy — the metrics still report the candidates.
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
    try std.testing.expectEqual(@as(usize, 5), best.used_slots);
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

test "switch routing chooser keeps small switches linear even with perfect dense windows" {
    // 4 cases average 2.5 exact checks on the linear chain — cheaper than any
    // table route's fixed dispatch overhead plus its landing-block check.
    const cases = [_]ir.SwitchCase{
        .{ .value = "0x00000001", .target = "a", .line = 1 },
        .{ .value = "0x00000011", .target = "b", .line = 2 },
        .{ .value = "0x00000021", .target = "c", .line = 3 },
        .{ .value = "0x00000031", .target = "d", .line = 4 },
    };
    const plan = choosePlan(.{ .selector = "selector", .cases = &cases, .default_target = "fallback" });
    try std.testing.expect(plan == .linear);
}

test "switch routing chooser picks dense bit-window for compact case sets" {
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
    try std.testing.expect(plan == .dense);
    // Sequential 0..19 sits collision-free in the 5-bit window at shift 0 —
    // the cheapest table shape for the set (no MUL, 32 slots).
    try std.testing.expectEqual(DensePlanKind.bit_window, plan.dense.kind);
    try std.testing.expectEqual(@as(?u8, 5), plan.dense.index_bits);
}

test "switch routing multiplicative hashing rescues sets no bit window can table" {
    // The pair 0x…01/0x…02 differs only in bits [0,2) and the pair
    // 0x00000001/0x10000000 collides in every low window, so no (bits, shift)
    // bit-window is collision-free across the set, and the value span exceeds
    // dense_max_table_slots so no range plan exists. Before the multiplicative
    // kind this set fell back to sparse buckets; the constant search finds a
    // collision-free multiply-shift table, so it now routes dense. Sparse's
    // own liveness witness is the >256-case switch in release_generic_backend
    // ("lowers sparse switch routing"), which no dense table can hold.
    const cases = [_]ir.SwitchCase{
        .{ .value = "0x00000001", .target = "a", .line = 1 },
        .{ .value = "0x00000002", .target = "b", .line = 2 },
        .{ .value = "0x10000000", .target = "c", .line = 3 },
        .{ .value = "0x20000000", .target = "d", .line = 4 },
        .{ .value = "0x00000003", .target = "e", .line = 5 },
        .{ .value = "0x00000101", .target = "f", .line = 6 },
        .{ .value = "0x00000102", .target = "g", .line = 7 },
        .{ .value = "0x10000103", .target = "h", .line = 8 },
        .{ .value = "0x00000104", .target = "i", .line = 9 },
        .{ .value = "0x00000105", .target = "j", .line = 10 },
        .{ .value = "0x00000201", .target = "k", .line = 11 },
        .{ .value = "0x00000202", .target = "l", .line = 12 },
        .{ .value = "0x00000203", .target = "m", .line = 13 },
        .{ .value = "0x00000204", .target = "n", .line = 14 },
        .{ .value = "0x00000205", .target = "o", .line = 15 },
        .{ .value = "0x00000301", .target = "p", .line = 16 },
        .{ .value = "0x00000302", .target = "q", .line = 17 },
        .{ .value = "0x00000303", .target = "r", .line = 18 },
        .{ .value = "0x00000304", .target = "s", .line = 19 },
        .{ .value = "0x00000305", .target = "t", .line = 20 },
    };
    const plan = choosePlan(.{ .selector = "selector", .cases = &cases, .default_target = "fallback" });
    try std.testing.expect(plan == .dense);
    try std.testing.expectEqual(DensePlanKind.multiplicative, plan.dense.kind);
}

test "switch routing chooser picks dense for the dispatcher liveness contract" {
    // Liveness pin: the 16 keccak selectors of d0()..d15() from
    // tests/conformance/dispatcher_dense_routing.ora. A 7-bit window at
    // shift 0 is collision-free (128 slots), but the multiplicative search
    // finds a smaller collision-free table, and smaller tables win the
    // comparator — if this stops choosing dense, the dispatcher execution
    // coverage in that conformance spec silently degrades.
    const cases = [_]ir.SwitchCase{
        .{ .value = "0xa9874b2a", .target = "a", .line = 1 },
        .{ .value = "0x8c18f2f1", .target = "b", .line = 2 },
        .{ .value = "0x7ef1dbea", .target = "c", .line = 3 },
        .{ .value = "0x1946b5be", .target = "d", .line = 4 },
        .{ .value = "0x848ea577", .target = "e", .line = 5 },
        .{ .value = "0x56f897ef", .target = "f", .line = 6 },
        .{ .value = "0x5fc75cb5", .target = "g", .line = 7 },
        .{ .value = "0x80b3bd9b", .target = "h", .line = 8 },
        .{ .value = "0x8b7d553d", .target = "i", .line = 9 },
        .{ .value = "0xbaf74a09", .target = "j", .line = 10 },
        .{ .value = "0xe0265090", .target = "k", .line = 11 },
        .{ .value = "0x13f2ec66", .target = "l", .line = 12 },
        .{ .value = "0x1d12652e", .target = "m", .line = 13 },
        .{ .value = "0xc57076af", .target = "n", .line = 14 },
        .{ .value = "0x1b15e99d", .target = "o", .line = 15 },
        .{ .value = "0x29a6a38d", .target = "p", .line = 16 },
    };
    const plan = choosePlan(.{ .selector = "selector", .cases = &cases, .default_target = "fallback" });
    try std.testing.expect(plan == .dense);
    try std.testing.expectEqual(DensePlanKind.multiplicative, plan.dense.kind);
    try std.testing.expect(plan.dense.table_slots < 128);
}

test "switch routing policies pick distinct shapes for the dispatcher liveness contract" {
    // Same 16 keccak selectors as the liveness pin. `gas` ignores bytes and
    // takes the cheapest runtime (bit window, no MUL); `balanced` pays 8 gas
    // of MUL for a table a third the size; `size` refuses the table bytes
    // entirely and stays on the linear chain.
    const cases = [_]ir.SwitchCase{
        .{ .value = "0xa9874b2a", .target = "a", .line = 1 },
        .{ .value = "0x8c18f2f1", .target = "b", .line = 2 },
        .{ .value = "0x7ef1dbea", .target = "c", .line = 3 },
        .{ .value = "0x1946b5be", .target = "d", .line = 4 },
        .{ .value = "0x848ea577", .target = "e", .line = 5 },
        .{ .value = "0x56f897ef", .target = "f", .line = 6 },
        .{ .value = "0x5fc75cb5", .target = "g", .line = 7 },
        .{ .value = "0x80b3bd9b", .target = "h", .line = 8 },
        .{ .value = "0x8b7d553d", .target = "i", .line = 9 },
        .{ .value = "0xbaf74a09", .target = "j", .line = 10 },
        .{ .value = "0xe0265090", .target = "k", .line = 11 },
        .{ .value = "0x13f2ec66", .target = "l", .line = 12 },
        .{ .value = "0x1d12652e", .target = "m", .line = 13 },
        .{ .value = "0xc57076af", .target = "n", .line = 14 },
        .{ .value = "0x1b15e99d", .target = "o", .line = 15 },
        .{ .value = "0x29a6a38d", .target = "p", .line = 16 },
    };
    const term = ir.SwitchTerminator{ .selector = "selector", .cases = &cases, .default_target = "fallback" };
    defer dispatch_policy = .balanced;

    dispatch_policy = .gas;
    const gas_plan = choosePlan(term);
    try std.testing.expect(gas_plan == .dense);
    try std.testing.expectEqual(DensePlanKind.bit_window, gas_plan.dense.kind);

    dispatch_policy = .balanced;
    const balanced_plan = choosePlan(term);
    try std.testing.expect(balanced_plan == .dense);
    try std.testing.expectEqual(DensePlanKind.multiplicative, balanced_plan.dense.kind);

    dispatch_policy = .size;
    const size_plan = choosePlan(term);
    try std.testing.expect(size_plan == .linear);
}

test "switch routing keeps sparse as the large-switch strategy" {
    // Retirement-watch close-out (verdict: KEEP). Above ~60 cases the
    // 256-slot table cap plus birthday collisions defeat every dense
    // candidate (P(collision-free) ~ exp(-n^2/512) per constant), so sparse
    // buckets are the only sub-linear shape for large selector sets.
    var value_bufs: [80][12]u8 = undefined;
    var cases: [80]ir.SwitchCase = undefined;
    for (&cases, 0..) |*case, i| {
        const selector = multiplicativeCandidate(@intCast(i));
        const text = std.fmt.bufPrint(&value_bufs[i], "0x{x:0>8}", .{selector}) catch unreachable;
        case.* = .{ .value = text, .target = "hit", .line = @intCast(i + 1) };
    }
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
