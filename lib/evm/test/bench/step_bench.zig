//! Step-rate benchmark for the debugger driver.
//!
//! Builds an in-memory contract that loops 50_000 times over a few opcodes,
//! drives the Debugger for the entire trace, and reports the average per-
//! opcode wall-clock time. The acceptance bar from the A2 perf-cliff phase
//! is "<100µs avg per step on a fixture" (see plan-file Phase 2/A2). The
//! per-step cost is dominated by EVM step + statement-boundary check, both
//! of which are now O(1) thanks to the hash-index + precompute work.
//!
//! This bench is meant to be tracked in CI so we notice the day a future
//! change reintroduces a per-step linear scan. It runs under `zig build
//! bench`.

const std = @import("std");
const primitives = @import("voltaire");
const Address = primitives.Address.Address;
const ora_evm = @import("ora_evm");

const Evm = ora_evm.Evm(.{});
const Frame = ora_evm.Frame(.{});
const Debugger = ora_evm.Debugger(.{});
const SourceMap = ora_evm.SourceMap;
const DebugInfo = ora_evm.DebugInfo;
const BlockContext = ora_evm.BlockContext;

/// Number of full iterations of the inner loop the benchmark runs.
const ITERATIONS: usize = 50_000;

/// Per-step time budget enforced by the bench. Bumping this means the
/// debugger has slowed down on the per-step path. Investigate before
/// raising.
const STEP_BUDGET_NS: u64 = 100_000; // 100 µs

pub fn main() !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const allocator = gpa_state.allocator();

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_file = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_file.interface;

    // Build a tight loop: PUSH1 N; JUMPDEST; ADD-ish work; PUSH1 jumpdest; JUMP.
    // We construct the bytecode ourselves so the loop count is known to the
    // bench rather than depending on the compiler.
    const bytecode = try makeLoopBytecode(allocator, ITERATIONS);
    defer allocator.free(bytecode);

    var evm = try buildEvm(allocator);
    defer {
        evm.deinit();
        allocator.destroy(evm);
    }
    try buildFrame(evm, bytecode);

    // Source-map: one statement entry per opcode position (worst case for the
    // statement-boundary check; every step is a candidate).
    var sm_entries: std.ArrayList(SourceMap.Entry) = .{};
    defer sm_entries.deinit(allocator);
    for (0..bytecode.len) |i| {
        try sm_entries.append(allocator, .{
            .idx = @intCast(i),
            .pc = @intCast(i),
            .file = "bench.ora",
            .line = 1,
            .col = 1,
            .statement_id = @intCast(i),
            .is_statement = true,
            .kind = .runtime,
        });
    }
    var src_map = try SourceMap.fromEntries(allocator, sm_entries.items);
    errdefer src_map.deinit();

    // Build a synthetic DebugInfo with one op per source-map idx, all real
    // opcodes (no "invalid"). This means the ignored_invalid_idx set ends up
    // empty — but the debugger still walks the per-step path on every step.
    var debug_info_buf: std.ArrayList(u8) = .{};
    defer debug_info_buf.deinit(allocator);
    const dw = debug_info_buf.writer(allocator);
    try dw.writeAll(
        \\{"version":2,"ops":[
    );
    for (0..bytecode.len) |i| {
        if (i != 0) try dw.writeAll(",");
        try dw.print(
            \\{{"idx":{d},"op":"opcode","function":"f","block":"bb0"}}
        , .{i});
    }
    try dw.writeAll("]}");

    var debug_info: ?DebugInfo = try DebugInfo.loadFromJson(allocator, debug_info_buf.items);
    var debugger = try Debugger.initWithDebugInfo(allocator, evm, src_map, debug_info.?, "bench source\n");
    debug_info = null;
    defer debugger.deinit();
    debugger.max_steps = std.math.maxInt(u64);

    var timer = try std.time.Timer.start();
    var steps: u64 = 0;
    while (!debugger.isHalted()) {
        try debugger.stepOpcode();
        steps += 1;
        if (steps >= ITERATIONS * 4) break; // hard ceiling; loop body is ~4 ops
    }
    const elapsed_ns = timer.read();
    const per_step_ns: u64 = if (steps > 0) elapsed_ns / steps else 0;

    try stdout.print(
        "step_bench: {d} steps in {d}ns ({d}ns/step, budget {d}ns/step)\n",
        .{ steps, elapsed_ns, per_step_ns, STEP_BUDGET_NS },
    );
    try stdout.flush();

    if (per_step_ns > STEP_BUDGET_NS) {
        try stdout.print(
            "step_bench: FAIL — per-step time {d}ns exceeds budget {d}ns\n",
            .{ per_step_ns, STEP_BUDGET_NS },
        );
        try stdout.flush();
        std.process.exit(1);
    }
}

fn makeLoopBytecode(allocator: std.mem.Allocator, iterations: usize) ![]u8 {
    // A simple decrement loop:
    //   PUSH2 N            // 0..2
    //   JUMPDEST           // 3   <- loop top
    //   PUSH1 1            // 4..5
    //   SWAP1              // 6
    //   SUB                // 7
    //   DUP1               // 8
    //   PUSH1 0x03         // 9..10
    //   JUMPI              // 11  <- jumps back to JUMPDEST while top != 0
    //   STOP               // 12
    var buf: std.ArrayList(u8) = .{};
    errdefer buf.deinit(allocator);

    const n: u16 = @intCast(@min(iterations, std.math.maxInt(u16)));
    try buf.append(allocator, 0x61); // PUSH2
    try buf.append(allocator, @intCast(n >> 8));
    try buf.append(allocator, @intCast(n & 0xff));
    try buf.append(allocator, 0x5b); // JUMPDEST  (PC 3)
    try buf.append(allocator, 0x60); // PUSH1
    try buf.append(allocator, 0x01);
    try buf.append(allocator, 0x90); // SWAP1
    try buf.append(allocator, 0x03); // SUB
    try buf.append(allocator, 0x80); // DUP1
    try buf.append(allocator, 0x60); // PUSH1
    try buf.append(allocator, 0x03); // jumpdest at PC 3
    try buf.append(allocator, 0x57); // JUMPI
    try buf.append(allocator, 0x00); // STOP
    return try buf.toOwnedSlice(allocator);
}

fn buildEvm(allocator: std.mem.Allocator) !*Evm {
    const evm = try allocator.create(Evm);
    errdefer allocator.destroy(evm);
    const block_context = BlockContext{
        .chain_id = 1,
        .block_number = 1,
        .block_timestamp = 1000,
        .block_difficulty = 0,
        .block_prevrandao = 0,
        .block_coinbase = try Address.fromHex("0x0000000000000000000000000000000000000000"),
        .block_gas_limit = 1_000_000_000,
        .block_base_fee = 1,
        .blob_base_fee = 1,
    };
    try evm.init(allocator, null, .CANCUN, block_context, primitives.ZERO_ADDRESS, 0, null);
    errdefer evm.deinit();
    try evm.initTransactionState(null);
    return evm;
}

fn buildFrame(evm: *Evm, bytecode: []const u8) !void {
    const caller = try Address.fromHex("0x1111111111111111111111111111111111111111");
    const address = try Address.fromHex("0x2222222222222222222222222222222222222222");
    try evm.preWarmTransaction(address);
    try evm.frames.append(evm.arena.allocator(), try Frame.init(
        evm.arena.allocator(),
        bytecode,
        500_000_000, // generous; we want to step lots
        caller,
        address,
        0,
        &.{},
        @ptrCast(evm),
        .CANCUN,
        false,
    ));
}
