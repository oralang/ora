//! Unit tests for the framework-agnostic debugger driver in debugger.zig.
//!
//! These tests construct a minimal Evm + Frame + SourceMap (and optionally
//! DebugInfo) by hand, without going through the compiler, so the suite
//! exercises stepping, breakpoint, scope, and binding-read/write semantics
//! in isolation from the compiler artifact emission.

const std = @import("std");
const testing = std.testing;
const primitives = @import("voltaire");
const Address = primitives.Address.Address;
const Hardfork = primitives.Hardfork;

const evm_mod = @import("evm.zig");
const Evm = evm_mod.Evm(.{});
const Frame = @import("frame.zig").Frame(.{});
const Debugger = @import("debugger.zig").Debugger(.{});
const SourceMap = @import("source_map.zig").SourceMap;
const DebugInfo = @import("debug_info.zig").DebugInfo;
const debug_session = @import("debug_session.zig");

// =============================================================================
// EVM/Frame/Debugger fixture builders
// =============================================================================

const Fixture = struct {
    allocator: std.mem.Allocator,
    evm: *Evm,
    debugger: Debugger,
    source_text: []u8,

    fn deinit(self: *Fixture) void {
        // Debugger owns src_map and debug_info via its deinit. The Debugger
        // borrows source_text, so the fixture frees it after the debugger
        // is torn down.
        self.debugger.deinit();
        self.evm.deinit();
        self.allocator.destroy(self.evm);
        self.allocator.free(self.source_text);
    }
};

fn buildEvm(allocator: std.mem.Allocator) !*Evm {
    const evm = try allocator.create(Evm);
    errdefer allocator.destroy(evm);
    // Use the same pinned block context the production debug-session
    // helpers use, so the unit-test determinism story matches the runtime
    // determinism story.
    try evm.init(allocator, null, .CANCUN, debug_session.deterministicBlockContext(), primitives.ZERO_ADDRESS, 0, null);
    errdefer evm.deinit();
    try evm.initTransactionState(null);
    return evm;
}

fn buildFrameInPlace(
    evm: *Evm,
    bytecode: []const u8,
    gas: i64,
) !void {
    const caller = try Address.fromHex("0x1111111111111111111111111111111111111111");
    const address = try Address.fromHex("0x2222222222222222222222222222222222222222");
    try evm.preWarmTransaction(address);
    try evm.frames.append(evm.arena.allocator(), try Frame.init(
        evm.arena.allocator(),
        bytecode,
        gas,
        caller,
        address,
        0,
        &.{},
        @ptrCast(evm),
        .CANCUN,
        false,
    ));
}

fn buildFixture(
    allocator: std.mem.Allocator,
    bytecode: []const u8,
    source_text: []const u8,
    entries: []const SourceMap.Entry,
    debug_info_json: ?[]const u8,
) !Fixture {
    const evm = try buildEvm(allocator);
    errdefer {
        evm.deinit();
        allocator.destroy(evm);
    }

    try buildFrameInPlace(evm, bytecode, 1_000_000);

    var src_map = try SourceMap.fromEntries(allocator, entries);
    errdefer src_map.deinit();

    var debug_info: ?DebugInfo = if (debug_info_json) |json|
        try DebugInfo.loadFromJson(allocator, json)
    else
        null;
    errdefer if (debug_info) |*info| info.deinit();

    // dupe source_text so the Debugger doesn't outlive the literal it was given
    const text_dup = try allocator.dupe(u8, source_text);
    errdefer allocator.free(text_dup);

    const debugger = if (debug_info) |info| blk: {
        const built = try Debugger.initWithDebugInfo(allocator, evm, src_map, info, text_dup);
        debug_info = null;
        break :blk built;
    } else try Debugger.init(allocator, evm, src_map, text_dup);

    return .{
        .allocator = allocator,
        .evm = evm,
        .debugger = debugger,
        .source_text = text_dup,
    };
}

// =============================================================================
// Stepping & breakpoint tests
// =============================================================================

// A two-statement program:
//   PC 0: PUSH1 0x01      ; statement A (line 1)
//   PC 2: PUSH1 0x02      ; statement A (continued)
//   PC 4: ADD             ; statement A (continued)
//   PC 5: PUSH1 0x00      ; statement B (line 2)
//   PC 7: MSTORE          ; statement B (continued)
//   PC 8: STOP            ; halt
const TWO_STATEMENT_BYTECODE = &[_]u8{
    0x60, 0x01, // PUSH1 1
    0x60, 0x02, // PUSH1 2
    0x01, // ADD
    0x60, 0x00, // PUSH1 0
    0x52, // MSTORE
    0x00, // STOP
};

const TWO_STATEMENT_SOURCE =
    \\let x = 1 + 2;
    \\mem[0] = x;
;

fn twoStatementEntries() []const SourceMap.Entry {
    const entries = &[_]SourceMap.Entry{
        .{ .pc = 0, .file = "fixture.ora", .line = 1, .col = 1, .statement_id = 1, .is_statement = true },
        .{ .pc = 2, .file = "fixture.ora", .line = 1, .col = 1, .statement_id = 1, .is_statement = false },
        .{ .pc = 4, .file = "fixture.ora", .line = 1, .col = 1, .statement_id = 1, .is_statement = false },
        .{ .pc = 5, .file = "fixture.ora", .line = 2, .col = 1, .statement_id = 2, .is_statement = true },
        .{ .pc = 7, .file = "fixture.ora", .line = 2, .col = 1, .statement_id = 2, .is_statement = false },
        .{ .pc = 8, .file = "fixture.ora", .line = 2, .col = 1, .statement_id = 2, .is_statement = false },
    };
    return entries;
}

test "Debugger: stepIn advances at statement boundaries" {
    const allocator = testing.allocator;
    var fx = try buildFixture(
        allocator,
        TWO_STATEMENT_BYTECODE,
        TWO_STATEMENT_SOURCE,
        twoStatementEntries(),
        null,
    );
    defer fx.deinit();

    // Initial location: PC=0, line 1.
    try testing.expectEqual(@as(?u32, 1), fx.debugger.currentSourceLine());

    // First stepIn: should advance to the second statement (line 2).
    try fx.debugger.stepIn();
    try testing.expectEqual(Debugger.StopReason.step_complete, fx.debugger.stop_reason);
    try testing.expectEqual(Debugger.State.paused, fx.debugger.state);
    try testing.expectEqual(@as(?u32, 2), fx.debugger.currentSourceLine());

    // Second stepIn: program halts (STOP at pc=8, all done).
    try fx.debugger.stepIn();
    try testing.expect(fx.debugger.isHalted());
    try testing.expectEqual(Debugger.StopReason.execution_finished, fx.debugger.stop_reason);
    try testing.expect(fx.debugger.isSuccess());
}

test "Debugger: stepOpcode advances exactly one opcode" {
    const allocator = testing.allocator;
    var fx = try buildFixture(
        allocator,
        TWO_STATEMENT_BYTECODE,
        TWO_STATEMENT_SOURCE,
        twoStatementEntries(),
        null,
    );
    defer fx.deinit();

    try testing.expectEqual(@as(u32, 0), fx.debugger.getPC());

    try fx.debugger.stepOpcode();
    // PUSH1 0x01 advances PC by 2 (opcode + 1 immediate byte).
    try testing.expectEqual(@as(u32, 2), fx.debugger.getPC());
    try testing.expectEqual(Debugger.StopReason.step_complete, fx.debugger.stop_reason);

    try fx.debugger.stepOpcode();
    try testing.expectEqual(@as(u32, 4), fx.debugger.getPC());
}

test "Debugger: stepOver halts at next same-or-shallower-depth statement" {
    const allocator = testing.allocator;
    var fx = try buildFixture(
        allocator,
        TWO_STATEMENT_BYTECODE,
        TWO_STATEMENT_SOURCE,
        twoStatementEntries(),
        null,
    );
    defer fx.deinit();

    try fx.debugger.stepOver();
    try testing.expectEqual(@as(?u32, 2), fx.debugger.currentSourceLine());
    try testing.expectEqual(Debugger.StopReason.step_complete, fx.debugger.stop_reason);
}

test "Debugger: continue_ runs to halt with no breakpoints" {
    const allocator = testing.allocator;
    var fx = try buildFixture(
        allocator,
        TWO_STATEMENT_BYTECODE,
        TWO_STATEMENT_SOURCE,
        twoStatementEntries(),
        null,
    );
    defer fx.deinit();

    try fx.debugger.continue_();
    try testing.expect(fx.debugger.isHalted());
    try testing.expectEqual(Debugger.StopReason.execution_finished, fx.debugger.stop_reason);
}

test "Debugger: breakpoints set, hit, and clear" {
    const allocator = testing.allocator;
    var fx = try buildFixture(
        allocator,
        TWO_STATEMENT_BYTECODE,
        TWO_STATEMENT_SOURCE,
        twoStatementEntries(),
        null,
    );
    defer fx.deinit();

    // Setting on line 2 resolves to PC 5.
    try testing.expect(fx.debugger.setBreakpoint("fixture.ora", 2));
    try testing.expect(fx.debugger.hasBreakpoint("fixture.ora", 2));
    try testing.expect(!fx.debugger.hasBreakpoint("fixture.ora", 99));

    // Setting on a line with no statement entry returns false.
    try testing.expect(!fx.debugger.setBreakpoint("fixture.ora", 99));

    try fx.debugger.continue_();
    try testing.expectEqual(Debugger.StopReason.breakpoint_hit, fx.debugger.stop_reason);
    try testing.expectEqual(Debugger.State.paused, fx.debugger.state);
    try testing.expectEqual(@as(?u32, 2), fx.debugger.currentSourceLine());

    // Removing the breakpoint and continuing finishes execution.
    fx.debugger.removeBreakpoint("fixture.ora", 2);
    try testing.expect(!fx.debugger.hasBreakpoint("fixture.ora", 2));

    try fx.debugger.continue_();
    try testing.expect(fx.debugger.isHalted());
    try testing.expectEqual(Debugger.StopReason.execution_finished, fx.debugger.stop_reason);
}

test "Debugger: toggleBreakpoint flips state" {
    const allocator = testing.allocator;
    var fx = try buildFixture(
        allocator,
        TWO_STATEMENT_BYTECODE,
        TWO_STATEMENT_SOURCE,
        twoStatementEntries(),
        null,
    );
    defer fx.deinit();

    try testing.expect(fx.debugger.toggleBreakpoint("fixture.ora", 2));
    try testing.expect(fx.debugger.hasBreakpoint("fixture.ora", 2));

    try testing.expect(!fx.debugger.toggleBreakpoint("fixture.ora", 2));
    try testing.expect(!fx.debugger.hasBreakpoint("fixture.ora", 2));
}

test "Debugger: step limit halts with step_limit_reached" {
    const allocator = testing.allocator;
    var fx = try buildFixture(
        allocator,
        TWO_STATEMENT_BYTECODE,
        TWO_STATEMENT_SOURCE,
        twoStatementEntries(),
        null,
    );
    defer fx.deinit();

    // The fixture has two statements; the first stepIn would normally cross
    // multiple opcodes. Cap to 1 to force the limit before the next boundary.
    fx.debugger.max_steps = 1;
    try fx.debugger.stepIn();
    try testing.expectEqual(Debugger.StopReason.step_limit_reached, fx.debugger.stop_reason);
    try testing.expectEqual(Debugger.State.paused, fx.debugger.state);
    try testing.expect(!fx.debugger.isHalted());
}

test "Debugger: getSourceLineText returns the right line" {
    const allocator = testing.allocator;
    var fx = try buildFixture(
        allocator,
        TWO_STATEMENT_BYTECODE,
        TWO_STATEMENT_SOURCE,
        twoStatementEntries(),
        null,
    );
    defer fx.deinit();

    try testing.expectEqualStrings("let x = 1 + 2;", fx.debugger.getSourceLineText(1).?);
    try testing.expectEqualStrings("mem[0] = x;", fx.debugger.getSourceLineText(2).?);
    try testing.expect(fx.debugger.getSourceLineText(0) == null);
    try testing.expect(fx.debugger.getSourceLineText(99) == null);
}

test "Debugger: stepOver replay produces an identical trace" {
    // Running the same bytecode twice must produce identical PC/opcode-name
    // observations. Pinning a deterministic BlockContext (A4) means
    // replay-determinism holds even when the contract reads block-context
    // opcodes — see the dedicated test below.
    const allocator = testing.allocator;

    const Trace = struct {
        pc: u32,
        line: ?u32,
        op: []const u8,
    };

    var first: std.ArrayList(Trace) = .{};
    defer first.deinit(allocator);
    var second: std.ArrayList(Trace) = .{};
    defer second.deinit(allocator);

    inline for (.{ &first, &second }) |out| {
        var fx = try buildFixture(
            allocator,
            TWO_STATEMENT_BYTECODE,
            TWO_STATEMENT_SOURCE,
            twoStatementEntries(),
            null,
        );
        defer fx.deinit();

        while (!fx.debugger.isHalted()) {
            try out.append(allocator, .{
                .pc = fx.debugger.getPC(),
                .line = fx.debugger.currentSourceLine(),
                .op = fx.debugger.getCurrentOpcodeName(),
            });
            try fx.debugger.stepIn();
        }
    }

    try testing.expectEqual(first.items.len, second.items.len);
    for (first.items, second.items) |a, b| {
        try testing.expectEqual(a.pc, b.pc);
        try testing.expectEqual(a.line, b.line);
        try testing.expectEqualStrings(a.op, b.op);
    }
}

test "Debugger: replay determinism across block-context opcodes" {
    // Stress the determinism contract: the contract pushes the values of
    // every block-context opcode (TIMESTAMP, NUMBER, CHAINID, COINBASE,
    // BASEFEE, PREVRANDAO, DIFFICULTY, GASLIMIT) onto the stack. With
    // ora_evm.deterministicBlockContext() pinned, two runs must produce
    // identical stacks at the STOP boundary.
    const allocator = testing.allocator;

    // Bytecode: TIMESTAMP, NUMBER, CHAINID, COINBASE, BASEFEE, PREVRANDAO,
    // GASLIMIT, then STOP. We omit DIFFICULTY because in CANCUN it aliases
    // PREVRANDAO and the EVM may reject the redundant op; the seven we keep
    // are independent and exercise the rest of the surface.
    const bytecode = &[_]u8{
        0x42, // TIMESTAMP
        0x43, // NUMBER
        0x46, // CHAINID
        0x41, // COINBASE
        0x48, // BASEFEE
        0x44, // PREVRANDAO
        0x45, // GASLIMIT
        0x00, // STOP
    };

    const entries = &[_]SourceMap.Entry{
        .{ .pc = 0, .file = "fx.ora", .line = 1, .col = 1, .statement_id = 1, .is_statement = true },
        .{ .pc = 7, .file = "fx.ora", .line = 1, .col = 1, .statement_id = 1, .is_statement = false },
    };

    var first_stack: std.ArrayList(u256) = .{};
    defer first_stack.deinit(allocator);
    var second_stack: std.ArrayList(u256) = .{};
    defer second_stack.deinit(allocator);

    inline for (.{ &first_stack, &second_stack }) |out| {
        var fx = try buildFixture(allocator, bytecode, "block ctx\n", entries, null);
        defer fx.deinit();

        try fx.debugger.continue_();
        // Snapshot the final stack contents.
        for (fx.debugger.getStack()) |word| try out.append(allocator, word);
    }

    try testing.expectEqual(first_stack.items.len, second_stack.items.len);
    for (first_stack.items, second_stack.items) |a, b| try testing.expectEqual(a, b);
}

// =============================================================================
// shouldIgnoreStatementBoundary
// =============================================================================
//
// A statement_id appears at two PCs: one is a real opcode, the other is a
// compiler-emitted invalid/unreachable. The boundary at the invalid PC must
// be skipped — stepIn should reach the next *distinct* statement_id.
test "Debugger: shouldIgnoreStatementBoundary skips lone invalid sites" {
    const allocator = testing.allocator;

    // Two statements; statement 1 has TWO PCs claiming the same statement_id:
    // PC 0 (real PUSH1) and PC 5 (invalid). With debug-info marking PC 5's idx
    // as op="invalid", the debugger should skip PC 5 as a stop point and
    // surface the boundary at PC 8 (statement_id=2).
    // PC 5 is a real JUMPDEST in the bytecode (a no-op at runtime), but the
    // accompanying debug_info marks its idx as op="invalid". The debugger
    // should treat PC 5's statement boundary as compiler-generated and skip
    // past it — never stopping there during stepIn even though the source
    // map tags it as a statement_id=1 boundary.
    const bytecode = &[_]u8{
        0x60, 0x01, // PC 0: PUSH1 1   (statement 1)
        0x60, 0x02, // PC 2: PUSH1 2   (statement 1)
        0x01, // PC 4: ADD       (statement 1)
        0x5b, // PC 5: JUMPDEST  (no-op; debug-info tags idx=3 as "invalid")
        0x50, // PC 6: POP       (statement 2 — drop the result)
        0x00, // PC 7: STOP      (statement 2)
    };

    const entries = &[_]SourceMap.Entry{
        .{ .idx = 0, .pc = 0, .file = "fx.ora", .line = 1, .col = 1, .statement_id = 1, .is_statement = true },
        .{ .idx = 1, .pc = 2, .file = "fx.ora", .line = 1, .col = 1, .statement_id = 1, .is_statement = false },
        .{ .idx = 2, .pc = 4, .file = "fx.ora", .line = 1, .col = 1, .statement_id = 1, .is_statement = false },
        .{ .idx = 3, .pc = 5, .file = "fx.ora", .line = 1, .col = 1, .statement_id = 1, .is_statement = true },
        .{ .idx = 4, .pc = 6, .file = "fx.ora", .line = 2, .col = 1, .statement_id = 2, .is_statement = true },
        .{ .idx = 5, .pc = 7, .file = "fx.ora", .line = 2, .col = 1, .statement_id = 2, .is_statement = false },
    };

    const debug_info_json =
        \\{
        \\  "version": 2,
        \\  "ops": [
        \\    {"idx": 0, "op": "push1",   "function": "f", "block": "bb0"},
        \\    {"idx": 1, "op": "push1",   "function": "f", "block": "bb0"},
        \\    {"idx": 2, "op": "add",     "function": "f", "block": "bb0"},
        \\    {"idx": 3, "op": "invalid", "function": "f", "block": "bb0"},
        \\    {"idx": 4, "op": "pop",     "function": "f", "block": "bb0"},
        \\    {"idx": 5, "op": "stop",    "function": "f", "block": "bb0"}
        \\  ]
        \\}
    ;

    var fx = try buildFixture(allocator, bytecode, "stmt 1\nstmt 2\n", entries, debug_info_json);
    defer fx.deinit();

    // First stepIn should land at line 2 (statement_id 2), having skipped the
    // lone-invalid boundary at PC 5.
    try fx.debugger.stepIn();
    try testing.expectEqual(@as(?u32, 2), fx.debugger.currentSourceLine());
}

// =============================================================================
// Visible bindings: storage/memory/tstore round-trip
// =============================================================================

const SINGLE_STATEMENT_BYTECODE = &[_]u8{
    0x60, 0x00, // PC 0: PUSH1 0
    0x00, // PC 2: STOP
};

const STORAGE_BINDING_DEBUG_INFO =
    \\{
    \\  "version": 2,
    \\  "ops": [
    \\    {"idx": 0, "op": "push1", "function": "f", "block": "bb0"}
    \\  ],
    \\  "source_scopes": [
    \\    {
    \\      "id": 1,
    \\      "parent": null,
    \\      "file": "fx.ora",
    \\      "function": "f",
    \\      "kind": "function",
    \\      "locals": [
    \\        {
    \\          "id": 10,
    \\          "name": "counter",
    \\          "kind": "field",
    \\          "storage_class": "storage",
    \\          "runtime": {
    \\            "kind": "storage_field",
    \\            "name": "counter",
    \\            "location": {"kind": "storage_root", "root": "counter", "slot": 0},
    \\            "editable": true
    \\          }
    \\        },
    \\        {
    \\          "id": 11,
    \\          "name": "scratch",
    \\          "kind": "field",
    \\          "storage_class": "memory",
    \\          "runtime": {
    \\            "kind": "memory_field",
    \\            "name": "scratch",
    \\            "location": {"kind": "memory_root", "root": "scratch", "slot": 0},
    \\            "editable": true
    \\          }
    \\        },
    \\        {
    \\          "id": 12,
    \\          "name": "ttl",
    \\          "kind": "field",
    \\          "storage_class": "tstore",
    \\          "runtime": {
    \\            "kind": "tstore_field",
    \\            "name": "ttl",
    \\            "location": {"kind": "tstore_root", "root": "ttl", "slot": 1},
    \\            "editable": true
    \\          }
    \\        }
    \\      ]
    \\    }
    \\  ],
    \\  "op_visibility": [
    \\    {"idx": 0, "scope_ids": [1], "visible_local_ids": [10, 11, 12]}
    \\  ]
    \\}
;

test "Debugger: storage binding round-trip via getVisible*RootValue" {
    const allocator = testing.allocator;
    const entries = &[_]SourceMap.Entry{
        .{ .idx = 0, .pc = 0, .file = "fx.ora", .line = 1, .col = 1, .statement_id = 1, .is_statement = true },
    };
    var fx = try buildFixture(
        allocator,
        SINGLE_STATEMENT_BYTECODE,
        "stmt\n",
        entries,
        STORAGE_BINDING_DEBUG_INFO,
    );
    defer fx.deinit();

    const counter = (try fx.debugger.findVisibleBindingByName(allocator, "counter")).?;
    try testing.expect(try fx.debugger.setVisibleStorageRootValue(counter, 42));
    try testing.expectEqual(@as(?u256, 42), try fx.debugger.getVisibleStorageRootValue(counter));

    // Also exercise the by-name path.
    try testing.expect(try fx.debugger.setVisibleBindingValueByName(allocator, "counter", 1234));
    try testing.expectEqual(@as(?u256, 1234), try fx.debugger.getVisibleBindingValueByName(allocator, "counter"));
}

test "Debugger: memory_field binding round-trip" {
    const allocator = testing.allocator;
    const entries = &[_]SourceMap.Entry{
        .{ .idx = 0, .pc = 0, .file = "fx.ora", .line = 1, .col = 1, .statement_id = 1, .is_statement = true },
    };
    var fx = try buildFixture(
        allocator,
        SINGLE_STATEMENT_BYTECODE,
        "stmt\n",
        entries,
        STORAGE_BINDING_DEBUG_INFO,
    );
    defer fx.deinit();

    const scratch = (try fx.debugger.findVisibleBindingByName(allocator, "scratch")).?;
    try testing.expect(try fx.debugger.setVisibleMemoryRootValue(scratch, 0xdeadbeef));
    try testing.expectEqual(@as(?u256, 0xdeadbeef), fx.debugger.getVisibleMemoryRootValue(scratch));
}

test "Debugger: tstore_field binding round-trip" {
    const allocator = testing.allocator;
    const entries = &[_]SourceMap.Entry{
        .{ .idx = 0, .pc = 0, .file = "fx.ora", .line = 1, .col = 1, .statement_id = 1, .is_statement = true },
    };
    var fx = try buildFixture(
        allocator,
        SINGLE_STATEMENT_BYTECODE,
        "stmt\n",
        entries,
        STORAGE_BINDING_DEBUG_INFO,
    );
    defer fx.deinit();

    const ttl = (try fx.debugger.findVisibleBindingByName(allocator, "ttl")).?;
    try testing.expect(try fx.debugger.setVisibleTStoreRootValue(ttl, 7));
    try testing.expectEqual(@as(?u256, 7), fx.debugger.getVisibleTStoreRootValue(ttl));
}

test "Debugger: getVisibleScopes returns the active scope" {
    const allocator = testing.allocator;
    const entries = &[_]SourceMap.Entry{
        .{ .idx = 0, .pc = 0, .file = "fx.ora", .line = 1, .col = 1, .statement_id = 1, .is_statement = true },
    };
    var fx = try buildFixture(
        allocator,
        SINGLE_STATEMENT_BYTECODE,
        "stmt\n",
        entries,
        STORAGE_BINDING_DEBUG_INFO,
    );
    defer fx.deinit();

    const scopes = try fx.debugger.getVisibleScopes(allocator);
    defer allocator.free(scopes);

    try testing.expectEqual(@as(usize, 1), scopes.len);
    try testing.expectEqualStrings("function", scopes[0].kind);
    try testing.expectEqualStrings("f", scopes[0].function);
}
