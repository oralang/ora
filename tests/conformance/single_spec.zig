//! Single-spec lib/evm conformance runner.
//!
//! Runs ONE `.ora` + `.spec.toml` pair through the in-process lib/evm, outside
//! the test harness. Used by the Anvil differential proof to exercise lib/evm on
//! exactly one (possibly hostile) call and observe its behavior — including a
//! crash. If lib/evm panics, this process aborts with a signal, which the caller
//! detects as the divergence.
//!
//! Usage: conformance-one <source.ora> <spec.toml>
//!   exit 0  -> spec passed on lib/evm
//!   exit 1  -> spec failed (assertion mismatch / clean error)
//!   signal  -> lib/evm crashed (panic) on this input

const std = @import("std");
const runner = @import("runner.zig");

fn exitCli(code: u8) noreturn {
    std.process.exit(code);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    if (args.len != 3) {
        std.debug.print("usage: conformance-one <source.ora> <spec.toml>\n", .{});
        exitCli(2);
    }

    runner.runConformanceSpec(allocator, args[1], args[2]) catch |err| {
        std.debug.print("conformance-one: spec failed: {s}\n", .{@errorName(err)});
        exitCli(1);
    };
    std.debug.print("conformance-one: ok\n", .{});
}
