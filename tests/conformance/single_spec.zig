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

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();

    _ = args.next();
    const source_path = args.next();
    const spec_path = args.next();
    if (source_path == null or spec_path == null or args.next() != null) {
        std.debug.print("usage: conformance-one <source.ora> <spec.toml>\n", .{});
        exitCli(2);
    }

    runner.runConformanceSpec(allocator, source_path.?, spec_path.?) catch |err| {
        std.debug.print("conformance-one: spec failed: {s}\n", .{@errorName(err)});
        exitCli(1);
    };
    std.debug.print("conformance-one: ok\n", .{});
}
