//! Metrics snapshot harness for change-quality metrics.
//!
//! Runs the whole conformance corpus and prints numeric compiler/runtime metrics
//! as a stable, deterministic table so any pipeline change can be judged
//! better/worse automatically:
//!   <spec>::__bytecode_bytes      <N>   — compiled bytecode size per contract
//!   <spec>::__deploy_gas          <gas> — metered gas for primary deployment
//!   <spec>::__deploy_gas:<name>   <gas> — metered gas for named secondary deploy
//!   <spec>::c<index>:<fn>         <gas> — metered gas per executed call
//!
//! Piped through scripts/metrics-check.py against a committed baseline, this
//! yields per-operation AND per-category (gas / size) deltas + totals — so a real
//! broad improvement is distinguishable from a narrow local trick, and a
//! regression in one dimension (e.g. smaller bytecode but more gas) is visible.
//!
//! Deterministic (lib/evm OSAKA, fixed block context + caller). Outcomes are
//! still asserted, so this only measures a green corpus.
//!
//! Experimental optimization reports can add compiler flags with
//! ORA_METRICS_COMPILER_ARGS. The value is split on ASCII whitespace and is only
//! meant for simple temporary compiler flags used by local optimization slices.

const std = @import("std");
const builtin = @import("builtin");
const runner = @import("runner.zig");
const types = @import("types.zig");

fn exitCli(code: u8) noreturn {
    std.process.exit(code);
}

fn freeStringList(allocator: std.mem.Allocator, values: []const []const u8) void {
    for (values) |value| allocator.free(value);
    allocator.free(values);
}

fn collectCompilerArgs(allocator: std.mem.Allocator) ![][]const u8 {
    var args: std.ArrayList([]const u8) = .empty;
    errdefer {
        for (args.items) |arg| allocator.free(arg);
        args.deinit(allocator);
    }

    try args.append(allocator, try allocator.dupe(u8, "--no-verify"));

    if (builtin.link_libc) {
        const raw = std.c.getenv("ORA_METRICS_COMPILER_ARGS") orelse return args.toOwnedSlice(allocator);
        const value = std.mem.span(raw);
        var tokens = std.mem.tokenizeAny(u8, value, " \t\r\n");
        while (tokens.next()) |token| {
            try args.append(allocator, try allocator.dupe(u8, token));
        }
    }

    return args.toOwnedSlice(allocator);
}

pub fn main() !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.Io.Dir.cwd().access(std.Io.Threaded.global_single_threaded.io(), types.ORA_BINARY_REL, .{}) catch {
        std.debug.print("metrics-snapshot: ora binary not found; run 'zig build' first\n", .{});
        exitCli(2);
    };

    const dir = types.CONFORMANCE_DIR_REL;
    const specs = try runner.collectSpecNames(allocator, dir);
    defer runner.freeStringList(allocator, specs);
    const compiler_args = try collectCompilerArgs(allocator);
    defer freeStringList(allocator, compiler_args);

    var stdout_buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(std.Io.Threaded.global_single_threaded.io(), &stdout_buf);
    const w = &stdout.interface;

    for (specs) |spec_name| {
        const stem = spec_name[0 .. spec_name.len - ".spec.toml".len];
        const source_name = try std.fmt.allocPrint(allocator, "{s}.ora", .{stem});
        defer allocator.free(source_name);
        const source_path = try std.fs.path.join(allocator, &.{ dir, source_name });
        defer allocator.free(source_path);
        const spec_path = try std.fs.path.join(allocator, &.{ dir, spec_name });
        defer allocator.free(spec_path);

        var list: std.ArrayList(runner.MetricSample) = .empty;
        defer {
            for (list.items) |s| allocator.free(s.key);
            list.deinit(allocator);
        }
        const sink = runner.MetricSink{ .allocator = allocator, .list = &list };

        runner.runConformanceSpecMetricsWithExtraArgs(allocator, source_path, spec_path, sink, compiler_args) catch |err| {
            // Fail closed: skipping would emit a partial snapshot whose
            // missing rows read as bytecode-size drift in the gate — a
            // confusing lie. Better one loud failure naming the spec.
            std.debug.print("metrics-snapshot: spec failed: {s}: {s}\n", .{ spec_path, @errorName(err) });
            return err;
        };

        var call_index: usize = 0;
        for (list.items) |s| {
            if (std.mem.startsWith(u8, s.key, "__")) {
                try w.print("{s}::{s}\t{d}\n", .{ stem, s.key, s.value });
            } else {
                try w.print("{s}::c{d}:{s}\t{d}\n", .{ stem, call_index, s.key, s.value });
                call_index += 1;
            }
        }
    }
    try w.flush();
}
