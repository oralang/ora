//! Compiler metrics snapshot harness.
//!
//! Compiles expected-success Ora examples through the package-mode frontend and
//! prints deterministic Tier-A metrics:
//!   <file>::<phase>::invocations       <N>
//!   <file>::<phase>::work_count        <N>
//!   <file>::<phase>::alloc_calls       <N>
//!   <file>::<phase>::bytes_allocated   <N>
//!   <file>::__source_bytes             <N>
//!   <file>::__source_lines             <N>
//!   <file>::__bytes_peak               <N>
//!
//! Each measured run uses a fresh CompilerDb. Repeated runs must produce the
//! same deterministic counts; wall time is emitted only to an explicit local
//! `--time-output` file, never to the committed Tier-A snapshot.

const std = @import("std");
const ora_root = @import("ora_root");

const compiler = ora_root.compiler;
const allocation_stats = ora_root.lsp.allocation_stats;
const metrics_mod = ora_root.metrics;

const default_warmup_runs: usize = 1;
const default_measured_runs: usize = 5;
const default_corpus_root = "ora-example";
const expected_pass_exception = "ora-example/refinements/negative_tests/fail_refinement_bounds.ora";

fn exitCli(code: u8) noreturn {
    std.process.exit(code);
}

const RunResult = struct {
    phases: []metrics_mod.Phase,
    source_bytes: u64,
    source_lines: u64,
    bytes_peak: u64,
    time_min_ns: []u64 = &.{},
    time_median_ns: []u64 = &.{},
    time_noise_ns: []u64 = &.{},

    fn deinit(self: *RunResult, allocator: std.mem.Allocator) void {
        allocator.free(self.phases);
        if (self.time_min_ns.len > 0) allocator.free(self.time_min_ns);
        if (self.time_median_ns.len > 0) allocator.free(self.time_median_ns);
        if (self.time_noise_ns.len > 0) allocator.free(self.time_noise_ns);
        self.* = undefined;
    }
};

const Options = struct {
    warmup_runs: usize = default_warmup_runs,
    measured_runs: usize = default_measured_runs,
    time_output: ?[]const u8 = null,
    paths: std.ArrayList([]const u8) = .empty,

    fn deinit(self: *Options, allocator: std.mem.Allocator) void {
        self.paths.deinit(allocator);
    }
};

pub fn main(init: std.process.Init) !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try collectArgs(allocator, init.minimal.args);
    defer freeArgs(allocator, args);

    var options = try parseOptions(allocator, args[1..]);
    defer options.deinit(allocator);
    if (options.measured_runs == 0) {
        std.debug.print("compile-metrics: --runs must be at least 1\n", .{});
        exitCli(2);
    }

    var paths: std.ArrayList([]const u8) = .empty;
    defer {
        freeStringList(allocator, paths.items);
        paths.deinit(allocator);
    }
    if (options.paths.items.len == 0) {
        try collectOraFiles(allocator, default_corpus_root, &paths);
    } else {
        for (options.paths.items) |path| {
            try collectPath(allocator, path, &paths);
        }
    }
    sortStrings(paths.items);

    const io = std.Io.Threaded.global_single_threaded.io();
    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    var time_file: ?std.Io.File = if (options.time_output) |path| try std.Io.Dir.cwd().createFile(io, path, .{}) else null;
    defer if (time_file) |*file| file.close(io);
    var time_buffer: [4096]u8 = undefined;
    var time_writer = if (time_file) |*file| file.writer(io, &time_buffer) else null;

    for (paths.items) |path| {
        if (expectedFailure(path)) continue;
        var result = measurePath(allocator, path, options.warmup_runs, options.measured_runs) catch |err| {
            std.debug.print("compile-metrics: failed for {s}: {s}\n", .{ path, @errorName(err) });
            return err;
        };
        defer result.deinit(allocator);
        try emitSnapshot(stdout, path, &result);
        if (time_writer) |*writer| {
            try emitTiming(&writer.interface, path, &result);
        }
    }
    if (time_writer) |*writer| try writer.interface.flush();
    try stdout.flush();
}

fn collectArgs(allocator: std.mem.Allocator, process_args: std.process.Args) ![][]u8 {
    var iterator = try std.process.Args.Iterator.initAllocator(process_args, allocator);
    defer iterator.deinit();

    var list: std.ArrayList([]u8) = .empty;
    errdefer {
        for (list.items) |arg| allocator.free(arg);
        list.deinit(allocator);
    }

    while (iterator.next()) |arg| {
        try list.append(allocator, try allocator.dupe(u8, arg));
    }
    return list.toOwnedSlice(allocator);
}

fn freeArgs(allocator: std.mem.Allocator, args: [][]u8) void {
    for (args) |arg| allocator.free(arg);
    allocator.free(args);
}

fn parseOptions(allocator: std.mem.Allocator, args: []const []const u8) !Options {
    var options = Options{};
    errdefer options.deinit(allocator);

    var index: usize = 0;
    while (index < args.len) {
        const arg = args[index];
        if (std.mem.eql(u8, arg, "--runs")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            options.measured_runs = try std.fmt.parseInt(usize, args[index], 10);
        } else if (std.mem.startsWith(u8, arg, "--runs=")) {
            options.measured_runs = try std.fmt.parseInt(usize, arg["--runs=".len..], 10);
        } else if (std.mem.eql(u8, arg, "--warmup")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            options.warmup_runs = try std.fmt.parseInt(usize, args[index], 10);
        } else if (std.mem.startsWith(u8, arg, "--warmup=")) {
            options.warmup_runs = try std.fmt.parseInt(usize, arg["--warmup=".len..], 10);
        } else if (std.mem.eql(u8, arg, "--time-output")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            options.time_output = args[index];
        } else if (std.mem.startsWith(u8, arg, "--time-output=")) {
            options.time_output = arg["--time-output=".len..];
        } else if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            try printUsage();
            exitCli(0);
        } else if (std.mem.startsWith(u8, arg, "-")) {
            return error.UnknownArgument;
        } else {
            try options.paths.append(allocator, arg);
        }
        index += 1;
    }

    return options;
}

fn printUsage() !void {
    const io = std.Io.Threaded.global_single_threaded.io();
    var stderr_buffer: [1024]u8 = undefined;
    var stderr_writer = std.Io.File.stderr().writer(io, &stderr_buffer);
    const stderr = &stderr_writer.interface;
    try stderr.writeAll(
        \\Usage: compile-metrics [--warmup N] [--runs N] [path...]
        \\
        \\If no paths are passed, the harness scans ora-example/ recursively and
        \\skips expected-fail fixtures whose filename contains "fail".
        \\Use --time-output <path> to write local min/median wall-time metrics.
        \\
    );
    try stderr.flush();
}

fn collectPath(allocator: std.mem.Allocator, path: []const u8, out: *std.ArrayList([]const u8)) !void {
    const io = std.Io.Threaded.global_single_threaded.io();
    const stat = try std.Io.Dir.cwd().statFile(io, path, .{});
    switch (stat.kind) {
        .file => {
            if (std.mem.endsWith(u8, path, ".ora")) {
                try out.append(allocator, try allocator.dupe(u8, path));
            }
        },
        .directory => try collectOraFiles(allocator, path, out),
        else => {},
    }
}

fn collectOraFiles(allocator: std.mem.Allocator, root_path: []const u8, out: *std.ArrayList([]const u8)) !void {
    const io = std.Io.Threaded.global_single_threaded.io();
    var dir = try std.Io.Dir.cwd().openDir(io, root_path, .{ .iterate = true });
    defer dir.close(io);

    var walker = try dir.walk(allocator);
    defer walker.deinit();
    while (try walker.next(io)) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.path, ".ora")) continue;
        const full_path = try std.fs.path.join(allocator, &.{ root_path, entry.path });
        try out.append(allocator, full_path);
    }
}

fn measurePath(allocator: std.mem.Allocator, path: []const u8, warmup_runs: usize, measured_runs: usize) !RunResult {
    for (0..warmup_runs) |_| {
        var warmup = try compileOnce(allocator, path);
        warmup.deinit(allocator);
    }

    var first = try compileOnce(allocator, path);
    errdefer first.deinit(allocator);
    const timing_count = first.phases.len * measured_runs;
    var timing_matrix = try allocator.alloc(u64, timing_count);
    defer allocator.free(timing_matrix);
    for (first.phases, 0..) |phase, phase_index| {
        timing_matrix[phase_index * measured_runs] = phase.wall_ns;
    }
    for (1..measured_runs) |run_index| {
        var next = try compileOnce(allocator, path);
        defer next.deinit(allocator);
        if (!runResultsEqual(&first, &next)) {
            std.debug.print("compile-metrics: nondeterministic Tier-A metrics for {s}\n", .{path});
            return error.NondeterministicMetrics;
        }
        for (next.phases, 0..) |phase, phase_index| {
            timing_matrix[phase_index * measured_runs + run_index] = phase.wall_ns;
        }
    }
    try attachTimingSummaries(allocator, &first, timing_matrix, measured_runs);
    return first;
}

fn compileOnce(allocator: std.mem.Allocator, path: []const u8) !RunResult {
    var counting_allocator = allocation_stats.CountingAllocator.init(allocator);
    var metrics = metrics_mod.Metrics.init(true);
    metrics.setAllocationStats(&counting_allocator.stats);

    var compilation = try compiler.compilePackageWithOptions(counting_allocator.allocator(), path, .{
        .compile_options = .{ .instrumentation = &metrics },
    });
    defer compilation.deinit();
    if (!compilation.isArtifactEmittable()) return error.NonEmittable;

    return .{
        .phases = try allocator.dupe(metrics_mod.Phase, metrics.phases[0..metrics.count]),
        .source_bytes = packageSourceBytes(&compilation),
        .source_lines = packageSourceLines(&compilation),
        .bytes_peak = std.math.cast(u64, counting_allocator.stats.bytes_peak) orelse std.math.maxInt(u64),
    };
}

fn runResultsEqual(lhs: *const RunResult, rhs: *const RunResult) bool {
    if (lhs.source_bytes != rhs.source_bytes) return false;
    if (lhs.source_lines != rhs.source_lines) return false;
    if (lhs.bytes_peak != rhs.bytes_peak) return false;
    if (lhs.phases.len != rhs.phases.len) return false;
    for (lhs.phases, rhs.phases) |a, b| {
        if (!std.mem.eql(u8, a.name, b.name)) return false;
        if (a.invocations != b.invocations) return false;
        if (a.work_count != b.work_count) return false;
        if (a.alloc_calls != b.alloc_calls) return false;
        if (a.bytes_allocated != b.bytes_allocated) return false;
    }
    return true;
}

fn emitSnapshot(writer: anytype, path: []const u8, result: *const RunResult) !void {
    for (result.phases) |phase| {
        try writer.print("{s}::{s}::invocations\t{d}\n", .{ path, phase.name, phase.invocations });
        try writer.print("{s}::{s}::work_count\t{d}\n", .{ path, phase.name, phase.work_count });
        try writer.print("{s}::{s}::alloc_calls\t{d}\n", .{ path, phase.name, phase.alloc_calls });
        try writer.print("{s}::{s}::bytes_allocated\t{d}\n", .{ path, phase.name, phase.bytes_allocated });
    }
    try writer.print("{s}::__source_bytes\t{d}\n", .{ path, result.source_bytes });
    try writer.print("{s}::__source_lines\t{d}\n", .{ path, result.source_lines });
    try writer.print("{s}::__bytes_peak\t{d}\n", .{ path, result.bytes_peak });
}

fn packageSourceBytes(compilation: *const compiler.driver.Compilation) u64 {
    const sources = &compilation.db.sources;
    const package = sources.package(compilation.package_id);
    var total: u64 = 0;
    for (package.modules.items) |module_id| {
        const module = sources.module(module_id);
        const source_file = sources.file(module.file_id);
        total = addSat(total, source_file.text.len);
    }
    return total;
}

fn packageSourceLines(compilation: *const compiler.driver.Compilation) u64 {
    const sources = &compilation.db.sources;
    const package = sources.package(compilation.package_id);
    var total: u64 = 0;
    for (package.modules.items) |module_id| {
        const module = sources.module(module_id);
        const source_file = sources.file(module.file_id);
        total = addSat(total, source_file.line_starts.len);
    }
    return total;
}

fn addSat(lhs: u64, rhs: usize) u64 {
    const rhs_u64 = std.math.cast(u64, rhs) orelse std.math.maxInt(u64);
    return std.math.add(u64, lhs, rhs_u64) catch std.math.maxInt(u64);
}

fn emitTiming(writer: anytype, path: []const u8, result: *const RunResult) !void {
    for (result.phases, 0..) |phase, index| {
        try writer.print("{s}::{s}::wall_ns_min\t{d}\n", .{ path, phase.name, result.time_min_ns[index] });
        try writer.print("{s}::{s}::wall_ns_median\t{d}\n", .{ path, phase.name, result.time_median_ns[index] });
        try writer.print("{s}::{s}::wall_ns_noise\t{d}\n", .{ path, phase.name, result.time_noise_ns[index] });
    }
}

fn attachTimingSummaries(allocator: std.mem.Allocator, result: *RunResult, timing_matrix: []const u64, measured_runs: usize) !void {
    result.time_min_ns = try allocator.alloc(u64, result.phases.len);
    result.time_median_ns = try allocator.alloc(u64, result.phases.len);
    result.time_noise_ns = try allocator.alloc(u64, result.phases.len);
    const scratch = try allocator.alloc(u64, measured_runs);
    defer allocator.free(scratch);

    for (result.phases, 0..) |_, phase_index| {
        const offset = phase_index * measured_runs;
        @memcpy(scratch, timing_matrix[offset .. offset + measured_runs]);
        std.mem.sort(u64, scratch, {}, struct {
            fn lessThan(_: void, lhs: u64, rhs: u64) bool {
                return lhs < rhs;
            }
        }.lessThan);
        result.time_min_ns[phase_index] = scratch[0];
        result.time_median_ns[phase_index] = scratch[measured_runs / 2];
        result.time_noise_ns[phase_index] = scratch[measured_runs - 1] - scratch[0];
    }
}

fn expectedFailure(path: []const u8) bool {
    if (std.mem.eql(u8, path, expected_pass_exception)) return false;
    const stem = std.fs.path.stem(path);
    return std.ascii.indexOfIgnoreCase(stem, "fail") != null;
}

fn sortStrings(items: [][]const u8) void {
    std.mem.sort([]const u8, items, {}, struct {
        fn lessThan(_: void, lhs: []const u8, rhs: []const u8) bool {
            return std.mem.lessThan(u8, lhs, rhs);
        }
    }.lessThan);
}

fn freeStringList(allocator: std.mem.Allocator, items: []const []const u8) void {
    for (items) |item| allocator.free(item);
}
