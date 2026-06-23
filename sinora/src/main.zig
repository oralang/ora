const std = @import("std");
const sinora = @import("sinora");

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try collectProcessArgs(allocator, init.minimal.args);
    defer allocator.free(args);

    if (args.len < 2) {
        try usage(io, args[0]);
        return error.InvalidArguments;
    }

    if (std.mem.eql(u8, args[1], "compare")) {
        const compare_options = parseCompareOptions(args[2..]) catch |err| {
            try usage(io, args[0]);
            return err;
        };
        try compareWithRustOracle(allocator, io, compare_options);
        return;
    }

    if (std.mem.eql(u8, args[1], "compare-corpus")) {
        const corpus_options = parseCorpusOptions(args[2..]) catch |err| {
            try usage(io, args[0]);
            return err;
        };
        try compareCorpusWithRustOracle(allocator, io, corpus_options);
        return;
    }

    if (std.mem.eql(u8, args[1], "emit-debug")) {
        if (args.len < 3 or args.len > 4) {
            try usage(io, args[0]);
            return error.InvalidArguments;
        }
        try emitDebugBytecode(allocator, io, args[2], if (args.len == 4) args[3] else null);
        return;
    }

    if (std.mem.eql(u8, args[1], "emit-release")) {
        if (args.len != 3) {
            try usage(io, args[0]);
            return error.InvalidArguments;
        }
        try emitReleaseBytecode(allocator, io, args[2]);
        return;
    }

    if (std.mem.eql(u8, args[1], "emit-release-generic")) {
        if (args.len != 3) {
            try usage(io, args[0]);
            return error.InvalidArguments;
        }
        try emitGenericReleaseBytecode(allocator, io, args[2]);
        return;
    }

    if (std.mem.eql(u8, args[1], "trace-release")) {
        if (args.len != 3) {
            try usage(io, args[0]);
            return error.InvalidArguments;
        }
        try traceGenericRelease(allocator, io, args[2]);
        return;
    }

    if (args.len > 3) {
        try usage(io, args[0]);
        return error.InvalidArguments;
    }

    const command = if (args.len == 3) args[1] else "check";
    const path = if (args.len == 3) args[2] else args[1];

    const source = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, std.Io.Limit.limited(64 * 1024 * 1024));
    defer allocator.free(source);

    var bag = sinora.DiagnosticBag.init(allocator);
    defer bag.deinit();

    var program = sinora.parse(allocator, source, &bag) catch |err| {
        try writeDiagnostics(io, &bag);
        return err;
    };
    defer program.deinit();

    sinora.validate(allocator, program, &bag) catch |err| {
        try writeDiagnostics(io, &bag);
        return err;
    };

    if (std.mem.eql(u8, command, "check")) {
        const stats = program.stats();
        var stdout_buffer: [1024]u8 = undefined;
        var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
        const stdout = &stdout_writer.interface;
        try stdout.print(
            "ok functions={d} data_segments={d} data_bytes={d} blocks={d} instructions={d} terminators={d} switches={d}\n",
            .{ stats.functions, stats.data_segments, stats.data_bytes, stats.blocks, stats.instructions, stats.terminators, stats.switches },
        );
        try stdout.flush();
    } else if (std.mem.eql(u8, command, "render")) {
        var stdout_buffer: [16 * 1024]u8 = undefined;
        var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
        const stdout = &stdout_writer.interface;
        try sinora.render.writeProgram(stdout, program);
        try stdout.flush();
    } else {
        try usage(io, args[0]);
        return error.InvalidArguments;
    }
}

fn collectProcessArgs(allocator: std.mem.Allocator, args: std.process.Args) ![]const []const u8 {
    var iterator = try std.process.Args.Iterator.initAllocator(args, allocator);
    defer iterator.deinit();

    var list: std.ArrayList([]const u8) = .empty;
    errdefer list.deinit(allocator);

    while (iterator.next()) |arg| {
        try list.append(allocator, arg);
    }

    return list.toOwnedSlice(allocator);
}

fn writeDiagnostics(io: std.Io, bag: *const sinora.DiagnosticBag) !void {
    var stderr_buffer: [4096]u8 = undefined;
    var stderr_writer = std.Io.File.stderr().writer(io, &stderr_buffer);
    const stderr = &stderr_writer.interface;
    try bag.writeTo(stderr);
    try stderr.flush();
}

fn usage(io: std.Io, argv0: []const u8) !void {
    var stderr_buffer: [1024]u8 = undefined;
    var stderr_writer = std.Io.File.stderr().writer(io, &stderr_buffer);
    const stderr = &stderr_writer.interface;
    try stderr.print(
        \\usage:
        \\  {s} <file.sir>
        \\  {s} check <file.sir>
        \\  {s} render <file.sir>
        \\  {s} compare [--release|--release-generic|--debug] [--text|--json] [--expect <classification>] <file.sir>
        \\  {s} compare-corpus [--release|--release-generic|--debug] [--text|--json] [--details] [--pending-only] <dir>
        \\  {s} emit-debug <file.sir> [function]
        \\  {s} emit-release <file.sir>
        \\  {s} emit-release-generic <file.sir>
        \\  {s} trace-release <file.sir>
        \\
        \\compare runs Sinora parse/legality plus the Rust Plank oracle.
        \\emit-debug emits conservative debug bytecode for one supported root.
        \\Set SINORA_PLANK_SIR or ORA_PLANK_SIR to override the Rust oracle path.
        \\
    , .{ argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0 });
    try stderr.flush();
}

const OutputFormat = enum {
    text,
    json,
};

const CompareOptions = struct {
    path: []const u8,
    mode: sinora.oracle.BackendMode = .release,
    output: OutputFormat = .text,
    expected: ?sinora.oracle.Classification = null,
};

const CorpusOptions = struct {
    root: []const u8,
    mode: sinora.oracle.BackendMode = .release,
    output: OutputFormat = .text,
    details: bool = false,
    pending_only: bool = false,
};

fn parseCompareOptions(args: []const []const u8) !CompareOptions {
    var result: CompareOptions = .{ .path = "" };

    var index: usize = 0;
    while (index < args.len) : (index += 1) {
        const arg = args[index];
        if (std.mem.eql(u8, arg, "--debug")) {
            result.mode = .debug;
        } else if (std.mem.eql(u8, arg, "--release")) {
            result.mode = .release;
        } else if (std.mem.eql(u8, arg, "--release-generic")) {
            result.mode = .release_generic;
        } else if (std.mem.eql(u8, arg, "--json")) {
            result.output = .json;
        } else if (std.mem.eql(u8, arg, "--text")) {
            result.output = .text;
        } else if (std.mem.eql(u8, arg, "--expect")) {
            index += 1;
            if (index >= args.len) return error.InvalidArguments;
            result.expected = sinora.oracle.Classification.parse(args[index]) orelse return error.InvalidArguments;
        } else if (std.mem.startsWith(u8, arg, "--")) {
            return error.InvalidArguments;
        } else if (result.path.len == 0) {
            result.path = arg;
        } else {
            return error.InvalidArguments;
        }
    }

    if (result.path.len == 0) return error.InvalidArguments;
    return result;
}

fn parseCorpusOptions(args: []const []const u8) !CorpusOptions {
    var result: CorpusOptions = .{ .root = "" };

    for (args) |arg| {
        if (std.mem.eql(u8, arg, "--debug")) {
            result.mode = .debug;
        } else if (std.mem.eql(u8, arg, "--release")) {
            result.mode = .release;
        } else if (std.mem.eql(u8, arg, "--release-generic")) {
            result.mode = .release_generic;
        } else if (std.mem.eql(u8, arg, "--json")) {
            result.output = .json;
        } else if (std.mem.eql(u8, arg, "--text")) {
            result.output = .text;
        } else if (std.mem.eql(u8, arg, "--details")) {
            result.details = true;
        } else if (std.mem.eql(u8, arg, "--pending-only")) {
            result.pending_only = true;
        } else if (std.mem.startsWith(u8, arg, "--")) {
            return error.InvalidArguments;
        } else if (result.root.len == 0) {
            result.root = arg;
        } else {
            return error.InvalidArguments;
        }
    }

    if (result.root.len == 0) return error.InvalidArguments;
    if (result.pending_only and !result.details) return error.InvalidArguments;
    return result;
}

fn compareWithRustOracle(allocator: std.mem.Allocator, io: std.Io, options: CompareOptions) !void {
    const rust_plank_path = sinora.oracle.resolveRustPlankPath(allocator, io) catch |err| {
        if (err == error.PlankSirNotFound) {
            var stderr_buffer: [512]u8 = undefined;
            var stderr_writer = std.Io.File.stderr().writer(io, &stderr_buffer);
            const stderr = &stderr_writer.interface;
            try stderr.writeAll("error: Rust Plank SIR oracle not found; run `zig build plank-sir` from Ora root or set SINORA_PLANK_SIR\n");
            try stderr.flush();
        }
        return err;
    };
    defer allocator.free(rust_plank_path);

    var comparison = try sinora.oracle.compareSirFile(allocator, io, options.path, rust_plank_path, options.mode);
    defer {
        sinora.oracle.freeDiagnostics(allocator, comparison.zig.diagnostics);
        comparison.deinit(allocator);
    }

    var stdout_buffer: [2048]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    switch (options.output) {
        .text => try writeComparisonText(stdout, comparison, options.mode),
        .json => try writeComparisonJson(stdout, comparison, options.mode),
    }
    try stdout.flush();

    if (options.expected) |expected| {
        if (comparison.classification != expected) {
            var stderr_buffer: [256]u8 = undefined;
            var stderr_writer = std.Io.File.stderr().writer(io, &stderr_buffer);
            const stderr = &stderr_writer.interface;
            try stderr.print(
                "error: expected classification {s}, got {s}\n",
                .{ expected.label(), comparison.classification.label() },
            );
            try stderr.flush();
            std.process.exit(1);
        }
    }
}

fn compareCorpusWithRustOracle(allocator: std.mem.Allocator, io: std.Io, options: CorpusOptions) !void {
    const rust_plank_path = sinora.oracle.resolveRustPlankPath(allocator, io) catch |err| {
        if (err == error.PlankSirNotFound) {
            var stderr_buffer: [512]u8 = undefined;
            var stderr_writer = std.Io.File.stderr().writer(io, &stderr_buffer);
            const stderr = &stderr_writer.interface;
            try stderr.writeAll("error: Rust Plank SIR oracle not found; run `zig build plank-sir` from Ora root or set SINORA_PLANK_SIR\n");
            try stderr.flush();
        }
        return err;
    };
    defer allocator.free(rust_plank_path);

    const expected = sinora.corpus.expectedLabel(options.mode);

    var stdout_buffer: [16 * 1024]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;

    if (options.details and options.output == .json) return error.InvalidArguments;

    var summary = if (options.details) blk: {
        const filter: sinora.corpus.RecordFilter = if (options.pending_only) .pending_only else .all;
        break :blk try sinora.corpus.compareDirectoryTextRecords(allocator, io, options.root, rust_plank_path, options.mode, filter, stdout);
    } else try sinora.corpus.compareDirectory(allocator, io, options.root, rust_plank_path, options.mode);
    defer summary.deinit(allocator);

    switch (options.output) {
        .text => try sinora.corpus.writeSummaryText(stdout, summary, options.mode, expected),
        .json => try sinora.corpus.writeSummaryJson(stdout, summary, options.mode, expected),
    }
    try stdout.flush();

    if (!summary.ok()) std.process.exit(1);
}

fn writeComparisonText(writer: anytype, comparison: sinora.oracle.Comparison, mode: sinora.oracle.BackendMode) !void {
    try writer.print("classification={s}\n", .{comparison.classification.label()});
    try writer.print("mode={s}\n", .{mode.label()});
    try writer.print("zig.accepted={}\n", .{comparison.zig.accepted});
    if (comparison.rust) |rust| {
        try writer.print("rust.accepted={}\n", .{rust.accepted});
        try writer.print("rust.stdout_bytes={d}\n", .{rust.stdout.len});
        try writer.print("rust.stderr_bytes={d}\n", .{rust.stderr.len});
    }
    if (comparison.debug_bytecode) |debug| {
        try writer.print("debug.zig_codegen_accepted={}\n", .{debug.zig_accepted});
        try writer.print("debug.zig_stdout_bytes={d}\n", .{debug.zig_stdout_bytes});
        try writer.print("debug.rust_stdout_bytes={d}\n", .{debug.rust_stdout_bytes});
        if (debug.equal) |equal| {
            try writer.print("debug.bytecode_equal={}\n", .{equal});
        }
    }
    if (comparison.release_bytecode) |release| {
        try writer.print("release.zig_codegen_accepted={}\n", .{release.zig_accepted});
        try writer.print("release.zig_stdout_bytes={d}\n", .{release.zig_stdout_bytes});
        try writer.print("release.rust_stdout_bytes={d}\n", .{release.rust_stdout_bytes});
        if (release.equal) |equal| {
            try writer.print("release.bytecode_equal={}\n", .{equal});
        }
    }
    if (comparison.zig.diagnostics.len > 0) {
        try writer.print("zig.diagnostics={d}\n", .{comparison.zig.diagnostics.len});
    }
}

fn writeComparisonJson(writer: anytype, comparison: sinora.oracle.Comparison, mode: sinora.oracle.BackendMode) !void {
    try writer.print(
        \\{{"classification":"{s}","mode":"{s}","zig":{{"accepted":{},"diagnostics":{d}}}
    , .{
        comparison.classification.label(),
        mode.label(),
        comparison.zig.accepted,
        comparison.zig.diagnostics.len,
    });
    if (comparison.rust) |rust| {
        try writer.print(
            \\,"rust":{{"accepted":{},"stdout_bytes":{d},"stderr_bytes":{d}}}
        , .{ rust.accepted, rust.stdout.len, rust.stderr.len });
    } else {
        try writer.writeAll(",\"rust\":null");
    }
    if (comparison.debug_bytecode) |debug| {
        try writer.print(
            ",\"debug_bytecode\":{{\"zig_codegen_accepted\":{},\"zig_stdout_bytes\":{d},\"rust_stdout_bytes\":{d},\"equal\":",
            .{ debug.zig_accepted, debug.zig_stdout_bytes, debug.rust_stdout_bytes },
        );
        if (debug.equal) |equal| {
            try writer.print("{}", .{equal});
        } else {
            try writer.writeAll("null");
        }
        try writer.writeAll("}");
    }
    if (comparison.release_bytecode) |release| {
        try writer.print(
            ",\"release_bytecode\":{{\"zig_codegen_accepted\":{},\"zig_stdout_bytes\":{d},\"rust_stdout_bytes\":{d},\"equal\":",
            .{ release.zig_accepted, release.zig_stdout_bytes, release.rust_stdout_bytes },
        );
        if (release.equal) |equal| {
            try writer.print("{}", .{equal});
        } else {
            try writer.writeAll("null");
        }
        try writer.writeAll("}");
    }
    try writer.writeAll("}\n");
}

fn emitDebugBytecode(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    function_name: ?[]const u8,
) !void {
    const source = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, std.Io.Limit.limited(64 * 1024 * 1024));
    defer allocator.free(source);

    var bag = sinora.DiagnosticBag.init(allocator);
    defer bag.deinit();

    var program = sinora.parse(allocator, source, &bag) catch |err| {
        try writeDiagnostics(io, &bag);
        return err;
    };
    defer program.deinit();

    sinora.validate(allocator, program, &bag) catch |err| {
        try writeDiagnostics(io, &bag);
        return err;
    };

    const bytes = sinora.debug_codegen.emitFunction(allocator, program, function_name, &bag) catch |err| {
        if (err == error.FunctionNotFound) {
            const name = function_name orelse "<default>";
            try bag.errorAt(0, 1, "function '{s}' not found", .{name});
        }
        try writeDiagnostics(io, &bag);
        return err;
    };
    defer allocator.free(bytes);

    var stdout_buffer: [16 * 1024]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    try sinora.bytecode.writeHex(stdout, bytes);
    try stdout.writeByte('\n');
    try stdout.flush();
}

fn emitReleaseBytecode(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
) !void {
    const source = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, std.Io.Limit.limited(64 * 1024 * 1024));
    defer allocator.free(source);

    var bag = sinora.DiagnosticBag.init(allocator);
    defer bag.deinit();

    var program = sinora.parse(allocator, source, &bag) catch |err| {
        try writeDiagnostics(io, &bag);
        return err;
    };
    defer program.deinit();

    sinora.validate(allocator, program, &bag) catch |err| {
        try writeDiagnostics(io, &bag);
        return err;
    };

    const bytes = sinora.release_generic_backend.emitRelease(allocator, program) catch |err| {
        try writeDiagnostics(io, &bag);
        return err;
    };
    defer allocator.free(bytes);

    var stdout_buffer: [16 * 1024]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    try sinora.bytecode.writeHex(stdout, bytes);
    try stdout.writeByte('\n');
    try stdout.flush();
}

fn traceGenericRelease(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
) !void {
    const source = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, std.Io.Limit.limited(64 * 1024 * 1024));
    defer allocator.free(source);

    var bag = sinora.DiagnosticBag.init(allocator);
    defer bag.deinit();

    var program = sinora.parse(allocator, source, &bag) catch |err| {
        try writeDiagnostics(io, &bag);
        return err;
    };
    defer program.deinit();

    sinora.validate(allocator, program, &bag) catch |err| {
        try writeDiagnostics(io, &bag);
        return err;
    };

    var stdout_buffer: [16 * 1024]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    try sinora.release_op_graph.traceProgram(
        stdout,
        allocator,
        program,
        sinora.release_schedule.ScheduleConfig.pre_amsterdam,
    );
    try stdout.flush();
}

fn emitGenericReleaseBytecode(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
) !void {
    const source = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, std.Io.Limit.limited(64 * 1024 * 1024));
    defer allocator.free(source);

    var bag = sinora.DiagnosticBag.init(allocator);
    defer bag.deinit();

    var program = sinora.parse(allocator, source, &bag) catch |err| {
        try writeDiagnostics(io, &bag);
        return err;
    };
    defer program.deinit();

    sinora.validate(allocator, program, &bag) catch |err| {
        try writeDiagnostics(io, &bag);
        return err;
    };

    const bytes = try sinora.release_generic_backend.emitRelease(allocator, program);
    defer allocator.free(bytes);

    var stdout_buffer: [16 * 1024]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    try sinora.bytecode.writeHex(stdout, bytes);
    try stdout.writeByte('\n');
    try stdout.flush();
}
