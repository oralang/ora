const std = @import("std");
const sinora = @import("sinora");

// Developer CLI for the standalone Sinora backend. This file intentionally
// stays thin: parse SIR text, run structural legality once, then hand the
// verified IR to the local bytecode emitter.
const max_sir_file_bytes = 64 * 1024 * 1024;

const SpecialCommand = enum {
    emit_debug,
    emit_release,
    emit_release_generic,
    trace_release,
};

const special_command_map = std.StaticStringMap(SpecialCommand).initComptime(.{
    .{ "emit-debug", .emit_debug },
    .{ "emit-release", .emit_release },
    .{ "emit-release-generic", .emit_release_generic },
    .{ "trace-release", .trace_release },
});

const DefaultCommand = enum {
    check,
    render,
};

const default_command_map = std.StaticStringMap(DefaultCommand).initComptime(.{
    .{ "check", .check },
    .{ "render", .render },
});

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

    if (special_command_map.get(args[1])) |special_command| {
        // Emitter commands have stricter arity than the default "check
        // file.sir" path, so dispatch them before interpreting the first
        // argument as a filename.
        switch (special_command) {
            .emit_debug => {
                if (args.len < 3 or args.len > 4) {
                    try usage(io, args[0]);
                    return error.InvalidArguments;
                }
                try emitDebugBytecode(allocator, io, args[2], if (args.len == 4) args[3] else null);
            },
            .emit_release => {
                const emit_options = parseEmitReleaseOptions(args[2..]) catch |err| {
                    try usage(io, args[0]);
                    return err;
                };
                if (emit_options.optimize) |policy| sinora.switch_routing.dispatch_policy = policy;
                try emitReleaseBytecode(
                    allocator,
                    io,
                    emit_options.path,
                    emit_options.source_map_path,
                    emit_options.metrics_path,
                    emit_options.dispatcher_report_path,
                );
            },
            .emit_release_generic => {
                const emit_options = parseEmitReleaseOptions(args[2..]) catch |err| {
                    try usage(io, args[0]);
                    return err;
                };
                if (emit_options.optimize) |policy| sinora.switch_routing.dispatch_policy = policy;
                try emitGenericReleaseBytecode(
                    allocator,
                    io,
                    emit_options.path,
                    emit_options.source_map_path,
                    emit_options.metrics_path,
                    emit_options.dispatcher_report_path,
                );
            },
            .trace_release => {
                if (args.len != 3) {
                    try usage(io, args[0]);
                    return error.InvalidArguments;
                }
                try traceGenericRelease(allocator, io, args[2]);
            },
        }
        return;
    }

    if (args.len > 3) {
        try usage(io, args[0]);
        return error.InvalidArguments;
    }

    const command_text = if (args.len == 3) args[1] else "check";
    const command = default_command_map.get(command_text) orelse {
        try usage(io, args[0]);
        return error.InvalidArguments;
    };
    const path = if (args.len == 3) args[2] else args[1];

    var bag = sinora.DiagnosticBag.init(allocator);
    defer bag.deinit();

    var program = try loadValidatedProgram(allocator, io, path, &bag);
    defer program.deinit();

    switch (command) {
        .check => try writeCheckSummary(io, &program),
        .render => try renderProgram(io, program),
    }
}

fn collectProcessArgs(allocator: std.mem.Allocator, args: std.process.Args) ![]const []const u8 {
    var iterator = try std.process.Args.Iterator.initAllocator(args, allocator);
    defer iterator.deinit();

    var list: std.ArrayList([]const u8) = .empty;
    errdefer list.deinit(allocator);
    try list.ensureTotalCapacity(allocator, 8);

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

fn loadValidatedProgram(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    bag: *sinora.DiagnosticBag,
) !sinora.Program {
    const source = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, std.Io.Limit.limited(max_sir_file_bytes));
    defer allocator.free(source);

    // The parser duplicates all IR tokens into Program's arena, so the file
    // buffer can be released before the caller runs codegen.
    var program = sinora.parse(allocator, source, bag) catch |err| {
        try writeDiagnostics(io, bag);
        return err;
    };
    errdefer program.deinit();

    // Validation is the single structural trust boundary for all Sinora entry
    // points. After this returns, backends can assume unique names, legal
    // arities, resolved CFG targets, and defined SSA value uses.
    sinora.validate(allocator, program, bag) catch |err| {
        try writeDiagnostics(io, bag);
        return err;
    };

    return program;
}

fn writeCheckSummary(io: std.Io, program: *const sinora.Program) !void {
    const stats = program.stats();
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    try stdout.print(
        "ok functions={d} data_segments={d} data_bytes={d} blocks={d} instructions={d} terminators={d} switches={d}\n",
        .{ stats.functions, stats.data_segments, stats.data_bytes, stats.blocks, stats.instructions, stats.terminators, stats.switches },
    );
    try stdout.flush();
}

fn renderProgram(io: std.Io, program: sinora.Program) !void {
    var stdout_buffer: [16 * 1024]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    try sinora.render.writeProgram(stdout, program);
    try stdout.flush();
}

fn writeBytecodeHexLine(io: std.Io, bytes: []const u8) !void {
    var stdout_buffer: [16 * 1024]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    try sinora.bytecode.writeHex(stdout, bytes);
    try stdout.writeByte('\n');
    try stdout.flush();
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
        \\  {s} emit-debug <file.sir> [function]
        \\  {s} emit-release [--source-map <path>] [--metrics <path>] [--optimize=<gas|balanced|size>] <file.sir>
        \\  {s} emit-release-generic [--source-map <path>] [--metrics <path>] [--optimize=<gas|balanced|size>] <file.sir>
        \\  {s} trace-release <file.sir>
        \\
        \\emit-debug emits conservative debug bytecode for one supported root.
        \\
    , .{ argv0, argv0, argv0, argv0, argv0, argv0, argv0 });
    try stderr.flush();
}

fn emitDebugBytecode(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    function_name: ?[]const u8,
) !void {
    var bag = sinora.DiagnosticBag.init(allocator);
    defer bag.deinit();

    var program = try loadValidatedProgram(allocator, io, path, &bag);
    defer program.deinit();

    const bytes = sinora.debug_codegen.emitFunction(allocator, program, function_name, &bag) catch |err| {
        if (err == error.FunctionNotFound) {
            const name = function_name orelse "<default>";
            try bag.errorAt(0, 1, "function '{s}' not found", .{name});
        }
        try writeDiagnostics(io, &bag);
        return err;
    };
    defer allocator.free(bytes);

    try writeBytecodeHexLine(io, bytes);
}

fn emitReleaseBytecode(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    source_map_path: ?[]const u8,
    metrics_path: ?[]const u8,
    dispatcher_report_path: ?[]const u8,
) !void {
    try emitReleaseBackendBytecode(
        allocator,
        io,
        path,
        source_map_path,
        metrics_path,
        dispatcher_report_path,
    );
}

fn traceGenericRelease(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
) !void {
    var bag = sinora.DiagnosticBag.init(allocator);
    defer bag.deinit();

    var program = try loadValidatedProgram(allocator, io, path, &bag);
    defer program.deinit();

    var stdout_buffer: [16 * 1024]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    // Tracing stops after release graph/schedule construction. It is a study
    // tool for stack scheduling, not another bytecode emitter.
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
    source_map_path: ?[]const u8,
    metrics_path: ?[]const u8,
    dispatcher_report_path: ?[]const u8,
) !void {
    try emitReleaseBackendBytecode(
        allocator,
        io,
        path,
        source_map_path,
        metrics_path,
        dispatcher_report_path,
    );
}

fn emitReleaseBackendBytecode(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    source_map_path: ?[]const u8,
    metrics_path: ?[]const u8,
    dispatcher_report_path: ?[]const u8,
) !void {
    var bag = sinora.DiagnosticBag.init(allocator);
    defer bag.deinit();

    var program = try loadValidatedProgram(allocator, io, path, &bag);
    defer program.deinit();

    if (source_map_path != null or dispatcher_report_path != null) {
        var result = sinora.release_generic_backend.emitReleaseWithSourceMap(allocator, program) catch |err| {
            try writeDiagnostics(io, &bag);
            return err;
        };
        defer result.deinit();
        if (source_map_path) |map_path| {
            try writeSourceMap(io, map_path, result.source_map, result.runtime_start_pc);
        }
        if (dispatcher_report_path) |report_path| {
            try writeDispatcherProofReport(io, report_path, result);
        }
        if (metrics_path) |out_path| {
            const release_metrics = try sinora.release_generic_backend.collectReleaseMetrics(
                allocator,
                program,
                result.bytes.len,
                result.source_map.len,
            );
            try writeReleaseMetrics(io, out_path, release_metrics);
        }
        try writeBytecodeHexLine(io, result.bytes);
        return;
    }

    // `emit-release` and `emit-release-generic` both target the generic release
    // backend today. Keeping both command names avoids breaking local scripts
    // while using one codegen entry point.
    const bytes = sinora.release_generic_backend.emitRelease(allocator, program) catch |err| {
        try writeDiagnostics(io, &bag);
        return err;
    };
    defer allocator.free(bytes);

    if (metrics_path) |out_path| {
        const release_metrics = try sinora.release_generic_backend.collectReleaseMetrics(
            allocator,
            program,
            bytes.len,
            0,
        );
        try writeReleaseMetrics(io, out_path, release_metrics);
    }

    try writeBytecodeHexLine(io, bytes);
}

const EmitReleaseOptions = struct {
    path: []const u8,
    source_map_path: ?[]const u8 = null,
    metrics_path: ?[]const u8 = null,
    dispatcher_report_path: ?[]const u8 = null,
    optimize: ?sinora.switch_routing.DispatchPolicy = null,
};

fn parseEmitReleaseOptions(args: []const []const u8) !EmitReleaseOptions {
    var result: EmitReleaseOptions = .{ .path = "" };
    var index: usize = 0;
    while (index < args.len) : (index += 1) {
        const arg = args[index];
        if (std.mem.eql(u8, arg, "--source-map")) {
            index += 1;
            if (index >= args.len) return error.InvalidArguments;
            if (result.source_map_path != null) return error.InvalidArguments;
            result.source_map_path = args[index];
            continue;
        }
        if (std.mem.eql(u8, arg, "--metrics")) {
            index += 1;
            if (index >= args.len) return error.InvalidArguments;
            if (result.metrics_path != null) return error.InvalidArguments;
            result.metrics_path = args[index];
            continue;
        }
        if (std.mem.eql(u8, arg, "--dispatcher-proof-report")) {
            index += 1;
            if (index >= args.len) return error.InvalidArguments;
            if (result.dispatcher_report_path != null) return error.InvalidArguments;
            result.dispatcher_report_path = args[index];
            continue;
        }
        if (std.mem.startsWith(u8, arg, "--optimize=")) {
            if (result.optimize != null) return error.InvalidArguments;
            result.optimize = sinora.switch_routing.parsePolicyName(arg["--optimize=".len..]) orelse
                return error.InvalidArguments;
            continue;
        }
        if (std.mem.startsWith(u8, arg, "--")) return error.InvalidArguments;
        if (result.path.len != 0) return error.InvalidArguments;
        result.path = arg;
    }
    if (result.path.len == 0) return error.InvalidArguments;
    return result;
}

fn writeSourceMap(
    io: std.Io,
    path: []const u8,
    entries: []const sinora.release_code_to_asm.SourceMapEntry,
    runtime_start_pc: u32,
) !void {
    var output: std.Io.Writer.Allocating = .init(std.heap.page_allocator);
    defer output.deinit();
    const writer = &output.writer;
    try writer.print("{{\"runtime_start_pc\":{},\"ops\":[", .{runtime_start_pc});
    for (entries, 0..) |entry, index| {
        if (index != 0) try writer.writeByte(',');
        try writer.print("{{\"idx\":{},\"pc\":{}}}", .{ entry.idx, entry.pc });
    }
    try writer.writeAll("]}");
    try std.Io.Dir.cwd().writeFile(io, .{
        .sub_path = path,
        .data = output.written(),
    });
}

fn writeDispatcherProofReport(
    io: std.Io,
    path: []const u8,
    result: sinora.release_code_to_asm.EmitResult,
) !void {
    for (result.dispatcher_facts) |fact| {
        if (!fact.templates_valid) return error.DispatcherBytecodeValidationFailed;
    }

    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(result.bytes, &digest, .{});
    const digest_hex = std.fmt.bytesToHex(digest, .lower);

    var output = std.Io.Writer.Allocating.init(std.heap.page_allocator);
    defer output.deinit();
    const writer = &output.writer;
    try writer.writeAll("{\n  \"schema_version\": 1,\n  \"bytecode_sha256\": \"");
    try writer.writeAll(&digest_hex);
    try writer.print("\",\n  \"runtime_start_pc\": {d},\n  \"switches\": [", .{
        result.runtime_start_pc,
    });
    for (result.dispatcher_facts, 0..) |fact, fact_index| {
        if (fact_index != 0) try writer.writeByte(',');
        try writer.writeAll("\n    {\"function\":");
        try writeJsonString(writer, fact.function_name);
        try writer.writeAll(",\"block\":");
        try writeJsonString(writer, fact.block_name);
        try writer.writeAll(",\"strategy\":");
        try writeJsonString(writer, fact.strategy);
        try writer.writeAll(",\"default_label\":");
        try writeJsonString(writer, fact.default_target);
        try writer.print(",\"default_pc\":{d},\"templates_valid\":true,\"cases\":[", .{
            fact.default_pc,
        });
        for (fact.cases, 0..) |case, case_index| {
            if (case_index != 0) try writer.writeByte(',');
            try writer.print("{{\"selector\":{d},\"target\":", .{case.selector});
            try writeJsonString(writer, case.target);
            try writer.print(",\"route_index\":{d},\"guard_pc\":{d},\"target_pc\":{d}}}", .{
                case.route_index,
                case.guard_pc,
                case.target_pc,
            });
        }
        try writer.writeAll("]}");
    }
    try writer.writeAll("\n  ]\n}\n");
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = output.written() });
}

fn writeJsonString(writer: anytype, value: []const u8) !void {
    try writer.writeByte('"');
    for (value) |byte| {
        switch (byte) {
            '"' => try writer.writeAll("\\\""),
            '\\' => try writer.writeAll("\\\\"),
            '\n' => try writer.writeAll("\\n"),
            '\r' => try writer.writeAll("\\r"),
            '\t' => try writer.writeAll("\\t"),
            else => try writer.writeByte(byte),
        }
    }
    try writer.writeByte('"');
}

fn writeReleaseMetrics(
    io: std.Io,
    path: []const u8,
    metrics: sinora.metrics.ReleaseMetrics,
) !void {
    var output: std.Io.Writer.Allocating = .init(std.heap.page_allocator);
    defer output.deinit();
    try sinora.metrics.writeReleaseMetricsJson(&output.writer, metrics);
    try std.Io.Dir.cwd().writeFile(io, .{
        .sub_path = path,
        .data = output.written(),
    });
}
