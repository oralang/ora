const std = @import("std");
const primitives = @import("voltaire");
const ora_evm = @import("ora_evm");

const Evm = ora_evm.Evm(.{});
const Frame = ora_evm.Frame(.{});
const Debugger = ora_evm.Debugger(.{});
const SourceMap = ora_evm.SourceMap;
const DebugInfo = ora_evm.DebugInfo;

const ProbeConfig = struct {
    bytecode_path: []const u8,
    source_map_path: []const u8,
    source_path: []const u8,
    debug_info_path: ?[]const u8 = null,
    calldata: []u8 = &.{},
    max_statements: usize = 64,

    fn deinit(self: *ProbeConfig, allocator: std.mem.Allocator) void {
        allocator.free(self.bytecode_path);
        allocator.free(self.source_map_path);
        allocator.free(self.source_path);
        if (self.debug_info_path) |path| allocator.free(path);
        allocator.free(self.calldata);
    }
};

pub fn main() !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const allocator = gpa_state.allocator();

    var config = try parseArgs(allocator);
    defer config.deinit(allocator);

    const bytecode_hex = try std.fs.cwd().readFileAlloc(allocator, config.bytecode_path, 16 * 1024 * 1024);
    defer allocator.free(bytecode_hex);
    const bytecode = try decodeHexAlloc(allocator, bytecode_hex);
    defer allocator.free(bytecode);

    const source_map_json = try std.fs.cwd().readFileAlloc(allocator, config.source_map_path, 16 * 1024 * 1024);
    defer allocator.free(source_map_json);
    var source_map = try SourceMap.loadFromJson(allocator, source_map_json);
    defer source_map.deinit();

    const source_text = try std.fs.cwd().readFileAlloc(allocator, config.source_path, 16 * 1024 * 1024);
    defer allocator.free(source_text);

    var debug_info_json: ?[]u8 = null;
    defer if (debug_info_json) |bytes| allocator.free(bytes);

    var debug_info: ?DebugInfo = null;
    if (config.debug_info_path) |path| {
        debug_info_json = try std.fs.cwd().readFileAlloc(allocator, path, 16 * 1024 * 1024);
        debug_info = try DebugInfo.loadFromJson(allocator, debug_info_json.?);
    }
    defer if (debug_info) |*info| info.deinit();

    var evm: Evm = undefined;
    try evm.init(allocator, null, null, null, primitives.ZERO_ADDRESS, 0, null);
    defer evm.deinit();

    const caller = primitives.Address.fromU256(0x100);
    const contract = primitives.Address.fromU256(0x200);

    const runtime_bytecode = try deployRuntimeBytecode(allocator, &evm, caller, contract, bytecode);
    defer allocator.free(runtime_bytecode);

    try evm.frames.append(evm.arena.allocator(), try Frame.init(
        evm.arena.allocator(),
        runtime_bytecode,
        5_000_000,
        caller,
        contract,
        0,
        config.calldata,
        @as(*anyopaque, @ptrCast(&evm)),
        evm.hardfork,
        false,
    ));

    var debugger = blk: {
        if (debug_info) |info| {
            debug_info = null;
            break :blk try Debugger.initWithDebugInfo(allocator, &evm, source_map, info, source_text);
        }
        break :blk try Debugger.init(allocator, &evm, source_map, source_text);
    };
    defer debugger.deinit();

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_file = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_file.interface;
    try stdout.print("debug probe\n", .{});
    try stdout.print("  bytecode: {s}\n", .{config.bytecode_path});
    try stdout.print("  source:   {s}\n", .{config.source_path});
    try stdout.print("  calldata: 0x", .{});
    try printHex(stdout, config.calldata);
    try stdout.print("\n", .{});

    var stops: usize = 0;
    while (!debugger.isHalted() and stops < config.max_statements) {
        try debugger.stepIn();
        if (debugger.currentEntry()) |entry| {
            stops += 1;
            try printStop(stdout, &debugger, entry, stops);
        }
    }

    try stdout.print("halted: {s}\n", .{@tagName(debugger.stop_reason)});
    if (evm.getCurrentFrame()) |frame| {
        try stdout.print("pc: {d}\n", .{frame.pc});
        try stdout.print("reverted: {}\n", .{frame.reverted});
        if (frame.output.len > 0) {
            try stdout.print("output: 0x", .{});
            try printHex(stdout, frame.output);
            try stdout.print("\n", .{});
        }
    }
    try stdout.flush();
}

fn deployRuntimeBytecode(
    allocator: std.mem.Allocator,
    evm: *Evm,
    caller: primitives.Address,
    contract: primitives.Address,
    deployment_bytecode: []const u8,
) ![]u8 {
    try evm.frames.append(evm.arena.allocator(), try Frame.init(
        evm.arena.allocator(),
        deployment_bytecode,
        5_000_000,
        caller,
        contract,
        0,
        &.{},
        @as(*anyopaque, @ptrCast(evm)),
        evm.hardfork,
        false,
    ));

    while (evm.getCurrentFrame()) |frame| {
        if (frame.stopped) break;
        try evm.step();
    }

    const frame = evm.getCurrentFrame() orelse return try allocator.dupe(u8, deployment_bytecode);
    defer evm.frames.clearRetainingCapacity();

    if (frame.reverted or frame.output.len == 0) {
        return try allocator.dupe(u8, deployment_bytecode);
    }
    return try allocator.dupe(u8, frame.output);
}

fn printStop(
    writer: anytype,
    debugger: *Debugger,
    entry: *const SourceMap.Entry,
    stop_index: usize,
) !void {
    try writer.print("\n[{d}] pc={d} idx={?d} {s}:{d}:{d}\n", .{
        stop_index,
        entry.pc,
        entry.idx,
        entry.file,
        entry.line,
        entry.col,
    });
    if (debugger.getSourceLineText(entry.line)) |line_text| {
        try writer.print("    {s}\n", .{std.mem.trim(u8, line_text, " \t")});
    }
    try writer.print("    opcode={s} depth={d} gas={d}\n", .{
        debugger.getCurrentOpcodeName(),
        debugger.getCallDepth(),
        debugger.getGasRemaining(),
    });

    const allocator = debugger.allocator;
    const bindings = try debugger.getVisibleBindings(allocator);
    defer allocator.free(bindings);
    for (bindings) |binding| {
        try writer.print("    binding {s} [{s}]", .{ binding.name, binding.runtime_kind });
        if (binding.folded_value) |folded| {
            try writer.print(" folded={s}", .{folded});
        }
        if (try debugger.getVisibleBindingValueByName(allocator, binding.name)) |value| {
            try writer.print(" value={d}", .{value});
        }
        try writer.print("\n", .{});
    }
}

fn parseArgs(allocator: std.mem.Allocator) !ProbeConfig {
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 4) {
        try printUsage();
        return error.InvalidArguments;
    }

    var config = ProbeConfig{
        .bytecode_path = try allocator.dupe(u8, args[1]),
        .source_map_path = try allocator.dupe(u8, args[2]),
        .source_path = try allocator.dupe(u8, args[3]),
        .calldata = try allocator.dupe(u8, &.{}),
    };
    errdefer config.deinit(allocator);

    var i: usize = 4;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--debug-info")) {
            i += 1;
            if (i >= args.len) return error.InvalidArguments;
            if (config.debug_info_path) |path| allocator.free(path);
            config.debug_info_path = try allocator.dupe(u8, args[i]);
        } else if (std.mem.eql(u8, arg, "--signature")) {
            i += 1;
            if (i >= args.len) return error.InvalidArguments;
            allocator.free(config.calldata);
            config.calldata = try selectorCalldataAlloc(allocator, args[i]);
        } else if (std.mem.eql(u8, arg, "--calldata-hex")) {
            i += 1;
            if (i >= args.len) return error.InvalidArguments;
            allocator.free(config.calldata);
            config.calldata = try decodeHexAlloc(allocator, args[i]);
        } else if (std.mem.eql(u8, arg, "--max-statements")) {
            i += 1;
            if (i >= args.len) return error.InvalidArguments;
            config.max_statements = try std.fmt.parseInt(usize, args[i], 10);
        } else {
            try printUsage();
            return error.InvalidArguments;
        }
    }

    return config;
}

fn printUsage() !void {
    var stderr_buffer: [2048]u8 = undefined;
    var stderr_file = std.fs.File.stderr().writer(&stderr_buffer);
    const stderr = &stderr_file.interface;
    try stderr.print(
        \\usage:
        \\  zig build debug-probe -- <bytecode.hex> <source-map.json> <source.ora> [--debug-info <debug.json>] [--signature <sig>] [--calldata-hex <hex>] [--max-statements <n>]
        \\
        \\examples:
        \\  zig build debug-probe -- ../artifacts/debugger_selector_probe/comptime_shift_probe.hex ../artifacts/debugger_selector_probe/comptime_shift_probe.sourcemap.json ../ora-example/comptime/comptime_shift_probe.ora --signature test_large_shr() --max-statements 8
        \\
    , .{});
    try stderr.flush();
}

fn selectorCalldataAlloc(allocator: std.mem.Allocator, signature: []const u8) ![]u8 {
    var hash: [32]u8 = undefined;
    std.crypto.hash.sha3.Keccak256.hash(signature, &hash, .{});
    return try allocator.dupe(u8, hash[0..4]);
}

fn decodeHexAlloc(allocator: std.mem.Allocator, text: []const u8) ![]u8 {
    var trimmed = std.mem.trim(u8, text, " \t\r\n");
    if (std.mem.startsWith(u8, trimmed, "0x") or std.mem.startsWith(u8, trimmed, "0X")) {
        trimmed = trimmed[2..];
    }

    if (trimmed.len == 0) return try allocator.dupe(u8, &.{});
    if (trimmed.len % 2 != 0) return error.InvalidHex;

    const out = try allocator.alloc(u8, trimmed.len / 2);
    errdefer allocator.free(out);

    var i: usize = 0;
    while (i < out.len) : (i += 1) {
        out[i] = try std.fmt.parseInt(u8, trimmed[i * 2 .. i * 2 + 2], 16);
    }
    return out;
}

fn printHex(writer: anytype, bytes: []const u8) !void {
    for (bytes) |byte| {
        const hi: u8 = @intCast((byte >> 4) & 0x0f);
        const lo: u8 = @intCast(byte & 0x0f);
        try writer.writeByte(hexNibble(hi));
        try writer.writeByte(hexNibble(lo));
    }
}

fn hexNibble(value: u8) u8 {
    return if (value < 10) '0' + value else 'a' + (value - 10);
}
