const std = @import("std");
const primitives = @import("voltaire");
const ora_evm = @import("ora_evm");

const Evm = ora_evm.Evm(.{});
const Frame = ora_evm.Frame(.{});
const Debugger = ora_evm.Debugger(.{});
const SourceMap = ora_evm.SourceMap;
const DebugInfo = ora_evm.DebugInfo;
const Session = ora_evm.DebugSession(.{});

const ProbeConfig = struct {
    bytecode_path: []const u8,
    source_map_path: []const u8,
    source_path: []const u8,
    debug_info_path: ?[]const u8 = null,
    abi_path: ?[]const u8 = null,
    calldata: []u8 = &.{},
    max_statements: usize = 64,
    limits: ora_evm.DebugLimits = .{},

    fn deinit(self: *ProbeConfig, allocator: std.mem.Allocator) void {
        allocator.free(self.bytecode_path);
        allocator.free(self.source_map_path);
        allocator.free(self.source_path);
        if (self.debug_info_path) |path| allocator.free(path);
        if (self.abi_path) |path| allocator.free(path);
        allocator.free(self.calldata);
    }
};

pub fn main() !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const allocator = gpa_state.allocator();

    var config = try parseArgs(allocator);
    defer config.deinit(allocator);

    const limits = config.limits;

    const bytecode_hex = try ora_evm.loadDebuggerArtifactWithCap(allocator, config.bytecode_path, limits.artifact_max_bytes);
    defer allocator.free(bytecode_hex);
    const bytecode = try decodeHexAlloc(allocator, bytecode_hex);
    defer allocator.free(bytecode);

    const source_map_json = try ora_evm.loadDebuggerArtifactWithCap(allocator, config.source_map_path, limits.artifact_max_bytes);
    defer allocator.free(source_map_json);
    var source_map = try SourceMap.loadFromJson(allocator, source_map_json);
    defer source_map.deinit();

    const source_text = try ora_evm.loadDebuggerArtifactWithCap(allocator, config.source_path, limits.artifact_max_bytes);
    defer allocator.free(source_text);

    var debug_info_json: ?[]u8 = null;
    defer if (debug_info_json) |bytes| allocator.free(bytes);
    var debug_info: ?DebugInfo = null;
    if (config.debug_info_path) |path| {
        debug_info_json = try ora_evm.loadDebuggerArtifactWithCap(allocator, path, limits.artifact_max_bytes);
        debug_info = try DebugInfo.loadFromJson(allocator, debug_info_json.?);
    }
    defer if (debug_info) |*info| info.deinit();

    var evm: Evm = undefined;
    try evm.init(allocator, null, null, ora_evm.deterministicBlockContext(), primitives.ZERO_ADDRESS, 0, null);
    defer evm.deinit();

    const caller = primitives.Address.fromU256(0x100);
    const contract = primitives.Address.fromU256(0x200);

    try evm.initTransactionState(null);
    try evm.preWarmTransaction(contract);

    const runtime_bytecode = try Session.deployRuntimeBytecode(allocator, &evm, .{
        .caller = caller,
        .contract = contract,
        .deployment_bytecode = bytecode,
        .gas_limit = limits.gas_limit,
        .step_cap = limits.deploy_step_cap,
    });
    defer allocator.free(runtime_bytecode);
    if (source_map.runtime_start_pc) |_| {
        const runtime_source_map = try Session.rebaseSourceMapForRuntime(allocator, &source_map, runtime_bytecode);
        source_map.deinit();
        source_map = runtime_source_map;
    }

    try evm.frames.append(evm.arena.allocator(), try Frame.init(
        evm.arena.allocator(),
        runtime_bytecode,
        limits.gas_limit,
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
    debugger.max_steps = limits.max_steps;

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
        if (debugger.stop_reason == .step_complete or debugger.stop_reason == .breakpoint_hit) {
            if (debugger.currentEntry()) |entry| {
                stops += 1;
                try printStop(stdout, &debugger, entry, stops);
            }
        } else if (debugger.isHalted()) {
            break;
        } else if (debugger.currentEntry()) |entry| {
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

    var signature: ?[]u8 = null;
    defer if (signature) |sig| allocator.free(sig);

    var raw_args: std.ArrayList([]u8) = .{};
    defer {
        for (raw_args.items) |item| allocator.free(item);
        raw_args.deinit(allocator);
    }

    var i: usize = 4;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--debug-info")) {
            i += 1;
            if (i >= args.len) return error.InvalidArguments;
            if (config.debug_info_path) |path| allocator.free(path);
            config.debug_info_path = try allocator.dupe(u8, args[i]);
        } else if (std.mem.eql(u8, arg, "--abi")) {
            i += 1;
            if (i >= args.len) return error.InvalidArguments;
            if (config.abi_path) |path| allocator.free(path);
            config.abi_path = try allocator.dupe(u8, args[i]);
        } else if (std.mem.eql(u8, arg, "--signature")) {
            i += 1;
            if (i >= args.len) return error.InvalidArguments;
            if (signature) |sig| allocator.free(sig);
            signature = try allocator.dupe(u8, args[i]);
        } else if (std.mem.eql(u8, arg, "--arg")) {
            i += 1;
            if (i >= args.len) return error.InvalidArguments;
            try raw_args.append(allocator, try allocator.dupe(u8, args[i]));
        } else if (std.mem.eql(u8, arg, "--calldata-hex")) {
            i += 1;
            if (i >= args.len) return error.InvalidArguments;
            allocator.free(config.calldata);
            config.calldata = try decodeHexAlloc(allocator, args[i]);
        } else if (std.mem.eql(u8, arg, "--max-statements")) {
            i += 1;
            if (i >= args.len) return error.InvalidArguments;
            config.max_statements = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--gas-limit")) {
            i += 1;
            if (i >= args.len) return error.InvalidArguments;
            config.limits.gas_limit = try std.fmt.parseInt(i64, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--max-steps")) {
            i += 1;
            if (i >= args.len) return error.InvalidArguments;
            config.limits.max_steps = try std.fmt.parseInt(u64, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--deploy-step-cap")) {
            i += 1;
            if (i >= args.len) return error.InvalidArguments;
            config.limits.deploy_step_cap = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--artifact-max-bytes")) {
            i += 1;
            if (i >= args.len) return error.InvalidArguments;
            config.limits.artifact_max_bytes = try std.fmt.parseInt(usize, args[i], 10);
        } else {
            try printUsage();
            return error.InvalidArguments;
        }
    }

    if (signature) |sig| {
        allocator.free(config.calldata);
        if (config.abi_path) |abi_path| {
            const abi_bytes = try ora_evm.loadDebuggerArtifact(allocator, abi_path);
            defer allocator.free(abi_bytes);
            config.calldata = try encodeAbiCallDataAlloc(allocator, abi_bytes, sig, raw_args.items);
        } else {
            config.calldata = try encodeSignatureCallDataAlloc(allocator, sig, raw_args.items);
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
        \\  zig build debug-probe -- <bytecode.hex> <source-map.json> <source.ora> [options]
        \\
        \\options:
        \\  --debug-info <debug.json>    Load source-scope debug info
        \\  --abi <abi.json>             Load ABI for signature-driven calldata
        \\  --signature <sig>            Function signature, e.g. add(u256,u256)
        \\  --arg <value>                Argument value (repeatable, in order)
        \\  --calldata-hex <hex>         Raw calldata as hex
        \\  --max-statements <n>         Cap source-stop count (default 64)
        \\  --gas-limit <i64>            Frame gas budget (default 5000000)
        \\  --max-steps <u64>            Per-command opcode safety cap (default 10000000)
        \\  --deploy-step-cap <usize>    Deployment opcode cap (default 200000)
        \\  --artifact-max-bytes <usize> Per-file artifact load cap (default 16777216)
        \\
        \\examples:
        \\  zig build debug-probe -- ../artifacts/debugger_selector_probe/comptime_shift_probe.hex ../artifacts/debugger_selector_probe/comptime_shift_probe.sourcemap.json ../ora-example/comptime/comptime_shift_probe.ora --signature test_large_shr() --max-statements 8
        \\  zig build debug-probe -- ../../artifacts/debugger_abi_probe/bin/arithmetic_test.hex ../../artifacts/debugger_abi_probe/arithmetic_test.sourcemap.json ../../ora-example/arithmetic_test.ora --abi ../../artifacts/debugger_abi_probe/abi/arithmetic_test.abi.json --signature add(u256,u256) --arg 7 --arg 9
        \\
    , .{});
    try stderr.flush();
}

fn selectorCalldataAlloc(allocator: std.mem.Allocator, signature: []const u8) ![]u8 {
    var hash: [32]u8 = undefined;
    std.crypto.hash.sha3.Keccak256.hash(signature, &hash, .{});
    return try allocator.dupe(u8, hash[0..4]);
}

fn encodeSignatureCallDataAlloc(
    allocator: std.mem.Allocator,
    signature: []const u8,
    arg_values: []const []const u8,
) ![]u8 {
    const open = std.mem.indexOfScalar(u8, signature, '(') orelse return error.InvalidSignature;
    const close = std.mem.lastIndexOfScalar(u8, signature, ')') orelse return error.InvalidSignature;
    if (close < open) return error.InvalidSignature;

    const type_list = signature[open + 1 .. close];
    var count: usize = 0;
    if (std.mem.trim(u8, type_list, " \t").len > 0) {
        var split_count = std.mem.splitScalar(u8, type_list, ',');
        while (split_count.next()) |_| count += 1;
    }
    if (count != arg_values.len) return error.ArgumentCountMismatch;

    const out = try allocator.alloc(u8, 4 + count * 32);
    errdefer allocator.free(out);

    var hash: [32]u8 = undefined;
    std.crypto.hash.sha3.Keccak256.hash(signature, &hash, .{});
    @memcpy(out[0..4], hash[0..4]);

    var split = std.mem.splitScalar(u8, type_list, ',');
    var arg_index: usize = 0;
    while (split.next()) |raw_type| : (arg_index += 1) {
        const ty = std.mem.trim(u8, raw_type, " \t");
        if (ty.len == 0) continue;
        const dest = out[4 + arg_index * 32 .. 4 + (arg_index + 1) * 32];
        @memset(dest, 0);
        try encodeAbiWord(allocator, dest, ty, arg_values[arg_index]);
    }

    return out;
}

fn encodeAbiCallDataAlloc(
    allocator: std.mem.Allocator,
    abi_json: []const u8,
    signature: []const u8,
    arg_values: []const []const u8,
) ![]u8 {
    const parsed = try std.json.parseFromSlice(OraAbi, allocator, abi_json, .{
        .ignore_unknown_fields = true,
    });
    defer parsed.deinit();

    const callable = findCallable(&parsed.value, signature) orelse return error.CallableNotFound;
    const wire = callable.wire orelse return error.MissingSelector;
    const selector_hex = wire.@"evm-default".selector orelse return error.MissingSelector;
    const selector = try decodeHexAlloc(allocator, selector_hex);
    defer allocator.free(selector);
    if (selector.len != 4) return error.InvalidSelector;

    if (callable.inputs.len != arg_values.len) return error.ArgumentCountMismatch;
    const out = try allocator.alloc(u8, 4 + callable.inputs.len * 32);
    errdefer allocator.free(out);
    @memcpy(out[0..4], selector);

    for (callable.inputs, 0..) |input, i| {
        const ty = resolveWireType(&parsed.value, input.typeId) orelse return error.UnsupportedArgType;
        const dest = out[4 + i * 32 .. 4 + (i + 1) * 32];
        @memset(dest, 0);
        try encodeAbiWord(allocator, dest, ty, arg_values[i]);
    }

    return out;
}

fn findCallable(abi: *const OraAbi, signature: []const u8) ?OraAbi.Callable {
    for (abi.callables) |callable| {
        if (std.mem.eql(u8, callable.signature, signature)) return callable;
        if (signatureEquivalent(callable.signature, signature)) return callable;
    }
    return null;
}

fn signatureEquivalent(abi_signature: []const u8, probe_signature: []const u8) bool {
    const abi_open = std.mem.indexOfScalar(u8, abi_signature, '(') orelse return false;
    const probe_open = std.mem.indexOfScalar(u8, probe_signature, '(') orelse return false;
    const abi_close = std.mem.lastIndexOfScalar(u8, abi_signature, ')') orelse return false;
    const probe_close = std.mem.lastIndexOfScalar(u8, probe_signature, ')') orelse return false;
    if (abi_close < abi_open or probe_close < probe_open) return false;

    if (!std.mem.eql(u8, std.mem.trim(u8, abi_signature[0..abi_open], " \t"), std.mem.trim(u8, probe_signature[0..probe_open], " \t"))) {
        return false;
    }

    var abi_split = std.mem.splitScalar(u8, abi_signature[abi_open + 1 .. abi_close], ',');
    var probe_split = std.mem.splitScalar(u8, probe_signature[probe_open + 1 .. probe_close], ',');
    while (true) {
        const abi_ty = abi_split.next();
        const probe_ty = probe_split.next();
        if (abi_ty == null or probe_ty == null) return abi_ty == null and probe_ty == null;
        if (!typeAliasEquivalent(std.mem.trim(u8, abi_ty.?, " \t"), std.mem.trim(u8, probe_ty.?, " \t"))) {
            return false;
        }
    }
}

fn typeAliasEquivalent(abi_ty: []const u8, probe_ty: []const u8) bool {
    if (std.mem.eql(u8, abi_ty, probe_ty)) return true;
    if (std.mem.startsWith(u8, abi_ty, "uint") and std.mem.startsWith(u8, probe_ty, "u")) {
        return std.mem.eql(u8, abi_ty[4..], probe_ty[1..]);
    }
    if (std.mem.startsWith(u8, abi_ty, "int") and std.mem.startsWith(u8, probe_ty, "i")) {
        return std.mem.eql(u8, abi_ty[3..], probe_ty[1..]);
    }
    return false;
}

fn resolveWireType(abi: *const OraAbi, type_id: []const u8) ?[]const u8 {
    const entry = abi.types.map.get(type_id) orelse return null;
    return entry.wire.@"evm-default".type;
}

fn encodeAbiWord(
    allocator: std.mem.Allocator,
    dest: []u8,
    ty: []const u8,
    value_text: []const u8,
) !void {
    if (dest.len != 32) return error.InvalidAbiWord;

    if (std.mem.eql(u8, ty, "bool")) {
        const normalized = std.mem.trim(u8, value_text, " \t");
        if (std.ascii.eqlIgnoreCase(normalized, "true")) {
            dest[31] = 1;
            return;
        }
        if (std.ascii.eqlIgnoreCase(normalized, "false")) {
            dest[31] = 0;
            return;
        }
        return error.InvalidBoolLiteral;
    }

    if (std.mem.eql(u8, ty, "address")) {
        const bytes = try decodeHexAlloc(allocator, value_text);
        defer allocator.free(bytes);
        if (bytes.len != 20) return error.InvalidAddressLiteral;
        @memcpy(dest[12..32], bytes);
        return;
    }

    if (std.mem.startsWith(u8, ty, "u") or std.mem.startsWith(u8, ty, "uint")) {
        const number = try parseUnsignedLiteral(value_text);
        writeU256BigEndian(dest, number);
        return;
    }

    return error.UnsupportedArgType;
}

fn parseUnsignedLiteral(text: []const u8) !u256 {
    const trimmed = std.mem.trim(u8, text, " \t");
    if (std.mem.startsWith(u8, trimmed, "0x") or std.mem.startsWith(u8, trimmed, "0X")) {
        return try std.fmt.parseInt(u256, trimmed[2..], 16);
    }
    return try std.fmt.parseInt(u256, trimmed, 10);
}

fn writeU256BigEndian(dest: []u8, value: u256) void {
    var i: usize = 0;
    while (i < 32) : (i += 1) {
        const shift: u8 = @intCast((31 - i) * 8);
        const chunk: u256 = value >> shift;
        dest[i] = @intCast(chunk & 0xff);
    }
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

const OraAbi = struct {
    callables: []const Callable = &.{},
    types: std.json.ArrayHashMap(TypeEntry) = .{},

    const Callable = struct {
        signature: []const u8,
        inputs: []const Input = &.{},
        wire: ?WireProfiles = null,
    };

    const Input = struct {
        typeId: []const u8,
    };

    const WireProfiles = struct {
        @"evm-default": EvmWire = .{},
    };

    const EvmWire = struct {
        selector: ?[]const u8 = null,
    };

    const TypeEntry = struct {
        wire: TypeWireProfiles = .{},
    };

    const TypeWireProfiles = struct {
        @"evm-default": TypeWire = .{},
    };

    const TypeWire = struct {
        type: ?[]const u8 = null,
    };
};
