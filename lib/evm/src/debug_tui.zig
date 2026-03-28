const std = @import("std");
const primitives = @import("voltaire");
const ora_evm = @import("ora_evm");

const Evm = ora_evm.Evm(.{});
const Frame = ora_evm.Frame(.{});
const Debugger = ora_evm.Debugger(.{});
const SourceMap = ora_evm.SourceMap;
const DebugInfo = ora_evm.DebugInfo;

const AppConfig = struct {
    bytecode_path: []u8,
    source_map_path: []u8,
    source_path: []u8,
    debug_info_path: ?[]u8 = null,
    abi_path: ?[]u8 = null,
    calldata: []u8 = &.{},

    fn deinit(self: *AppConfig, allocator: std.mem.Allocator) void {
        allocator.free(self.bytecode_path);
        allocator.free(self.source_map_path);
        allocator.free(self.source_path);
        if (self.debug_info_path) |path| allocator.free(path);
        if (self.abi_path) |path| allocator.free(path);
        allocator.free(self.calldata);
    }
};

const Terminal = struct {
    stdout: std.fs.File,
    stdin: std.fs.File,
    original: std.posix.termios,
    raw_enabled: bool = false,

    const Size = struct {
        rows: u16,
        cols: u16,
    };

    fn init() !Terminal {
        return .{
            .stdout = std.fs.File.stdout(),
            .stdin = std.fs.File.stdin(),
            .original = try std.posix.tcgetattr(std.posix.STDIN_FILENO),
        };
    }

    fn enableRawMode(self: *Terminal) !void {
        var raw = self.original;
        raw.iflag.IXON = false;
        raw.iflag.ICRNL = false;
        raw.lflag.ECHO = false;
        raw.lflag.ICANON = false;
        raw.lflag.ISIG = false;
        raw.lflag.IEXTEN = false;
        raw.oflag.OPOST = false;
        raw.cc[@intFromEnum(std.posix.V.MIN)] = 1;
        raw.cc[@intFromEnum(std.posix.V.TIME)] = 0;
        try std.posix.tcsetattr(std.posix.STDIN_FILENO, .FLUSH, raw);
        self.raw_enabled = true;
    }

    fn restore(self: *Terminal) void {
        if (self.raw_enabled) {
            std.posix.tcsetattr(std.posix.STDIN_FILENO, .FLUSH, self.original) catch {};
            self.raw_enabled = false;
        }
        self.writeAll("\x1b[0m\x1b[?25h\x1b[?1049l") catch {};
    }

    fn enterAltScreen(self: *Terminal) !void {
        try self.writeAll("\x1b[?1049h\x1b[?25l");
    }

    fn clear(self: *Terminal) !void {
        try self.writeAll("\x1b[2J\x1b[H");
    }

    fn size(self: *Terminal) Size {
        _ = self;
        var ws: std.posix.winsize = undefined;
        const rc = std.posix.system.ioctl(std.posix.STDOUT_FILENO, std.posix.T.IOCGWINSZ, @intFromPtr(&ws));
        if (std.posix.errno(rc) == .SUCCESS and ws.row > 0 and ws.col > 0) {
            return .{ .rows = ws.row, .cols = ws.col };
        }
        return .{ .rows = 24, .cols = 100 };
    }

    fn readByte(self: *Terminal) !u8 {
        var byte: [1]u8 = undefined;
        _ = try self.stdin.read(&byte);
        return byte[0];
    }

    fn writeAll(self: *Terminal, bytes: []const u8) !void {
        try self.stdout.writeAll(bytes);
    }
};

const App = struct {
    allocator: std.mem.Allocator,
    debugger: Debugger,
    source_map: SourceMap,
    source_text: []u8,
    terminal: Terminal,
    scroll_line: u32 = 1,
    status: []const u8 = "ready",
    calldata: []const u8,
    bytecode_path: []const u8,
    source_path: []const u8,

    fn deinit(self: *App) void {
        self.terminal.restore();
        self.debugger.deinit();
        self.source_map.deinit();
        self.allocator.free(self.source_text);
    }

    fn run(self: *App) !void {
        try self.primeInitialStop();
        try self.terminal.enableRawMode();
        try self.terminal.enterAltScreen();
        while (true) {
            try self.render();
            const ch = try self.terminal.readByte();
            switch (ch) {
                'q' => return,
                's' => try self.doStepIn(),
                'n' => try self.doStepOver(),
                'o' => try self.doStepOut(),
                'c' => try self.doContinue(),
                'j' => self.scrollDown(),
                'k' => self.scrollUp(),
                else => {},
            }
        }
    }

    fn primeInitialStop(self: *App) !void {
        if (self.debugger.isHalted()) {
            self.status = "halted";
            return;
        }

        if (self.debugger.currentEntry()) |entry| {
            if (entry.is_statement) {
                self.status = "ready";
                self.centerOnCurrentLine();
                return;
            }
        }

        var attempts: usize = 0;
        while (!self.debugger.isHalted() and attempts < 64) : (attempts += 1) {
            try self.debugger.stepIn();
            self.status = @tagName(self.debugger.stop_reason);
            if (self.debugger.currentEntry()) |entry| {
                if (entry.is_statement) break;
            }
        }

        if (self.debugger.isHalted()) {
            self.status = @tagName(self.debugger.stop_reason);
        }
        self.centerOnCurrentLine();
    }

    fn doStepIn(self: *App) !void {
        if (self.debugger.isHalted()) {
            self.status = "halted";
            return;
        }
        try self.debugger.stepIn();
        self.status = @tagName(self.debugger.stop_reason);
        self.centerOnCurrentLine();
    }

    fn doStepOver(self: *App) !void {
        if (self.debugger.isHalted()) {
            self.status = "halted";
            return;
        }
        try self.debugger.stepOver();
        self.status = @tagName(self.debugger.stop_reason);
        self.centerOnCurrentLine();
    }

    fn doStepOut(self: *App) !void {
        if (self.debugger.isHalted()) {
            self.status = "halted";
            return;
        }
        try self.debugger.stepOut();
        self.status = @tagName(self.debugger.stop_reason);
        self.centerOnCurrentLine();
    }

    fn doContinue(self: *App) !void {
        if (self.debugger.isHalted()) {
            self.status = "halted";
            return;
        }
        try self.debugger.continue_();
        self.status = @tagName(self.debugger.stop_reason);
        self.centerOnCurrentLine();
    }

    fn scrollDown(self: *App) void {
        const total = self.debugger.totalSourceLines();
        if (self.scroll_line < total) self.scroll_line += 1;
    }

    fn scrollUp(self: *App) void {
        if (self.scroll_line > 1) self.scroll_line -= 1;
    }

    fn centerOnCurrentLine(self: *App) void {
        const line = self.debugger.currentSourceLine() orelse return;
        const height: u32 = 18;
        if (line > height / 2) {
            self.scroll_line = line - height / 2;
        } else {
            self.scroll_line = 1;
        }
    }

    fn render(self: *App) !void {
        const size = self.terminal.size();
        const total_rows: usize = @intCast(size.rows);
        const total_cols: usize = @intCast(size.cols);
        const header_rows: usize = 3;
        const footer_rows: usize = 2;
        const content_rows = if (total_rows > header_rows + footer_rows) total_rows - header_rows - footer_rows else 1;
        const right_width: usize = if (total_cols >= 100) 36 else @min(@as(usize, 30), total_cols / 3);
        const left_width = if (total_cols > right_width + 1) total_cols - right_width - 1 else total_cols;

        try self.terminal.clear();

        var buffer = std.ArrayList(u8){};
        defer buffer.deinit(self.allocator);
        const writer = buffer.writer(self.allocator);

        const line = self.debugger.currentSourceLine() orelse 0;
        const entry = self.debugger.currentEntry();
        try writer.print("Ora EVM Debugger  {s}\n", .{self.source_path});
        try writer.print("status={s} pc={d} idx={?d} opcode={s} depth={d} gas={d} line={d}\n", .{
            self.status,
            self.debugger.getPC(),
            if (entry) |e| e.idx else null,
            self.debugger.getCurrentOpcodeName(),
            self.debugger.getCallDepth(),
            self.debugger.getGasRemaining(),
            line,
        });
        try writer.print("bytecode={s} calldata=0x", .{self.bytecode_path});
        try writeHex(writer, self.calldata);
        try writer.writeByte('\n');

        const start_line = self.scroll_line;
        const end_line = @min(self.debugger.totalSourceLines(), start_line + @as(u32, @intCast(content_rows)) - 1);
        var row: u32 = start_line;
        while (row <= end_line) : (row += 1) {
            const left_text = try self.renderSourceRow(self.allocator, row, left_width, line);
            defer self.allocator.free(left_text);
            try writer.writeAll(left_text);

            if (@as(usize, @intCast(row - start_line)) < content_rows) {
                const right_text = try self.renderBindingRow(self.allocator, @intCast(row - start_line), right_width);
                defer self.allocator.free(right_text);
                if (right_width > 0) {
                    try writer.writeByte(' ');
                    try writer.writeAll(right_text);
                }
            }
            try writer.writeByte('\n');
        }

        var remaining = content_rows - @as(usize, @intCast(end_line - start_line + 1));
        while (remaining > 0) : (remaining -= 1) {
            try writeSpaces(writer, left_width);
            if (right_width > 0) {
                try writer.writeByte(' ');
                const idx = content_rows - remaining;
                const right_text = try self.renderBindingRow(self.allocator, idx, right_width);
                defer self.allocator.free(right_text);
                try writer.writeAll(right_text);
            }
            try writer.writeByte('\n');
        }

        try writer.print("keys: s step-in  n step-over  o step-out  c continue  j/k scroll  q quit\n", .{});
        try writer.print("bindings: visible locals, folded values, and rooted runtime values when readable\n", .{});

        try self.terminal.writeAll(buffer.items);
    }

    fn renderSourceRow(
        self: *App,
        allocator: std.mem.Allocator,
        line_number: u32,
        width: usize,
        current_line: u32,
    ) ![]u8 {
        const raw_line = self.debugger.getSourceLineText(line_number) orelse "";
        const trimmed = std.mem.trimRight(u8, raw_line, "\r");
        const prefix = if (line_number == current_line) ">" else " ";
        var row = std.ArrayList(u8){};
        errdefer row.deinit(allocator);
        const writer = row.writer(allocator);
        if (line_number == current_line) try writer.writeAll("\x1b[7m");
        try writer.print("{s}{d:>4} | ", .{ prefix, line_number });
        const text_width = if (width > 8) width - 8 else 0;
        try writeTruncated(writer, trimmed, text_width);
        if (line_number == current_line) try writer.writeAll("\x1b[0m");
        const owned = try row.toOwnedSlice(allocator);
        return owned;
    }

    fn renderBindingRow(self: *App, allocator: std.mem.Allocator, index: usize, width: usize) ![]u8 {
        if (width == 0) return try allocator.dupe(u8, "");

        const bindings = try self.debugger.getVisibleBindings(allocator);
        defer allocator.free(bindings);

        var row = std.ArrayList(u8){};
        errdefer row.deinit(allocator);
        const writer = row.writer(allocator);

        if (index == 0) {
            try writeTruncated(writer, "Bindings", width);
            return try row.toOwnedSlice(allocator);
        }
        if (index - 1 >= bindings.len) {
            return try allocator.dupe(u8, "");
        }

        const binding = bindings[index - 1];
        try writer.print("{s}", .{binding.name});
        if (binding.folded_value) |folded| {
            try writer.print("={s}", .{folded});
        } else if (try self.debugger.getVisibleBindingValueByName(allocator, binding.name)) |value| {
            try writer.print("={d}", .{value});
        } else {
            try writer.print(" [{s}]", .{binding.runtime_kind});
        }

        const owned = try row.toOwnedSlice(allocator);
        if (owned.len <= width) return owned;
        defer allocator.free(owned);
        return try truncateOwned(allocator, owned, width);
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

    const source_text = try std.fs.cwd().readFileAlloc(allocator, config.source_path, 16 * 1024 * 1024);

    var debug_info_json: ?[]u8 = null;
    defer if (debug_info_json) |bytes| allocator.free(bytes);
    var debug_info: ?DebugInfo = null;
    if (config.debug_info_path) |path| {
        debug_info_json = try std.fs.cwd().readFileAlloc(allocator, path, 16 * 1024 * 1024);
        debug_info = try DebugInfo.loadFromJson(allocator, debug_info_json.?);
    }

    var evm: Evm = undefined;
    try evm.init(allocator, null, null, null, primitives.ZERO_ADDRESS, 0, null);
    errdefer evm.deinit();

    const caller = primitives.Address.fromU256(0x100);
    const contract = primitives.Address.fromU256(0x200);

    try evm.initTransactionState(null);
    try evm.preWarmTransaction(contract);

    const runtime_bytecode = try deployRuntimeBytecode(allocator, &evm, caller, contract, bytecode);
    defer allocator.free(runtime_bytecode);
    if (source_map.runtime_start_pc) |_| {
        const runtime_source_map = try rebaseSourceMapForRuntime(allocator, &source_map, runtime_bytecode);
        source_map.deinit();
        source_map = runtime_source_map;
    }

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

    const debugger = blk: {
        if (debug_info) |info| {
            debug_info = null;
            break :blk try Debugger.initWithDebugInfo(allocator, &evm, source_map, info, source_text);
        }
        break :blk try Debugger.init(allocator, &evm, source_map, source_text);
    };

    var app = App{
        .allocator = allocator,
        .debugger = debugger,
        .source_map = source_map,
        .source_text = source_text,
        .terminal = try Terminal.init(),
        .status = "ready",
        .calldata = config.calldata,
        .bytecode_path = config.bytecode_path,
        .source_path = config.source_path,
    };
    defer {
        app.deinit();
        evm.deinit();
    }
    app.centerOnCurrentLine();
    try app.run();
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

fn rebaseSourceMapForRuntime(
    allocator: std.mem.Allocator,
    creation_source_map: *const SourceMap,
    runtime_bytecode: []const u8,
) !SourceMap {
    const runtime_start_pc = creation_source_map.runtime_start_pc orelse {
        return try SourceMap.fromEntries(allocator, creation_source_map.entries);
    };
    if (runtime_bytecode.len == 0) {
        return try SourceMap.fromEntries(allocator, creation_source_map.entries);
    }

    var rebased: std.ArrayList(SourceMap.Entry) = .{};
    defer rebased.deinit(allocator);

    for (creation_source_map.entries) |entry| {
        if (entry.pc < runtime_start_pc) continue;
        const rebased_pc = entry.pc - runtime_start_pc;
        if (rebased_pc >= runtime_bytecode.len) continue;
        try rebased.append(allocator, .{
            .idx = entry.idx,
            .pc = @intCast(rebased_pc),
            .file = entry.file,
            .line = entry.line,
            .col = entry.col,
            .is_statement = entry.is_statement,
        });
    }

    if (rebased.items.len == 0) {
        return try SourceMap.fromEntries(allocator, creation_source_map.entries);
    }
    var runtime_map = try SourceMap.fromEntries(allocator, rebased.items);
    runtime_map.runtime_start_pc = 0;
    return runtime_map;
}

fn parseArgs(allocator: std.mem.Allocator) !AppConfig {
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 4) {
        try printUsage();
        return error.InvalidArguments;
    }

    var config = AppConfig{
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
        } else {
            try printUsage();
            return error.InvalidArguments;
        }
    }

    if (signature) |sig| {
        allocator.free(config.calldata);
        if (config.abi_path) |abi_path| {
            const abi_bytes = try std.fs.cwd().readFileAlloc(allocator, abi_path, 16 * 1024 * 1024);
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
        \\  zig build debug-tui -- <bytecode.hex> <source-map.json> <source.ora> [--debug-info <debug.json>] [--abi <abi.json>] [--signature <sig> [--arg <value>]...] [--calldata-hex <hex>]
        \\
    , .{});
    try stderr.flush();
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

fn writeHex(writer: anytype, bytes: []const u8) !void {
    for (bytes) |byte| {
        try writer.print("{x:0>2}", .{byte});
    }
}

fn writeTruncated(writer: anytype, text: []const u8, width: usize) !void {
    if (width == 0) return;
    if (text.len <= width) {
        try writer.writeAll(text);
        if (text.len < width) try writeSpaces(writer, width - text.len);
        return;
    }
    if (width <= 1) {
        try writer.writeByte(text[0]);
        return;
    }
    try writer.writeAll(text[0 .. width - 1]);
    try writer.writeByte('~');
}

fn writeSpaces(writer: anytype, count: usize) !void {
    var i: usize = 0;
    while (i < count) : (i += 1) try writer.writeByte(' ');
}

fn truncateOwned(allocator: std.mem.Allocator, text: []const u8, width: usize) ![]u8 {
    var out = std.ArrayList(u8){};
    errdefer out.deinit(allocator);
    const writer = out.writer(allocator);
    try writeTruncated(writer, text, width);
    return try out.toOwnedSlice(allocator);
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
