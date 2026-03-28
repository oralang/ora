const std = @import("std");
const tui = @import("tui");
const primitives = @import("voltaire");
const ora_evm = @import("ora_evm");

const Evm = ora_evm.Evm(.{});
const Frame = ora_evm.Frame(.{});
const Debugger = ora_evm.Debugger(.{});
const SourceMap = ora_evm.SourceMap;
const DebugInfo = ora_evm.DebugInfo;

const Color = tui.color.Color;
const Style = tui.style.Style;
const BorderStyle = tui.style.BorderStyle;

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

const DebuggerView = struct {
    allocator: std.mem.Allocator,
    debugger: *Debugger,
    app: ?*tui.App = null,
    source_path: []const u8,
    calldata: []const u8,
    scroll_line: u32 = 1,
    focus_line: ?u32 = null,
    status: []const u8 = "ready",

    fn init(
        allocator: std.mem.Allocator,
        debugger: *Debugger,
        source_path: []const u8,
        calldata: []const u8,
    ) !DebuggerView {
        var self = DebuggerView{
            .allocator = allocator,
            .debugger = debugger,
            .source_path = source_path,
            .calldata = calldata,
        };
        try self.primeInitialStop();
        return self;
    }

    pub fn render(self: *DebuggerView, ctx: *tui.RenderContext) void {
        const screen = ctx.screen;
        screen.clear();

        const width = ctx.bounds.width;
        const height = ctx.bounds.height;
        if (width < 40 or height < 10) {
            screen.setStyle(Style.default.setFg(Color.light_red).bold());
            putStringClipped(screen, 1, 1, width - 2, "Terminal too small for debugger view");
            return;
        }

        const left_width: u16 = if (width >= 120) width - 38 else if (width >= 90) width - 30 else width;
        const right_width: u16 = if (width > left_width + 2) width - left_width - 2 else 0;

        self.drawHeader(screen, width);
        self.drawSourcePane(screen, left_width, height);
        if (right_width > 0) self.drawBindingsPane(screen, left_width + 1, right_width, height);
        self.drawFooter(screen, width, height);
    }

    pub fn handleEvent(self: *DebuggerView, event: tui.Event) tui.EventResult {
        switch (event) {
            .key => |key_event| {
                if (key_event.modifiers.ctrl and key_event.key == .char) {
                    const c = key_event.key.char;
                    if (c == 'c' or c == 'q' or c == 'C' or c == 'Q') return .ignored;
                }

                switch (key_event.key) {
                    .char => |c| switch (c) {
                        's' => return self.runStep(.in),
                        'n' => return self.runStep(.over),
                        'o' => return self.runStep(.out),
                        'c' => return self.runStep(.continue_),
                        'q' => {
                            if (self.app) |app| app.quit();
                            return .consumed;
                        },
                        'j' => {
                            self.scrollDown();
                            return .needs_redraw;
                        },
                        'k' => {
                            self.scrollUp();
                            return .needs_redraw;
                        },
                        else => return .ignored,
                    },
                    .down => {
                        self.scrollDown();
                        return .needs_redraw;
                    },
                    .up => {
                        self.scrollUp();
                        return .needs_redraw;
                    },
                    .page_down => {
                        self.scroll_page(8);
                        return .needs_redraw;
                    },
                    .page_up => {
                        self.scroll_page(-8);
                        return .needs_redraw;
                    },
                    else => return .ignored,
                }
            },
            else => return .ignored,
        }
    }

    fn runStep(self: *DebuggerView, comptime mode: enum { in, over, out, continue_ }) tui.EventResult {
        if (self.debugger.isHalted()) {
            self.status = "halted";
            self.syncFocusFromDebugger();
            return .needs_redraw;
        }
        switch (mode) {
            .in => self.debugger.stepIn() catch {
                self.status = "execution_error";
                return .needs_redraw;
            },
            .over => self.debugger.stepOver() catch {
                self.status = "execution_error";
                return .needs_redraw;
            },
            .out => self.debugger.stepOut() catch {
                self.status = "execution_error";
                return .needs_redraw;
            },
            .continue_ => self.debugger.continue_() catch {
                self.status = "execution_error";
                return .needs_redraw;
            },
        }
        self.status = @tagName(self.debugger.stop_reason);
        if (!self.shouldPreserveFocusOnTerminalStop()) {
            self.syncFocusFromDebugger();
        }
        self.centerOnCurrentLine();
        return .needs_redraw;
    }

    fn primeInitialStop(self: *DebuggerView) !void {
        if (self.debugger.isHalted()) {
            self.status = "halted";
            return;
        }

        if (self.debugger.currentEntry()) |entry| {
            if (entry.is_statement) {
                self.status = "ready";
                self.syncFocusFromDebugger();
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

        if (self.debugger.isHalted()) self.status = @tagName(self.debugger.stop_reason);
        self.syncFocusFromDebugger();
        self.centerOnCurrentLine();
    }

    fn syncFocusFromDebugger(self: *DebuggerView) void {
        if (self.debugger.currentEntry()) |entry| {
            if (entry.is_statement) {
                self.focus_line = entry.line;
                return;
            }
        }
        if (self.focus_line == null) {
            self.focus_line = self.debugger.currentSourceLine();
        }
    }

    fn shouldPreserveFocusOnTerminalStop(self: *const DebuggerView) bool {
        if (!self.debugger.isHalted()) return false;
        return switch (self.debugger.stop_reason) {
            .execution_finished, .execution_reverted, .execution_error => true,
            else => false,
        };
    }

    fn centerOnCurrentLine(self: *DebuggerView) void {
        const line = self.focus_line orelse return;
        if (line > 8) self.scroll_line = line - 8 else self.scroll_line = 1;
    }

    fn scrollDown(self: *DebuggerView) void {
        const total = self.debugger.totalSourceLines();
        if (self.scroll_line < total) self.scroll_line += 1;
    }

    fn scrollUp(self: *DebuggerView) void {
        if (self.scroll_line > 1) self.scroll_line -= 1;
    }

    fn scroll_page(self: *DebuggerView, delta: i32) void {
        if (delta < 0) {
            const amount: u32 = @intCast(-delta);
            if (self.scroll_line > amount) self.scroll_line -= amount else self.scroll_line = 1;
        } else {
            self.scroll_line += @intCast(delta);
            const total = self.debugger.totalSourceLines();
            if (self.scroll_line > total) self.scroll_line = total;
        }
    }

    fn drawHeader(self: *DebuggerView, screen: anytype, width: u16) void {
        const source_name = std.fs.path.basename(self.source_path);
        const line = self.focus_line orelse self.debugger.currentSourceLine() orelse 0;
        const entry = self.debugger.currentEntry();

        screen.setStyle(Style.default.setBg(Color.fromRGB(232, 239, 246)).setFg(Color.fromRGB(25, 31, 36)).bold());
        screen.fill(0, 0, width, 1, ' ');

        var title: [256]u8 = undefined;
        const title_text = std.fmt.bufPrint(&title, " Ora EVM Debugger | {s}", .{source_name}) catch "Ora EVM Debugger";
        putStringClipped(screen, 0, 0, width, title_text);

        screen.setStyle(Style.default.setBg(Color.fromRGB(26, 29, 33)).setFg(Color.fromRGB(211, 219, 227)));
        screen.fill(0, 1, width, 2, ' ');

        var status: [512]u8 = undefined;
        const status_text = std.fmt.bufPrint(&status, " status={s}  line={d}  pc={d}  idx={?d}  opcode={s}  depth={d}  gas={d}", .{
            self.status,
            line,
            self.debugger.getPC(),
            if (entry) |e| e.idx else null,
            self.debugger.getCurrentOpcodeName(),
            self.debugger.getCallDepth(),
            self.debugger.getGasRemaining(),
        }) catch "status";
        putStringClipped(screen, 0, 1, width, status_text);

        var current: [512]u8 = undefined;
        const current_source = if (line != 0) blk: {
            if (self.debugger.getSourceLineText(line)) |line_text| {
                break :blk std.mem.trim(u8, std.mem.trimRight(u8, line_text, "\r"), " \t");
            }
            break :blk "";
        } else "";
        const current_text = std.fmt.bufPrint(&current, " current={s}", .{current_source}) catch "current=";
        putStringClipped(screen, 0, 2, width, current_text);
    }

    fn drawSourcePane(self: *DebuggerView, screen: anytype, width: u16, height: u16) void {
        if (width < 8 or height < 8) return;

        const pane_top: u16 = 3;
        const pane_height = height - 5;
        screen.setStyle(Style.default.setFg(Color.fromRGB(120, 131, 142)));
        screen.drawBox(0, pane_top, width, pane_height, BorderStyle.single);

        screen.setStyle(Style.default.setFg(Color.fromRGB(222, 228, 235)).bold());
        putStringClipped(screen, 2, pane_top, width - 4, " Source ");

        const content_top = pane_top + 1;
        const content_height = pane_height - 2;
        const current_line = self.focus_line orelse self.debugger.currentSourceLine() orelse 0;
        const start_line = self.scroll_line;
        const end_line = @min(self.debugger.totalSourceLines(), start_line + content_height - 1);

        var y: u16 = 0;
        var row = start_line;
        while (y < content_height) : (y += 1) {
            screen.setStyle(Style.default);
            screen.fill(1, content_top + y, width - 2, 1, ' ');
            if (row <= end_line) {
                const line_text = self.debugger.getSourceLineText(row) orelse "";
                const trimmed = std.mem.trimRight(u8, line_text, "\r");
                if (row == current_line) {
                    screen.setStyle(Style.default.setBg(Color.fromRGB(48, 58, 69)).setFg(Color.fromRGB(245, 247, 250)));
                    screen.fill(1, content_top + y, width - 2, 1, ' ');
                } else {
                    screen.setStyle(Style.default.setFg(Color.fromRGB(208, 214, 220)));
                }

                var label_buf: [16]u8 = undefined;
                const label = std.fmt.bufPrint(&label_buf, "{d:>4} ", .{row}) catch "   ?";
                putStringClipped(screen, 2, content_top + y, 5, label);
                screen.setStyle(if (row == current_line)
                    Style.default.setBg(Color.fromRGB(48, 58, 69)).setFg(Color.fromRGB(245, 247, 250))
                else
                    Style.default.setFg(Color.fromRGB(208, 214, 220)));
                putStringClipped(screen, 7, content_top + y, width - 9, trimmed);
                row += 1;
            }
        }
    }

    fn drawBindingsPane(self: *DebuggerView, screen: anytype, start_x: u16, width: u16, height: u16) void {
        if (width < 12 or height < 8) return;

        const pane_top: u16 = 3;
        const pane_height = height - 5;
        screen.setStyle(Style.default.setFg(Color.fromRGB(120, 131, 142)));
        screen.drawBox(start_x, pane_top, width, pane_height, BorderStyle.single);

        const bindings = self.debugger.getVisibleBindings(self.allocator) catch &.{};
        defer if (bindings.len > 0) self.allocator.free(bindings);

        var title_buf: [64]u8 = undefined;
        const title = std.fmt.bufPrint(&title_buf, " Bindings [{d}] ", .{bindings.len}) catch " Bindings ";
        screen.setStyle(Style.default.setFg(Color.fromRGB(222, 228, 235)).bold());
        putStringClipped(screen, start_x + 2, pane_top, width - 4, title);

        const content_top = pane_top + 1;
        const content_height = pane_height - 2;
        screen.setStyle(Style.default.setFg(Color.fromRGB(192, 199, 207)).dim());
        putStringClipped(screen, start_x + 2, content_top, width - 4, "name = value / runtime");

        if (bindings.len == 0) {
            screen.setStyle(Style.default.setFg(Color.fromRGB(150, 158, 166)).italic());
            putStringClipped(screen, start_x + 2, content_top + 2, width - 4, "no visible bindings");
            return;
        }

        var i: usize = 0;
        while (i < bindings.len and i + 1 < content_height) : (i += 1) {
            const binding = bindings[i];
            const y = content_top + 1 + @as(u16, @intCast(i));
            var row_buf: [256]u8 = undefined;
            const row_text = if (binding.folded_value) |folded|
                std.fmt.bufPrint(&row_buf, "{s} = {s}", .{ binding.name, folded }) catch binding.name
            else if ((self.debugger.getVisibleBindingValueByName(self.allocator, binding.name) catch null)) |value|
                std.fmt.bufPrint(&row_buf, "{s} = {d}", .{ binding.name, value }) catch binding.name
            else
                std.fmt.bufPrint(&row_buf, "{s} [{s}]", .{ binding.name, binding.runtime_kind }) catch binding.name;

            screen.setStyle(Style.default.setFg(Color.fromRGB(214, 220, 226)));
            putStringClipped(screen, start_x + 2, y, width - 4, row_text);
        }
    }

    fn drawFooter(self: *DebuggerView, screen: anytype, width: u16, height: u16) void {
        _ = self;
        screen.setStyle(Style.default.setBg(Color.fromRGB(232, 239, 246)).setFg(Color.fromRGB(25, 31, 36)).bold());
        screen.fill(0, height - 2, width, 1, ' ');
        putStringClipped(screen, 0, height - 2, width, " s step-in  n step-over  o step-out  c continue  arrows/jk scroll  q quit ");

        screen.setStyle(Style.default.setBg(Color.fromRGB(26, 29, 33)).setFg(Color.fromRGB(168, 176, 184)));
        screen.fill(0, height - 1, width, 1, ' ');
        putStringClipped(screen, 0, height - 1, width, " debugger values show folded constants or rooted runtime values when readable ");
    }
};

fn putStringClipped(screen: anytype, x: u16, y: u16, max_width: u16, text: []const u8) void {
    if (max_width == 0) return;
    const clipped = clipTextBytes(text, max_width);
    screen.putStringAt(x, y, clipped);
}

fn clipTextBytes(text: []const u8, max_width: u16) []const u8 {
    const limit: usize = @intCast(max_width);
    if (text.len <= limit) return text;
    if (limit == 0) return "";
    return text[0..limit];
}

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
    defer allocator.free(source_text);

    var debug_info_json: ?[]u8 = null;
    defer if (debug_info_json) |bytes| allocator.free(bytes);
    var debug_info: ?DebugInfo = null;
    if (config.debug_info_path) |path| {
        debug_info_json = try std.fs.cwd().readFileAlloc(allocator, path, 16 * 1024 * 1024);
        debug_info = try DebugInfo.loadFromJson(allocator, debug_info_json.?);
    }

    var evm: Evm = undefined;
    try evm.init(allocator, null, null, null, primitives.ZERO_ADDRESS, 0, null);
    defer evm.deinit();

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

    var debugger = blk: {
        if (debug_info) |info| {
            debug_info = null;
            break :blk try Debugger.initWithDebugInfo(allocator, &evm, source_map, info, source_text);
        }
        break :blk try Debugger.init(allocator, &evm, source_map, source_text);
    };
    defer {
        debugger.deinit();
        source_map.deinit();
    }

    var root = try DebuggerView.init(allocator, &debugger, config.source_path, config.calldata);

    var app = try tui.App.init(.{
        .alternate_screen = true,
        .hide_cursor = true,
        .enable_mouse = false,
        .enable_paste = false,
        .enable_focus = false,
        .target_fps = 30,
        .poll_timeout_ms = 16,
    });
    defer app.deinit();
    root.app = &app;
    try app.setRoot(&root);
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
