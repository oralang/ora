const std = @import("std");
const vaxis = @import("vaxis");
const primitives = @import("voltaire");
const ora_evm = @import("ora_evm");
const opcode_utils = ora_evm.opcode;

const Evm = ora_evm.Evm(.{});
const Frame = ora_evm.Frame(.{});
const Debugger = ora_evm.Debugger(.{});
const SourceMap = ora_evm.SourceMap;
const DebugInfo = ora_evm.DebugInfo;

const Style = vaxis.Style;
const Color = vaxis.Color;
const Segment = vaxis.Segment;
const Window = vaxis.Window;
const ascii_border_glyphs: [6][]const u8 = .{ "+", "-", "+", "|", "+", "+" };

const AppEvent = union(enum) {
    key_press: vaxis.Key,
    winsize: vaxis.Winsize,
    focus_in,
    focus_out,
    paste: []const u8,
};

const AppConfig = struct {
    bytecode_path: []u8,
    source_map_path: []u8,
    source_path: []u8,
    sir_path: ?[]u8 = null,
    debug_info_path: ?[]u8 = null,
    abi_path: ?[]u8 = null,
    init_calldata: []u8 = &.{},
    init_calldata_fallback: []u8 = &.{},
    calldata: []u8 = &.{},

    fn deinit(self: *AppConfig, allocator: std.mem.Allocator) void {
        allocator.free(self.bytecode_path);
        allocator.free(self.source_map_path);
        allocator.free(self.source_path);
        if (self.sir_path) |path| allocator.free(path);
        if (self.debug_info_path) |path| allocator.free(path);
        if (self.abi_path) |path| allocator.free(path);
        allocator.free(self.init_calldata);
        allocator.free(self.init_calldata_fallback);
        allocator.free(self.calldata);
    }
};

const Snapshot = struct {
    gas_remaining: i64 = 0,
    stack_len: usize = 0,
    stack_top_count: usize = 0,
    stack_top: [8]u256 = [_]u256{0} ** 8,
    memory_size: u32 = 0,
    memory_word_count: usize = 0,
    memory_words: [8]u256 = [_]u256{0} ** 8,
    storage_count: usize = 0,
    storage_top_count: usize = 0,
    storage_top_slots: [8]u256 = [_]u256{0} ** 8,
    storage_top_values: [8]u256 = [_]u256{0} ** 8,
    tstore_count: usize = 0,
    tstore_top_count: usize = 0,
    tstore_top_slots: [8]u256 = [_]u256{0} ** 8,
    tstore_top_values: [8]u256 = [_]u256{0} ** 8,
};

const BindingSnapshotEntry = struct {
    name: []u8,
    value: ?u256,
};

const EvmTabKind = enum { stack, memory, storage, tstore, calldata };
const StepMode = enum { in, opcode, over, out, continue_ };

const SavedSession = struct {
    const SavedCheckpoint = struct {
        id: u32,
        step_index: usize,
        scroll_line: u32,
        focus_line: ?u32 = null,
        active_evm_tab: []const u8 = "stack",
    };

    version: u32 = 1,
    bytecode_path: []const u8,
    source_map_path: []const u8,
    source_path: []const u8,
    sir_path: ?[]const u8 = null,
    debug_info_path: ?[]const u8 = null,
    abi_path: ?[]const u8 = null,
    calldata_hex: []const u8,
    scroll_line: u32 = 1,
    sir_scroll_line: u32 = 1,
    sir_follow: bool = true,
    focus_line: ?u32 = null,
    active_evm_tab: []const u8 = "stack",
    step_history: []const []const u8 = &.{},
    breakpoints: []const u32 = &.{},
    checkpoints: []const SavedCheckpoint = &.{},
};

const AbiDoc = struct {
    allocator: std.mem.Allocator,
    json_bytes: []u8,
    parsed: std.json.Parsed(std.json.Value),

    fn loadFromPath(allocator: std.mem.Allocator, path: []const u8) !AbiDoc {
        const json_bytes = try std.fs.cwd().readFileAlloc(allocator, path, 16 * 1024 * 1024);
        errdefer allocator.free(json_bytes);

        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_bytes, .{});
        errdefer parsed.deinit();

        return .{
            .allocator = allocator,
            .json_bytes = json_bytes,
            .parsed = parsed,
        };
    }

    fn deinit(self: *AbiDoc) void {
        self.parsed.deinit();
        self.allocator.free(self.json_bytes);
    }

    fn findCallableBySelector(self: *const AbiDoc, selector: [4]u8) ?std.json.Value {
        const callables = self.parsed.value.object.get("callables") orelse return null;
        if (callables != .array) return null;

        const selector_int = std.mem.readInt(u32, &selector, .big);
        var selector_buf: [10]u8 = undefined;
        const selector_text = std.fmt.bufPrint(&selector_buf, "0x{x:0>8}", .{selector_int}) catch return null;

        for (callables.array.items) |callable| {
            if (callable != .object) continue;
            const kind = callable.object.get("kind") orelse continue;
            if (kind != .string or !std.mem.eql(u8, kind.string, "function")) continue;

            const wire = callable.object.get("wire") orelse continue;
            if (wire != .object) continue;
            const evm_default = wire.object.get("evm-default") orelse continue;
            if (evm_default != .object) continue;
            const selector_value = evm_default.object.get("selector") orelse continue;
            if (selector_value != .string) continue;
            if (std.mem.eql(u8, selector_value.string, selector_text)) return callable;
        }

        return null;
    }

    fn wireTypeForTypeId(self: *const AbiDoc, type_id: []const u8) ?[]const u8 {
        const types = self.parsed.value.object.get("types") orelse return null;
        if (types != .object) return null;
        const type_value = types.object.get(type_id) orelse return null;
        if (type_value != .object) return null;

        if (type_value.object.get("wire")) |wire| {
            if (wire == .object) {
                if (wire.object.get("evm-default")) |evm_default| {
                    if (evm_default == .object) {
                        if (evm_default.object.get("type")) |wire_type| {
                            if (wire_type == .string) return wire_type.string;
                        }
                    }
                }
            }
        }

        if (type_value.object.get("name")) |name| {
            if (name == .string) return name.string;
        }

        return null;
    }

    fn findInputWireType(self: *const AbiDoc, callable: std.json.Value, input_name: []const u8) ?[]const u8 {
        if (callable != .object) return null;
        const inputs = callable.object.get("inputs") orelse return null;
        if (inputs != .array) return null;

        for (inputs.array.items) |input| {
            if (input != .object) continue;
            const name = input.object.get("name") orelse continue;
            if (name != .string or !std.mem.eql(u8, name.string, input_name)) continue;

            const type_id = input.object.get("typeId") orelse return null;
            if (type_id != .string) return null;
            return self.wireTypeForTypeId(type_id.string);
        }

        return null;
    }

    fn findInputIndex(self: *const AbiDoc, callable: std.json.Value, input_name: []const u8) ?usize {
        _ = self;
        if (callable != .object) return null;
        const inputs = callable.object.get("inputs") orelse return null;
        if (inputs != .array) return null;

        for (inputs.array.items, 0..) |input, i| {
            if (input != .object) continue;
            const name = input.object.get("name") orelse continue;
            if (name == .string and std.mem.eql(u8, name.string, input_name)) return i;
        }

        return null;
    }
};

const SessionSeed = struct {
    runtime_bytecode: []u8,
    source_map: SourceMap,
    debug_info_json: ?[]u8 = null,
    source_text: []u8,
    source_path: []u8,
    sir_text: ?[]u8 = null,
    calldata: []u8,
    caller: primitives.Address,
    contract: primitives.Address,

    fn deinit(self: *SessionSeed, allocator: std.mem.Allocator) void {
        allocator.free(self.runtime_bytecode);
        self.source_map.deinit();
        if (self.debug_info_json) |bytes| allocator.free(bytes);
        allocator.free(self.source_text);
        allocator.free(self.source_path);
        if (self.sir_text) |text| allocator.free(text);
        allocator.free(self.calldata);
    }
};

const Session = struct {
    evm: Evm,
    debugger: Debugger,

    fn init(self: *Session, allocator: std.mem.Allocator, seed: *const SessionSeed) !void {
        try self.evm.init(allocator, null, null, null, primitives.ZERO_ADDRESS, 0, null);
        errdefer self.evm.deinit();
        try self.evm.initTransactionState(null);
        try self.evm.preWarmTransaction(seed.contract);

        try self.evm.frames.append(self.evm.arena.allocator(), try Frame.init(
            self.evm.arena.allocator(),
            seed.runtime_bytecode,
            5_000_000,
            seed.caller,
            seed.contract,
            0,
            seed.calldata,
            @as(*anyopaque, @ptrCast(&self.evm)),
            self.evm.hardfork,
            false,
        ));

        var source_map = try SourceMap.fromEntries(allocator, seed.source_map.entries);
        source_map.runtime_start_pc = seed.source_map.runtime_start_pc;
        errdefer source_map.deinit();
        self.debugger = if (seed.debug_info_json) |json|
            try Debugger.initWithDebugInfo(allocator, &self.evm, source_map, try DebugInfo.loadFromJson(allocator, json), seed.source_text)
        else
            try Debugger.init(allocator, &self.evm, source_map, seed.source_text);
    }

    fn deinit(self: *Session) void {
        self.debugger.deinit();
        self.evm.deinit();
    }
};

const Checkpoint = struct {
    id: u32,
    step_index: usize,
    scroll_line: u32,
    focus_line: ?u32 = null,
    active_evm_tab: EvmTabKind = .stack,
};

const MappingWindow = struct {
    ora_line: u32,
    sir_start: u32,
    sir_end: u32,
    idx_start: ?u32 = null,
    idx_end: ?u32 = null,
    pc_start: u32,
    pc_end: u32,
};

const WriteEffectKind = enum {
    none,
    return_buffer,
    memory,
    storage,
    tstore,
};

const Ui = struct {
    allocator: std.mem.Allocator,
    config: AppConfig,
    abi_doc: ?AbiDoc = null,
    seed: SessionSeed,
    session: Session,
    status: []const u8 = "ready",
    scroll_line: u32 = 1,
    sir_scroll_line: u32 = 1,
    sir_follow: bool = true,
    focus_line: ?u32 = null,
    active_evm_tab: EvmTabKind = .stack,
    previous_snapshot: Snapshot = .{},
    source_buffer: vaxis.widgets.TextView.Buffer = .{},
    source_view: vaxis.widgets.CodeView = .{},
    sir_buffer: vaxis.widgets.TextView.Buffer = .{},
    sir_view: vaxis.widgets.CodeView = .{},
    command_mode: bool = false,
    command_buffer: std.ArrayList(u8) = .{},
    command_status: []const u8 = "ready",
    command_status_storage: std.ArrayList(u8) = .{},
    command_log: std.ArrayList([]u8) = .{},
    render_scratch: std.ArrayList(u8) = .{},
    step_history: std.ArrayList(StepMode) = .{},
    previous_bindings: std.ArrayList(BindingSnapshotEntry) = .{},
    breakpoints: std.ArrayList(u32) = .{},
    checkpoints: std.ArrayList(Checkpoint) = .{},
    next_checkpoint_id: u32 = 1,
    selected_frame_index: usize = 0,

    fn init(
        allocator: std.mem.Allocator,
        config: AppConfig,
        seed: SessionSeed,
    ) !Ui {
        var self = Ui{
            .allocator = allocator,
            .config = config,
            .abi_doc = if (config.abi_path) |path| try AbiDoc.loadFromPath(allocator, path) else null,
            .seed = seed,
            .session = undefined,
        };
        try Session.init(&self.session, allocator, &self.seed);
        try self.source_buffer.update(allocator, .{ .bytes = self.seed.source_text });
        self.source_view.highlighted_style = .{
            .bg = Color.rgbFromUint(0x303A45),
            .fg = Color.rgbFromUint(0xF5F7FA),
        };
        self.source_view.indentation_cell = .{
            .char = .{
                .grapheme = "|",
                .width = 1,
            },
            .style = .{ .dim = true },
        };
        self.sir_view.highlighted_style = .{
            .bg = Color.rgbFromUint(0x2C3540),
            .fg = Color.rgbFromUint(0xF5F7FA),
        };
        if (self.seed.sir_text) |sir_text| {
            try self.sir_buffer.update(allocator, .{ .bytes = sir_text });
        }
        try self.primeInitialStop();
        self.previous_snapshot = self.captureSnapshot();
        try self.refreshPreviousBindingsSnapshot();
        return self;
    }

    fn deinit(self: *Ui) void {
        self.config.deinit(self.allocator);
        if (self.abi_doc) |*abi_doc| abi_doc.deinit();
        self.session.deinit();
        self.seed.deinit(self.allocator);
        self.source_buffer.deinit(self.allocator);
        self.sir_buffer.deinit(self.allocator);
        self.command_buffer.deinit(self.allocator);
        self.command_status_storage.deinit(self.allocator);
        self.clearCommandLog();
        self.command_log.deinit(self.allocator);
        self.render_scratch.deinit(self.allocator);
        self.step_history.deinit(self.allocator);
        self.clearPreviousBindingsSnapshot();
        self.previous_bindings.deinit(self.allocator);
        self.breakpoints.deinit(self.allocator);
        self.checkpoints.deinit(self.allocator);
    }

    fn handleKey(self: *Ui, key: vaxis.Key) !bool {
        if (key.matches('q', .{}) or key.matches('c', .{ .ctrl = true })) return true;

        if (self.command_mode) {
            if (key.matches(vaxis.Key.escape, .{})) {
                self.command_mode = false;
                self.command_status = "command cancelled";
                return false;
            }
            if (key.matches(vaxis.Key.enter, .{})) {
                self.command_mode = false;
                const executed = try self.allocator.dupe(u8, self.command_buffer.items);
                defer self.command_buffer.clearRetainingCapacity();
                const should_quit = try self.executeCommand();
                self.appendCommandLog(executed, self.command_status) catch {};
                self.allocator.free(executed);
                return should_quit;
            }
            if (key.matches(vaxis.Key.backspace, .{})) {
                if (self.command_buffer.items.len > 0) _ = self.command_buffer.pop();
                return false;
            }
            if (key.text) |text| {
                if (text.len > 0 and text[0] >= 0x20 and text[0] != 0x7f) {
                    try self.command_buffer.appendSlice(self.allocator, text);
                }
            }
            return false;
        }

        if (key.matches('s', .{})) {
            self.runStep(.in, true);
            return false;
        }
        if (key.matches('x', .{})) {
            self.runStep(.opcode, true);
            return false;
        }
        if (key.matches('n', .{})) {
            self.runStep(.over, true);
            return false;
        }
        if (key.matches('o', .{})) {
            self.runStep(.out, true);
            return false;
        }
        if (key.matches('c', .{})) {
            self.runStep(.continue_, true);
            return false;
        }
        if (key.matches('p', .{})) {
            self.stepBack();
            return false;
        }
        if (key.matches(':', .{})) {
            self.command_mode = true;
            self.command_status = "command";
            self.command_buffer.clearRetainingCapacity();
            return false;
        }
        if (key.matches('j', .{}) or key.matches(vaxis.Key.down, .{})) {
            self.scrollDown();
            return false;
        }
        if (key.matches('k', .{}) or key.matches(vaxis.Key.up, .{})) {
            self.scrollUp();
            return false;
        }
        if (key.matches('J', .{})) {
            self.scrollSirDown();
            return false;
        }
        if (key.matches('K', .{})) {
            self.scrollSirUp();
            return false;
        }
        if (key.matches('=', .{})) {
            self.resyncSirView();
            self.command_status = "sir follow";
            return false;
        }
        if (key.matches(vaxis.Key.page_down, .{})) {
            self.scrollPage(8);
            return false;
        }
        if (key.matches(vaxis.Key.page_up, .{})) {
            self.scrollPage(-8);
            return false;
        }
        if (key.matches('[', .{})) {
            self.prevEvmTab();
            return false;
        }
        if (key.matches(']', .{})) {
            self.nextEvmTab();
            return false;
        }
        if (key.matches('1', .{})) self.active_evm_tab = .stack;
        if (key.matches('2', .{})) self.active_evm_tab = .memory;
        if (key.matches('3', .{})) self.active_evm_tab = .storage;
        if (key.matches('4', .{})) self.active_evm_tab = .tstore;
        if (key.matches('5', .{})) self.active_evm_tab = .calldata;
        return false;
    }

    fn executeCommand(self: *Ui) !bool {
        const raw = std.mem.trim(u8, self.command_buffer.items, " \t");
        if (raw.len == 0) {
            self.command_status = "empty command";
            return false;
        }

        if (std.mem.eql(u8, raw, "q") or std.mem.eql(u8, raw, "quit")) {
            self.command_status = "quit";
            return true;
        }
        if (std.mem.eql(u8, raw, "c") or std.mem.eql(u8, raw, "continue")) {
            self.runStep(.continue_, true);
            self.command_status = "continue";
            return false;
        }
        if (std.mem.eql(u8, raw, "r") or std.mem.eql(u8, raw, "run") or std.mem.eql(u8, raw, "rerun")) {
            try self.rerunToHistory(0);
            self.command_status = "run";
            return false;
        }
        if (std.mem.eql(u8, raw, "s") or std.mem.eql(u8, raw, "step") or std.mem.eql(u8, raw, "si")) {
            self.runStep(.in, true);
            self.command_status = "step";
            return false;
        }
        if (std.mem.eql(u8, raw, "x") or std.mem.eql(u8, raw, "op") or std.mem.eql(u8, raw, "opcode")) {
            self.runStep(.opcode, true);
            self.command_status = "opcode";
            return false;
        }
        if (std.mem.eql(u8, raw, "n") or std.mem.eql(u8, raw, "next") or std.mem.eql(u8, raw, "so")) {
            self.runStep(.over, true);
            self.command_status = "next";
            return false;
        }
        if (std.mem.eql(u8, raw, "o") or std.mem.eql(u8, raw, "out") or std.mem.eql(u8, raw, "finish")) {
            self.runStep(.out, true);
            self.command_status = "out";
            return false;
        }
        if (std.mem.eql(u8, raw, "p") or std.mem.eql(u8, raw, "prev") or std.mem.eql(u8, raw, "previous")) {
            self.stepBack();
            return false;
        }
        if (std.mem.startsWith(u8, raw, "line ")) {
            const rest = std.mem.trim(u8, raw["line ".len..], " \t");
            const line = std.fmt.parseUnsigned(u32, rest, 10) catch {
                self.command_status = "invalid line";
                return false;
            };
            self.focus_line = line;
            self.centerOnCurrentLine();
            self.command_status = "line";
            return false;
        }
        if (std.mem.startsWith(u8, raw, "sirline ")) {
            const rest = std.mem.trim(u8, raw["sirline ".len..], " \t");
            const line = std.fmt.parseUnsigned(u32, rest, 10) catch {
                self.command_status = "invalid sir line";
                return false;
            };
            self.sir_scroll_line = if (line > 0) line else 1;
            self.sir_follow = false;
            self.command_status = "sirline";
            return false;
        }
        if (std.mem.eql(u8, raw, "sirfollow") or std.mem.eql(u8, raw, "syncsir")) {
            self.resyncSirView();
            self.command_status = "sir follow";
            return false;
        }
        if (std.mem.eql(u8, raw, "checkpoint")) {
            try self.addCheckpoint();
            return false;
        }
        if (std.mem.eql(u8, raw, "checkpoints")) {
            try self.describeCheckpoints();
            return false;
        }
        if (std.mem.eql(u8, raw, "bt") or std.mem.eql(u8, raw, "backtrace")) {
            try self.describeBacktrace();
            return false;
        }
        if (std.mem.startsWith(u8, raw, "frame ")) {
            const rest = std.mem.trim(u8, raw["frame ".len..], " \t");
            const idx = std.fmt.parseUnsigned(usize, rest, 10) catch {
                self.command_status = "invalid frame index";
                return false;
            };
            try self.selectFrame(idx);
            return false;
        }
        if (std.mem.startsWith(u8, raw, "restart ")) {
            const rest = std.mem.trim(u8, raw["restart ".len..], " \t");
            const id = std.fmt.parseUnsigned(u32, rest, 10) catch {
                self.command_status = "invalid checkpoint id";
                return false;
            };
            try self.restartCheckpoint(id);
            return false;
        }
        if (std.mem.startsWith(u8, raw, "break ")) {
            const rest = std.mem.trim(u8, raw["break ".len..], " \t");
            try self.handleBreakpointSet(rest);
            return false;
        }
        if (std.mem.startsWith(u8, raw, "delete ")) {
            const rest = std.mem.trim(u8, raw["delete ".len..], " \t");
            try self.handleBreakpointDelete(rest);
            return false;
        }
        if (std.mem.eql(u8, raw, "info break")) {
            try self.describeBreakpoints();
            return false;
        }
        if (std.mem.eql(u8, raw, "gas")) {
            const gas_remaining = self.session.debugger.getGasRemaining();
            const gas_spent_step = if (self.previous_snapshot.gas_remaining >= gas_remaining)
                self.previous_snapshot.gas_remaining - gas_remaining
            else
                0;
            const gas_spent_total = 5_000_000 - gas_remaining;
            try self.setCommandStatusFmt("gas={d} step_spent={d} total_spent={d}", .{
                gas_remaining,
                gas_spent_step,
                gas_spent_total,
            });
            return false;
        }
        if (std.mem.startsWith(u8, raw, "gas ")) {
            const rest = std.mem.trim(u8, raw["gas ".len..], " \t");
            const gas = std.fmt.parseInt(i64, rest, 0) catch {
                self.command_status = "invalid gas value";
                return false;
            };
            const frame = self.session.evm.getCurrentFrame() orelse {
                self.command_status = "no active frame";
                return false;
            };
            if (gas < 0) {
                self.command_status = "gas must be >= 0";
                return false;
            }
            frame.gas_remaining = gas;
            self.previous_snapshot = self.captureSnapshot();
            try self.refreshPreviousBindingsSnapshot();
            try self.setCommandStatusFmt("gas set to {d}", .{gas});
            return false;
        }
        if (std.mem.startsWith(u8, raw, "print ")) {
            const target = std.mem.trim(u8, raw["print ".len..], " \t");
            if (target.len == 0) {
                self.command_status = "missing print target";
                return false;
            }
            try self.handlePrintCommand(target);
            return false;
        }
        if (std.mem.startsWith(u8, raw, "set ")) {
            const rest = std.mem.trim(u8, raw["set ".len..], " \t");
            try self.handleSetCommand(rest);
            return false;
        }
        if (std.mem.startsWith(u8, raw, "write-session ")) {
            const path = std.mem.trim(u8, raw["write-session ".len..], " \t");
            if (path.len == 0) {
                self.command_status = "missing session path";
                return false;
            }
            self.writeSession(path) catch {
                self.command_status = "failed to write session";
                return false;
            };
            self.command_status = "session saved";
            return false;
        }
        if (std.mem.startsWith(u8, raw, "ws ")) {
            const path = std.mem.trim(u8, raw["ws ".len..], " \t");
            if (path.len == 0) {
                self.command_status = "missing session path";
                return false;
            }
            self.writeSession(path) catch {
                self.command_status = "failed to write session";
                return false;
            };
            self.command_status = "session saved";
            return false;
        }
        if (std.mem.startsWith(u8, raw, "load-session ")) {
            const path = std.mem.trim(u8, raw["load-session ".len..], " \t");
            if (path.len == 0) {
                self.command_status = "missing session path";
                return false;
            }
            self.loadSession(path) catch {
                self.command_status = "failed to load session";
                return false;
            };
            self.command_status = "session loaded";
            return false;
        }
        if (std.mem.startsWith(u8, raw, "ls ")) {
            const path = std.mem.trim(u8, raw["ls ".len..], " \t");
            if (path.len == 0) {
                self.command_status = "missing session path";
                return false;
            }
            self.loadSession(path) catch {
                self.command_status = "failed to load session";
                return false;
            };
            self.command_status = "session loaded";
            return false;
        }

        self.command_status = "unknown command";
        return false;
    }

    fn handlePrintCommand(self: *Ui, target: []const u8) !void {
        if (std.mem.eql(u8, target, "gas")) {
            const gas_remaining = self.session.debugger.getGasRemaining();
            try self.setCommandStatusFmt("gas={d}", .{gas_remaining});
            return;
        }
        if (std.mem.eql(u8, target, "calldata")) {
            const calldata = if (self.selectedFrame()) |frame| frame.calldata else self.seed.calldata;
            try self.setCommandStatusFmt("calldata 0x{x}", .{calldata});
            return;
        }
        if (std.mem.eql(u8, target, "storage")) {
            try self.handlePrintStorageCommand(false);
            return;
        }
        if (std.mem.eql(u8, target, "tstore")) {
            try self.handlePrintStorageCommand(true);
            return;
        }
        if (std.mem.startsWith(u8, target, "mem ")) {
            const rest = std.mem.trim(u8, target["mem ".len..], " \t");
            try self.handlePrintMemoryCommand(rest);
            return;
        }
        if (std.mem.startsWith(u8, target, "stack[")) {
            try self.handlePrintStackCommand(target);
            return;
        }
        if (std.mem.startsWith(u8, target, "slot ")) {
            const rest = std.mem.trim(u8, target["slot ".len..], " \t");
            try self.handlePrintSlotCommand(rest);
            return;
        }
        const binding = try self.session.debugger.findVisibleBindingByName(self.allocator, target) orelse {
            self.command_status = "binding not visible";
            return;
        };
        const binding_label = self.bindingLabel(&binding);
        if (binding.folded_value) |folded| {
            try self.setCommandStatusFmt("{s} [{s}] = {s} [folded]", .{ binding.name, binding_label, folded });
            return;
        }
        if (try self.resolvedBindingValue(&binding)) |value| {
            switch (value) {
                .numeric => |numeric| try self.setCommandStatusFmt("{s} [{s}] = {d}", .{ binding.name, binding_label, numeric }),
                .text => |text| try self.setCommandStatusFmt("{s} [{s}] = {s}", .{ binding.name, binding_label, text }),
            }
            return;
        }
        try self.setCommandStatusFmt("{s} [{s}]", .{ binding.name, binding_label });
    }

    fn handlePrintMemoryCommand(self: *Ui, rest: []const u8) !void {
        const frame = self.selectedFrame() orelse {
            self.command_status = "no active frame";
            return;
        };
        var split = std.mem.tokenizeScalar(u8, rest, ' ');
        const offset_text = split.next() orelse {
            self.command_status = "expected mem <offset> <words>";
            return;
        };
        const words_text = split.next() orelse {
            self.command_status = "expected mem <offset> <words>";
            return;
        };
        const offset = std.fmt.parseUnsigned(u32, offset_text, 0) catch {
            self.command_status = "invalid memory offset";
            return;
        };
        const words = std.fmt.parseUnsigned(u32, words_text, 10) catch {
            self.command_status = "invalid word count";
            return;
        };
        if (words == 0) {
            self.command_status = "word count must be > 0";
            return;
        }

        self.command_status_storage.clearRetainingCapacity();
        var writer = self.command_status_storage.writer(self.allocator);
        try writer.print("mem:", .{});
        var word_index: u32 = 0;
        while (word_index < words) : (word_index += 1) {
            const word_offset = std.math.add(u32, offset, word_index * 32) catch break;
            var value: u256 = 0;
            var j: u32 = 0;
            while (j < 32 and word_offset + j < frame.memory_size) : (j += 1) {
                value = (value << 8) | frame.readMemory(word_offset + j);
            }
            try writer.print(" 0x{X:0>4}={s}", .{ word_offset, self.scratchShortU256(value) });
            if (word_index + 1 < words) try writer.writeAll(" |");
        }
        self.command_status = self.command_status_storage.items;
    }

    fn handlePrintStackCommand(self: *Ui, target: []const u8) !void {
        const frame = self.selectedFrame() orelse {
            self.command_status = "no active frame";
            return;
        };
        if (!std.mem.endsWith(u8, target, "]")) {
            self.command_status = "expected stack[index]";
            return;
        }
        const idx_text = std.mem.trim(u8, target["stack[".len .. target.len - 1], " \t");
        const idx = std.fmt.parseUnsigned(usize, idx_text, 10) catch {
            self.command_status = "invalid stack index";
            return;
        };
        if (idx >= frame.stack.items.len) {
            self.command_status = "stack index out of range";
            return;
        }
        const value = frame.stack.items[frame.stack.items.len - 1 - idx];
        try self.setCommandStatusFmt("stack[{d}] = {d} ({s})", .{ idx, value, self.scratchFullU256(value) });
    }

    fn handlePrintSlotCommand(self: *Ui, slot_text: []const u8) !void {
        const frame = self.selectedFrame() orelse {
            self.command_status = "no active frame";
            return;
        };
        const slot = std.fmt.parseUnsigned(u256, slot_text, 0) catch {
            self.command_status = "invalid slot";
            return;
        };
        const storage_value = try self.session.evm.storage.get(frame.address, slot);
        const transient_value = self.session.evm.storage.get_transient(frame.address, slot);
        try self.setCommandStatusFmt("slot {s} storage={d} transient={d}", .{
            self.scratchFullU256(slot),
            storage_value,
            transient_value,
        });
    }

    fn handlePrintStorageCommand(self: *Ui, transient: bool) !void {
        const frame = self.selectedFrame() orelse {
            self.command_status = "no active frame";
            return;
        };
        self.command_status_storage.clearRetainingCapacity();
        var writer = self.command_status_storage.writer(self.allocator);
        try writer.print("{s}:", .{if (transient) "tstore" else "storage"});

        var count: usize = 0;
        if (transient) {
            var it = self.session.evm.storage.transient.iterator();
            while (it.next()) |entry| {
                if (!std.mem.eql(u8, entry.key_ptr.address[0..], frame.address.bytes[0..])) continue;
                try writer.print(" {s}={s}", .{
                    self.scratchShortU256(entry.key_ptr.slot),
                    self.scratchShortU256(entry.value_ptr.*),
                });
                count += 1;
                if (count >= 6) break;
                try writer.writeAll(" |");
            }
        } else {
            var it = self.session.evm.storage.storage.iterator();
            while (it.next()) |entry| {
                if (!std.mem.eql(u8, entry.key_ptr.address[0..], frame.address.bytes[0..])) continue;
                try writer.print(" {s}={s}", .{
                    self.scratchShortU256(entry.key_ptr.slot),
                    self.scratchShortU256(entry.value_ptr.*),
                });
                count += 1;
                if (count >= 6) break;
                try writer.writeAll(" |");
            }
        }

        if (count == 0) {
            try writer.writeAll(" empty");
        }
        self.command_status = self.command_status_storage.items;
    }

    fn handleSetCommand(self: *Ui, rest: []const u8) !void {
        if (std.mem.startsWith(u8, rest, "slot ")) {
            try self.handleSetSlotCommand(std.mem.trim(u8, rest["slot ".len..], " \t"));
            return;
        }
        if (std.mem.startsWith(u8, rest, "mem ")) {
            try self.handleSetMemoryCommand(std.mem.trim(u8, rest["mem ".len..], " \t"));
            return;
        }
        const eq_index = std.mem.indexOfScalar(u8, rest, '=') orelse {
            self.command_status = "expected name = value";
            return;
        };
        const name = std.mem.trim(u8, rest[0..eq_index], " \t");
        const value_text = std.mem.trim(u8, rest[eq_index + 1 ..], " \t");
        if (name.len == 0 or value_text.len == 0) {
            self.command_status = "expected name = value";
            return;
        }

        if (std.mem.eql(u8, name, "gas")) {
            const gas = std.fmt.parseInt(i64, value_text, 0) catch {
                self.command_status = "invalid gas value";
                return;
            };
            const frame = self.session.evm.getCurrentFrame() orelse {
                self.command_status = "no active frame";
                return;
            };
            if (gas < 0) {
                self.command_status = "gas must be >= 0";
                return;
            }
            frame.gas_remaining = gas;
            self.previous_snapshot = self.captureSnapshot();
            try self.refreshPreviousBindingsSnapshot();
            try self.setCommandStatusFmt("gas set to {d}", .{gas});
            return;
        }

        const value = std.fmt.parseUnsigned(u256, value_text, 0) catch {
            self.command_status = "invalid binding value";
            return;
        };
        const ok = try self.session.debugger.setVisibleBindingValueByName(self.allocator, name, value);
        if (!ok) {
            self.command_status = "binding not writable";
            return;
        }
        self.previous_snapshot = self.captureSnapshot();
        try self.refreshPreviousBindingsSnapshot();
        try self.setCommandStatusFmt("{s} set to {d}", .{ name, value });
    }

    fn handleSetSlotCommand(self: *Ui, rest: []const u8) !void {
        const eq_index = std.mem.indexOfScalar(u8, rest, '=') orelse {
            self.command_status = "expected slot <hex> = <value>";
            return;
        };
        const slot_text = std.mem.trim(u8, rest[0..eq_index], " \t");
        const value_text = std.mem.trim(u8, rest[eq_index + 1 ..], " \t");
        const frame = self.selectedFrame() orelse {
            self.command_status = "no active frame";
            return;
        };
        const slot = std.fmt.parseUnsigned(u256, slot_text, 0) catch {
            self.command_status = "invalid slot";
            return;
        };
        const value = std.fmt.parseUnsigned(u256, value_text, 0) catch {
            self.command_status = "invalid slot value";
            return;
        };
        try self.session.evm.storage.set(frame.address, slot, value);
        self.previous_snapshot = self.captureSnapshot();
        try self.refreshPreviousBindingsSnapshot();
        try self.setCommandStatusFmt("slot {s} set to {d}", .{ self.scratchFullU256(slot), value });
    }

    fn handleSetMemoryCommand(self: *Ui, rest: []const u8) !void {
        const eq_index = std.mem.indexOfScalar(u8, rest, '=') orelse {
            self.command_status = "expected mem <offset> = <value>";
            return;
        };
        const offset_text = std.mem.trim(u8, rest[0..eq_index], " \t");
        const value_text = std.mem.trim(u8, rest[eq_index + 1 ..], " \t");
        const frame = self.selectedFrame() orelse {
            self.command_status = "no active frame";
            return;
        };
        const offset = std.fmt.parseUnsigned(u32, offset_text, 0) catch {
            self.command_status = "invalid memory offset";
            return;
        };
        const value = std.fmt.parseUnsigned(u256, value_text, 0) catch {
            self.command_status = "invalid memory value";
            return;
        };

        var idx: u32 = 0;
        while (idx < 32) : (idx += 1) {
            const byte = @as(u8, @truncate(value >> @intCast((31 - idx) * 8)));
            const addr = std.math.add(u32, offset, idx) catch {
                self.command_status = "memory write out of range";
                return;
            };
            try frame.writeMemory(addr, byte);
        }
        self.previous_snapshot = self.captureSnapshot();
        try self.refreshPreviousBindingsSnapshot();
        try self.setCommandStatusFmt("mem 0x{X:0>4} set to {s}", .{ offset, self.scratchShortU256(value) });
    }

    fn handleBreakpointSet(self: *Ui, rest: []const u8) !void {
        const line = try self.parseBreakpointLine(rest);
        for (self.breakpoints.items) |existing| {
            if (existing == line) {
                self.command_status = "breakpoint already set";
                return;
            }
        }
        if (!self.session.debugger.setBreakpoint(self.seed.source_path, line)) {
            self.command_status = self.breakpointFailureMessage(line);
            return;
        }
        try self.breakpoints.append(self.allocator, line);
        try self.setCommandStatusFmt("breakpoint set on line {d}", .{line});
    }

    fn handleBreakpointDelete(self: *Ui, rest: []const u8) !void {
        const line = try self.parseBreakpointLine(rest);
        var found = false;
        var i: usize = 0;
        while (i < self.breakpoints.items.len) : (i += 1) {
            if (self.breakpoints.items[i] == line) {
                _ = self.breakpoints.swapRemove(i);
                found = true;
                break;
            }
        }
        self.session.debugger.removeBreakpoint(self.seed.source_path, line);
        if (!found) {
            self.command_status = "breakpoint not set";
            return;
        }
        try self.setCommandStatusFmt("breakpoint removed from line {d}", .{line});
    }

    fn parseBreakpointLine(self: *Ui, rest: []const u8) !u32 {
        _ = self;
        const trimmed = std.mem.trim(u8, rest, " \t");
        if (trimmed.len == 0) return error.InvalidArguments;
        if (std.mem.indexOfScalar(u8, trimmed, ':')) |colon| {
            return std.fmt.parseUnsigned(u32, std.mem.trim(u8, trimmed[colon + 1 ..], " \t"), 10);
        }
        return std.fmt.parseUnsigned(u32, trimmed, 10);
    }

    fn breakpointFailureMessage(self: *Ui, line: u32) []const u8 {
        const src_map = &self.session.debugger.src_map;
        const file = self.seed.source_path;
        if (!src_map.hasAnyEntryForLine(file, line)) return "no runtime mapping on that line";
        if (src_map.getStatementKindForLine(file, line) == null) return "non-executable spec line";
        return "no executable statement on that line";
    }

    fn describeBreakpoints(self: *Ui) !void {
        if (self.breakpoints.items.len == 0) {
            self.command_status = "no breakpoints";
            return;
        }
        self.command_status_storage.clearRetainingCapacity();
        var writer = self.command_status_storage.writer(self.allocator);
        try writer.writeAll("breakpoints:");
        for (self.breakpoints.items, 0..) |line, i| {
            if (i == 0) {
                try writer.print(" {d}", .{line});
            } else {
                try writer.print(", {d}", .{line});
            }
        }
        self.command_status = self.command_status_storage.items;
    }

    fn addCheckpoint(self: *Ui) !void {
        const id = self.next_checkpoint_id;
        self.next_checkpoint_id += 1;
        try self.checkpoints.append(self.allocator, .{
            .id = id,
            .step_index = self.step_history.items.len,
            .scroll_line = self.scroll_line,
            .focus_line = self.focus_line,
            .active_evm_tab = self.active_evm_tab,
        });
        try self.setCommandStatusFmt("checkpoint {d} saved at step {d}", .{ id, self.step_history.items.len });
    }

    fn describeCheckpoints(self: *Ui) !void {
        if (self.checkpoints.items.len == 0) {
            self.command_status = "no checkpoints";
            return;
        }
        self.command_status_storage.clearRetainingCapacity();
        var writer = self.command_status_storage.writer(self.allocator);
        try writer.writeAll("checkpoints:");
        for (self.checkpoints.items, 0..) |cp, i| {
            if (i == 0) {
                try writer.print(" #{d}@{d}", .{ cp.id, cp.step_index });
            } else {
                try writer.print(", #{d}@{d}", .{ cp.id, cp.step_index });
            }
        }
        self.command_status = self.command_status_storage.items;
    }

    fn describeBacktrace(self: *Ui) !void {
        if (self.session.evm.frames.items.len == 0) {
            self.command_status = "no frames";
            return;
        }
        self.command_status_storage.clearRetainingCapacity();
        var writer = self.command_status_storage.writer(self.allocator);
        try writer.writeAll("bt:");
        const frames = self.session.evm.frames.items;
        var logical: usize = 0;
        var i: usize = frames.len;
        while (i > 0) {
            i -= 1;
            const frame = frames[i];
            if (logical == 0) {
                try writer.print(" #{d} pc={d} gas={d}", .{ logical, frame.pc, frame.gas_remaining });
            } else {
                try writer.print(", #{d} pc={d} gas={d}", .{ logical, frame.pc, frame.gas_remaining });
            }
            logical += 1;
        }
        self.command_status = self.command_status_storage.items;
    }

    fn selectFrame(self: *Ui, index_from_top: usize) !void {
        if (index_from_top >= self.session.evm.frames.items.len) {
            self.command_status = "frame out of range";
            return;
        }
        self.selected_frame_index = index_from_top;
        try self.setCommandStatusFmt("frame #{d} selected", .{index_from_top});
    }

    fn restartCheckpoint(self: *Ui, id: u32) !void {
        for (self.checkpoints.items) |cp| {
            if (cp.id != id) continue;
            try self.rerunToHistory(cp.step_index);
            self.scroll_line = cp.scroll_line;
            self.focus_line = cp.focus_line;
            self.active_evm_tab = cp.active_evm_tab;
            try self.setCommandStatusFmt("restarted to checkpoint {d}", .{id});
            return;
        }
        self.command_status = "checkpoint not found";
    }

    fn rerunToHistory(self: *Ui, step_count: usize) !void {
        const replay_items = try self.allocator.dupe(StepMode, self.step_history.items[0..@min(step_count, self.step_history.items.len)]);
        defer self.allocator.free(replay_items);

        self.session.deinit();
        try Session.init(&self.session, self.allocator, &self.seed);
        self.command_mode = false;
        self.command_buffer.clearRetainingCapacity();
        self.selected_frame_index = 0;
        self.step_history.clearRetainingCapacity();
        try self.primeInitialStop();
        try self.applyBreakpoints();
        for (replay_items) |mode| {
            self.runStep(mode, true);
            if (std.mem.eql(u8, self.status, "execution_error")) break;
        }
        self.previous_snapshot = self.captureSnapshot();
        try self.refreshPreviousBindingsSnapshot();
    }

    fn applyBreakpoints(self: *Ui) !void {
        for (self.breakpoints.items) |line| {
            if (!self.session.debugger.setBreakpoint(self.seed.source_path, line)) {
                return error.InvalidArguments;
            }
        }
    }

    fn setCommandStatusFmt(self: *Ui, comptime fmt: []const u8, args: anytype) !void {
        self.command_status_storage.clearRetainingCapacity();
        try self.command_status_storage.writer(self.allocator).print(fmt, args);
        self.command_status = self.command_status_storage.items;
    }

    fn runStep(self: *Ui, mode: StepMode, record_history: bool) void {
        if (self.session.debugger.isHalted()) {
            self.status = "halted";
            self.syncFocusFromDebugger();
            return;
        }
        self.previous_snapshot = self.captureSnapshot();
        self.refreshPreviousBindingsSnapshot() catch {};
        switch (mode) {
            .in => self.session.debugger.stepIn() catch {
                self.status = self.session.debugger.lastErrorName() orelse "execution_error";
                self.command_status = self.status;
                return;
            },
            .opcode => self.session.debugger.stepOpcode() catch {
                self.status = self.session.debugger.lastErrorName() orelse "execution_error";
                self.command_status = self.status;
                return;
            },
            .over => self.session.debugger.stepOver() catch {
                self.status = self.session.debugger.lastErrorName() orelse "execution_error";
                self.command_status = self.status;
                return;
            },
            .out => self.session.debugger.stepOut() catch {
                self.status = self.session.debugger.lastErrorName() orelse "execution_error";
                self.command_status = self.status;
                return;
            },
            .continue_ => self.session.debugger.continue_() catch {
                self.status = self.session.debugger.lastErrorName() orelse "execution_error";
                self.command_status = self.status;
                return;
            },
        }
        if (record_history) self.step_history.append(self.allocator, mode) catch {};
        self.status = @tagName(self.session.debugger.stop_reason);
        if (!self.shouldPreserveFocusOnTerminalStop()) self.syncFocusFromDebugger();
        self.centerOnCurrentLine();
    }

    fn stepBack(self: *Ui) void {
        if (self.step_history.items.len == 0) {
            self.command_status = "at first stop";
            return;
        }
        _ = self.step_history.pop();
        self.session.deinit();
        Session.init(&self.session, self.allocator, &self.seed) catch {
            self.status = "execution_error";
            self.command_status = "failed to rebuild session";
            return;
        };
        self.primeInitialStop() catch {
            self.status = "execution_error";
            self.command_status = "failed to prime session";
            return;
        };
        for (self.step_history.items) |mode| {
            self.runStep(mode, false);
            if (std.mem.eql(u8, self.status, "execution_error")) {
                self.command_status = "failed to replay history";
                return;
            }
        }
        self.previous_snapshot = self.captureSnapshot();
        self.refreshPreviousBindingsSnapshot() catch {};
        self.command_status = "previous stop";
    }

    fn primeInitialStop(self: *Ui) !void {
        if (self.session.debugger.isHalted()) {
            self.status = "halted";
            return;
        }

        if (self.session.debugger.currentEntry()) |entry| {
            if (entry.is_statement) {
                self.status = "ready";
                self.syncFocusFromDebugger();
                self.centerOnCurrentLine();
                return;
            }
        }

        var attempts: usize = 0;
        while (!self.session.debugger.isHalted() and attempts < 64) : (attempts += 1) {
            try self.session.debugger.stepIn();
            self.status = @tagName(self.session.debugger.stop_reason);
            if (self.session.debugger.currentEntry()) |entry| {
                if (entry.is_statement) break;
            }
        }

        if (self.session.debugger.isHalted()) self.status = @tagName(self.session.debugger.stop_reason);
        self.syncFocusFromDebugger();
        self.centerOnCurrentLine();
    }

    fn syncFocusFromDebugger(self: *Ui) void {
        if (self.session.debugger.lastStatementLine()) |line| {
            self.focus_line = line;
            return;
        }
        if (self.session.debugger.currentEntry()) |entry| {
            if (entry.is_statement) {
                self.focus_line = entry.line;
                return;
            }
        }
        if (self.focus_line == null) self.focus_line = self.session.debugger.currentSourceLine();
    }

    fn shouldPreserveFocusOnTerminalStop(self: *const Ui) bool {
        if (!self.session.debugger.isHalted()) return false;
        return switch (self.session.debugger.stop_reason) {
            .execution_finished, .execution_reverted, .execution_error => true,
            else => false,
        };
    }

    fn centerOnCurrentLine(self: *Ui) void {
        const line = self.focus_line orelse return;
        if (line > 8) self.scroll_line = line - 8 else self.scroll_line = 1;
        if (self.sir_follow) self.centerSirOnCurrentMapping();
    }

    fn currentSirLine(self: *const Ui) ?u32 {
        if (self.session.debugger.currentEntry()) |entry| {
            if (entry.sir_line) |sir_line| return sir_line;
        }
        return self.session.debugger.lastStatementSirLine();
    }

    fn currentMappingWindow(self: *const Ui) ?MappingWindow {
        const file = self.seed.source_path;
        const ora_line = self.focus_line orelse self.session.debugger.currentSourceLine() orelse return null;
        const entry = self.session.debugger.currentEntry() orelse return null;

        var found = false;
        var sir_start: u32 = 0;
        var sir_end: u32 = 0;
        var idx_start: ?u32 = null;
        var idx_end: ?u32 = null;
        var pc_start: u32 = 0;
        var pc_end: u32 = 0;

        for (self.session.debugger.src_map.entries) |map_entry| {
            if (!std.mem.eql(u8, map_entry.file, file)) continue;
            if (map_entry.line != ora_line) continue;

            if (!found) {
                sir_start = map_entry.sir_line orelse 0;
                sir_end = map_entry.sir_line orelse 0;
                idx_start = map_entry.idx;
                idx_end = map_entry.idx;
                pc_start = map_entry.pc;
                pc_end = map_entry.pc;
                found = true;
            } else {
                if (map_entry.sir_line) |sir_line| {
                    if (sir_start == 0 or sir_line < sir_start) sir_start = sir_line;
                    if (sir_line > sir_end) sir_end = sir_line;
                }
                if (map_entry.idx) |idx| {
                    if (idx_start == null or idx < idx_start.?) idx_start = idx;
                    if (idx_end == null or idx > idx_end.?) idx_end = idx;
                }
                if (map_entry.pc < pc_start) pc_start = map_entry.pc;
                if (map_entry.pc > pc_end) pc_end = map_entry.pc;
            }
        }

        if (!found) {
            return .{
                .ora_line = ora_line,
                .sir_start = entry.sir_line orelse 0,
                .sir_end = entry.sir_line orelse 0,
                .idx_start = entry.idx,
                .idx_end = entry.idx,
                .pc_start = entry.pc,
                .pc_end = entry.pc,
            };
        }

        return .{
            .ora_line = ora_line,
            .sir_start = sir_start,
            .sir_end = sir_end,
            .idx_start = idx_start,
            .idx_end = idx_end,
            .pc_start = pc_start,
            .pc_end = pc_end,
        };
    }

    fn scrollDown(self: *Ui) void {
        const total = self.session.debugger.totalSourceLines();
        if (self.scroll_line < total) self.scroll_line += 1;
    }

    fn scrollUp(self: *Ui) void {
        if (self.scroll_line > 1) self.scroll_line -= 1;
    }

    fn scrollPage(self: *Ui, delta: i32) void {
        if (delta < 0) {
            const amount: u32 = @intCast(-delta);
            if (self.scroll_line > amount) self.scroll_line -= amount else self.scroll_line = 1;
        } else {
            self.scroll_line += @intCast(delta);
            const total = self.session.debugger.totalSourceLines();
            if (self.scroll_line > total) self.scroll_line = total;
        }
    }

    fn centerSirOnCurrentMapping(self: *Ui) void {
        const mapping = self.currentMappingWindow() orelse {
            self.sir_scroll_line = 1;
            return;
        };
        const target = if (mapping.sir_start != 0 and mapping.sir_end >= mapping.sir_start)
            @as(u32, @intCast((mapping.sir_start + mapping.sir_end) / 2))
        else
            self.currentSirLine() orelse 1;
        self.sir_scroll_line = if (target > 6) target - 6 else 1;
    }

    fn scrollSirDown(self: *Ui) void {
        self.sir_follow = false;
        self.sir_scroll_line += 1;
    }

    fn scrollSirUp(self: *Ui) void {
        self.sir_follow = false;
        if (self.sir_scroll_line > 1) self.sir_scroll_line -= 1;
    }

    fn resyncSirView(self: *Ui) void {
        self.sir_follow = true;
        self.centerSirOnCurrentMapping();
    }

    fn nextEvmTab(self: *Ui) void {
        self.active_evm_tab = switch (self.active_evm_tab) {
            .stack => .memory,
            .memory => .storage,
            .storage => .tstore,
            .tstore => .calldata,
            .calldata => .stack,
        };
    }

    fn prevEvmTab(self: *Ui) void {
        self.active_evm_tab = switch (self.active_evm_tab) {
            .stack => .calldata,
            .memory => .stack,
            .storage => .memory,
            .tstore => .storage,
            .calldata => .tstore,
        };
    }

    fn captureSnapshot(self: *Ui) Snapshot {
        var snapshot = Snapshot{};
        snapshot.gas_remaining = self.session.debugger.getGasRemaining();
        const stack = self.session.debugger.getStack();
        snapshot.stack_len = stack.len;
        snapshot.stack_top_count = @min(stack.len, snapshot.stack_top.len);
        var i: usize = 0;
        while (i < snapshot.stack_top_count) : (i += 1) {
            snapshot.stack_top[i] = stack[stack.len - 1 - i];
        }

        const frame = self.session.evm.getCurrentFrame() orelse return snapshot;
        snapshot.memory_size = frame.memory_size;
        snapshot.memory_word_count = @min(@as(usize, @intCast((frame.memory_size + 31) / 32)), snapshot.memory_words.len);
        var word: usize = 0;
        while (word < snapshot.memory_word_count) : (word += 1) {
            const offset: u32 = @intCast(word * 32);
            var value: u256 = 0;
            var j: u32 = 0;
            while (j < 32 and offset + j < frame.memory_size) : (j += 1) {
                value = (value << 8) | frame.readMemory(offset + j);
            }
            snapshot.memory_words[word] = value;
        }

        snapshot.storage_count = self.captureStorageSnapshot(false, &snapshot.storage_top_slots, &snapshot.storage_top_values);
        snapshot.storage_top_count = @min(snapshot.storage_count, snapshot.storage_top_slots.len);
        snapshot.tstore_count = self.captureStorageSnapshot(true, &snapshot.tstore_top_slots, &snapshot.tstore_top_values);
        snapshot.tstore_top_count = @min(snapshot.tstore_count, snapshot.tstore_top_slots.len);
        return snapshot;
    }

    fn captureStorageSnapshot(self: *Ui, transient: bool, slots_out: *[8]u256, values_out: *[8]u256) usize {
        const frame = self.selectedFrame() orelse return 0;
        var count: usize = 0;
        var captured: usize = 0;

        if (transient) {
            var it = self.session.evm.storage.transient.iterator();
            while (it.next()) |entry| {
                if (!std.mem.eql(u8, entry.key_ptr.address[0..], frame.address.bytes[0..])) continue;
                if (captured < slots_out.len) {
                    slots_out[captured] = entry.key_ptr.slot;
                    values_out[captured] = entry.value_ptr.*;
                    captured += 1;
                }
                count += 1;
            }
        } else {
            var it = self.session.evm.storage.storage.iterator();
            while (it.next()) |entry| {
                if (!std.mem.eql(u8, entry.key_ptr.address[0..], frame.address.bytes[0..])) continue;
                if (captured < slots_out.len) {
                    slots_out[captured] = entry.key_ptr.slot;
                    values_out[captured] = entry.value_ptr.*;
                    captured += 1;
                }
                count += 1;
            }
        }

        return count;
    }

    fn clearPreviousBindingsSnapshot(self: *Ui) void {
        for (self.previous_bindings.items) |entry| {
            self.allocator.free(entry.name);
        }
        self.previous_bindings.clearRetainingCapacity();
    }

    fn refreshPreviousBindingsSnapshot(self: *Ui) !void {
        self.clearPreviousBindingsSnapshot();
        const bindings = try self.session.debugger.getVisibleBindings(self.allocator);
        defer if (bindings.len > 0) self.allocator.free(bindings);

        for (bindings) |binding| {
            try self.previous_bindings.append(self.allocator, .{
                .name = try self.allocator.dupe(u8, binding.name),
                .value = try self.numericBindingValue(&binding),
            });
        }
    }

    fn previousBindingValue(self: *const Ui, name: []const u8) ?u256 {
        for (self.previous_bindings.items) |entry| {
            if (std.mem.eql(u8, entry.name, name)) return entry.value;
        }
        return null;
    }

    fn render(self: *Ui, vx: *vaxis.Vaxis) void {
        self.render_scratch.clearRetainingCapacity();
        const win = vx.window();
        const scratch_target: usize = @as(usize, win.width) * @as(usize, win.height) * 8;
        self.render_scratch.ensureTotalCapacity(self.allocator, scratch_target) catch {};
        win.clear();
        if (win.width < 50 or win.height < 12) {
            drawSegments(win, 1, 1, &.{seg("Terminal too small for debugger view", style_error())});
            return;
        }

        self.drawHeader(win);
        self.drawFooter(win);

        const content_y: u16 = 3;
        const content_h: u16 = win.height - 8;
        const source_w: u16 = if (win.width >= 140)
            @max(20, @as(u16, @intCast((@as(u32, win.width) * 60) / 100)))
        else
            @max(20, @as(u16, @intCast((@as(u32, win.width) * 58) / 100)));
        const right_w: u16 = win.width - source_w;
        if (content_h < 10 or right_w < 24) return;

        const top_h: u16 = @max(8, @as(u16, @intCast((@as(u32, content_h) * 52) / 100)));
        const bottom_h: u16 = content_h - top_h;
        const current_source_line = self.focus_line orelse self.session.debugger.currentSourceLine() orelse 0;
        const mapping = self.currentMappingWindow();
        const current_sir_line = if (mapping) |m| m.sir_start else self.currentSirLine() orelse 0;
        const current_idx = if (mapping) |m| m.idx_start else if (self.session.debugger.currentEntry()) |entry| entry.idx else null;
        const ora_w: u16 = if (self.seed.sir_text != null)
            @max(20, @as(u16, @intCast((@as(u32, source_w) * 52) / 100)))
        else
            source_w;
        const sir_w: u16 = source_w - ora_w;

        const source_outer = win.child(.{
            .x_off = 0,
            .y_off = @intCast(content_y),
            .width = ora_w,
            .height = top_h,
            .border = .{ .where = .all, .glyphs = .{ .custom = ascii_border_glyphs }, .style = style_border() },
        });
        const source_title = self.scratchFmt(" Ora Source  line {d} ", .{current_source_line}) catch " Ora Source ";
        Ui.drawPanelTitle(win, 2, content_y, source_title);
        self.drawSourcePane(source_outer);

        if (self.seed.sir_text != null and sir_w >= 24) {
            const sir_outer = win.child(.{
                .x_off = ora_w,
                .y_off = @intCast(content_y),
                .width = sir_w,
                .height = top_h,
                .border = .{ .where = .all, .glyphs = .{ .custom = ascii_border_glyphs }, .style = style_border() },
            });
            const sir_title = if (mapping) |m|
                if (m.idx_start != null and m.idx_end != null)
                    self.scratchFmt(" SIR Text  lines {d}..{d}  idx {d}..{d} ", .{ m.sir_start, m.sir_end, m.idx_start.?, m.idx_end.? }) catch " SIR Text "
                else
                    self.scratchFmt(" SIR Text  lines {d}..{d} ", .{ m.sir_start, m.sir_end }) catch " SIR Text "
            else if (current_idx) |idx|
                self.scratchFmt(" SIR Text  line {d}  idx {d} ", .{ current_sir_line, idx }) catch " SIR Text "
            else
                self.scratchFmt(" SIR Text  line {d} ", .{current_sir_line}) catch " SIR Text ";
            Ui.drawPanelTitle(win, ora_w + 2, content_y, sir_title);
            self.drawSirPane(sir_outer);
        }

        const right_x = source_w;
        const right_top_h: u16 = top_h;
        const bindings_h: u16 = @max(6, @as(u16, @intCast((@as(u32, right_top_h) * 55) / 100)));
        const machine_h: u16 = right_top_h - bindings_h;

        const bindings_outer = win.child(.{
            .x_off = @intCast(right_x),
            .y_off = @intCast(content_y),
            .width = right_w,
            .height = bindings_h,
            .border = .{ .where = .all, .glyphs = .{ .custom = ascii_border_glyphs }, .style = style_border() },
        });
        Ui.drawPanelTitle(win, right_x + 2, content_y, " Bindings ");
        self.drawBindingsPane(bindings_outer);

        const machine_outer = win.child(.{
            .x_off = @intCast(right_x),
            .y_off = @intCast(content_y + bindings_h),
            .width = right_w,
            .height = machine_h,
            .border = .{ .where = .all, .glyphs = .{ .custom = ascii_border_glyphs }, .style = style_border() },
        });
        Ui.drawPanelTitle(win, right_x + 2, content_y + bindings_h, " Machine ");
        self.drawMachinePane(machine_outer);

        const state_outer = win.child(.{
            .x_off = 0,
            .y_off = @intCast(content_y + top_h),
            .width = win.width,
            .height = bottom_h,
            .border = .{ .where = .all, .glyphs = .{ .custom = ascii_border_glyphs }, .style = style_border() },
        });
        Ui.drawPanelTitle(win, 2, content_y + top_h, " State ");
        self.drawEvmPane(state_outer);
    }

    fn drawPanelTitle(win: Window, x: u16, y: u16, title: []const u8) void {
        drawSegments(win, x, y, &.{seg(title, style_title())});
    }

    fn drawHeader(self: *Ui, win: Window) void {
        const source_name = std.fs.path.basename(self.seed.source_path);
        const line = self.focus_line orelse self.session.debugger.currentSourceLine() orelse 0;
        const mapping = self.currentMappingWindow();
        const sir_line = if (mapping) |m| m.sir_start else self.currentSirLine() orelse 0;
        const entry = self.session.debugger.currentEntry();
        const frame = self.session.evm.getCurrentFrame();
        const opcode = if (frame != null) self.session.debugger.getCurrentOpcodeName() else "no-frame";
        const gas_remaining: i64 = if (frame) |f| f.gas_remaining else 0;
        const gas_spent_step: i64 = if (frame != null and self.previous_snapshot.gas_remaining >= gas_remaining)
            self.previous_snapshot.gas_remaining - gas_remaining
        else
            0;
        const gas_spent_total: i64 = if (frame != null)
            5_000_000 - gas_remaining
        else if (self.session.debugger.isSuccess())
            5_000_000
        else
            0;

        const top = win.child(.{ .height = 1 });
        top.fill(.{ .char = .{ .grapheme = " " }, .style = style_header_title() });
        const title = self.scratchFmt(" Ora EVM Debugger | {s}", .{source_name}) catch "Ora EVM Debugger";
        drawSegments(top, 0, 0, &.{seg(title, style_header_title())});

        const meta = win.child(.{ .y_off = 1, .height = 2 });
        meta.fill(.{ .char = .{ .grapheme = " " }, .style = style_header_meta() });

        const status_text = if (frame == null)
            self.scratchFmt(" {s}  |  ora {d}  |  sir {d}  |  no active frame  |  result {s}", .{
                self.status,
                line,
                sir_line,
                if (self.session.debugger.isSuccess()) "success" else "reverted",
            }) catch "status"
        else if (self.session.debugger.lastErrorName()) |err_name|
            if (entry) |e| blk: {
                if (e.idx) |idx| {
                    break :blk self.scratchFmt(" error {s}  |  ora {d}  |  sir {d}  |  pc {d} idx {d}  |  {s}  |  depth {d}  |  gas {d}", .{
                        err_name,
                        line,
                        sir_line,
                        self.session.debugger.getPC(),
                        idx,
                        opcode,
                        self.session.debugger.getCallDepth(),
                        gas_remaining,
                    }) catch "status";
                }
                break :blk self.scratchFmt(" error {s}  |  ora {d}  |  sir {d}  |  pc {d}  |  {s}  |  depth {d}  |  gas {d}", .{
                    err_name,
                    line,
                    sir_line,
                    self.session.debugger.getPC(),
                    opcode,
                    self.session.debugger.getCallDepth(),
                    gas_remaining,
                }) catch "status";
            } else
                self.scratchFmt(" error {s}  |  ora {d}  |  sir {d}  |  pc {d}  |  {s}  |  depth {d}  |  gas {d}", .{
                    err_name,
                    line,
                    sir_line,
                    self.session.debugger.getPC(),
                    opcode,
                    self.session.debugger.getCallDepth(),
                    gas_remaining,
                }) catch "status"
        else if (entry) |e| blk: {
            if (e.idx) |idx| {
                break :blk self.scratchFmt(" {s}  |  ora {d}  |  sir {d}  |  pc {d} idx {d}  |  {s}  |  depth {d}  |  gas {d}  |  step -{d}  |  total -{d}", .{
                    self.status,
                    line,
                    sir_line,
                    self.session.debugger.getPC(),
                    idx,
                    opcode,
                    self.session.debugger.getCallDepth(),
                    gas_remaining,
                    gas_spent_step,
                    gas_spent_total,
                }) catch "status";
            }
            break :blk self.scratchFmt(" {s}  |  ora {d}  |  sir {d}  |  pc {d}  |  {s}  |  depth {d}  |  gas {d}  |  step -{d}  |  total -{d}", .{
                self.status,
                line,
                sir_line,
                self.session.debugger.getPC(),
                opcode,
                self.session.debugger.getCallDepth(),
                gas_remaining,
                gas_spent_step,
                gas_spent_total,
            }) catch "status";
        } else
            self.scratchFmt(" {s}  |  ora {d}  |  sir {d}  |  pc {d}  |  {s}  |  depth {d}  |  gas {d}  |  step -{d}  |  total -{d}", .{
                self.status,
                line,
                sir_line,
                self.session.debugger.getPC(),
                opcode,
                self.session.debugger.getCallDepth(),
                gas_remaining,
                gas_spent_step,
                gas_spent_total,
            }) catch "status";
        drawSegments(meta, 0, 0, &.{seg(status_text, style_header_meta())});

        const current_source = if (line != 0) blk: {
            if (self.session.debugger.getSourceLineText(line)) |line_text| {
                break :blk std.mem.trim(u8, std.mem.trimRight(u8, line_text, "\r"), " \t");
            }
            break :blk "";
        } else "";
        const current_text = if (mapping) |m|
            if (m.idx_start != null and m.idx_end != null)
                self.scratchFmt(" map  |  ora {d}  ->  sir {d}..{d}  ->  idx {d}..{d}  ->  pc {d}..{d}  |  {s}", .{
                    m.ora_line,
                    m.sir_start,
                    m.sir_end,
                    m.idx_start.?,
                    m.idx_end.?,
                    m.pc_start,
                    m.pc_end,
                    current_source,
                }) catch "map"
            else
                self.scratchFmt(" map  |  ora {d}  ->  sir {d}..{d}  ->  pc {d}..{d}  |  {s}", .{
                    m.ora_line,
                    m.sir_start,
                    m.sir_end,
                    m.pc_start,
                    m.pc_end,
                    current_source,
                }) catch "map"
        else if (entry) |e|
            if (e.idx) |idx|
                self.scratchFmt(" map  |  ora {d}  ->  sir {d}  ->  idx {d}  ->  pc {d}  |  {s}", .{
                    line,
                    sir_line,
                    idx,
                    e.pc,
                    current_source,
                }) catch "map"
            else
                self.scratchFmt(" map  |  ora {d}  ->  sir {d}  ->  pc {d}  |  {s}", .{
                    line,
                    sir_line,
                    e.pc,
                    current_source,
                }) catch "map"
        else
            self.scratchFmt(" source  |  {s}", .{current_source}) catch "source";
        drawSegments(meta, 0, 1, &.{seg(current_text, style_header_meta())});
    }

    fn drawFooter(self: *Ui, win: Window) void {
        const cmd = win.child(.{ .y_off = @intCast(win.height - 4), .height = 1 });
        cmd.fill(.{ .char = .{ .grapheme = " " }, .style = style_command_bg() });
        const prompt = if (self.command_mode)
            self.scratchFmt(":{s}", .{self.command_buffer.items}) catch ":"
        else
            self.scratchFmt(":{s}", .{self.command_status}) catch ":";
        drawSegments(cmd, 0, 0, &.{seg(prompt, style_command())});

        const console_height: u16 = 2;
        const console = win.child(.{ .y_off = @intCast(win.height - 3), .height = console_height });
        console.fill(.{ .char = .{ .grapheme = " " }, .style = style_header_meta() });
        self.drawFooterConsole(console);

        const help = win.child(.{ .y_off = @intCast(win.height - 1), .height = 1 });
        help.fill(.{ .char = .{ .grapheme = " " }, .style = style_header_title() });
        drawSegments(help, 0, 0, &.{seg(" s step-in  x opcode  n step-over  o step-out  c continue  p previous  : command  j/k Ora  J/K SIR  = sync SIR  1..5 tabs  [/] cycle  q quit  |  . runtime  ! guard  * break  >|< sir-range ", style_header_title())});
    }

    fn clearCommandLog(self: *Ui) void {
        for (self.command_log.items) |line| self.allocator.free(line);
        self.command_log.clearRetainingCapacity();
    }

    fn appendCommandLog(self: *Ui, command: []const u8, result: []const u8) !void {
        const line = try std.fmt.allocPrint(self.allocator, ":{s} => {s}", .{ command, result });
        if (self.command_log.items.len >= 6) {
            const oldest = self.command_log.orderedRemove(0);
            self.allocator.free(oldest);
        }
        try self.command_log.append(self.allocator, line);
    }

    fn footerConsoleText(self: *Ui) ?[]const u8 {
        if (self.command_log.items.len == 0) return null;
        if (self.command_log.items.len == 1) return self.command_log.items[0];
        return self.command_log.items[self.command_log.items.len - 1];
    }

    fn drawFooterConsole(self: *Ui, win: Window) void {
        if (self.command_log.items.len == 0) {
            drawSegments(win, 0, 0, &.{seg(" debugger values show folded constants or rooted runtime values when readable ", style_footer_note())});
            return;
        }

        const visible = @min(@as(usize, win.height), self.command_log.items.len);
        const start = self.command_log.items.len - visible;
        var row: u16 = 0;
        var i: usize = start;
        while (i < self.command_log.items.len and row < win.height) : ({
            i += 1;
            row += 1;
        }) {
            drawSegments(win, 0, row, &.{seg(self.command_log.items[i], style_footer_note())});
        }
    }

    fn drawSourcePane(self: *Ui, win: Window) void {
        const current_line = self.focus_line orelse self.session.debugger.currentSourceLine() orelse 0;
        const mapping = self.currentMappingWindow();
        const summary = if (mapping) |m|
            if (m.idx_start != null and m.idx_end != null)
                self.scratchFmt(" runtime {s} | ora {d} => sir {d}..{d} | idx {d}..{d} | pc {d}..{d}", .{
                    self.statementKindLabel(current_line),
                    m.ora_line,
                    m.sir_start,
                    m.sir_end,
                    m.idx_start.?,
                    m.idx_end.?,
                    m.pc_start,
                    m.pc_end,
                }) catch "runtime mapping"
            else
                self.scratchFmt(" runtime {s} | ora {d} => sir {d}..{d} | pc {d}..{d}", .{
                    self.statementKindLabel(current_line),
                    m.ora_line,
                    m.sir_start,
                    m.sir_end,
                    m.pc_start,
                    m.pc_end,
                }) catch "runtime mapping"
        else
            self.scratchFmt(" runtime {s} | ora {d}", .{
                self.statementKindLabel(current_line),
                current_line,
            }) catch "runtime mapping";
        drawSegments(win, 1, 0, &.{seg(summary, style_muted())});

        const content_y: u16 = if (win.height > 1) 1 else 0;
        const content_h: u16 = if (win.height > 1) win.height - 1 else win.height;
        if (content_h == 0) return;
        const content = win.child(.{ .y_off = content_y, .height = content_h });

        self.source_view.scroll_view.scroll.y = if (self.scroll_line > 0) self.scroll_line - 1 else 0;
        self.source_view.highlighted_style = .{
            .bg = Color.rgbFromUint(0x303A45),
            .fg = Color.rgbFromUint(0xF5F7FA),
        };
        self.source_view.draw(content, self.source_buffer, .{
            .highlighted_line = @intCast(current_line),
            .draw_line_numbers = true,
            .indentation = 4,
        });
        self.drawSourceGutterMarkers(content, current_line);
    }

    fn drawSirPane(self: *Ui, win: Window) void {
        if (self.seed.sir_text == null) {
            drawSegments(win, 1, 1, &.{seg("no SIR text artifact for this session", style_hint())});
            return;
        }
        const mapping = self.currentMappingWindow();
        const current_sir_line = if (mapping) |m| if (m.sir_start != 0) m.sir_start else self.currentSirLine() orelse 0 else self.currentSirLine() orelse 0;
        const effect = self.currentWriteEffectKind();
        const summary = if (mapping) |m|
            if (m.idx_start != null and m.idx_end != null)
                self.scratchFmt(" lowered region | ora {d} => sir {d}..{d} | idx {d}..{d} | pc {d}..{d} | effect {s}", .{
                    m.ora_line,
                    m.sir_start,
                    m.sir_end,
                    m.idx_start.?,
                    m.idx_end.?,
                    m.pc_start,
                    m.pc_end,
                    self.writeEffectLabel(effect),
                }) catch "lowered region"
            else
                self.scratchFmt(" lowered region | ora {d} => sir {d}..{d} | pc {d}..{d} | effect {s}", .{
                    m.ora_line,
                    m.sir_start,
                    m.sir_end,
                    m.pc_start,
                    m.pc_end,
                    self.writeEffectLabel(effect),
                }) catch "lowered region"
        else if (self.session.debugger.currentOpMeta()) |meta|
            self.scratchFmt(" lowered op | {s} [{s}:{s}] | effect {s}", .{
                meta.op,
                meta.function,
                meta.block,
                self.writeEffectLabel(effect),
            }) catch "lowered op"
        else
            "lowered region";
        drawSegments(win, 1, 0, &.{seg(summary, style_muted())});

        const content_y: u16 = if (win.height > 1) 1 else 0;
        const content_h: u16 = if (win.height > 1) win.height - 1 else win.height;
        if (content_h == 0) return;
        const content = win.child(.{ .y_off = content_y, .height = content_h });

        if (self.sir_follow) self.centerSirOnCurrentMapping();
        self.sir_view.scroll_view.scroll.y = if (self.sir_scroll_line > 0) self.sir_scroll_line - 1 else 0;
        self.sir_view.draw(content, self.sir_buffer, .{
            .highlighted_line = @intCast(current_sir_line),
            .draw_line_numbers = true,
            .indentation = 4,
        });
        if (mapping) |m| self.drawSirRangeMarkers(content, m);
    }

    fn statementKindLabel(self: *Ui, line: u32) []const u8 {
        const kind = self.statementKindForLine(line) orelse return "none";
        return switch (kind) {
            .runtime => "stmt",
            .runtime_guard => "guard",
        };
    }

    fn currentWriteEffectKind(self: *Ui) WriteEffectKind {
        const meta = self.session.debugger.currentOpMeta() orelse return .none;
        const current_line = (self.focus_line orelse self.session.debugger.currentSourceLine()) orelse 0;
        const source_text = if (current_line != 0)
            if (self.session.debugger.getSourceLineText(current_line)) |line_text|
                std.mem.trim(u8, std.mem.trimRight(u8, line_text, "\r"), " \t")
            else
                ""
        else
            "";

        if (std.mem.indexOf(u8, meta.op, "tstore") != null) return .tstore;
        if (std.mem.indexOf(u8, meta.op, "sstore") != null) return .storage;

        const memory_store = std.mem.indexOf(u8, meta.op, "mstore") != null or
            std.mem.eql(u8, meta.op, "sir.store") or
            std.mem.endsWith(u8, meta.op, ".store");
        if (!memory_store) return .none;

        if (std.mem.startsWith(u8, source_text, "return ")) return .return_buffer;
        return .memory;
    }

    fn writeEffectLabel(self: *Ui, effect: WriteEffectKind) []const u8 {
        _ = self;
        return switch (effect) {
            .none => "read/compute",
            .return_buffer => "return-buffer write",
            .memory => "memory write",
            .storage => "storage write",
            .tstore => "tstore write",
        };
    }

    fn drawBindingsPane(self: *Ui, root: Window) void {
        const bindings = self.session.debugger.getVisibleBindings(self.allocator) catch &.{};
        defer if (bindings.len > 0) self.allocator.free(bindings);

        drawSegments(root, 1, 0, &.{seg("name = value / runtime", style_muted())});
        const breakpoints_summary = self.scratchBreakpointsSummary() catch "breakpoints: ?";
        drawSegments(root, 1, 1, &.{seg(breakpoints_summary, style_muted())});
        const checkpoints_summary = self.scratchCheckpointSummaryCompact() catch "checkpoints: ?";
        drawSegments(root, 1, 2, &.{seg(checkpoints_summary, style_muted())});

        if (bindings.len == 0) {
            drawSegments(root, 1, 4, &.{seg("no visible bindings", style_hint())});
            return;
        }

        var i: usize = 0;
        while (i < bindings.len and i + 4 < root.height) : (i += 1) {
            const binding = bindings[i];
            const binding_label = self.bindingLabel(&binding);
            const current_value = self.numericBindingValue(&binding) catch null;
            const resolved_value = self.resolvedBindingValue(&binding) catch null;
            const previous_value = self.previousBindingValue(binding.name);
            const changed = current_value != previous_value;
            const marker = if (changed) "*" else " ";
            const row_text = if (binding.folded_value) |folded|
                self.scratchFmt("{s} {s} [{s}] = {s} [folded]", .{ marker, binding.name, binding_label, folded }) catch binding.name
            else if (changed)
                if (previous_value) |prev|
                    if (current_value) |value|
                        self.scratchFmt("{s} {s} [{s}]: {d} -> {d}", .{ marker, binding.name, binding_label, prev, value }) catch binding.name
                    else
                        self.scratchFmt("{s} {s} [{s}]: {d} -> ?", .{ marker, binding.name, binding_label, prev }) catch binding.name
                else if (current_value) |value|
                    self.scratchFmt("{s} {s} [{s}]: ? -> {d}", .{ marker, binding.name, binding_label, value }) catch binding.name
                else
                    self.scratchFmt("{s} {s} [{s}]", .{ marker, binding.name, binding_label }) catch binding.name
            else if (resolved_value) |value|
                switch (value) {
                    .numeric => |numeric| self.scratchFmt("{s} {s} [{s}] = {d}", .{ marker, binding.name, binding_label, numeric }) catch binding.name,
                    .text => |text| self.scratchFmt("{s} {s} [{s}] = {s}", .{ marker, binding.name, binding_label, text }) catch binding.name,
                }
            else
                self.scratchFmt("{s} {s} [{s}]", .{ marker, binding.name, binding_label }) catch binding.name;
            drawSegments(root, 1, @intCast(4 + i), &.{seg(row_text, if (changed) style_changed() else style_text())});
        }
    }

    fn drawMachinePane(self: *Ui, win: Window) void {
        const frame = self.selectedFrame() orelse {
            drawSegments(win, 1, 1, &.{seg("no active frame", style_hint())});
            return;
        };
        const gas_remaining = frame.gas_remaining;
        const gas_spent_step = if (self.previous_snapshot.gas_remaining >= gas_remaining)
            self.previous_snapshot.gas_remaining - gas_remaining
        else
            0;

        var row: u16 = 0;

        drawSegments(win, 1, row, &.{seg(self.scratchFmt("frame={d}/{d}  pc={d}  opcode={s}", .{
            self.selected_frame_index,
            self.session.evm.frames.items.len,
            frame.pc,
            self.opcodeNameForFrame(frame),
        }) catch "pc", style_text())});
        row += 1;
        drawSegments(win, 1, row, &.{seg(self.scratchFmt("gas={d}  spent_step={d}  stack={d}  stopped={} reverted={}", .{
            gas_remaining,
            gas_spent_step,
            frame.stack.items.len,
            frame.stopped,
            frame.reverted,
        }) catch "gas", style_text())});
        row += 1;
        drawSegments(win, 1, row, &.{seg(self.scratchFmt("caller={s}", .{self.scratchAddress(frame.caller)}) catch "caller", style_text())});
        row += 1;
        drawSegments(win, 1, row, &.{seg(self.scratchFmt("callee={s}", .{self.scratchAddress(frame.address)}) catch "callee", style_text())});
        row += 1;
        drawSegments(win, 1, row, &.{seg(self.scratchFmt("value={d}  calldata={d}B  mem={d}B", .{
            frame.value,
            frame.calldata.len,
            frame.memory_size,
        }) catch "value", style_text())});
        row += 2;
        drawSegments(win, 1, row, &.{seg(self.scratchFmt("effect={s}", .{
            self.writeEffectLabel(self.currentWriteEffectKind()),
        }) catch "effect", style_muted())});
        row += 1;
        drawSegments(win, 1, row, &.{seg(self.scratchFmt("state={s}  stop={s}", .{
            if (self.session.debugger.isHalted()) "halted" else "running",
            @tagName(self.session.debugger.stop_reason),
        }) catch "state", style_muted())});
        row += 2;
        if (self.checkpoints.items.len > 0 and row < win.height) {
            drawSegments(win, 1, row, &.{seg(self.scratchCheckpointSummary() catch "checkpoints", style_muted())});
            row += 2;
        }

        if (row >= win.height) return;
        const opcode_rows: u16 = @min(4, win.height - row);
        var pc = frame.pc;
        var shown: u16 = 0;
        while (shown < opcode_rows) : (shown += 1) {
            const opcode = frame.bytecode.getOpcode(pc) orelse break;
            const is_current = shown == 0;
            const prefix = if (is_current) ">" else " ";
            const line = self.scratchFmt("{s} {d:>4}  {s}", .{ prefix, pc, opcode_utils.getOpName(opcode) }) catch "op";
            drawSegments(win, 1, row + shown, &.{seg(line, if (is_current) style_emphasis() else style_muted())});
            pc = self.nextOpcodePc(frame.bytecode, pc) orelse break;
        }
    }

    fn drawEvmPane(self: *Ui, root: Window) void {
        self.drawTabBar(root);
        const frame = self.selectedFrame();
        const stack_len = if (frame) |f| f.stack.items.len else 0;
        const mem_size = if (frame) |f| f.memory_size else 0;
        const storage_count = self.countStorageSlots(false);
        const tstore_count = self.countStorageSlots(true);
        const calldata_len = if (frame) |f| f.calldata.len else self.seed.calldata.len;
        const summary = self.scratchFmt(
            "frame={d}/{d}  tab={s}  stack={d}  mem={d}B  stor={d}  tstor={d}  calldata={d}B",
            .{
                self.selected_frame_index,
                self.session.evm.frames.items.len,
                tabName(self.active_evm_tab),
                stack_len,
                mem_size,
                storage_count,
                tstore_count,
                calldata_len,
            },
        ) catch "state";
        drawSegments(root, 1, 1, &.{seg(summary, style_muted())});
        const body = root.child(.{ .y_off = 2, .height = root.height - 2 });
        switch (self.active_evm_tab) {
            .stack => self.drawStackTab(body),
            .memory => self.drawMemoryTab(body),
            .storage => self.drawStorageTab(body, false),
            .tstore => self.drawStorageTab(body, true),
            .calldata => self.drawCalldataTab(body),
        }
    }

    fn drawTabBar(self: *Ui, win: Window) void {
        const tabs = [_]EvmTabKind{ .stack, .memory, .storage, .tstore, .calldata };
        var x: u16 = 1;
        for (tabs, 0..) |tab, i| {
            const label = tabLabel(tab);
            const style = if (tab == self.active_evm_tab) style_tab_active() else style_tab_inactive();
            const text = self.scratchFmt("[{d}] {s}", .{ i + 1, label }) catch label;
            drawSegments(win, x, 0, &.{seg(text, style)});
            x += @intCast(text.len + 2);
        }
    }

    fn drawStackTab(self: *Ui, win: Window) void {
        const frame = self.selectedFrame() orelse {
            drawSegments(win, 1, 1, &.{seg("no active frame", style_hint())});
            return;
        };
        const stack = frame.stack.items;
        const delta = @as(i64, @intCast(stack.len)) - @as(i64, @intCast(self.previous_snapshot.stack_len));
        const summary = self.scratchFmt("len={d}  delta={d}", .{ stack.len, delta }) catch "stack";
        drawSegments(win, 1, 0, &.{seg(summary, if (delta != 0) style_changed() else style_muted())});
        if (stack.len == 0) {
            drawSegments(win, 1, 2, &.{seg("stack empty at current stop", style_hint())});
            drawSegments(win, 1, 3, &.{seg("use x or :op to inspect operand stack between statement stops", style_hint())});
            return;
        }
        const shown = @min(stack.len, @as(usize, @intCast(@max(@as(u16, 0), win.height - 1))));
        var i: usize = 0;
        while (i < shown) : (i += 1) {
            const idx = stack.len - 1 - i;
            const prefix = if (i == 0) "top" else "   ";
            const line = self.scratchFmt("{s} [{d:>2}] {s}", .{ prefix, idx, self.scratchShortU256(stack[idx]) }) catch "stack";
            const changed = i >= self.previous_snapshot.stack_top_count or self.previous_snapshot.stack_top[i] != stack[idx] or self.previous_snapshot.stack_len != stack.len;
            drawSegments(win, 1, @intCast(i + 1), &.{seg(line, if (changed) style_changed() else if (i == 0) style_emphasis() else style_text())});
        }
    }

    fn drawMemoryTab(self: *Ui, win: Window) void {
        const frame = self.selectedFrame() orelse {
            drawSegments(win, 1, 1, &.{seg("no active frame", style_hint())});
            return;
        };
        const effect = self.currentWriteEffectKind();
        const summary = if (effect == .return_buffer)
            "memory view | current lowered op is writing the return buffer"
        else
            "memory view | transient EVM memory, not persistent contract state";
        drawSegments(win, 1, 0, &.{seg(summary, style_muted())});
        if (frame.memory_size == 0) {
            drawSegments(win, 1, 2, &.{seg("memory is empty", style_hint())});
            return;
        }
        const max_words: u32 = @min(@as(u32, if (win.height > 1) win.height - 1 else 0), 8);
        var word: u32 = 0;
        while (word < max_words) : (word += 1) {
            const offset = word * 32;
            if (offset >= frame.memory_size) break;
            var value: u256 = 0;
            var j: u32 = 0;
            while (j < 32 and offset + j < frame.memory_size) : (j += 1) {
                value = (value << 8) | frame.readMemory(offset + j);
            }
            const line = self.scratchFmt("0x{X:0>4}  {s}", .{ offset, self.scratchShortU256(value) }) catch "mem";
            const changed = word >= self.previous_snapshot.memory_word_count or self.previous_snapshot.memory_words[word] != value or self.previous_snapshot.memory_size != frame.memory_size;
            drawSegments(win, 1, @intCast(word + 1), &.{seg(line, if (changed) style_changed() else style_text())});
        }
    }

    fn drawStorageTab(self: *Ui, win: Window, transient: bool) void {
        const frame = self.selectedFrame() orelse {
            drawSegments(win, 1, 1, &.{seg("no active frame", style_hint())});
            return;
        };

        var count: usize = 0;
        var row: u16 = 1;
        drawSegments(win, 1, 0, &.{seg(if (transient)
            "transient storage view | stateful writes for current contract"
        else
            "storage view | persistent contract state", style_muted())});
        if (transient) {
            var it = self.session.evm.storage.transient.iterator();
            while (it.next()) |entry| {
                if (!std.mem.eql(u8, entry.key_ptr.address[0..], frame.address.bytes[0..])) continue;
                if (row >= win.height) break;
                const changed = self.storageSlotChanged(true, entry.key_ptr.slot, entry.value_ptr.*);
                const line = self.scratchFmt("slot {s}", .{
                    self.scratchFullU256(entry.key_ptr.slot),
                }) catch "slot";
                drawSegments(win, 1, row, &.{seg(line, if (changed) style_changed() else style_muted())});
                row += 1;
                if (row >= win.height) break;
                const value_line = self.scratchFmt("  value {s}", .{
                    self.scratchShortU256(entry.value_ptr.*),
                }) catch "value";
                drawSegments(win, 1, row, &.{seg(value_line, if (changed) style_changed() else style_text())});
                row += 1;
                count += 1;
            }
        } else {
            var it = self.session.evm.storage.storage.iterator();
            while (it.next()) |entry| {
                if (!std.mem.eql(u8, entry.key_ptr.address[0..], frame.address.bytes[0..])) continue;
                if (row >= win.height) break;
                const changed = self.storageSlotChanged(false, entry.key_ptr.slot, entry.value_ptr.*);
                const line = self.scratchFmt("slot {s}", .{
                    self.scratchFullU256(entry.key_ptr.slot),
                }) catch "slot";
                drawSegments(win, 1, row, &.{seg(line, if (changed) style_changed() else style_muted())});
                row += 1;
                if (row >= win.height) break;
                const value_line = self.scratchFmt("  value {s}", .{
                    self.scratchShortU256(entry.value_ptr.*),
                }) catch "value";
                drawSegments(win, 1, row, &.{seg(value_line, if (changed) style_changed() else style_text())});
                row += 1;
                count += 1;
            }
        }

        if (count == 0) {
            drawSegments(win, 1, 2, &.{seg(if (transient) "no transient slots for current contract" else "no storage slots for current contract", style_hint())});
        }
    }

    fn storageSlotChanged(self: *Ui, transient: bool, slot: u256, value: u256) bool {
        if (transient) {
            if (self.previous_snapshot.tstore_count != self.countStorageSlots(true)) return true;
            var i: usize = 0;
            while (i < self.previous_snapshot.tstore_top_count) : (i += 1) {
                if (self.previous_snapshot.tstore_top_slots[i] == slot) {
                    return self.previous_snapshot.tstore_top_values[i] != value;
                }
            }
            return true;
        }

        if (self.previous_snapshot.storage_count != self.countStorageSlots(false)) return true;
        var i: usize = 0;
        while (i < self.previous_snapshot.storage_top_count) : (i += 1) {
            if (self.previous_snapshot.storage_top_slots[i] == slot) {
                return self.previous_snapshot.storage_top_values[i] != value;
            }
        }
        return true;
    }

    fn countStorageSlots(self: *Ui, transient: bool) usize {
        const frame = self.selectedFrame() orelse return 0;
        var count: usize = 0;
        if (transient) {
            var it = self.session.evm.storage.transient.iterator();
            while (it.next()) |entry| {
                if (std.mem.eql(u8, entry.key_ptr.address[0..], frame.address.bytes[0..])) count += 1;
            }
        } else {
            var it = self.session.evm.storage.storage.iterator();
            while (it.next()) |entry| {
                if (std.mem.eql(u8, entry.key_ptr.address[0..], frame.address.bytes[0..])) count += 1;
            }
        }
        return count;
    }

    fn drawCalldataTab(self: *Ui, win: Window) void {
        const calldata = if (self.selectedFrame()) |frame| frame.calldata else self.seed.calldata;
        if (calldata.len == 0) {
            drawSegments(win, 1, 1, &.{seg("calldata is empty", style_hint())});
            return;
        }

        var row: u16 = 0;
        var offset: usize = 0;
        while (offset < calldata.len and row < win.height) : ({
            offset += 16;
            row += 1;
        }) {
            const end = @min(offset + 16, calldata.len);
            var hex_buf: [128]u8 = undefined;
            var stream = std.io.fixedBufferStream(&hex_buf);
            const writer = stream.writer();
            for (calldata[offset..end], 0..) |byte, i| {
                if (i != 0) writer.writeByte(' ') catch break;
                writer.print("{X:0>2}", .{byte}) catch break;
            }
            const line = stream.getWritten();
            const text = self.scratchFmt("0x{X:0>4}  {s}", .{ offset, line }) catch "calldata";
            drawSegments(win, 1, row, &.{seg(text, style_text())});
        }
    }

    fn scratchFmt(self: *Ui, comptime fmt: []const u8, args: anytype) ![]const u8 {
        const start = self.render_scratch.items.len;
        try self.render_scratch.writer(self.allocator).print(fmt, args);
        return self.render_scratch.items[start..];
    }

    fn scratchShortU256(self: *Ui, value: u256) []const u8 {
        const start = self.render_scratch.items.len;
        self.render_scratch.writer(self.allocator).print("{x}", .{value}) catch return "0x?";
        const full = self.render_scratch.items[start..];
        if (full.len <= 14) {
            const prefix_start = self.render_scratch.items.len;
            self.render_scratch.writer(self.allocator).print("0x{s}", .{full}) catch return "0x?";
            return self.render_scratch.items[prefix_start..];
        }
        const short_start = self.render_scratch.items.len;
        self.render_scratch.writer(self.allocator).print("0x{s}..{s}", .{ full[0..6], full[full.len - 4 ..] }) catch return "0x?";
        return self.render_scratch.items[short_start..];
    }

    fn scratchFullU256(self: *Ui, value: u256) []const u8 {
        const start = self.render_scratch.items.len;
        self.render_scratch.writer(self.allocator).print("0x{x}", .{value}) catch return "0x?";
        return self.render_scratch.items[start..];
    }

    fn scratchAddress(self: *Ui, address: primitives.Address) []const u8 {
        const start = self.render_scratch.items.len;
        self.render_scratch.writer(self.allocator).print("0x{x}", .{address.bytes}) catch return "0x?";
        return self.render_scratch.items[start..];
    }

    fn bindingLabel(self: *Ui, binding: *const DebugInfo.VisibleBinding) []const u8 {
        if (std.mem.eql(u8, binding.runtime_kind, "ssa")) return binding.kind;
        if (std.mem.eql(u8, binding.kind, "field")) return binding.runtime_kind;
        const start = self.render_scratch.items.len;
        self.render_scratch.writer(self.allocator).print("{s}/{s}", .{ binding.kind, binding.runtime_kind }) catch return binding.kind;
        return self.render_scratch.items[start..];
    }

    const ResolvedBindingValue = union(enum) {
        numeric: u256,
        text: []const u8,
    };

    fn numericBindingValue(self: *Ui, binding: *const DebugInfo.VisibleBinding) !?u256 {
        return try self.session.debugger.getVisibleBindingValueByName(self.allocator, binding.name);
    }

    fn resolvedBindingValue(self: *Ui, binding: *const DebugInfo.VisibleBinding) !?ResolvedBindingValue {
        if (try self.numericBindingValue(binding)) |value| {
            return .{ .numeric = value };
        }
        return self.resolveAbiParamValue(binding);
    }

    fn resolveAbiParamValue(self: *Ui, binding: *const DebugInfo.VisibleBinding) ?ResolvedBindingValue {
        if (!std.mem.eql(u8, binding.kind, "param")) return null;
        if (!std.mem.eql(u8, binding.runtime_kind, "ssa")) return null;

        const abi_doc = self.abi_doc orelse return null;
        const frame = self.session.evm.getCurrentFrame() orelse return null;
        if (frame.calldata.len < 4) return null;

        var selector: [4]u8 = undefined;
        @memcpy(selector[0..], frame.calldata[0..4]);
        const callable = abi_doc.findCallableBySelector(selector) orelse return null;
        const input_index = abi_doc.findInputIndex(callable, binding.name) orelse return null;
        const wire_type = abi_doc.findInputWireType(callable, binding.name) orelse return null;

        const start = 4 + input_index * 32;
        const end = start + 32;
        if (frame.calldata.len < end) return null;
        return self.decodeAbiWordValue(wire_type, frame.calldata[start..end]);
    }

    fn decodeAbiWordValue(self: *Ui, wire_type: []const u8, word: []const u8) ?ResolvedBindingValue {
        if (word.len != 32) return null;

        if (std.mem.eql(u8, wire_type, "address")) {
            const start = self.render_scratch.items.len;
            self.render_scratch.writer(self.allocator).print("0x{x}", .{word[12..32]}) catch return null;
            return .{ .text = self.render_scratch.items[start..] };
        }

        if (std.mem.eql(u8, wire_type, "bool")) {
            return .{ .text = if (word[31] == 0) "false" else "true" };
        }

        var value: u256 = 0;
        for (word) |byte| value = (value << 8) | byte;
        return .{ .numeric = value };
    }

    fn selectedFrame(self: *Ui) ?*Frame {
        const frames = self.session.evm.frames.items;
        if (frames.len == 0) return null;
        if (self.selected_frame_index >= frames.len) self.selected_frame_index = 0;
        return &frames[frames.len - 1 - self.selected_frame_index];
    }

    fn opcodeNameForFrame(self: *Ui, frame: *Frame) []const u8 {
        _ = self;
        const opcode = frame.bytecode.getOpcode(frame.pc) orelse return "???";
        return opcode_utils.getOpName(opcode);
    }

    fn scratchCheckpointSummary(self: *Ui) ![]const u8 {
        const start = self.render_scratch.items.len;
        try self.render_scratch.writer(self.allocator).writeAll("checkpoints:");
        for (self.checkpoints.items, 0..) |cp, i| {
            if (i == 0) {
                try self.render_scratch.writer(self.allocator).print(" #{d}@{d}", .{ cp.id, cp.step_index });
            } else {
                try self.render_scratch.writer(self.allocator).print(", #{d}@{d}", .{ cp.id, cp.step_index });
            }
        }
        return self.render_scratch.items[start..];
    }

    fn scratchCheckpointSummaryCompact(self: *Ui) ![]const u8 {
        const start = self.render_scratch.items.len;
        if (self.checkpoints.items.len == 0) {
            try self.render_scratch.writer(self.allocator).writeAll("checkpoints: none");
            return self.render_scratch.items[start..];
        }
        try self.render_scratch.writer(self.allocator).print("checkpoints: {d}", .{self.checkpoints.items.len});
        const tail = @min(self.checkpoints.items.len, 3);
        var i = self.checkpoints.items.len - tail;
        while (i < self.checkpoints.items.len) : (i += 1) {
            const cp = self.checkpoints.items[i];
            try self.render_scratch.writer(self.allocator).print("  #{d}@{d}", .{ cp.id, cp.step_index });
        }
        return self.render_scratch.items[start..];
    }

    fn scratchBreakpointsSummary(self: *Ui) ![]const u8 {
        const start = self.render_scratch.items.len;
        if (self.breakpoints.items.len == 0) {
            try self.render_scratch.writer(self.allocator).writeAll("breakpoints: none");
            return self.render_scratch.items[start..];
        }
        try self.render_scratch.writer(self.allocator).print("breakpoints: {d}", .{self.breakpoints.items.len});
        const tail = @min(self.breakpoints.items.len, 4);
        var i = self.breakpoints.items.len - tail;
        while (i < self.breakpoints.items.len) : (i += 1) {
            try self.render_scratch.writer(self.allocator).print("  L{d}", .{self.breakpoints.items[i]});
        }
        return self.render_scratch.items[start..];
    }

    fn drawSourceGutterMarkers(self: *Ui, win: Window, current_line: u32) void {
        var visible_row: u16 = 0;
        while (visible_row < win.height) : (visible_row += 1) {
            const line = self.scroll_line + visible_row;
            if (line > self.session.debugger.totalSourceLines()) break;
            const has_break = self.hasBreakpointLine(line);
            const is_current = line == current_line;
            const kind = self.statementKindForLine(line);
            const marker = if (has_break and is_current)
                ">"
            else if (has_break)
                "*"
            else if (kind == .runtime_guard)
                "!"
            else if (kind != null)
                "."
            else
                " ";
            const style = if (has_break)
                style_changed()
            else if (kind == .runtime_guard)
                style_guard()
            else if (kind != null)
                style_muted()
            else
                style_muted();
            drawSegments(win, 0, visible_row, &.{seg(marker, style)});
        }
    }

    fn drawSirRangeMarkers(self: *Ui, win: Window, mapping: MappingWindow) void {
        if (mapping.sir_start == 0 or mapping.sir_end == 0) return;
        const scroll_y: u32 = @intCast(self.sir_view.scroll_view.scroll.y);
        var visible_row: u16 = 0;
        while (visible_row < win.height) : (visible_row += 1) {
            const line = scroll_y + visible_row + 1;
            if (line < mapping.sir_start or line > mapping.sir_end) continue;
            const marker = if (mapping.sir_start == mapping.sir_end)
                ">"
            else if (line == mapping.sir_start)
                ">"
            else if (line == mapping.sir_end)
                "<"
            else
                "|";
            const style = if (line == mapping.sir_start or line == mapping.sir_end)
                style_emphasis()
            else
                style_guard();
            drawSegments(win, 0, visible_row, &.{seg(marker, style)});
        }
    }

    fn statementKindForLine(self: *Ui, line: u32) ?ora_evm.SourceMap.StatementKind {
        return self.session.debugger.src_map.getStatementKindForLine(self.config.source_path, line);
    }

    fn hasBreakpointLine(self: *Ui, line: u32) bool {
        for (self.breakpoints.items) |bp| {
            if (bp == line) return true;
        }
        return false;
    }

    fn nextOpcodePc(self: *Ui, bytecode: anytype, pc: u32) ?u32 {
        _ = self;
        const opcode = bytecode.getOpcode(pc) orelse return null;
        const step: u32 = if (opcode >= 0x60 and opcode <= 0x7f)
            1 + @as(u32, opcode - 0x5f)
        else
            1;
        const next = pc + step;
        if (next >= bytecode.len()) return null;
        return next;
    }

    fn writeSession(self: *Ui, path: []const u8) !void {
        var step_names = try self.allocator.alloc([]const u8, self.step_history.items.len);
        defer self.allocator.free(step_names);
        for (self.step_history.items, 0..) |mode, i| {
            step_names[i] = stepModeName(mode);
        }
        const breakpoints = try self.allocator.dupe(u32, self.breakpoints.items);
        defer self.allocator.free(breakpoints);
        var saved_checkpoints = try self.allocator.alloc(SavedSession.SavedCheckpoint, self.checkpoints.items.len);
        defer self.allocator.free(saved_checkpoints);
        for (self.checkpoints.items, 0..) |cp, i| {
            saved_checkpoints[i] = .{
                .id = cp.id,
                .step_index = cp.step_index,
                .scroll_line = cp.scroll_line,
                .focus_line = cp.focus_line,
                .active_evm_tab = tabName(cp.active_evm_tab),
            };
        }

        const calldata_hex = try std.fmt.allocPrint(self.allocator, "{x}", .{self.seed.calldata});
        defer self.allocator.free(calldata_hex);

        const session = SavedSession{
            .bytecode_path = self.config.bytecode_path,
            .source_map_path = self.config.source_map_path,
            .source_path = self.config.source_path,
            .sir_path = self.config.sir_path,
            .debug_info_path = self.config.debug_info_path,
            .abi_path = self.config.abi_path,
            .calldata_hex = calldata_hex,
            .scroll_line = self.scroll_line,
            .sir_scroll_line = self.sir_scroll_line,
            .sir_follow = self.sir_follow,
            .focus_line = self.focus_line,
            .active_evm_tab = tabName(self.active_evm_tab),
            .step_history = step_names,
            .breakpoints = breakpoints,
            .checkpoints = saved_checkpoints,
        };

        var json_buf: std.ArrayList(u8) = .{};
        defer json_buf.deinit(self.allocator);
        var writer = json_buf.writer(self.allocator);
        var adapter_buf: [256]u8 = undefined;
        var adapter = writer.adaptToNewApi(&adapter_buf);
        var jw: std.json.Stringify = .{
            .writer = &adapter.new_interface,
            .options = .{ .whitespace = .indent_2 },
        };
        try jw.write(session);
        if (adapter.err) |err| return err;
        try std.fs.cwd().writeFile(.{ .sub_path = path, .data = json_buf.items });
    }

    fn loadSession(self: *Ui, path: []const u8) !void {
        const json_bytes = try std.fs.cwd().readFileAlloc(self.allocator, path, 16 * 1024 * 1024);
        defer self.allocator.free(json_bytes);

        const parsed = try std.json.parseFromSlice(SavedSession, self.allocator, json_bytes, .{
            .ignore_unknown_fields = true,
        });
        defer parsed.deinit();

        var new_config = AppConfig{
            .bytecode_path = try self.allocator.dupe(u8, parsed.value.bytecode_path),
            .source_map_path = try self.allocator.dupe(u8, parsed.value.source_map_path),
            .source_path = try self.allocator.dupe(u8, parsed.value.source_path),
            .sir_path = if (parsed.value.sir_path) |p| try self.allocator.dupe(u8, p) else null,
            .debug_info_path = if (parsed.value.debug_info_path) |p| try self.allocator.dupe(u8, p) else null,
            .abi_path = if (parsed.value.abi_path) |p| try self.allocator.dupe(u8, p) else null,
            .init_calldata = try self.allocator.dupe(u8, &.{}),
            .init_calldata_fallback = try self.allocator.dupe(u8, &.{}),
            .calldata = try decodeHexAlloc(self.allocator, parsed.value.calldata_hex),
        };
        errdefer new_config.deinit(self.allocator);

        var new_seed = try loadSeedFromConfig(self.allocator, &new_config);
        errdefer new_seed.deinit(self.allocator);

        self.config.deinit(self.allocator);
        self.session.deinit();
        self.seed.deinit(self.allocator);
        self.clearPreviousBindingsSnapshot();
        self.step_history.clearRetainingCapacity();

        self.config = new_config;
        self.seed = new_seed;
        try Session.init(&self.session, self.allocator, &self.seed);

        self.command_mode = false;
        self.command_buffer.clearRetainingCapacity();
        self.active_evm_tab = parseTabName(parsed.value.active_evm_tab) orelse .stack;
        self.breakpoints.clearRetainingCapacity();
        for (parsed.value.breakpoints) |line| {
            try self.breakpoints.append(self.allocator, line);
        }
        self.checkpoints.clearRetainingCapacity();
        self.next_checkpoint_id = 1;
        for (parsed.value.checkpoints) |cp| {
            try self.checkpoints.append(self.allocator, .{
                .id = cp.id,
                .step_index = cp.step_index,
                .scroll_line = cp.scroll_line,
                .focus_line = cp.focus_line,
                .active_evm_tab = parseTabName(cp.active_evm_tab) orelse .stack,
            });
            if (cp.id >= self.next_checkpoint_id) self.next_checkpoint_id = cp.id + 1;
        }

        self.source_buffer.deinit(self.allocator);
        self.source_buffer = .{};
        try self.source_buffer.update(self.allocator, .{ .bytes = self.seed.source_text });

        try self.primeInitialStop();
        try self.applyBreakpoints();
        for (parsed.value.step_history) |name| {
            const mode = parseStepMode(name) orelse continue;
            self.runStep(mode, true);
            if (std.mem.eql(u8, self.status, "execution_error")) break;
        }
        self.scroll_line = parsed.value.scroll_line;
        self.sir_scroll_line = parsed.value.sir_scroll_line;
        self.sir_follow = parsed.value.sir_follow;
        self.focus_line = parsed.value.focus_line;
        self.previous_snapshot = self.captureSnapshot();
        try self.refreshPreviousBindingsSnapshot();
        if (self.focus_line == null) self.syncFocusFromDebugger();
    }
};

fn seg(text: []const u8, style: Style) Segment {
    return .{ .text = text, .style = style };
}

fn drawSegments(win: Window, col: u16, row: u16, segments: []const Segment) void {
    _ = win.print(segments, .{ .col_offset = col, .row_offset = row, .wrap = .none });
}

fn style_header_title() Style {
    return .{ .fg = Color.rgbFromUint(0x191F24), .bg = Color.rgbFromUint(0xE8EFF6), .bold = true };
}

fn style_header_meta() Style {
    return .{ .fg = Color.rgbFromUint(0xD3DBE3), .bg = Color.rgbFromUint(0x1A1D21) };
}

fn style_footer_note() Style {
    return .{ .fg = Color.rgbFromUint(0xA8B0B8), .bg = Color.rgbFromUint(0x1A1D21) };
}

fn style_border() Style {
    return .{ .fg = Color.rgbFromUint(0x78838E) };
}

fn style_title() Style {
    return .{ .fg = Color.rgbFromUint(0xDEE4EB), .bold = true };
}

fn style_text() Style {
    return .{ .fg = Color.rgbFromUint(0xD6DCE2) };
}

fn style_emphasis() Style {
    return .{ .fg = Color.rgbFromUint(0xF5F7FA), .bold = true };
}

fn style_changed() Style {
    return .{ .fg = Color.rgbFromUint(0xFFD666), .bold = true };
}

fn style_guard() Style {
    return .{ .fg = Color.rgbFromUint(0xFFAD66), .bold = true };
}

fn style_hint() Style {
    return .{ .fg = Color.rgbFromUint(0x969EA6), .italic = true };
}

fn style_muted() Style {
    return .{ .fg = Color.rgbFromUint(0xC0C7CF), .dim = true };
}

fn style_tab_active() Style {
    return .{ .fg = Color.rgbFromUint(0xEEF2F8), .bold = true, .ul_style = .single };
}

fn style_tab_inactive() Style {
    return .{ .fg = Color.rgbFromUint(0xA0AAB4) };
}

fn style_error() Style {
    return .{ .fg = Color.rgbFromUint(0xFF6B6B), .bold = true };
}

fn style_command_bg() Style {
    return .{ .fg = Color.rgbFromUint(0xDDE5ED), .bg = Color.rgbFromUint(0x111417) };
}

fn style_command() Style {
    return .{ .fg = Color.rgbFromUint(0xE7EDF4), .bg = Color.rgbFromUint(0x111417), .bold = true };
}

fn tabLabel(tab: EvmTabKind) []const u8 {
    return switch (tab) {
        .stack => "Stack",
        .memory => "Memory",
        .storage => "Storage",
        .tstore => "TStore",
        .calldata => "Calldata",
    };
}

fn tabName(tab: EvmTabKind) []const u8 {
    return switch (tab) {
        .stack => "stack",
        .memory => "memory",
        .storage => "storage",
        .tstore => "tstore",
        .calldata => "calldata",
    };
}

fn parseTabName(name: []const u8) ?EvmTabKind {
    if (std.mem.eql(u8, name, "stack")) return .stack;
    if (std.mem.eql(u8, name, "memory")) return .memory;
    if (std.mem.eql(u8, name, "storage")) return .storage;
    if (std.mem.eql(u8, name, "tstore")) return .tstore;
    if (std.mem.eql(u8, name, "calldata")) return .calldata;
    return null;
}

fn stepModeName(mode: StepMode) []const u8 {
    return switch (mode) {
        .in => "in",
        .opcode => "opcode",
        .over => "over",
        .out => "out",
        .continue_ => "continue",
    };
}

fn parseStepMode(name: []const u8) ?StepMode {
    if (std.mem.eql(u8, name, "in")) return .in;
    if (std.mem.eql(u8, name, "opcode")) return .opcode;
    if (std.mem.eql(u8, name, "over")) return .over;
    if (std.mem.eql(u8, name, "out")) return .out;
    if (std.mem.eql(u8, name, "continue")) return .continue_;
    return null;
}

pub fn main() !void {
    runMain() catch |err| {
        var stderr_buffer: [1024]u8 = undefined;
        var stderr_writer = std.fs.File.stderr().writer(&stderr_buffer);
        const stderr = &stderr_writer.interface;
        switch (err) {
            error.DeploymentRevertedWithNoRuntime => {
                try stderr.print("error: constructor/init reverted during deployment\n", .{});
                try stderr.flush();
                std.process.exit(1);
            },
            else => return err,
        }
    };
}

fn runMain() !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const allocator = gpa_state.allocator();

    var config = try parseArgs(allocator);
    var seed = try loadSeedFromConfig(allocator, &config);
    var ui = Ui.init(allocator, config, seed) catch |err| {
        seed.deinit(allocator);
        config.deinit(allocator);
        return err;
    };
    defer ui.deinit();

    var tty_buf: [4096]u8 = undefined;
    var tty = try vaxis.Tty.init(&tty_buf);
    defer tty.deinit();

    var vx = try vaxis.init(allocator, .{});
    defer vx.deinit(allocator, tty.writer());

    var loop: vaxis.Loop(AppEvent) = .{ .tty = &tty, .vaxis = &vx };
    try loop.init();
    try loop.start();
    defer loop.stop();

    try vx.enterAltScreen(tty.writer());
    try vx.queryTerminal(tty.writer(), 1 * std.time.ns_per_s);
    const initial_winsize = try vaxis.Tty.getWinsize(tty.fd);
    try vx.resize(allocator, tty.writer(), initial_winsize);
    ui.render(&vx);
    try vx.render(tty.writer());

    var running = true;
    while (running) {
        const event = loop.nextEvent();
        switch (event) {
            .winsize => |winsize| {
                try vx.resize(allocator, tty.writer(), winsize);
                ui.render(&vx);
                try vx.render(tty.writer());
            },
            .key_press => |key| {
                running = !(try ui.handleKey(key));
                ui.render(&vx);
                try vx.render(tty.writer());
            },
            .paste => |bytes| allocator.free(bytes),
            else => {},
        }
    }
}

fn loadSeedFromConfig(allocator: std.mem.Allocator, config: *const AppConfig) !SessionSeed {
    const bytecode_hex = try std.fs.cwd().readFileAlloc(allocator, config.bytecode_path, 16 * 1024 * 1024);
    defer allocator.free(bytecode_hex);
    const bytecode = try decodeHexAlloc(allocator, bytecode_hex);
    defer allocator.free(bytecode);

    const source_map_json = try std.fs.cwd().readFileAlloc(allocator, config.source_map_path, 16 * 1024 * 1024);
    defer allocator.free(source_map_json);
    var source_map = try SourceMap.loadFromJson(allocator, source_map_json);
    errdefer source_map.deinit();

    const source_text = try std.fs.cwd().readFileAlloc(allocator, config.source_path, 16 * 1024 * 1024);
    errdefer allocator.free(source_text);

    var debug_info_json: ?[]u8 = null;
    errdefer if (debug_info_json) |bytes| allocator.free(bytes);
    if (config.debug_info_path) |path| {
        debug_info_json = try std.fs.cwd().readFileAlloc(allocator, path, 16 * 1024 * 1024);
    }

    var sir_text: ?[]u8 = null;
    errdefer if (sir_text) |bytes| allocator.free(bytes);
    if (config.sir_path) |path| {
        sir_text = std.fs.cwd().readFileAlloc(allocator, path, 16 * 1024 * 1024) catch null;
    }

    const caller = primitives.Address.fromU256(0x100);
    const contract = primitives.Address.fromU256(0x200);

    var evm: Evm = undefined;
    try evm.init(allocator, null, null, null, primitives.ZERO_ADDRESS, 0, null);
    defer evm.deinit();
    try evm.initTransactionState(null);
    try evm.preWarmTransaction(contract);

    const runtime_bytecode = deployRuntimeBytecode(allocator, &evm, caller, contract, bytecode, config.init_calldata) catch |err| blk: {
        if (err != error.DeploymentRevertedWithNoRuntime or config.init_calldata_fallback.len == 0) {
            return err;
        }
        evm.deinit();
        try evm.init(allocator, null, null, null, primitives.ZERO_ADDRESS, 0, null);
        try evm.initTransactionState(null);
        try evm.preWarmTransaction(contract);
        break :blk try deployRuntimeBytecode(allocator, &evm, caller, contract, bytecode, config.init_calldata_fallback);
    };
    errdefer allocator.free(runtime_bytecode);

    if (source_map.runtime_start_pc) |_| {
        const runtime_source_map = try rebaseSourceMapForRuntime(allocator, &source_map, runtime_bytecode);
        source_map.deinit();
        source_map = runtime_source_map;
    }

    return .{
        .runtime_bytecode = runtime_bytecode,
        .source_map = source_map,
        .debug_info_json = debug_info_json,
        .source_text = source_text,
        .source_path = try allocator.dupe(u8, config.source_path),
        .sir_text = sir_text,
        .calldata = try allocator.dupe(u8, config.calldata),
        .caller = caller,
        .contract = contract,
    };
}

fn deployRuntimeBytecode(
    allocator: std.mem.Allocator,
    evm: *Evm,
    caller: primitives.Address,
    contract: primitives.Address,
    deployment_bytecode: []const u8,
    init_calldata: []const u8,
) ![]u8 {
    var stderr_buffer: [1024]u8 = undefined;
    var stderr_writer = std.fs.File.stderr().writer(&stderr_buffer);
    const stderr = &stderr_writer.interface;
    try evm.frames.append(evm.arena.allocator(), try Frame.init(
        evm.arena.allocator(),
        deployment_bytecode,
        5_000_000,
        caller,
        contract,
        0,
        init_calldata,
        @as(*anyopaque, @ptrCast(evm)),
        evm.hardfork,
        false,
    ));

    var steps: usize = 0;
    while (evm.getCurrentFrame()) |frame| {
        if (frame.stopped or frame.reverted) break;
        if (steps >= 200_000) {
            try stderr.print("debug-tui: deployment did not halt after {d} steps (pc={d})\n", .{
                steps,
                frame.pc,
            });
            try stderr.flush();
            return error.DeploymentDidNotHalt;
        }
        try evm.step();
        steps += 1;
    }

    const frame = evm.getCurrentFrame() orelse return try allocator.dupe(u8, deployment_bytecode);
    defer {
        for (evm.frames.items) |*live_frame| {
            live_frame.deinit();
        }
        evm.frames.clearRetainingCapacity();
    }

    if (frame.reverted or frame.output.len == 0) {
        try stderr.print("debug-tui: deployment completed without runtime output (reverted={any}, output_len={d}, steps={d})\n", .{
            frame.reverted,
            frame.output.len,
            steps,
        });
        try stderr.flush();
        return error.DeploymentRevertedWithNoRuntime;
    }
    try stderr.print("debug-tui: deployment completed in {d} steps, runtime_len={d}\n", .{ steps, frame.output.len });
    try stderr.flush();
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
            .sir_line = entry.sir_line,
            .is_statement = entry.is_statement,
            .kind = entry.kind,
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

    var init_signature: ?[]u8 = null;
    defer if (init_signature) |sig| allocator.free(sig);
    var signature: ?[]u8 = null;
    defer if (signature) |sig| allocator.free(sig);

    var init_raw_args: std.ArrayList([]u8) = .{};
    defer {
        for (init_raw_args.items) |item| allocator.free(item);
        init_raw_args.deinit(allocator);
    }

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
        } else if (std.mem.eql(u8, arg, "--init-signature")) {
            i += 1;
            if (i >= args.len) return error.InvalidArguments;
            if (init_signature) |sig| allocator.free(sig);
            init_signature = try allocator.dupe(u8, args[i]);
        } else if (std.mem.eql(u8, arg, "--init-arg")) {
            i += 1;
            if (i >= args.len) return error.InvalidArguments;
            try init_raw_args.append(allocator, try allocator.dupe(u8, args[i]));
        } else if (std.mem.eql(u8, arg, "--init-calldata-hex")) {
            i += 1;
            if (i >= args.len) return error.InvalidArguments;
            allocator.free(config.init_calldata);
            config.init_calldata = try decodeHexAlloc(allocator, args[i]);
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

    if (init_signature) |sig| {
        allocator.free(config.init_calldata);
        allocator.free(config.init_calldata_fallback);
        if (config.abi_path) |abi_path| {
            const abi_bytes = try std.fs.cwd().readFileAlloc(allocator, abi_path, 16 * 1024 * 1024);
            defer allocator.free(abi_bytes);
            config.init_calldata = try encodeConstructorArgsAlloc(allocator, abi_bytes, sig, init_raw_args.items);
        } else {
            config.init_calldata = try encodeConstructorArgsAlloc(allocator, null, sig, init_raw_args.items);
        }
        config.init_calldata_fallback = try encodeSignatureCallDataAlloc(allocator, sig, init_raw_args.items);
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

    if (config.sir_path == null) {
        if (inferSirPath(allocator, config.source_map_path)) |sir_path| {
            if (pathExists(sir_path)) {
                config.sir_path = sir_path;
            } else {
                allocator.free(sir_path);
            }
        } else |_| {}
    }

    return config;
}

fn printUsage() !void {
    var stderr_buffer: [2048]u8 = undefined;
    var stderr_file = std.fs.File.stderr().writer(&stderr_buffer);
    const stderr = &stderr_file.interface;
    try stderr.print(
        \\usage:
        \\  ora-evm-debug-tui <bytecode.hex> <source-map.json> <source.ora> [--debug-info <debug.json>] [--abi <abi.json>] [--init-signature <sig> [--init-arg <value>]...] [--init-calldata-hex <hex>] [--signature <sig> [--arg <value>]...] [--calldata-hex <hex>]
        \\
        \\example:
        \\  zig build install
        \\  ./zig-out/bin/ora-evm-debug-tui <bytecode.hex> <source-map.json> <source.ora> --abi <abi.json> --init-signature 'init(u256)' --init-arg 1000 --signature 'foo(u256)' --arg 1
        \\
    , .{});
    try stderr.flush();
}

fn pathExists(path: []const u8) bool {
    std.fs.cwd().access(path, .{}) catch return false;
    return true;
}

fn inferSirPath(allocator: std.mem.Allocator, source_map_path: []const u8) ![]u8 {
    const dir = std.fs.path.dirname(source_map_path) orelse ".";
    const base = std.fs.path.basename(source_map_path);
    const suffix = ".sourcemap.json";
    if (!std.mem.endsWith(u8, base, suffix)) return error.InvalidArguments;
    const stem = base[0 .. base.len - suffix.len];
    const sir_name = try std.fmt.allocPrint(allocator, "{s}.sir", .{stem});
    defer allocator.free(sir_name);
    return try std.fs.path.join(allocator, &.{ dir, sir_name });
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

    var arg_index: usize = 0;
    if (count > 0) {
        var split = std.mem.splitScalar(u8, type_list, ',');
        while (split.next()) |raw_type| : (arg_index += 1) {
            const type_name = std.mem.trim(u8, raw_type, " \t");
            try encodeAbiWord(type_name, arg_values[arg_index], out[4 + arg_index * 32 ..][0..32]);
        }
    }

    return out;
}

fn encodeConstructorArgsAlloc(
    allocator: std.mem.Allocator,
    abi_json_opt: ?[]const u8,
    signature: []const u8,
    arg_values: []const []const u8,
) ![]u8 {
    _ = abi_json_opt;
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

    const out = try allocator.alloc(u8, count * 32);
    errdefer allocator.free(out);

    var arg_index: usize = 0;
    if (count > 0) {
        var split = std.mem.splitScalar(u8, type_list, ',');
        while (split.next()) |raw_type| : (arg_index += 1) {
            const type_name = std.mem.trim(u8, raw_type, " \t");
            try encodeAbiWord(switchAbiType(type_name), arg_values[arg_index], out[arg_index * 32 ..][0..32]);
        }
    }

    return out;
}

fn encodeAbiCallDataAlloc(
    allocator: std.mem.Allocator,
    abi_json: []const u8,
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

    const selector = try extractAbiSelector(abi_json, signature);
    @memcpy(out[0..4], selector[0..4]);

    var arg_index: usize = 0;
    if (count > 0) {
        var split = std.mem.splitScalar(u8, type_list, ',');
        while (split.next()) |raw_type| : (arg_index += 1) {
            const type_name = std.mem.trim(u8, raw_type, " \t");
            try encodeAbiWord(switchAbiType(type_name), arg_values[arg_index], out[4 + arg_index * 32 ..][0..32]);
        }
    }

    return out;
}

fn extractAbiSelector(abi_json: []const u8, signature: []const u8) ![4]u8 {
    const normalized = try normalizeAbiSignature(std.heap.page_allocator, signature);
    defer std.heap.page_allocator.free(normalized);

    var search_index: usize = 0;
    while (std.mem.indexOfPos(u8, abi_json, search_index, "\"signature\"")) |field_pos| {
        const after_field = std.mem.indexOfPos(u8, abi_json, field_pos, ":") orelse return error.InvalidAbi;
        const sig_start_quote = std.mem.indexOfPos(u8, abi_json, after_field, "\"") orelse return error.InvalidAbi;
        const sig_end_quote = std.mem.indexOfPos(u8, abi_json, sig_start_quote + 1, "\"") orelse return error.InvalidAbi;
        const abi_sig = abi_json[sig_start_quote + 1 .. sig_end_quote];
        if (!std.mem.eql(u8, abi_sig, normalized)) {
            search_index = sig_end_quote + 1;
            continue;
        }

        const selector_key = std.mem.indexOfPos(u8, abi_json, sig_end_quote, "\"selector\"") orelse return error.InvalidAbi;
        const selector_colon = std.mem.indexOfPos(u8, abi_json, selector_key, ":") orelse return error.InvalidAbi;
        const selector_quote = std.mem.indexOfPos(u8, abi_json, selector_colon, "\"") orelse return error.InvalidAbi;
        const selector_end = std.mem.indexOfPos(u8, abi_json, selector_quote + 1, "\"") orelse return error.InvalidAbi;
        const selector_text = abi_json[selector_quote + 1 .. selector_end];
        if (selector_text.len != 10 or !std.mem.startsWith(u8, selector_text, "0x")) return error.InvalidAbi;

        var selector: [4]u8 = undefined;
        _ = try std.fmt.hexToBytes(&selector, selector_text[2..]);
        return selector;
    }

    return error.FunctionNotFound;
}

fn normalizeAbiSignature(allocator: std.mem.Allocator, signature: []const u8) ![]u8 {
    const open = std.mem.indexOfScalar(u8, signature, '(') orelse return error.InvalidSignature;
    const close = std.mem.lastIndexOfScalar(u8, signature, ')') orelse return error.InvalidSignature;
    if (close < open) return error.InvalidSignature;

    const name = std.mem.trim(u8, signature[0..open], " \t");
    const raw_types = signature[open + 1 .. close];

    var out = std.ArrayList(u8){};
    errdefer out.deinit(allocator);

    try out.appendSlice(allocator, name);
    try out.append(allocator, '(');

    var first = true;
    var split = std.mem.splitScalar(u8, raw_types, ',');
    while (split.next()) |raw_type| {
        const trimmed = std.mem.trim(u8, raw_type, " \t");
        if (trimmed.len == 0) continue;
        if (!first) try out.append(allocator, ',');
        first = false;
        try out.appendSlice(allocator, switchAbiType(trimmed));
    }

    try out.append(allocator, ')');
    return out.toOwnedSlice(allocator);
}

fn switchAbiType(type_name: []const u8) []const u8 {
    if (std.mem.eql(u8, type_name, "u256")) return "uint256";
    if (std.mem.eql(u8, type_name, "u128")) return "uint128";
    if (std.mem.eql(u8, type_name, "u64")) return "uint64";
    if (std.mem.eql(u8, type_name, "u32")) return "uint32";
    if (std.mem.eql(u8, type_name, "u16")) return "uint16";
    if (std.mem.eql(u8, type_name, "u8")) return "uint8";
    if (std.mem.eql(u8, type_name, "i256")) return "int256";
    if (std.mem.eql(u8, type_name, "i128")) return "int128";
    if (std.mem.eql(u8, type_name, "i64")) return "int64";
    if (std.mem.eql(u8, type_name, "i32")) return "int32";
    if (std.mem.eql(u8, type_name, "i16")) return "int16";
    if (std.mem.eql(u8, type_name, "i8")) return "int8";
    return type_name;
}

fn encodeAbiWord(type_name: []const u8, value_text: []const u8, out_word: []u8) !void {
    if (out_word.len != 32) return error.InvalidAbiType;
    @memset(out_word, 0);

    if (std.mem.eql(u8, type_name, "bool")) {
        const is_true = std.mem.eql(u8, value_text, "true") or std.mem.eql(u8, value_text, "1");
        const is_false = std.mem.eql(u8, value_text, "false") or std.mem.eql(u8, value_text, "0");
        if (!is_true and !is_false) return error.InvalidBoolean;
        out_word[31] = if (is_true) 1 else 0;
        return;
    }

    if (std.mem.eql(u8, type_name, "address")) {
        const text = if (std.mem.startsWith(u8, value_text, "0x")) value_text[2..] else value_text;
        if (text.len > 40) return error.InvalidAddress;
        var buf: [20]u8 = [_]u8{0} ** 20;
        const start = 20 - text.len / 2;
        _ = try std.fmt.hexToBytes(buf[start..], text);
        @memcpy(out_word[12..32], &buf);
        return;
    }

    const value = try std.fmt.parseUnsigned(u256, value_text, 0);
    var tmp = value;
    var i: usize = 0;
    while (i < 32) : (i += 1) {
        out_word[31 - i] = @truncate(tmp & 0xff);
        tmp >>= 8;
    }
}

fn decodeHexAlloc(allocator: std.mem.Allocator, text: []const u8) ![]u8 {
    const trimmed = std.mem.trim(u8, text, " \t\r\n");
    const hex = if (std.mem.startsWith(u8, trimmed, "0x")) trimmed[2..] else trimmed;
    if (hex.len % 2 != 0) return error.InvalidHex;

    const out = try allocator.alloc(u8, hex.len / 2);
    errdefer allocator.free(out);
    _ = try std.fmt.hexToBytes(out, hex);
    return out;
}
