const std = @import("std");
const builtin = @import("builtin");
const vaxis = @import("vaxis");
const primitives = @import("voltaire");
const ora_evm = @import("ora_evm");
const opcode_utils = ora_evm.opcode;

const Evm = ora_evm.Evm(.{});
const Frame = ora_evm.Frame(.{});
const Debugger = ora_evm.Debugger(.{});
const SourceMap = ora_evm.SourceMap;
const DebugInfo = ora_evm.DebugInfo;
const SessionHelpers = ora_evm.DebugSession(.{});
const AbiDoc = ora_evm.debug_abi.AbiDoc;
const writeAbiWord = ora_evm.debug_abi.writeAbiWord;
const readU256BE = ora_evm.debug_abi.readU256BE;
const writeU256BE = ora_evm.debug_abi.writeU256BE;
const ParsedBreakpoint = ora_evm.debug_breakpoint.ParsedBreakpoint;
const parseBreakpointArgsImpl = ora_evm.debug_breakpoint.parse;

const Style = vaxis.Style;
const Color = vaxis.Color;
const Segment = vaxis.Segment;
const Window = vaxis.Window;
const ascii_border_glyphs = @import("debug_tui_draw.zig").ascii_border_glyphs;

const AppEvent = union(enum) {
    key_press: vaxis.Key,
    winsize: vaxis.Winsize,
    focus_in,
    focus_out,
    paste: []const u8,
};

/// Pairs an external contract address with the path to its
/// `.abi.json`. Multiple `--abi <addr>=<path>` invocations on the
/// command line each produce one of these. The TUI loads them into
/// a hashmap so per-frame ABI lookup can decode external callees.
const SecondaryAbiSpec = struct {
    address: [20]u8,
    path: []u8,

    fn deinit(self: *SecondaryAbiSpec, allocator: std.mem.Allocator) void {
        allocator.free(self.path);
    }
};

pub const AppConfig = struct {
    bytecode_path: []u8,
    source_map_path: []u8,
    source_path: []u8,
    sir_path: ?[]u8 = null,
    debug_info_path: ?[]u8 = null,
    abi_path: ?[]u8 = null,
    secondary_abi_specs: std.ArrayList(SecondaryAbiSpec) = .{},
    init_calldata: []u8 = &.{},
    init_calldata_fallback: []u8 = &.{},
    calldata: []u8 = &.{},
    limits: ora_evm.DebugLimits = .{},

    pub fn deinit(self: *AppConfig, allocator: std.mem.Allocator) void {
        allocator.free(self.bytecode_path);
        allocator.free(self.source_map_path);
        allocator.free(self.source_path);
        if (self.sir_path) |path| allocator.free(path);
        if (self.debug_info_path) |path| allocator.free(path);
        if (self.abi_path) |path| allocator.free(path);
        for (self.secondary_abi_specs.items) |*spec| spec.deinit(allocator);
        self.secondary_abi_specs.deinit(allocator);
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

pub const EvmTabKind = enum { stack, memory, storage, tstore, calldata };
pub const StepMode = enum { in, opcode, over, out, continue_ };

// SavedSession + writeSession/loadSession/exportTrace live in
// debug_tui_session.zig; the methods on `Ui` below are 1-line
// delegators. Keep imports tight: we don't need to surface
// SavedSession at this scope.

pub const SessionSeed = struct {
    runtime_bytecode: []u8,
    source_map: SourceMap,
    debug_info_json: ?[]u8 = null,
    source_text: []u8,
    source_path: []u8,
    sir_text: ?[]u8 = null,
    calldata: []u8,
    caller: primitives.Address,
    contract: primitives.Address,
    limits: ora_evm.DebugLimits = .{},

    pub fn deinit(self: *SessionSeed, allocator: std.mem.Allocator) void {
        allocator.free(self.runtime_bytecode);
        self.source_map.deinit();
        if (self.debug_info_json) |bytes| allocator.free(bytes);
        allocator.free(self.source_text);
        allocator.free(self.source_path);
        if (self.sir_text) |text| allocator.free(text);
        allocator.free(self.calldata);
    }
};

pub const Session = struct {
    evm: Evm,
    debugger: Debugger,

    pub fn init(self: *Session, allocator: std.mem.Allocator, seed: *const SessionSeed) !void {
        try self.evm.init(allocator, null, null, ora_evm.deterministicBlockContext(), primitives.ZERO_ADDRESS, 0, null);
        errdefer self.evm.deinit();
        try self.evm.initTransactionState(null);
        try self.evm.preWarmTransaction(seed.contract);

        try self.evm.frames.append(self.evm.arena.allocator(), try Frame.init(
            self.evm.arena.allocator(),
            seed.runtime_bytecode,
            seed.limits.gas_limit,
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
        self.debugger.max_steps = seed.limits.max_steps;
    }

    pub fn deinit(self: *Session) void {
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

const OverlayMode = ora_evm.debug_overlay.OverlayMode;

/// A user-set breakpoint. The debugger core only knows about lines (it
/// halts on every hit); the TUI layers a side-effect-free predicate
/// (`condition`) and an optional N-th-hit target on top, transparently
/// resuming when neither gate is satisfied.
pub const Breakpoint = struct {
    line: u32,
    condition: ?[]u8 = null,
    hit_target: ?u32 = null,
    hit_count: u32 = 0,

    pub fn deinit(self: *Breakpoint, allocator: std.mem.Allocator) void {
        if (self.condition) |c| allocator.free(c);
    }
};

const MappingWindow = struct {
    statement_id: ?u32 = null,
    execution_region_id: ?u32 = null,
    statement_run_index: ?u32 = null,
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

const lock_prefix: u256 = (@as(u256, 1) << 255);

/// Safety cap on the number of transparent resumes runStep will issue
/// when a conditional / hit-count breakpoint keeps short-circuiting.
/// This is only a guard against malformed user predicates (e.g.
/// `when false`); under normal use the loop terminates within a
/// handful of iterations.
const kMaxConditionalBreakpointGatingIters: u32 = 1_000_000;

pub const Ui = struct {
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
    breakpoints: std.ArrayList(Breakpoint) = .{},
    checkpoints: std.ArrayList(Checkpoint) = .{},
    next_checkpoint_id: u32 = 1,
    selected_frame_index: usize = 0,
    overlay_mode: OverlayMode = .none,
    /// External-contract ABIs keyed by 20-byte address. Populated
    /// from `--abi <hex-address>=<path>` CLI args; consulted by
    /// `abiDocForFrame` after the primary check fails. Empty when no
    /// secondary ABIs were provided.
    secondary_abi: std.AutoHashMap([20]u8, AbiDoc),

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
            .secondary_abi = std.AutoHashMap([20]u8, AbiDoc).init(allocator),
        };
        errdefer self.secondary_abi.deinit();
        errdefer if (self.abi_doc) |*doc| doc.deinit();

        // Load secondary ABIs. A failure mid-loop frees what was
        // already loaded so we don't leak.
        for (config.secondary_abi_specs.items) |spec| {
            const doc = AbiDoc.loadFromPath(allocator, spec.path) catch |err| {
                var it = self.secondary_abi.valueIterator();
                while (it.next()) |loaded| loaded.deinit();
                return err;
            };
            // Last spec wins on duplicate addresses; free the prior
            // one before overwriting.
            if (self.secondary_abi.fetchRemove(spec.address)) |kv| {
                var prev = kv.value;
                prev.deinit();
            }
            try self.secondary_abi.put(spec.address, doc);
        }

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
        var sec_it = self.secondary_abi.valueIterator();
        while (sec_it.next()) |doc| doc.deinit();
        self.secondary_abi.deinit();
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
        for (self.breakpoints.items) |*bp| bp.deinit(self.allocator);
        self.breakpoints.deinit(self.allocator);
        self.checkpoints.deinit(self.allocator);
    }

    /// Action a key triggers. The exact key→action mapping lives in
    /// `kKeymap` below (see KEYBINDINGS.md for the user-facing surface).
    /// Adding a new binding is one new row in the table plus one switch
    /// arm here.
    const KeyAction = enum {
        quit,
        enter_command_mode,
        step_in,
        step_opcode,
        step_over,
        step_out,
        step_continue,
        step_back,
        scroll_source_down,
        scroll_source_up,
        scroll_source_page_down,
        scroll_source_page_up,
        scroll_sir_down,
        scroll_sir_up,
        sir_follow,
        evm_tab_prev,
        evm_tab_next,
        evm_tab_stack,
        evm_tab_memory,
        evm_tab_storage,
        evm_tab_tstore,
        evm_tab_calldata,
        cycle_overlay,
    };

    const KeyBinding = struct {
        key: vaxis.Key,
        action: KeyAction,
    };

    fn keymap() [23]KeyBinding {
        return .{
            .{ .key = .{ .codepoint = 'q' }, .action = .quit },
            .{ .key = .{ .codepoint = 'c', .mods = .{ .ctrl = true } }, .action = .quit },
            .{ .key = .{ .codepoint = ':' }, .action = .enter_command_mode },
            .{ .key = .{ .codepoint = 's' }, .action = .step_in },
            .{ .key = .{ .codepoint = 'x' }, .action = .step_opcode },
            .{ .key = .{ .codepoint = 'n' }, .action = .step_over },
            .{ .key = .{ .codepoint = 'o' }, .action = .step_out },
            .{ .key = .{ .codepoint = 'c' }, .action = .step_continue },
            .{ .key = .{ .codepoint = 'p' }, .action = .step_back },
            .{ .key = .{ .codepoint = 'j' }, .action = .scroll_source_down },
            .{ .key = .{ .codepoint = 'k' }, .action = .scroll_source_up },
            .{ .key = vaxis.Key{ .codepoint = vaxis.Key.down }, .action = .scroll_source_down },
            .{ .key = vaxis.Key{ .codepoint = vaxis.Key.up }, .action = .scroll_source_up },
            .{ .key = vaxis.Key{ .codepoint = vaxis.Key.page_down }, .action = .scroll_source_page_down },
            .{ .key = vaxis.Key{ .codepoint = vaxis.Key.page_up }, .action = .scroll_source_page_up },
            .{ .key = .{ .codepoint = 'J' }, .action = .scroll_sir_down },
            .{ .key = .{ .codepoint = 'K' }, .action = .scroll_sir_up },
            .{ .key = .{ .codepoint = '=' }, .action = .sir_follow },
            .{ .key = .{ .codepoint = '[' }, .action = .evm_tab_prev },
            .{ .key = .{ .codepoint = ']' }, .action = .evm_tab_next },
            .{ .key = .{ .codepoint = '1' }, .action = .evm_tab_stack },
            .{ .key = .{ .codepoint = '2' }, .action = .evm_tab_memory },
            .{ .key = .{ .codepoint = 'O' }, .action = .cycle_overlay },
        };
    }

    // The `handleKey` table fits in 22 rows. Numeric tab keys 3/4/5 are
    // handled directly because keymap() returns a fixed-size array; if a
    // future binding pushes us beyond that we'll grow the array literal.
    fn dispatchKeyAction(self: *Ui, action: KeyAction) bool {
        switch (action) {
            .quit => return true,
            .enter_command_mode => {
                self.command_mode = true;
                self.command_status = "command";
                self.command_buffer.clearRetainingCapacity();
            },
            .step_in => self.runStep(.in, true),
            .step_opcode => self.runStep(.opcode, true),
            .step_over => self.runStep(.over, true),
            .step_out => self.runStep(.out, true),
            .step_continue => self.runStep(.continue_, true),
            .step_back => self.stepBack(),
            .scroll_source_down => self.scrollDown(),
            .scroll_source_up => self.scrollUp(),
            .scroll_source_page_down => self.scrollPage(8),
            .scroll_source_page_up => self.scrollPage(-8),
            .scroll_sir_down => self.scrollSirDown(),
            .scroll_sir_up => self.scrollSirUp(),
            .sir_follow => {
                self.resyncSirView();
                self.command_status = "sir follow";
            },
            .evm_tab_prev => self.prevEvmTab(),
            .evm_tab_next => self.nextEvmTab(),
            .evm_tab_stack => self.active_evm_tab = .stack,
            .evm_tab_memory => self.active_evm_tab = .memory,
            .evm_tab_storage => self.active_evm_tab = .storage,
            .evm_tab_tstore => self.active_evm_tab = .tstore,
            .evm_tab_calldata => self.active_evm_tab = .calldata,
            .cycle_overlay => {
                self.overlay_mode = self.overlay_mode.next();
                self.setCommandStatusFmt("overlay = {s}", .{self.overlay_mode.name()}) catch {
                    self.command_status = "overlay";
                };
            },
        }
        return false;
    }

    fn handleKey(self: *Ui, key: vaxis.Key) !bool {
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
                self.appendCommandLog(executed, self.command_status) catch {
                    // Couldn't record the command in the rolling log; the
                    // command itself ran fine. Surface so the user knows
                    // history is incomplete.
                    self.command_status = "warning: command log truncated (OOM)";
                };
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

        // Keymap dispatch — the canonical binding table. KEYBINDINGS.md is
        // authored from this list; adding a new binding is one row in
        // keymap() plus one switch arm in dispatchKeyAction.
        for (keymap()) |binding| {
            if (key.matches(binding.key.codepoint, binding.key.mods)) {
                return self.dispatchKeyAction(binding.action);
            }
        }

        // Tail bindings that don't fit the fixed-size keymap array.
        if (key.matches('3', .{})) self.active_evm_tab = .storage;
        if (key.matches('4', .{})) self.active_evm_tab = .tstore;
        if (key.matches('5', .{})) self.active_evm_tab = .calldata;
        return false;
    }

    /// Outcome of a single command dispatch.
    const CommandOutcome = enum { ok, quit };

    /// One command-table entry. `name` is the canonical token; `aliases`
    /// are alternates that resolve to the same handler (the first-match
    /// rule is `name | aliases[i]` checked in order).
    /// - `match = .exact` requires `raw == name | aliases[i]`.
    /// - `match = .prefix` matches `name ` or `name<space>` followed by an
    ///   argument tail; `arg` receives the trimmed remainder. Aliases on
    ///   prefix commands match the same way.
    /// - `help` is a one-line summary used to render `:help`.
    const CommandMatch = enum { exact, prefix };
    const CommandHandler = *const fn (*Ui, arg: []const u8) anyerror!CommandOutcome;
    const CommandSpec = struct {
        name: []const u8,
        aliases: []const []const u8 = &.{},
        match: CommandMatch,
        handler: CommandHandler,
        help: []const u8,
    };

    fn cmdQuit(_: *Ui, _: []const u8) anyerror!CommandOutcome {
        return .quit;
    }
    fn cmdHelp(self: *Ui, _: []const u8) anyerror!CommandOutcome {
        try self.describeHelp();
        return .ok;
    }
    fn cmdContinue(self: *Ui, _: []const u8) anyerror!CommandOutcome {
        self.runStep(.continue_, true);
        return .ok;
    }
    fn cmdRun(self: *Ui, _: []const u8) anyerror!CommandOutcome {
        try self.rerunToHistory(0);
        try self.updateCommandStatusForCurrentStop("run");
        return .ok;
    }
    fn cmdStepIn(self: *Ui, _: []const u8) anyerror!CommandOutcome {
        self.runStep(.in, true);
        return .ok;
    }
    fn cmdStepOpcode(self: *Ui, _: []const u8) anyerror!CommandOutcome {
        self.runStep(.opcode, true);
        return .ok;
    }
    fn cmdStepOver(self: *Ui, _: []const u8) anyerror!CommandOutcome {
        self.runStep(.over, true);
        return .ok;
    }
    fn cmdStepOut(self: *Ui, _: []const u8) anyerror!CommandOutcome {
        self.runStep(.out, true);
        return .ok;
    }
    fn cmdStepBack(self: *Ui, _: []const u8) anyerror!CommandOutcome {
        self.stepBack();
        return .ok;
    }
    fn cmdLine(self: *Ui, arg: []const u8) anyerror!CommandOutcome {
        const line = std.fmt.parseUnsigned(u32, arg, 10) catch {
            self.command_status = "invalid line";
            return .ok;
        };
        self.focus_line = line;
        self.centerOnCurrentLine();
        self.command_status = "line";
        return .ok;
    }
    fn cmdLineInfo(self: *Ui, arg: []const u8) anyerror!CommandOutcome {
        const line = std.fmt.parseUnsigned(u32, arg, 10) catch {
            self.command_status = "invalid line";
            return .ok;
        };
        try self.describeLineInfo(line);
        return .ok;
    }
    fn cmdWhere(self: *Ui, _: []const u8) anyerror!CommandOutcome {
        try self.describeCurrentStop();
        return .ok;
    }
    fn cmdOrigin(self: *Ui, _: []const u8) anyerror!CommandOutcome {
        const origin_line = self.currentOriginLine() orelse {
            self.command_status = "no distinct origin line";
            return .ok;
        };
        self.focus_line = origin_line;
        self.centerOnCurrentLine();
        try self.setCommandStatusFmt("origin line {d}", .{origin_line});
        return .ok;
    }
    fn cmdSirLine(self: *Ui, arg: []const u8) anyerror!CommandOutcome {
        const line = std.fmt.parseUnsigned(u32, arg, 10) catch {
            self.command_status = "invalid sir line";
            return .ok;
        };
        self.sir_scroll_line = if (line > 0) line else 1;
        self.sir_follow = false;
        self.command_status = "sirline";
        return .ok;
    }
    fn cmdSirFollow(self: *Ui, _: []const u8) anyerror!CommandOutcome {
        self.resyncSirView();
        self.command_status = "sir follow";
        return .ok;
    }
    fn cmdCheckpoint(self: *Ui, _: []const u8) anyerror!CommandOutcome {
        try self.addCheckpoint();
        return .ok;
    }
    fn cmdCheckpoints(self: *Ui, _: []const u8) anyerror!CommandOutcome {
        try self.describeCheckpoints();
        return .ok;
    }
    fn cmdBacktrace(self: *Ui, _: []const u8) anyerror!CommandOutcome {
        try self.describeBacktrace();
        return .ok;
    }
    fn cmdFrame(self: *Ui, arg: []const u8) anyerror!CommandOutcome {
        const idx = std.fmt.parseUnsigned(usize, arg, 10) catch {
            self.command_status = "invalid frame index";
            return .ok;
        };
        try self.selectFrame(idx);
        return .ok;
    }
    fn cmdRestart(self: *Ui, arg: []const u8) anyerror!CommandOutcome {
        const id = std.fmt.parseUnsigned(u32, arg, 10) catch {
            self.command_status = "invalid checkpoint id";
            return .ok;
        };
        try self.restartCheckpoint(id);
        return .ok;
    }
    fn cmdBreakpointSet(self: *Ui, arg: []const u8) anyerror!CommandOutcome {
        try self.handleBreakpointSet(arg);
        return .ok;
    }
    fn cmdBreakpointDelete(self: *Ui, arg: []const u8) anyerror!CommandOutcome {
        try self.handleBreakpointDelete(arg);
        return .ok;
    }
    fn cmdInfoBreak(self: *Ui, _: []const u8) anyerror!CommandOutcome {
        try self.describeBreakpoints();
        return .ok;
    }
    fn cmdGasShow(self: *Ui, _: []const u8) anyerror!CommandOutcome {
        const gas_remaining = self.session.debugger.getGasRemaining();
        const gas_spent_step = if (self.previous_snapshot.gas_remaining >= gas_remaining)
            self.previous_snapshot.gas_remaining - gas_remaining
        else
            0;
        const gas_spent_total = self.seed.limits.gas_limit - gas_remaining;
        try self.setCommandStatusFmt("gas={d} step_spent={d} total_spent={d}", .{
            gas_remaining, gas_spent_step, gas_spent_total,
        });
        return .ok;
    }
    fn cmdGasSet(self: *Ui, arg: []const u8) anyerror!CommandOutcome {
        const gas = std.fmt.parseInt(i64, arg, 0) catch {
            self.command_status = "invalid gas value";
            return .ok;
        };
        const frame = self.session.evm.getCurrentFrame() orelse {
            self.command_status = "no active frame";
            return .ok;
        };
        if (gas < 0) {
            self.command_status = "gas must be >= 0";
            return .ok;
        }
        frame.gas_remaining = gas;
        self.previous_snapshot = self.captureSnapshot();
        try self.refreshPreviousBindingsSnapshot();
        try self.setCommandStatusFmt("gas set to {d}", .{gas});
        return .ok;
    }
    fn cmdPrint(self: *Ui, arg: []const u8) anyerror!CommandOutcome {
        if (arg.len == 0) {
            self.command_status = "missing print target";
            return .ok;
        }
        try self.handlePrintCommand(arg);
        return .ok;
    }
    fn cmdTraceExport(self: *Ui, arg: []const u8) anyerror!CommandOutcome {
        const trimmed = std.mem.trim(u8, arg, " \t");
        if (!std.mem.startsWith(u8, trimmed, "export ") and !std.mem.startsWith(u8, trimmed, "export\t")) {
            self.command_status = "usage: :trace export <path>";
            return .ok;
        }
        const path = std.mem.trim(u8, trimmed[7..], " \t");
        if (path.len == 0) {
            self.command_status = "missing trace path";
            return .ok;
        }
        const wrote = self.exportTrace(path) catch |err| {
            switch (err) {
                error.OutOfMemory => self.command_status = "out of memory",
                else => self.command_status = "failed to export trace",
            }
            return .ok;
        };
        try self.setCommandStatusFmt("trace exported: {d} steps -> {s}", .{ wrote, path });
        return .ok;
    }
    fn cmdOverlay(self: *Ui, arg: []const u8) anyerror!CommandOutcome {
        const trimmed = std.mem.trim(u8, arg, " \t");
        if (trimmed.len == 0) {
            try self.setCommandStatusFmt("overlay = {s} (modes: none, coverage, gas, folded, hoist)", .{self.overlay_mode.name()});
            return .ok;
        }
        const mode = OverlayMode.parse(trimmed) orelse {
            self.command_status = "unknown overlay (modes: none, coverage, gas, folded, hoist)";
            return .ok;
        };
        self.overlay_mode = mode;
        try self.setCommandStatusFmt("overlay = {s}", .{mode.name()});
        return .ok;
    }
    fn cmdCoverage(self: *Ui, arg: []const u8) anyerror!CommandOutcome {
        const total = self.session.debugger.lineHitsCount();
        if (total == 0) {
            self.command_status = "no lines hit yet";
            return .ok;
        }
        var n: usize = 10;
        const trimmed = std.mem.trim(u8, arg, " \t");
        if (trimmed.len > 0) {
            n = std.fmt.parseUnsigned(usize, trimmed, 10) catch {
                self.command_status = "usage: :cov [N]";
                return .ok;
            };
            if (n == 0) n = 10;
        }
        const top = try self.session.debugger.getLineHitsTopN(self.allocator, n);
        defer self.allocator.free(top);

        self.command_status_storage.clearRetainingCapacity();
        var writer = self.command_status_storage.writer(self.allocator);
        try writer.print("cov: {d} lines hit; top {d}:", .{ total, top.len });
        for (top) |hit| {
            try writer.print(" L{d}={d}", .{ hit.line, hit.count });
        }
        self.command_status = self.command_status_storage.items;
        return .ok;
    }
    fn cmdGasCoverage(self: *Ui, arg: []const u8) anyerror!CommandOutcome {
        const total = self.session.debugger.lineGasCount();
        if (total == 0) {
            self.command_status = "no gas attributed yet";
            return .ok;
        }
        var n: usize = 10;
        const trimmed = std.mem.trim(u8, arg, " \t");
        if (trimmed.len > 0) {
            n = std.fmt.parseUnsigned(usize, trimmed, 10) catch {
                self.command_status = "usage: :gascov [N]";
                return .ok;
            };
            if (n == 0) n = 10;
        }
        const top = try self.session.debugger.getLineGasTopN(self.allocator, n);
        defer self.allocator.free(top);

        self.command_status_storage.clearRetainingCapacity();
        var writer = self.command_status_storage.writer(self.allocator);
        try writer.print("gas: {d} lines with gas; top {d}:", .{ total, top.len });
        for (top) |entry| {
            try writer.print(" L{d}={d}", .{ entry.line, entry.gas });
        }
        self.command_status = self.command_status_storage.items;
        return .ok;
    }
    fn cmdEval(self: *Ui, arg: []const u8) anyerror!CommandOutcome {
        if (arg.len == 0) {
            self.command_status = "missing expression";
            return .ok;
        }
        const value = self.evaluateExpr(arg) catch |err| {
            switch (err) {
                error.ParseError => self.command_status = "parse error",
                error.UnknownIdentifier => self.command_status = "unknown identifier",
                error.BindingUnavailable => self.command_status = "binding unavailable",
                error.DivisionByZero => self.command_status = "division by zero",
                error.Overflow => self.command_status = "literal overflow",
                error.OutOfMemory => self.command_status = "out of memory",
            }
            return .ok;
        };
        switch (value) {
            .num => |n| try self.setCommandStatusFmt("=> {d}", .{n}),
            .bool_ => |b| try self.setCommandStatusFmt("=> {s}", .{if (b) "true" else "false"}),
        }
        return .ok;
    }
    fn cmdSet(self: *Ui, arg: []const u8) anyerror!CommandOutcome {
        try self.handleSetCommand(arg);
        return .ok;
    }
    fn cmdWriteSession(self: *Ui, arg: []const u8) anyerror!CommandOutcome {
        if (arg.len == 0) {
            self.command_status = "missing session path";
            return .ok;
        }
        self.writeSession(arg) catch {
            self.command_status = "failed to write session";
            return .ok;
        };
        self.command_status = "session saved";
        return .ok;
    }
    fn cmdLoadSession(self: *Ui, arg: []const u8) anyerror!CommandOutcome {
        if (arg.len == 0) {
            self.command_status = "missing session path";
            return .ok;
        }
        self.loadSession(arg) catch {
            self.command_status = "failed to load session";
            return .ok;
        };
        self.command_status = "session loaded";
        return .ok;
    }

    fn cmdWatch(self: *Ui, arg: []const u8) anyerror!CommandOutcome {
        if (arg.len == 0) {
            self.command_status = "usage: :watch <binding-name | slot-hex>";
            return .ok;
        }
        // Try slot first (decimal or hex), then fall back to binding name.
        if (std.fmt.parseUnsigned(u256, arg, 0)) |slot| {
            const id = self.session.debugger.addWatchpointBySlot(slot) catch {
                self.command_status = "failed to add watchpoint";
                return .ok;
            };
            try self.setCommandStatusFmt("watchpoint #{d} on slot {d}", .{ id, slot });
            return .ok;
        } else |_| {}
        const id_opt = self.session.debugger.addWatchpointByBindingName(arg) catch {
            self.command_status = "failed to add watchpoint";
            return .ok;
        };
        if (id_opt) |id| {
            try self.setCommandStatusFmt("watchpoint #{d} on {s}", .{ id, arg });
        } else {
            self.command_status = "binding not visible or not backed by a writable storage slot";
        }
        return .ok;
    }

    fn cmdUnwatch(self: *Ui, arg: []const u8) anyerror!CommandOutcome {
        if (arg.len == 0) {
            self.command_status = "usage: :unwatch <id>";
            return .ok;
        }
        const id = std.fmt.parseUnsigned(u32, arg, 10) catch {
            self.command_status = "invalid watchpoint id";
            return .ok;
        };
        if (self.session.debugger.removeWatchpoint(id)) {
            try self.setCommandStatusFmt("watchpoint #{d} removed", .{id});
        } else {
            self.command_status = "no watchpoint with that id";
        }
        return .ok;
    }

    fn cmdInfoWatch(self: *Ui, _: []const u8) anyerror!CommandOutcome {
        const wps = self.session.debugger.getWatchpoints();
        if (wps.len == 0) {
            self.command_status = "no watchpoints";
            return .ok;
        }
        self.clearCommandLog();
        for (wps) |wp| {
            const line = self.scratchFmt(
                "wp #{d}  slot={d}  last={d}  ({s})",
                .{ wp.id, wp.slot, wp.last_seen, wp.name },
            ) catch continue;
            try self.appendLogLine(line);
        }
        self.command_status = "watchpoints listed";
        return .ok;
    }

    /// The full command surface. Adding a new command is one row here
    /// plus one handler. `:help` is generated from this table.
    const kCommands = [_]CommandSpec{
        .{ .name = "q", .aliases = &.{"quit"}, .match = .exact, .handler = cmdQuit, .help = "quit the debugger" },
        .{ .name = "h", .aliases = &.{ "help", "legend", "marks" }, .match = .exact, .handler = cmdHelp, .help = "show help / mark legend" },
        .{ .name = "c", .aliases = &.{"continue"}, .match = .exact, .handler = cmdContinue, .help = "continue until breakpoint or halt" },
        .{ .name = "r", .aliases = &.{ "run", "rerun" }, .match = .exact, .handler = cmdRun, .help = "restart from seed, replay nothing" },
        .{ .name = "s", .aliases = &.{ "step", "si", "in" }, .match = .exact, .handler = cmdStepIn, .help = "step in (next source statement)" },
        .{ .name = "x", .aliases = &.{ "op", "opcode" }, .match = .exact, .handler = cmdStepOpcode, .help = "step exactly one EVM opcode" },
        .{ .name = "n", .aliases = &.{ "next", "so" }, .match = .exact, .handler = cmdStepOver, .help = "step over (skip nested calls)" },
        .{ .name = "o", .aliases = &.{ "out", "finish" }, .match = .exact, .handler = cmdStepOut, .help = "step out of current frame" },
        .{ .name = "p", .aliases = &.{ "prev", "previous" }, .match = .exact, .handler = cmdStepBack, .help = "step back (replay history minus 1)" },
        .{ .name = "where", .aliases = &.{"why-here"}, .match = .exact, .handler = cmdWhere, .help = "describe the current stop" },
        .{ .name = "origin", .aliases = &.{"origin-line"}, .match = .exact, .handler = cmdOrigin, .help = "jump to origin source line" },
        .{ .name = "sirfollow", .aliases = &.{"syncsir"}, .match = .exact, .handler = cmdSirFollow, .help = "resync SIR pane to current op" },
        .{ .name = "checkpoint", .match = .exact, .handler = cmdCheckpoint, .help = "save a checkpoint at current step" },
        .{ .name = "checkpoints", .match = .exact, .handler = cmdCheckpoints, .help = "list checkpoints" },
        .{ .name = "bt", .aliases = &.{"backtrace"}, .match = .exact, .handler = cmdBacktrace, .help = "print call-frame stack" },
        .{ .name = "info break", .match = .exact, .handler = cmdInfoBreak, .help = "list breakpoints" },
        .{ .name = "info watch", .match = .exact, .handler = cmdInfoWatch, .help = "list watchpoints" },
        .{ .name = "gas", .match = .exact, .handler = cmdGasShow, .help = "show gas (remaining/spent)" },
        // Prefix commands. Order matters: longer prefixes that share a
        // leading word with a shorter one must come first ("line-info "
        // before "line ", "why-line " before plain words, etc.).
        .{ .name = "line-info ", .aliases = &.{"why-line "}, .match = .prefix, .handler = cmdLineInfo, .help = "explain provenance for source line" },
        .{ .name = "line ", .match = .prefix, .handler = cmdLine, .help = "focus source pane on line N" },
        .{ .name = "sirline ", .match = .prefix, .handler = cmdSirLine, .help = "pin SIR pane to line N" },
        .{ .name = "frame ", .match = .prefix, .handler = cmdFrame, .help = "select call-frame index" },
        .{ .name = "restart ", .match = .prefix, .handler = cmdRestart, .help = "rewind to checkpoint id" },
        .{ .name = "break ", .match = .prefix, .handler = cmdBreakpointSet, .help = "set breakpoint on line" },
        .{ .name = "delete ", .match = .prefix, .handler = cmdBreakpointDelete, .help = "delete breakpoint on line" },
        .{ .name = "watch ", .match = .prefix, .handler = cmdWatch, .help = "watch storage slot or binding (halt on change)" },
        .{ .name = "unwatch ", .match = .prefix, .handler = cmdUnwatch, .help = "remove watchpoint by id" },
        .{ .name = "gas ", .match = .prefix, .handler = cmdGasSet, .help = "override frame gas_remaining" },
        .{ .name = "print ", .match = .prefix, .handler = cmdPrint, .help = "print binding/state target" },
        .{ .name = "eval ", .match = .prefix, .handler = cmdEval, .help = "evaluate side-effect-free expression (numbers, bindings, +-*/%, == != < > <= >=, && ||, !)" },
        .{ .name = "cov", .match = .exact, .handler = cmdCoverage, .help = "report top-10 hottest source lines (statement-boundary hits)" },
        .{ .name = "cov ", .match = .prefix, .handler = cmdCoverage, .help = "report top-N hottest source lines (`:cov 20`)" },
        .{ .name = "gascov", .match = .exact, .handler = cmdGasCoverage, .help = "report top-10 source lines by cumulative gas spent" },
        .{ .name = "gascov ", .match = .prefix, .handler = cmdGasCoverage, .help = "report top-N source lines by cumulative gas spent" },
        .{ .name = "overlay", .match = .exact, .handler = cmdOverlay, .help = "show current overlay mode" },
        .{ .name = "overlay ", .match = .prefix, .handler = cmdOverlay, .help = "switch overlay mode (none, coverage, gas, folded, hoist)" },
        .{ .name = "trace ", .match = .prefix, .handler = cmdTraceExport, .help = "export EIP-3155 trace (`:trace export <path>`)" },
        .{ .name = "set ", .match = .prefix, .handler = cmdSet, .help = "write binding/state target" },
        .{ .name = "write-session ", .aliases = &.{"ws "}, .match = .prefix, .handler = cmdWriteSession, .help = "save session to path" },
        .{ .name = "load-session ", .aliases = &.{"ls "}, .match = .prefix, .handler = cmdLoadSession, .help = "load session from path" },
    };

    fn matchCommand(spec: CommandSpec, raw: []const u8) ?[]const u8 {
        switch (spec.match) {
            .exact => {
                if (std.mem.eql(u8, raw, spec.name)) return raw[raw.len..];
                for (spec.aliases) |alias| {
                    if (std.mem.eql(u8, raw, alias)) return raw[raw.len..];
                }
                return null;
            },
            .prefix => {
                if (std.mem.startsWith(u8, raw, spec.name)) return std.mem.trim(u8, raw[spec.name.len..], " \t");
                for (spec.aliases) |alias| {
                    if (std.mem.startsWith(u8, raw, alias)) return std.mem.trim(u8, raw[alias.len..], " \t");
                }
                return null;
            },
        }
    }

    fn executeCommand(self: *Ui) !bool {
        const raw = std.mem.trim(u8, self.command_buffer.items, " \t");
        if (raw.len == 0) {
            self.command_status = "empty command";
            return false;
        }

        for (kCommands) |spec| {
            if (matchCommand(spec, raw)) |arg| {
                const outcome = try spec.handler(self, arg);
                return outcome == .quit;
            }
        }

        self.command_status = "unknown command";
        return false;
    }

    fn describeHelp(self: *Ui) !void {
        self.clearCommandLog();
        // Auto-generated from kCommands so the docs and the dispatcher
        // can't drift. Layout: ":<name>  <help>", with prefix commands
        // showing the trailing space as a hint that an argument follows.
        for (kCommands) |spec| {
            const name_buf = self.scratchFmt("{s}{s}{s}  -  {s}", .{
                ":",
                spec.name,
                if (spec.aliases.len > 0) "" else "",
                spec.help,
            }) catch continue;
            try self.appendLogLine(name_buf);
        }
        try self.appendLogLine("keys: see KEYBINDINGS.md (or `man 1 ora-evm-debug-tui`)");
        try self.appendLogLine("legend: . direct  ~ synthetic  + mixed  = folded  ! guard  - removed  * breakpoint  ^ origin  >|< sir-range");
        self.command_status = "help";
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
        if (std.mem.eql(u8, target, "logs")) {
            try self.handlePrintLogsCommand();
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
        const resolved_value = try self.resolvedBindingValue(&binding);
        const availability = self.bindingAvailability(&binding, resolved_value);
        if (binding.folded_value) |folded| {
            try self.setCommandStatusFmt("{s} [{s}] = {s} [derived: folded]", .{ binding.name, binding_label, folded });
            return;
        }
        if (resolved_value) |value| {
            switch (value) {
                .numeric => |numeric| if (availability == .derived)
                    try self.setCommandStatusFmt("{s} [{s}] = {d} [derived]", .{ binding.name, binding_label, numeric })
                else
                    try self.setCommandStatusFmt("{s} [{s}] = {d}", .{ binding.name, binding_label, numeric }),
                .text => |text| if (availability == .derived)
                    try self.setCommandStatusFmt("{s} [{s}] = {s} [derived]", .{ binding.name, binding_label, text })
                else
                    try self.setCommandStatusFmt("{s} [{s}] = {s}", .{ binding.name, binding_label, text }),
            }
            return;
        }
        switch (availability) {
            .optimized_away => try self.setCommandStatusFmt("{s} [{s}] = optimized away", .{ binding.name, binding_label }),
            .out_of_scope => try self.setCommandStatusFmt("{s} [{s}] = out of scope", .{ binding.name, binding_label }),
            .not_initialized => try self.setCommandStatusFmt("{s} [{s}] = not initialized", .{ binding.name, binding_label }),
            .unavailable => if (std.mem.eql(u8, binding.runtime_kind, "ssa"))
                try self.setCommandStatusFmt("{s} [{s}] = unavailable [not recoverable at this stop]", .{ binding.name, binding_label })
            else
                try self.setCommandStatusFmt("{s} [{s}] = unavailable", .{ binding.name, binding_label }),
            else => try self.setCommandStatusFmt("{s} [{s}]", .{ binding.name, binding_label }),
        }
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

    fn handlePrintLogsCommand(self: *Ui) !void {
        const logs = self.session.evm.logs.items;
        if (logs.len == 0) {
            self.command_status = "no logs emitted";
            return;
        }

        self.command_status_storage.clearRetainingCapacity();
        var writer = self.command_status_storage.writer(self.allocator);
        try writer.print("logs ({d}):", .{logs.len});
        for (logs, 0..) |log_entry, i| {
            self.render_scratch.clearRetainingCapacity();
            if (self.formatDecodedLog(log_entry)) |decoded| {
                try writer.print(" #{d}={s}", .{ i, decoded });
            } else {
                try writer.print(" #{d}=<{d} topics, {d} data bytes>", .{
                    i, log_entry.topics.len, log_entry.data.len,
                });
            }
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
        const parsed = self.parseBreakpointArgs(rest) catch {
            self.command_status = "usage: :break <line> [when <expr>] [hit <n>]";
            return;
        };
        const line = parsed.line;
        for (self.breakpoints.items) |existing| {
            if (existing.line == line) {
                self.command_status = "breakpoint already set";
                return;
            }
        }
        if (!self.session.debugger.setBreakpoint(self.seed.source_path, line)) {
            self.command_status = self.breakpointFailureMessage(line);
            return;
        }
        const condition_dup: ?[]u8 = if (parsed.condition) |c| try self.allocator.dupe(u8, c) else null;
        errdefer if (condition_dup) |c| self.allocator.free(c);
        try self.breakpoints.append(self.allocator, .{
            .line = line,
            .condition = condition_dup,
            .hit_target = parsed.hit_target,
        });
        const prov_label_opt = self.session.debugger.src_map.getLineProvenance(self.seed.source_path, line);
        const prov_str = if (prov_label_opt) |p| p.label() else "";
        if (parsed.condition != null and parsed.hit_target != null) {
            try self.setCommandStatusFmt("breakpoint set on line {d}{s}{s}{s} when '{s}' hit {d}", .{
                line,
                if (prov_label_opt != null) " [" else "",
                prov_str,
                if (prov_label_opt != null) "]" else "",
                parsed.condition.?,
                parsed.hit_target.?,
            });
        } else if (parsed.condition) |cond| {
            try self.setCommandStatusFmt("breakpoint set on line {d}{s}{s}{s} when '{s}'", .{
                line,
                if (prov_label_opt != null) " [" else "",
                prov_str,
                if (prov_label_opt != null) "]" else "",
                cond,
            });
        } else if (parsed.hit_target) |target| {
            try self.setCommandStatusFmt("breakpoint set on line {d}{s}{s}{s} hit {d}", .{
                line,
                if (prov_label_opt != null) " [" else "",
                prov_str,
                if (prov_label_opt != null) "]" else "",
                target,
            });
        } else if (prov_label_opt) |prov| {
            try self.setCommandStatusFmt("breakpoint set on line {d} [{s}]", .{ line, prov.label() });
        } else {
            try self.setCommandStatusFmt("breakpoint set on line {d}", .{line});
        }
    }

    fn handleBreakpointDelete(self: *Ui, rest: []const u8) !void {
        const line = try self.parseBreakpointLine(rest);
        var found = false;
        var i: usize = 0;
        while (i < self.breakpoints.items.len) : (i += 1) {
            if (self.breakpoints.items[i].line == line) {
                self.breakpoints.items[i].deinit(self.allocator);
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

    fn parseBreakpointArgs(self: *Ui, rest: []const u8) !ParsedBreakpoint {
        _ = self;
        return parseBreakpointArgsImpl(rest);
    }

    fn breakpointFailureMessage(self: *Ui, line: u32) []const u8 {
        const src_map = &self.session.debugger.src_map;
        const file = self.seed.source_path;
        if (!src_map.hasAnyEntryForLine(file, line)) return "no runtime mapping on that line";
        if (src_map.getStatementKindForLine(file, line) == null) return "non-executable spec line";
        return "no executable statement on that line";
    }

    fn describeLineInfo(self: *Ui, line: u32) !void {
        const src_map = &self.session.debugger.src_map;
        const file = self.seed.source_path;
        if (self.isFoldedSourceLine(line)) {
            try self.setCommandStatusFmt("line {d}: folded source line; {s}", .{
                line,
                self.foldedLineExplanation(line),
            });
            return;
        }
        if (self.isRemovedSourceLine(line)) {
            try self.setCommandStatusFmt("line {d}: removed source line; {s}", .{
                line,
                self.removedLineExplanation(line),
            });
            return;
        }
        if (self.isStructuralSourceLine(line)) {
            try self.setCommandStatusFmt("line {d}: structural source line; syntax only, no standalone runtime stop", .{line});
            return;
        }
        if (!src_map.hasAnyEntryForLine(file, line)) {
            self.command_status = "no runtime mapping on that line";
            return;
        }

        const kind = src_map.getStatementKindForLine(file, line);
        const provenance = src_map.getLineProvenance(file, line);
        var stmt_id: ?u32 = null;
        var origin_stmt_id: ?u32 = null;
        var function_name: ?[]const u8 = null;
        for (src_map.entries) |entry| {
            if (entry.line != line) continue;
            if (!std.mem.eql(u8, entry.file, file)) continue;
            if (!entry.is_statement) continue;
            if (stmt_id == null and entry.statement_id != null) stmt_id = entry.statement_id;
            if (origin_stmt_id == null and entry.origin_statement_id != null) origin_stmt_id = entry.origin_statement_id;
            if (function_name == null and entry.function != null) function_name = entry.function;
        }

        const kind_label = self.statementKindExplanation(kind);
        const prov_label = self.lineProvenanceExplanation(provenance);

        if (stmt_id) |stmt|
            if (origin_stmt_id) |origin|
                if (function_name) |function|
                    try self.setCommandStatusFmt("line {d}: {s}; {s}; stmt={d}; origin={d}; fn={s}", .{
                        line, kind_label, prov_label, stmt, origin, function,
                    })
                else
                    try self.setCommandStatusFmt("line {d}: {s}; {s}; stmt={d}; origin={d}", .{
                        line, kind_label, prov_label, stmt, origin,
                    })
            else if (function_name) |function|
                try self.setCommandStatusFmt("line {d}: {s}; {s}; stmt={d}; fn={s}", .{
                    line, kind_label, prov_label, stmt, function,
                })
            else
                try self.setCommandStatusFmt("line {d}: {s}; {s}; stmt={d}", .{
                    line, kind_label, prov_label, stmt,
                })
        else if (origin_stmt_id) |origin|
            if (function_name) |function|
                try self.setCommandStatusFmt("line {d}: {s}; {s}; origin={d}; fn={s}", .{
                    line, kind_label, prov_label, origin, function,
                })
            else
                try self.setCommandStatusFmt("line {d}: {s}; {s}; origin={d}", .{
                    line, kind_label, prov_label, origin,
                })
        else if (function_name) |function|
            try self.setCommandStatusFmt("line {d}: {s}; {s}; fn={s}", .{
                line, kind_label, prov_label, function,
            })
        else
            try self.setCommandStatusFmt("line {d}: {s}; {s}", .{
                line, kind_label, prov_label,
            });
    }

    fn describeCurrentStop(self: *Ui) !void {
        const current_line = self.focus_line orelse self.session.debugger.currentSourceLine() orelse 0;
        const entry = self.session.debugger.currentEntry() orelse {
            self.command_status = "no active stop";
            return;
        };
        const prov = self.currentProvenanceLabel();
        const kind = self.statementKindExplanation(self.statementKindForLine(current_line));
        const origin_line = self.currentOriginLine();
        const synthetic_path = self.currentSyntheticPath();

        if (origin_line) |origin|
            if (origin != current_line)
                if (entry.statement_id) |stmt|
                    if (entry.execution_region_id) |region|
                        if (entry.statement_run_index) |run|
                            if (synthetic_path) |path|
                                try self.setCommandStatusFmt("here: {s}; {s}; stmt={d}; line={d}; origin={d}; region={d}.{d}; copies={s}", .{
                                    prov, kind, stmt, current_line, origin, region, run, self.scratchSyntheticPathSummary(path),
                                })
                            else
                                try self.setCommandStatusFmt("here: {s}; {s}; stmt={d}; line={d}; origin={d}; region={d}.{d}", .{
                                    prov, kind, stmt, current_line, origin, region, run,
                                })
                        else
                            try self.setCommandStatusFmt("here: {s}; {s}; stmt={d}; line={d}; origin={d}; region={d}", .{
                                prov, kind, stmt, current_line, origin, region,
                            })
                    else
                        try self.setCommandStatusFmt("here: {s}; {s}; stmt={d}; line={d}; origin={d}", .{
                            prov, kind, stmt, current_line, origin,
                        })
                else if (entry.origin_statement_id) |origin_stmt|
                    try self.setCommandStatusFmt("here: {s}; {s}; line={d}; origin line={d}; origin stmt={d}", .{
                        prov, kind, current_line, origin, origin_stmt,
                    })
                else
                    try self.setCommandStatusFmt("here: {s}; {s}; line={d}; origin line={d}", .{
                        prov, kind, current_line, origin,
                    })
            else
                try self.setCommandStatusFmt("here: {s}; {s}; line={d}", .{ prov, kind, current_line })
        else
            try self.setCommandStatusFmt("here: {s}; {s}; line={d}", .{ prov, kind, current_line });

        // Append the inlined-from chain when present. Today this is
        // always empty (no MLIR-level inliner yet); the render path
        // is here so future compiler-side work that populates the
        // chain immediately surfaces in the TUI without further
        // plumbing.
        //
        // `setCommandStatusFmt` left `command_status` pointing into
        // `command_status_storage.items`, so we just extend the
        // buffer in place and re-slice — no clear, no copy.
        if (self.currentInlinedFromChain()) |chain| {
            if (chain.len > 0) {
                const writer = self.command_status_storage.writer(self.allocator);
                try appendInlinedChain(writer, chain);
                self.command_status = self.command_status_storage.items;
            }
        }
    }

    fn describeBreakpoints(self: *Ui) !void {
        if (self.breakpoints.items.len == 0) {
            self.command_status = "no breakpoints";
            return;
        }
        self.command_status_storage.clearRetainingCapacity();
        var writer = self.command_status_storage.writer(self.allocator);
        try writer.writeAll("breakpoints:");
        for (self.breakpoints.items, 0..) |bp, i| {
            const prov = self.session.debugger.src_map.getLineProvenance(self.seed.source_path, bp.line);
            const sep: []const u8 = if (i == 0) " " else ", ";
            if (prov) |p|
                try writer.print("{s}{d}[{s}]", .{ sep, bp.line, p.label() })
            else
                try writer.print("{s}{d}", .{ sep, bp.line });
            if (bp.condition) |cond| try writer.print(" when '{s}'", .{cond});
            if (bp.hit_target) |target| try writer.print(" hit {d}/{d}", .{ bp.hit_count, target });
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
            const sep: []const u8 = if (logical == 0) " " else ", ";
            try writer.print("{s}#{d} {x} pc={d} gas={d}", .{ sep, logical, frame.address, frame.pc, frame.gas_remaining });
            if (frame.is_static) try writer.writeAll(" static");
            if (frame.calldata.len >= 4) {
                var selector_buf: [4]u8 = undefined;
                @memcpy(&selector_buf, frame.calldata[0..4]);
                if (self.abiDocForFrame(&frame)) |abi_doc| {
                    if (abi_doc.findCallableBySelector(selector_buf)) |callable| {
                        if (callable.object.get("name")) |n| {
                            if (n == .string) {
                                try writer.print(" -> {s}", .{n.string});
                                logical += 1;
                                continue;
                            }
                        }
                    }
                }
                const sel_int = std.mem.readInt(u32, &selector_buf, .big);
                try writer.print(" sel=0x{x:0>8}", .{sel_int});
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
        // Reset hit counters before replay — the replay re-traverses the
        // same opcodes so re-counting from zero keeps the count
        // consistent with what the user just saw.
        self.resetBreakpointHitCounts();
        try self.primeInitialStop();
        try self.applyBreakpoints();
        for (replay_items) |mode| {
            self.runStep(mode, true);
            if (std.mem.eql(u8, self.status, "execution_error")) break;
        }
        self.previous_snapshot = self.captureSnapshot();
        try self.refreshPreviousBindingsSnapshot();
    }

    /// Clear every breakpoint's hit_count back to 0. Called before any
    /// session rebuild that will replay history, so a `:break <line>
    /// hit <n>` predicate doesn't get "stuck" past N.
    pub fn resetBreakpointHitCounts(self: *Ui) void {
        for (self.breakpoints.items) |*bp| bp.hit_count = 0;
    }

    pub fn applyBreakpoints(self: *Ui) !void {
        // Continue past per-line failures so a single stale breakpoint
        // (e.g. after a session reload onto recompiled bytecode) doesn't
        // disable the rest. Surface the count via the status line.
        var failed: usize = 0;
        var first_failed_line: u32 = 0;
        for (self.breakpoints.items) |bp| {
            if (!self.session.debugger.setBreakpoint(self.seed.source_path, bp.line)) {
                if (failed == 0) first_failed_line = bp.line;
                failed += 1;
            }
        }
        if (failed != 0) {
            try self.setCommandStatusFmt(
                "{d} breakpoint(s) skipped (e.g. line {d} has no statement entry)",
                .{ failed, first_failed_line },
            );
        }
    }

    fn setCommandStatusFmt(self: *Ui, comptime fmt: []const u8, args: anytype) !void {
        self.command_status_storage.clearRetainingCapacity();
        try self.command_status_storage.writer(self.allocator).print(fmt, args);
        self.command_status = self.command_status_storage.items;
    }

    fn runDebuggerCommand(self: *Ui, mode: StepMode) anyerror!void {
        return switch (mode) {
            .in => self.session.debugger.stepIn(),
            .opcode => self.session.debugger.stepOpcode(),
            .over => self.session.debugger.stepOver(),
            .out => self.session.debugger.stepOut(),
            .continue_ => self.session.debugger.continue_(),
        };
    }

    pub fn runStep(self: *Ui, mode: StepMode, record_history: bool) void {
        if (self.session.debugger.isHalted()) {
            self.status = "halted";
            self.syncFocusFromDebugger();
            self.updateCommandStatusForCurrentStop(stepModeName(mode)) catch {};
            return;
        }
        self.previous_snapshot = self.captureSnapshot();
        self.refreshPreviousBindingsSnapshot() catch {};
        self.runDebuggerCommand(mode) catch {
            self.status = self.session.debugger.lastErrorName() orelse "execution_error";
            self.updateExecutionErrorStatus(stepModeName(mode)) catch {
                self.command_status = self.status;
            };
            return;
        };
        // Apply conditional / hit-count gating: if the breakpoint we
        // halted at has predicates that aren't satisfied, transparently
        // resume. Capped to avoid spinning on a malformed predicate that
        // always falses out — caller can disable the breakpoint manually.
        var gating_iters: u32 = 0;
        while (self.shouldSkipCurrentBreakpoint()) : (gating_iters += 1) {
            if (gating_iters >= kMaxConditionalBreakpointGatingIters) break;
            self.runDebuggerCommand(.continue_) catch {
                self.status = self.session.debugger.lastErrorName() orelse "execution_error";
                self.updateExecutionErrorStatus(stepModeName(mode)) catch {
                    self.command_status = self.status;
                };
                return;
            };
            if (self.session.debugger.isHalted()) break;
        }
        if (record_history) self.step_history.append(self.allocator, mode) catch {
            // Out of memory recording history. Step result is still valid;
            // step-back / replay will be incomplete past this point.
            self.command_status = "warning: step history truncated (OOM)";
        };
        self.status = @tagName(self.session.debugger.stop_reason);
        if (!self.shouldPreserveFocusOnTerminalStop()) self.syncFocusFromDebugger();
        self.centerOnCurrentLine();
        self.updateCommandStatusForCurrentStop(stepModeName(mode)) catch {};
    }

    /// Check whether the current breakpoint hit should be skipped because
    /// its `when <expr>` predicate evaluated to false or its `hit <n>`
    /// target hasn't been reached yet. Bumps the matching `Breakpoint`'s
    /// `hit_count` as a side effect when we hit one.
    fn shouldSkipCurrentBreakpoint(self: *Ui) bool {
        if (self.session.debugger.stop_reason != .breakpoint_hit) return false;
        const current_line = self.session.debugger.currentSourceLine() orelse return false;
        var bp_ptr: ?*Breakpoint = null;
        for (self.breakpoints.items) |*bp| {
            if (bp.line == current_line) {
                bp_ptr = bp;
                break;
            }
        }
        const bp = bp_ptr orelse return false;
        bp.hit_count +%= 1;

        var skip = false;
        if (bp.hit_target) |target| {
            if (bp.hit_count != target) skip = true;
        }
        if (!skip) {
            if (bp.condition) |cond| {
                const value = self.evaluateExpr(cond) catch {
                    // Predicate failed to evaluate — fail open (halt) so
                    // the user can fix the predicate; don't silently
                    // skip.
                    return false;
                };
                if (!value.asBool()) skip = true;
            }
        }
        return skip;
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
        // Same reasoning as in rerunToHistory: replay re-traverses the
        // same opcodes, so re-counting hit_count from zero keeps the
        // breakpoint state consistent with the user's view.
        self.resetBreakpointHitCounts();
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

    pub fn primeInitialStop(self: *Ui) !void {
        if (self.session.debugger.isHalted()) {
            self.status = "halted";
            return;
        }

        var initial_function: ?[]const u8 = null;
        if (self.session.debugger.currentEntry()) |entry| {
            initial_function = entry.function;
            if (entry.is_statement) {
                if (self.shouldAcceptInitialStop(entry, initial_function)) {
                    self.status = "ready";
                    self.syncInitialFocusFromDebugger();
                    self.alignInitialSourceView();
                    return;
                }
            }
        }

        var attempts: usize = 0;
        while (!self.session.debugger.isHalted() and attempts < 256) : (attempts += 1) {
            try self.session.debugger.stepIn();
            self.status = @tagName(self.session.debugger.stop_reason);
            if (self.session.debugger.currentEntry()) |entry| {
                if (initial_function == null) initial_function = entry.function;
                if (entry.is_statement and self.shouldAcceptInitialStop(entry, initial_function)) break;
            }
        }

        if (self.session.debugger.isHalted()) self.status = @tagName(self.session.debugger.stop_reason);
        self.syncInitialFocusFromDebugger();
        self.alignInitialSourceView();
    }

    fn shouldAcceptInitialStop(self: *const Ui, entry: *const SourceMap.Entry, initial_function: ?[]const u8) bool {
        if (!entry.is_statement) return false;
        const function_name = initial_function orelse entry.function orelse return true;
        if (entry.function) |entry_function| {
            if (!std.mem.eql(u8, entry_function, function_name)) return true;
        }

        if (!entry.is_synthetic and !entry.is_hoisted) return true;
        return !self.hasPreferredInitialStatementLater(function_name, entry.pc);
    }

    fn hasPreferredInitialStatementLater(self: *const Ui, function_name: []const u8, after_pc: u32) bool {
        for (self.session.debugger.src_map.entries) |entry| {
            if (!entry.is_statement) continue;
            if (entry.pc <= after_pc) continue;
            const entry_function = entry.function orelse continue;
            if (!std.mem.eql(u8, entry_function, function_name)) continue;
            if (entry.is_synthetic or entry.is_hoisted) continue;
            return true;
        }
        return false;
    }

    pub fn syncFocusFromDebugger(self: *Ui) void {
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

    fn syncInitialFocusFromDebugger(self: *Ui) void {
        if (self.session.debugger.currentEntry()) |entry| {
            if (entry.is_synthetic or entry.is_hoisted or entry.is_duplicated) {
                if (self.currentOriginLine()) |origin_line| {
                    self.focus_line = origin_line;
                    return;
                }
            }
        }
        self.syncFocusFromDebugger();
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

    fn alignInitialSourceView(self: *Ui) void {
        const function_start_line = self.currentFunctionDeclarationLine();
        if (function_start_line) |line| {
            self.scroll_line = if (line > 0) line else 1;
        } else {
            self.centerOnCurrentLine();
            return;
        }
        if (self.sir_follow) self.centerSirOnCurrentMapping();
    }

    fn currentFunctionDeclarationLine(self: *Ui) ?u32 {
        const entry = self.session.debugger.currentEntry() orelse return null;
        const function_name = entry.function orelse return null;
        const scopes = self.session.debugger.getVisibleScopes(self.allocator) catch return null;
        defer if (scopes.len > 0) self.allocator.free(scopes);

        for (scopes) |scope| {
            if (!std.mem.eql(u8, scope.kind, "function")) continue;
            if (!std.mem.eql(u8, scope.function, function_name)) continue;
            const range = scope.range orelse continue;
            return range.start.line;
        }
        return null;
    }

    /// Look up the innermost visible function-scope's
    /// `inlined_from` chain, if any. The chain is empty for scopes
    /// that the compiler lowered from their lexical site (the common
    /// case today). Once the MLIR-level inliner lands and populates
    /// the chain, this becomes the source of "you're inside an
    /// inlined callee" hints surfaced by `describeCurrentStop` and
    /// related status renders.
    ///
    /// Returns `null` when no debug info is loaded or no function
    /// scope is visible at the current PC.
    fn currentInlinedFromChain(self: *Ui) ?[]const DebugInfo.InlinedFrame {
        const entry = self.session.debugger.currentEntry() orelse return null;
        const function_name = entry.function orelse return null;
        const scopes = self.session.debugger.getVisibleScopes(self.allocator) catch return null;
        defer if (scopes.len > 0) self.allocator.free(scopes);

        for (scopes) |scope| {
            if (!std.mem.eql(u8, scope.kind, "function")) continue;
            if (!std.mem.eql(u8, scope.function, function_name)) continue;
            return scope.inlined_from;
        }
        return null;
    }

    /// Append the inlined-from chain to `writer` as a human-readable
    /// suffix, e.g. ` (inlined from bar() at app.ora:42 <- baz() at
    /// app.ora:99)`. No-op when the chain is empty. Outermost entry
    /// renders first (matches `InlinedFrame` doc on outermost-first
    /// ordering).
    fn appendInlinedChain(writer: anytype, chain: []const DebugInfo.InlinedFrame) !void {
        if (chain.len == 0) return;
        try writer.writeAll(" (inlined from ");
        for (chain, 0..) |frame, i| {
            if (i != 0) try writer.writeAll(" <- ");
            try writer.print("{s}()", .{frame.function});
            if (frame.call_site) |range| {
                // Render the file basename, not the full path, to keep
                // the status line readable.
                const file = std.fs.path.basename(frame.file);
                try writer.print(" at {s}:{d}", .{ file, range.start.line });
            } else {
                const file = std.fs.path.basename(frame.file);
                try writer.print(" at {s}", .{file});
            }
        }
        try writer.writeAll(")");
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
        const statement_id = entry.statement_id orelse self.session.debugger.lastStatementId();
        const execution_region_id = entry.execution_region_id;
        const statement_run_index = entry.statement_run_index;

        var found = false;
        var sir_start: u32 = 0;
        var sir_end: u32 = 0;
        var idx_start: ?u32 = null;
        var idx_end: ?u32 = null;
        var pc_start: u32 = 0;
        var pc_end: u32 = 0;

        for (self.session.debugger.src_map.entries) |map_entry| {
            if (!std.mem.eql(u8, map_entry.file, file)) continue;
            if (statement_id) |stmt_id| {
                if (map_entry.statement_id != stmt_id) continue;
            } else {
                if (map_entry.line != ora_line) continue;
            }

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
                .statement_id = statement_id,
                .execution_region_id = execution_region_id,
                .statement_run_index = statement_run_index,
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
            .statement_id = statement_id,
            .execution_region_id = execution_region_id,
            .statement_run_index = statement_run_index,
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

    pub fn captureSnapshot(self: *Ui) Snapshot {
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

    pub fn clearPreviousBindingsSnapshot(self: *Ui) void {
        for (self.previous_bindings.items) |entry| {
            self.allocator.free(entry.name);
        }
        self.previous_bindings.clearRetainingCapacity();
    }

    pub fn refreshPreviousBindingsSnapshot(self: *Ui) !void {
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
                    if (m.statement_id) |stmt_id|
                        if (m.execution_region_id) |region_id|
                            if (m.statement_run_index) |run_index|
                                self.scratchFmt(" SIR Text  lines {d}..{d}  idx {d}..{d}  stmt {d}  region {d}.{d} ", .{ m.sir_start, m.sir_end, m.idx_start.?, m.idx_end.?, stmt_id, region_id, run_index }) catch " SIR Text "
                            else
                                self.scratchFmt(" SIR Text  lines {d}..{d}  idx {d}..{d}  stmt {d}  region {d} ", .{ m.sir_start, m.sir_end, m.idx_start.?, m.idx_end.?, stmt_id, region_id }) catch " SIR Text "
                        else
                            self.scratchFmt(" SIR Text  lines {d}..{d}  idx {d}..{d}  stmt {d} ", .{ m.sir_start, m.sir_end, m.idx_start.?, m.idx_end.?, stmt_id }) catch " SIR Text "
                    else
                        self.scratchFmt(" SIR Text  lines {d}..{d}  idx {d}..{d} ", .{ m.sir_start, m.sir_end, m.idx_start.?, m.idx_end.? }) catch " SIR Text "
                else
                    if (m.statement_id) |stmt_id|
                        if (m.execution_region_id) |region_id|
                            if (m.statement_run_index) |run_index|
                                self.scratchFmt(" SIR Text  lines {d}..{d}  stmt {d}  region {d}.{d} ", .{ m.sir_start, m.sir_end, stmt_id, region_id, run_index }) catch " SIR Text "
                            else
                                self.scratchFmt(" SIR Text  lines {d}..{d}  stmt {d}  region {d} ", .{ m.sir_start, m.sir_end, stmt_id, region_id }) catch " SIR Text "
                        else
                            self.scratchFmt(" SIR Text  lines {d}..{d}  stmt {d} ", .{ m.sir_start, m.sir_end, stmt_id }) catch " SIR Text "
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

        const bottom_state_w: u16 = @max(28, @as(u16, @intCast((@as(u32, win.width) * 58) / 100)));
        const bottom_trace_w: u16 = win.width - bottom_state_w;

        const state_outer = win.child(.{
            .x_off = 0,
            .y_off = @intCast(content_y + top_h),
            .width = bottom_state_w,
            .height = bottom_h,
            .border = .{ .where = .all, .glyphs = .{ .custom = ascii_border_glyphs }, .style = style_border() },
        });
        Ui.drawPanelTitle(win, 2, content_y + top_h, " State ");
        self.drawEvmPane(state_outer);

        if (bottom_trace_w >= 24) {
            const trace_outer = win.child(.{
                .x_off = bottom_state_w,
                .y_off = @intCast(content_y + top_h),
                .width = bottom_trace_w,
                .height = bottom_h,
                .border = .{ .where = .all, .glyphs = .{ .custom = ascii_border_glyphs }, .style = style_border() },
            });
            Ui.drawPanelTitle(win, bottom_state_w + 2, content_y + top_h, " Trace ");
            self.drawTracePane(trace_outer);
        }
    }

    fn drawPanelTitle(win: Window, x: u16, y: u16, title: []const u8) void {
        drawSegments(win, x, y, &.{seg(title, style_title())});
    }

    fn drawHeader(self: *Ui, win: Window) void {
        const source_name = std.fs.path.basename(self.seed.source_path);
        const line = self.focus_line orelse self.session.debugger.currentSourceLine() orelse 0;
        const mapping = self.currentMappingWindow();
        const sir_line = if (mapping) |m| m.sir_start else self.currentSirLine() orelse 0;
        const origin_line = self.currentOriginLine();
        const entry = self.session.debugger.currentEntry();
        const frame = self.session.evm.getCurrentFrame();
        const opcode = if (frame != null) self.session.debugger.getCurrentOpcodeName() else "no-frame";
        const gas_remaining: i64 = if (frame) |f| f.gas_remaining else 0;
        const gas_spent_step: i64 = if (frame != null and self.previous_snapshot.gas_remaining >= gas_remaining)
            self.previous_snapshot.gas_remaining - gas_remaining
        else
            0;
        const gas_spent_total: i64 = if (frame != null)
            self.seed.limits.gas_limit - gas_remaining
        else if (self.session.debugger.isSuccess())
            self.seed.limits.gas_limit
        else
            0;

        const top = win.child(.{ .height = 1 });
        top.fill(.{ .char = .{ .grapheme = " " }, .style = style_header_title() });
        const title = self.scratchFmt(" Ora EVM Debugger | {s}", .{source_name}) catch "Ora EVM Debugger";
        drawSegments(top, 0, 0, &.{seg(title, style_header_title())});

        const meta = win.child(.{ .y_off = 1, .height = 2 });
        meta.fill(.{ .char = .{ .grapheme = " " }, .style = style_header_meta() });

        const status_text = if (frame == null)
            if (mapping) |m|
                if (m.statement_id) |stmt_id|
                    if (m.execution_region_id) |region_id|
                        if (m.statement_run_index) |run_index|
                            self.scratchFmt(" {s}  |  stmt {d}  |  region {d}.{d}  |  ora {d}  |  sir {d}  |  no active frame  |  result {s}", .{
                                self.status, stmt_id, region_id, run_index, line, sir_line, if (self.session.debugger.isSuccess()) "success" else "reverted",
                            }) catch "status"
                        else
                            self.scratchFmt(" {s}  |  stmt {d}  |  region {d}  |  ora {d}  |  sir {d}  |  no active frame  |  result {s}", .{
                                self.status, stmt_id, region_id, line, sir_line, if (self.session.debugger.isSuccess()) "success" else "reverted",
                            }) catch "status"
                    else
                        self.scratchFmt(" {s}  |  stmt {d}  |  ora {d}  |  sir {d}  |  no active frame  |  result {s}", .{
                            self.status,
                            stmt_id,
                            line,
                            sir_line,
                            if (self.session.debugger.isSuccess()) "success" else "reverted",
                        }) catch "status"
                else
                    self.scratchFmt(" {s}  |  ora {d}  |  sir {d}  |  no active frame  |  result {s}", .{
                        self.status,
                        line,
                        sir_line,
                        if (self.session.debugger.isSuccess()) "success" else "reverted",
                    }) catch "status"
            else
                self.scratchFmt(" {s}  |  ora {d}  |  sir {d}  |  no active frame  |  result {s}", .{
                    self.status,
                    line,
                    sir_line,
                    if (self.session.debugger.isSuccess()) "success" else "reverted",
                }) catch "status"
        else if (self.session.debugger.lastErrorName()) |err_name|
            if (entry) |e| blk: {
                if (e.idx) |idx| {
                    if (mapping) |m| {
                        if (m.statement_id) |stmt_id| {
                            break :blk self.scratchFmt(" error {s}  |  stmt {d}  |  ora {d}  |  sir {d}  |  pc {d} idx {d}  |  {s}  |  depth {d}  |  gas {d}", .{
                                err_name,
                                stmt_id,
                                line,
                                sir_line,
                                self.session.debugger.getPC(),
                                idx,
                                opcode,
                                self.session.debugger.getCallDepth(),
                                gas_remaining,
                            }) catch "status";
                        }
                    }
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
                if (mapping) |m| {
                    if (m.statement_id) |stmt_id| {
                        break :blk self.scratchFmt(" {s}  |  stmt {d}  |  ora {d}  |  sir {d}  |  pc {d} idx {d}  |  {s}  |  depth {d}  |  gas {d}  |  step -{d}  |  total -{d}", .{
                            self.status,
                            stmt_id,
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
                }
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
                if (m.statement_id) |stmt_id|
                    if (origin_line) |origin|
                        if (origin != m.ora_line)
                            self.scratchFmt(" map  |  stmt {d}  ->  ora {d}  (origin {d})  ->  sir {d}..{d}  ->  idx {d}..{d}  ->  pc {d}..{d}  |  {s}", .{
                                stmt_id, m.ora_line, origin, m.sir_start, m.sir_end, m.idx_start.?, m.idx_end.?, m.pc_start, m.pc_end, current_source,
                            }) catch "map"
                        else
                            self.scratchFmt(" map  |  stmt {d}  ->  ora {d}  ->  sir {d}..{d}  ->  idx {d}..{d}  ->  pc {d}..{d}  |  {s}", .{
                                stmt_id, m.ora_line, m.sir_start, m.sir_end, m.idx_start.?, m.idx_end.?, m.pc_start, m.pc_end, current_source,
                            }) catch "map"
                    else
                        self.scratchFmt(" map  |  stmt {d}  ->  ora {d}  ->  sir {d}..{d}  ->  idx {d}..{d}  ->  pc {d}..{d}  |  {s}", .{
                            stmt_id, m.ora_line, m.sir_start, m.sir_end, m.idx_start.?, m.idx_end.?, m.pc_start, m.pc_end, current_source,
                        }) catch "map"
                else
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
                if (m.statement_id) |stmt_id|
                    self.scratchFmt(" map  |  stmt {d}  ->  ora {d}  ->  sir {d}..{d}  ->  pc {d}..{d}  |  {s}", .{
                        stmt_id,
                        m.ora_line,
                        m.sir_start,
                        m.sir_end,
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
        drawSegments(help, 0, 0, &.{seg(" s step-in  x opcode  n step-over  o step-out  c continue  p previous  : command  j/k Ora  J/K SIR  = sync SIR  1..5 tabs  [/] cycle  q quit  |  . direct  ~ synthetic  + mixed  = folded  ! guard  - removed  * break  ^ origin  >|< sir-range ", style_header_title())});
    }

    fn clearCommandLog(self: *Ui) void {
        for (self.command_log.items) |line| self.allocator.free(line);
        self.command_log.clearRetainingCapacity();
    }

    fn appendCommandLog(self: *Ui, command: []const u8, result: []const u8) !void {
        const line = try std.fmt.allocPrint(self.allocator, ":{s} => {s}", .{ command, result });
        try self.appendLogEntry(line);
    }

    fn appendLogLine(self: *Ui, line: []const u8) !void {
        const owned = try self.allocator.dupe(u8, line);
        try self.appendLogEntry(owned);
    }

    fn appendLogEntry(self: *Ui, line: []u8) !void {
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
            drawSegments(win, 0, 0, &.{seg(" :help for commands and marker legend  |  values show folded constants or readable runtime roots ", style_footer_note())});
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
        const origin_line = self.currentOriginLine();
        const summary = if (mapping) |m|
            if (m.idx_start != null and m.idx_end != null)
                if (m.statement_id) |stmt_id|
                    if (m.execution_region_id) |region_id|
                        if (m.statement_run_index) |run_index|
                            if (origin_line) |origin|
                                if (origin != m.ora_line)
                                    self.scratchFmt(" runtime {s}/{s} | stmt {d} | origin {d} | region {d}.{d} | {s} | ora {d} => sir {d}..{d} | idx {d}..{d} | pc {d}..{d}", .{
                                        self.statementKindLabel(current_line), self.lineProvenanceLabel(current_line), stmt_id, origin, region_id, run_index, self.currentProvenanceLabel(), m.ora_line, m.sir_start, m.sir_end, m.idx_start.?, m.idx_end.?, m.pc_start, m.pc_end,
                                    }) catch "runtime mapping"
                                else
                                    self.scratchFmt(" runtime {s}/{s} | stmt {d} | region {d}.{d} | {s} | ora {d} => sir {d}..{d} | idx {d}..{d} | pc {d}..{d}", .{
                                        self.statementKindLabel(current_line), self.lineProvenanceLabel(current_line), stmt_id, region_id, run_index, self.currentProvenanceLabel(), m.ora_line, m.sir_start, m.sir_end, m.idx_start.?, m.idx_end.?, m.pc_start, m.pc_end,
                                    }) catch "runtime mapping"
                            else
                                self.scratchFmt(" runtime {s}/{s} | stmt {d} | region {d}.{d} | {s} | ora {d} => sir {d}..{d} | idx {d}..{d} | pc {d}..{d}", .{
                                    self.statementKindLabel(current_line), self.lineProvenanceLabel(current_line), stmt_id, region_id, run_index, self.currentProvenanceLabel(), m.ora_line, m.sir_start, m.sir_end, m.idx_start.?, m.idx_end.?, m.pc_start, m.pc_end,
                                }) catch "runtime mapping"
                        else
                            self.scratchFmt(" runtime {s}/{s} | stmt {d} | region {d} | {s} | ora {d} => sir {d}..{d} | idx {d}..{d} | pc {d}..{d}", .{
                                self.statementKindLabel(current_line), self.lineProvenanceLabel(current_line), stmt_id, region_id, self.currentProvenanceLabel(), m.ora_line, m.sir_start, m.sir_end, m.idx_start.?, m.idx_end.?, m.pc_start, m.pc_end,
                            }) catch "runtime mapping"
                    else
                        self.scratchFmt(" runtime {s}/{s} | stmt {d} | {s} | ora {d} => sir {d}..{d} | idx {d}..{d} | pc {d}..{d}", .{
                            self.statementKindLabel(current_line),
                            self.lineProvenanceLabel(current_line),
                            stmt_id,
                            self.currentProvenanceLabel(),
                            m.ora_line,
                            m.sir_start,
                            m.sir_end,
                            m.idx_start.?,
                            m.idx_end.?,
                            m.pc_start,
                            m.pc_end,
                        }) catch "runtime mapping"
                else
                    self.scratchFmt(" runtime {s}/{s} | {s} | ora {d} => sir {d}..{d} | idx {d}..{d} | pc {d}..{d}", .{
                        self.statementKindLabel(current_line),
                        self.lineProvenanceLabel(current_line),
                        self.currentProvenanceLabel(),
                        m.ora_line,
                        m.sir_start,
                        m.sir_end,
                        m.idx_start.?,
                        m.idx_end.?,
                        m.pc_start,
                        m.pc_end,
                    }) catch "runtime mapping"
            else
                if (m.statement_id) |stmt_id|
                    self.scratchFmt(" runtime {s}/{s} | stmt {d} | {s} | ora {d} => sir {d}..{d} | pc {d}..{d}", .{
                        self.statementKindLabel(current_line),
                        self.lineProvenanceLabel(current_line),
                        stmt_id,
                        self.currentProvenanceLabel(),
                        m.ora_line,
                        m.sir_start,
                        m.sir_end,
                        m.pc_start,
                        m.pc_end,
                    }) catch "runtime mapping"
                else
                    self.scratchFmt(" runtime {s}/{s} | {s} | ora {d} => sir {d}..{d} | pc {d}..{d}", .{
                        self.statementKindLabel(current_line),
                        self.lineProvenanceLabel(current_line),
                        self.currentProvenanceLabel(),
                        m.ora_line,
                        m.sir_start,
                        m.sir_end,
                        m.pc_start,
                        m.pc_end,
                    }) catch "runtime mapping"
        else
            self.scratchFmt(" runtime {s}/{s} | {s} | ora {d}", .{
                self.statementKindLabel(current_line),
                self.lineProvenanceLabel(current_line),
                self.currentProvenanceLabel(),
                current_line,
            }) catch "runtime mapping";
        drawSegments(win, 1, 0, &.{seg(summary, style_muted())});

        const content_y: u16 = if (win.height > 1) 1 else 0;
        const content_h: u16 = if (win.height > 1) win.height - 1 else win.height;
        if (content_h == 0) return;
        const content = win.child(.{ .y_off = content_y, .height = content_h });
        const gutter_width: u16 = 9;
        const source_content = if (content.width > gutter_width)
            content.child(.{ .x_off = gutter_width, .width = content.width - gutter_width })
        else
            content;
        self.drawSourceText(source_content, current_line);
        self.drawSourceGutter(content, current_line);
    }

    fn drawSourceText(self: *Ui, win: Window, current_line: u32) void {
        var visible_row: u16 = 0;
        while (visible_row < win.height) : (visible_row += 1) {
            const line = self.scroll_line + visible_row;
            if (line > self.session.debugger.totalSourceLines()) break;
            const line_text = self.session.debugger.getSourceLineText(line) orelse continue;
            const text = std.mem.trimRight(u8, line_text, "\r");
            const style = if (line == current_line)
                Style{ .bg = Color.rgbFromUint(0x303A45), .fg = Color.rgbFromUint(0xF5F7FA) }
            else if (self.isFoldedSourceLine(line))
                style_hint()
            else if (self.isRemovedSourceLine(line))
                style_dead()
            else
                style_text();
            drawSegments(win, 0, visible_row, &.{seg(text, style)});
        }
    }

    fn drawSirPane(self: *Ui, win: Window) void {
        if (self.seed.sir_text == null) {
            drawSegments(win, 1, 1, &.{seg("no SIR text artifact for this session", style_hint())});
            return;
        }
        const mapping = self.currentMappingWindow();
        const current_sir_line = if (mapping) |m| if (m.sir_start != 0) m.sir_start else self.currentSirLine() orelse 0 else self.currentSirLine() orelse 0;
        const origin_line = self.currentOriginLine();
        const effect = self.currentWriteEffectKind();
        const summary = if (mapping) |m|
            if (m.idx_start != null and m.idx_end != null)
                if (m.statement_id) |stmt_id|
                    if (m.execution_region_id) |region_id|
                        if (m.statement_run_index) |run_index|
                            if (origin_line) |origin|
                                if (origin != m.ora_line)
                                    self.scratchFmt(" lowered region | stmt {d} | origin {d} | region {d}.{d} | {s} | ora {d} => sir {d}..{d} | idx {d}..{d} | pc {d}..{d} | effect {s}", .{
                                        stmt_id, origin, region_id, run_index, self.currentProvenanceLabel(), m.ora_line, m.sir_start, m.sir_end, m.idx_start.?, m.idx_end.?, m.pc_start, m.pc_end, self.writeEffectLabel(effect),
                                    }) catch "lowered region"
                                else
                                    self.scratchFmt(" lowered region | stmt {d} | region {d}.{d} | {s} | ora {d} => sir {d}..{d} | idx {d}..{d} | pc {d}..{d} | effect {s}", .{
                                        stmt_id, region_id, run_index, self.currentProvenanceLabel(), m.ora_line, m.sir_start, m.sir_end, m.idx_start.?, m.idx_end.?, m.pc_start, m.pc_end, self.writeEffectLabel(effect),
                                    }) catch "lowered region"
                            else
                                self.scratchFmt(" lowered region | stmt {d} | region {d}.{d} | {s} | ora {d} => sir {d}..{d} | idx {d}..{d} | pc {d}..{d} | effect {s}", .{
                                    stmt_id, region_id, run_index, self.currentProvenanceLabel(), m.ora_line, m.sir_start, m.sir_end, m.idx_start.?, m.idx_end.?, m.pc_start, m.pc_end, self.writeEffectLabel(effect),
                                }) catch "lowered region"
                        else
                            self.scratchFmt(" lowered region | stmt {d} | region {d} | {s} | ora {d} => sir {d}..{d} | idx {d}..{d} | pc {d}..{d} | effect {s}", .{
                                stmt_id, region_id, self.currentProvenanceLabel(), m.ora_line, m.sir_start, m.sir_end, m.idx_start.?, m.idx_end.?, m.pc_start, m.pc_end, self.writeEffectLabel(effect),
                            }) catch "lowered region"
                    else
                        self.scratchFmt(" lowered region | stmt {d} | {s} | ora {d} => sir {d}..{d} | idx {d}..{d} | pc {d}..{d} | effect {s}", .{
                            stmt_id,
                            self.currentProvenanceLabel(),
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
                    self.scratchFmt(" lowered region | {s} | ora {d} => sir {d}..{d} | idx {d}..{d} | pc {d}..{d} | effect {s}", .{
                        self.currentProvenanceLabel(),
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
                if (m.statement_id) |stmt_id|
                    self.scratchFmt(" lowered region | stmt {d} | {s} | ora {d} => sir {d}..{d} | pc {d}..{d} | effect {s}", .{
                        stmt_id,
                        self.currentProvenanceLabel(),
                        m.ora_line,
                        m.sir_start,
                        m.sir_end,
                        m.pc_start,
                        m.pc_end,
                        self.writeEffectLabel(effect),
                    }) catch "lowered region"
                else
                    self.scratchFmt(" lowered region | {s} | ora {d} => sir {d}..{d} | pc {d}..{d} | effect {s}", .{
                        self.currentProvenanceLabel(),
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
        if (self.isFoldedSourceLine(line)) return "folded";
        if (self.isRemovedSourceLine(line)) return "removed";
        const kind = self.statementKindForLine(line) orelse return "none";
        return switch (kind) {
            .runtime => "stmt",
            .runtime_guard => "guard",
        };
    }

    fn statementKindExplanation(self: *Ui, kind: ?ora_evm.SourceMap.StatementKind) []const u8 {
        _ = self;
        return switch (kind orelse return "non-executable source line") {
            .runtime => "runtime statement",
            .runtime_guard => "runtime guard check",
        };
    }

    fn lineProvenanceLabel(self: *Ui, line: u32) []const u8 {
        if (self.isFoldedSourceLine(line)) return "folded";
        if (self.isRemovedSourceLine(line)) return "removed";
        const provenance = self.session.debugger.src_map.getLineProvenance(self.config.source_path, line) orelse return "none";
        return provenance.label();
    }

    fn lineProvenanceExplanation(self: *Ui, provenance: ?ora_evm.SourceMap.LineProvenance) []const u8 {
        _ = self;
        return switch (provenance orelse return "no runtime coverage") {
            .direct => "direct coverage: executes as written",
            .synthetic => "synthetic coverage: reached through compiler-generated lowering only",
            .mixed => "mixed coverage: partly executes as written and partly through lowered ops",
        };
    }

    fn foldedLineExplanation(self: *Ui, line: u32) []const u8 {
        _ = self;
        _ = line;
        return "folded at compile time: source declaration stays visible, but runtime uses the folded value directly";
    }

    fn removedLineExplanation(self: *Ui, line: u32) []const u8 {
        _ = self;
        _ = line;
        return "removed from runtime: source declaration has no SIR or bytecode coverage";
    }

    fn isFoldedSourceLine(self: *Ui, line: u32) bool {
        if (self.session.debugger.src_map.hasAnyEntryForLine(self.config.source_path, line)) return false;
        const info = self.session.debugger.debug_info orelse return false;
        if (!self.lineInAnyFunctionScope(info, line)) return false;

        for (info.parsed.value.source_scopes) |scope| {
            if (!std.mem.eql(u8, scope.kind, "function")) continue;
            const range = scope.range orelse continue;
            if (line < range.start.line or line > range.end.line) continue;
            for (scope.locals) |local| {
                const decl = local.decl orelse continue;
                if (decl.start.line != line) continue;
                if (local.is_folded) return true;
            }
        }
        return false;
    }

    /// Return the folded value text for a folded source line, or null
    /// if the line isn't folded or the local doesn't carry a literal
    /// folded_value (e.g. only `is_folded` was set).
    fn foldedValueForLine(self: *Ui, line: u32) ?[]const u8 {
        const info = self.session.debugger.debug_info orelse return null;
        for (info.parsed.value.source_scopes) |scope| {
            if (!std.mem.eql(u8, scope.kind, "function")) continue;
            const range = scope.range orelse continue;
            if (line < range.start.line or line > range.end.line) continue;
            for (scope.locals) |local| {
                const decl = local.decl orelse continue;
                if (decl.start.line != line) continue;
                if (!local.is_folded) continue;
                if (local.folded_value) |fv| return fv;
            }
        }
        return null;
    }

    /// Return the origin_statement_id of the first source-map entry
    /// on `line` that's marked hoisted (either at the source-map entry
    /// level or via its op_meta). null when nothing on the line is
    /// hoisted.
    ///
    /// Limitation: only the first statement boundary on the line is
    /// inspected. If a line carries multiple statements with different
    /// hoist statuses (rare in practice), the second statement's hoist
    /// won't surface in the overlay. SourceMap doesn't expose an
    /// all-entries-for-line iterator, and adding one for an overlay
    /// hint isn't justified yet.
    fn hoistOriginForLine(self: *Ui, line: u32) ?u32 {
        const entry = self.session.debugger.src_map.getFirstStatementEntryForLine(self.config.source_path, line) orelse return null;
        if (entry.is_hoisted) {
            if (entry.origin_statement_id) |o| return o;
        }
        const info = self.session.debugger.debug_info orelse return null;
        const idx = entry.idx orelse return null;
        if (info.getOpMetaForIdx(idx)) |meta| {
            if (meta.is_hoisted) {
                if (entry.origin_statement_id) |o| return o;
                if (meta.origin_statement_id) |o| return o;
            }
        }
        return null;
    }

    fn isRemovedSourceLine(self: *Ui, line: u32) bool {
        if (self.session.debugger.src_map.hasAnyEntryForLine(self.config.source_path, line)) return false;
        const info = self.session.debugger.debug_info orelse return false;
        if (!self.lineInAnyFunctionScope(info, line)) return false;
        if (!self.isMeaningfulRemovedSourceText(line)) return false;

        for (info.parsed.value.source_scopes) |scope| {
            if (!std.mem.eql(u8, scope.kind, "function")) continue;
            const range = scope.range orelse continue;
            if (line < range.start.line or line > range.end.line) continue;
            for (scope.locals) |local| {
                const decl = local.decl orelse continue;
                if (decl.start.line != line) continue;
                if (std.mem.eql(u8, local.runtime.kind, "ssa")) return true;
            }
        }
        return true;
    }

    fn lineInAnyFunctionScope(self: *Ui, info: DebugInfo, line: u32) bool {
        _ = self;
        for (info.parsed.value.source_scopes) |scope| {
            if (!std.mem.eql(u8, scope.kind, "function")) continue;
            const range = scope.range orelse continue;
            if (line >= range.start.line and line <= range.end.line) return true;
        }
        return false;
    }

    fn isMeaningfulRemovedSourceText(self: *Ui, line: u32) bool {
        const raw = self.session.debugger.getSourceLineText(line) orelse return false;
        const text = std.mem.trim(u8, std.mem.trimRight(u8, raw, "\r"), " \t");
        if (text.len == 0) return false;
        if (std.mem.eql(u8, text, "{")) return false;
        if (std.mem.eql(u8, text, "}")) return false;
        if (std.mem.eql(u8, text, "} else {")) return false;
        if (std.mem.startsWith(u8, text, "else")) return false;
        if (std.mem.startsWith(u8, text, "if ")) return true;
        if (std.mem.startsWith(u8, text, "const ")) return true;
        if (std.mem.startsWith(u8, text, "let ")) return true;
        if (std.mem.startsWith(u8, text, "return ")) return true;
        if (std.mem.indexOfScalar(u8, text, '=') != null) return true;
        return false;
    }

    fn isStructuralSourceLine(self: *Ui, line: u32) bool {
        const raw = self.session.debugger.getSourceLineText(line) orelse return false;
        const text = std.mem.trim(u8, std.mem.trimRight(u8, raw, "\r"), " \t");
        if (text.len == 0) return false;
        if (std.mem.eql(u8, text, "{")) return true;
        if (std.mem.eql(u8, text, "}")) return true;
        if (std.mem.eql(u8, text, "} else {")) return true;
        if (std.mem.startsWith(u8, text, "else")) return true;
        return false;
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

    fn currentProvenanceLabel(self: *Ui) []const u8 {
        const current_line = self.focus_line orelse self.session.debugger.currentSourceLine() orelse 0;
        const line_prov = self.session.debugger.src_map.getLineProvenance(self.config.source_path, current_line);
        if (self.session.debugger.currentOpMeta()) |meta| {
            switch (line_prov orelse .direct) {
                .direct => return "direct",
                .mixed => {
                    if (meta.is_hoisted and meta.is_duplicated) return "mixed hoisted duplicated";
                    if (meta.is_hoisted) return "mixed hoisted";
                    if (meta.is_duplicated) return "mixed duplicated";
                    return "mixed";
                },
                .synthetic => {
                    if (meta.is_hoisted and meta.is_duplicated) return "synthetic hoisted duplicated";
                    if (meta.is_hoisted) return "synthetic hoisted";
                    if (meta.is_duplicated) return "synthetic duplicated";
                    return "synthetic";
                },
            }
        }
        const entry = self.session.debugger.currentEntry() orelse return "direct";
        switch (line_prov orelse .direct) {
            .direct => return "direct",
            .mixed => {
                if (entry.is_hoisted and entry.is_duplicated) return "mixed hoisted duplicated";
                if (entry.is_hoisted) return "mixed hoisted";
                if (entry.is_duplicated) return "mixed duplicated";
                return "mixed";
            },
            .synthetic => {
                if (entry.is_synthetic) {
                    if (entry.is_hoisted and entry.is_duplicated) return "synthetic hoisted duplicated";
                    if (entry.is_hoisted) return "synthetic hoisted";
                    if (entry.is_duplicated) return "synthetic duplicated";
                    return "synthetic";
                }
                if (entry.is_hoisted and entry.is_duplicated) return "synthetic hoisted duplicated";
                if (entry.is_hoisted) return "synthetic hoisted";
                if (entry.is_duplicated) return "synthetic duplicated";
                return "synthetic";
            },
        }
    }

    fn currentOriginLine(self: *Ui) ?u32 {
        const entry = self.session.debugger.currentEntry() orelse return null;
        const origin_stmt_id = entry.origin_statement_id orelse return null;
        return self.session.debugger.src_map.getFirstLineForStatementId(self.config.source_path, origin_stmt_id) orelse
            self.session.debugger.src_map.getFirstLineForOriginStatementId(self.config.source_path, origin_stmt_id);
    }

    fn currentSyntheticPath(self: *Ui) ?[]const u8 {
        if (self.session.debugger.currentOpMeta()) |meta| {
            if (meta.synthetic_path) |path| return path;
        }
        if (self.session.debugger.currentEntry()) |entry| {
            return entry.synthetic_path;
        }
        return null;
    }

    fn scratchSyntheticPathSummary(self: *Ui, path: []const u8) []const u8 {
        const start = self.render_scratch.items.len;
        const writer = self.render_scratch.writer(self.allocator);
        writer.writeAll("copies ") catch return path;

        var it = std.mem.splitScalar(u8, path, '/');
        var wrote_any = false;
        while (it.next()) |segment| {
            const dot = std.mem.indexOfScalar(u8, segment, '.') orelse {
                writer.print("{s}", .{path}) catch return path;
                return self.render_scratch.items[start..];
            };
            const idx = std.fmt.parseInt(u32, segment[0..dot], 10) catch {
                writer.print("{s}", .{path}) catch return path;
                return self.render_scratch.items[start..];
            };
            const count = std.fmt.parseInt(u32, segment[dot + 1 ..], 10) catch {
                writer.print("{s}", .{path}) catch return path;
                return self.render_scratch.items[start..];
            };
            if (wrote_any) writer.writeAll(" -> ") catch return path;
            writer.print("{d}/{d}", .{ idx + 1, count }) catch return path;
            wrote_any = true;
        }

        return self.render_scratch.items[start..];
    }

    fn updateCommandStatusForCurrentStop(self: *Ui, action: []const u8) !void {
        const entry = self.session.debugger.currentEntry();
        const current_line = self.focus_line orelse self.session.debugger.currentSourceLine() orelse 0;
        const origin_line = self.currentOriginLine();

        if (self.session.debugger.lastErrorName()) |err_name| {
            try self.setCommandStatusFmt("{s} => error {s}", .{ action, err_name });
            return;
        }
        if (self.session.debugger.stop_reason == .watchpoint_hit) {
            if (self.session.debugger.lastWatchpointId()) |wp_id| {
                try self.setCommandStatusFmt("{s} => watchpoint #{d} hit at line {d}", .{ action, wp_id, current_line });
                return;
            }
        }
        if (self.session.debugger.stop_reason == .execution_reverted) {
            const frames = self.session.evm.frames.items;
            if (frames.len > 0) {
                const top = &frames[frames.len - 1];
                if (top.output.len > 0) {
                    if (self.formatDecodedRevert(top.output)) |decoded| {
                        try self.setCommandStatusFmt("{s} => reverted: {s}", .{ action, decoded });
                        return;
                    }
                }
            }
            try self.setCommandStatusFmt("{s} => reverted (no decoded payload)", .{action});
            return;
        }

        if (entry) |e| {
            const stmt_id = e.statement_id;
            const origin_stmt_id = e.origin_statement_id;
            const region_id = e.execution_region_id;
            const run_index = e.statement_run_index;
            const prov = self.currentProvenanceLabel();

            if (origin_line) |origin|
                if (origin != current_line) {
                    if (stmt_id) |stmt|
                        if (region_id) |region|
                            if (run_index) |run|
                                try self.setCommandStatusFmt("{s} => {s} stmt {d} at line {d}, origin line {d}, region {d}.{d}", .{
                                    action, prov, stmt, current_line, origin, region, run,
                                })
                            else
                                try self.setCommandStatusFmt("{s} => {s} stmt {d} at line {d}, origin line {d}, region {d}", .{
                                    action, prov, stmt, current_line, origin, region,
                                })
                        else
                            try self.setCommandStatusFmt("{s} => {s} stmt {d} at line {d}, origin line {d}", .{
                                action, prov, stmt, current_line, origin,
                            })
                    else if (origin_stmt_id) |origin_stmt|
                        if (region_id) |region|
                            if (run_index) |run|
                                try self.setCommandStatusFmt("{s} => {s} region from stmt {d}, line {d}, origin line {d}, region {d}.{d}", .{
                                    action, prov, origin_stmt, current_line, origin, region, run,
                                })
                            else
                                try self.setCommandStatusFmt("{s} => {s} region from stmt {d}, line {d}, origin line {d}, region {d}", .{
                                    action, prov, origin_stmt, current_line, origin, region,
                                })
                        else
                            try self.setCommandStatusFmt("{s} => {s} region from stmt {d}, line {d}, origin line {d}", .{
                                action, prov, origin_stmt, current_line, origin,
                            })
                    else
                        try self.setCommandStatusFmt("{s} => {s} line {d}, origin line {d}", .{
                            action, prov, current_line, origin,
                        });
                    return;
                };

            if (stmt_id) |stmt|
                if (region_id) |region|
                    if (run_index) |run|
                        try self.setCommandStatusFmt("{s} => {s} stmt {d} at line {d}, region {d}.{d}", .{
                            action, prov, stmt, current_line, region, run,
                        })
                    else
                        try self.setCommandStatusFmt("{s} => {s} stmt {d} at line {d}, region {d}", .{
                            action, prov, stmt, current_line, region,
                        })
                else
                    try self.setCommandStatusFmt("{s} => {s} stmt {d} at line {d}", .{
                        action, prov, stmt, current_line,
                    })
            else if (origin_stmt_id) |origin_stmt|
                if (region_id) |region|
                    if (run_index) |run|
                        try self.setCommandStatusFmt("{s} => {s} origin stmt {d} at line {d}, region {d}.{d}", .{
                            action, prov, origin_stmt, current_line, region, run,
                        })
                    else
                        try self.setCommandStatusFmt("{s} => {s} origin stmt {d} at line {d}, region {d}", .{
                            action, prov, origin_stmt, current_line, region,
                        })
                else
                    try self.setCommandStatusFmt("{s} => {s} origin stmt {d} at line {d}", .{
                        action, prov, origin_stmt, current_line,
                    })
            else
                try self.setCommandStatusFmt("{s} => {s} line {d}", .{
                    action, prov, current_line,
                });
            return;
        }

        if (self.session.debugger.isHalted()) {
            try self.setCommandStatusFmt("{s} => {s}", .{ action, self.status });
            return;
        }
        try self.setCommandStatusFmt("{s} => line {d}", .{ action, current_line });
    }

    fn updateExecutionErrorStatus(self: *Ui, action: []const u8) !void {
        const err_name = self.session.debugger.lastErrorName() orelse "execution_error";
        const current_line = self.focus_line orelse self.session.debugger.currentSourceLine() orelse 0;
        const source_text = if (current_line != 0) self.session.debugger.getSourceLineText(current_line) else null;
        const opcode_name = self.session.debugger.getCurrentOpcodeName();

        if (std.mem.eql(u8, err_name, "InvalidOpcode")) {
            if (source_text) |line_text| {
                const trimmed = std.mem.trim(u8, line_text, " \t");
                try self.setCommandStatusFmt(
                    "{s} => trap {s} at line {d}: {s} [likely failed checked arithmetic, failed guard, or invalid input]",
                    .{ action, opcode_name, current_line, trimmed },
                );
                return;
            }
            try self.setCommandStatusFmt(
                "{s} => trap {s} at line {d} [likely failed checked arithmetic, failed guard, or invalid input]",
                .{ action, opcode_name, current_line },
            );
            return;
        }

        if (source_text) |line_text| {
            const trimmed = std.mem.trim(u8, line_text, " \t");
            try self.setCommandStatusFmt("{s} => error {s} at line {d}: {s}", .{
                action,
                err_name,
                current_line,
                trimmed,
            });
            return;
        }

        try self.setCommandStatusFmt("{s} => error {s}", .{ action, err_name });
    }

    fn commandActionLabel(self: *const Ui) []const u8 {
        if (self.step_history.items.len == 0) return "ready";
        return switch (self.step_history.items[self.step_history.items.len - 1]) {
            .in => "step-in",
            .opcode => "opcode",
            .over => "next",
            .out => "step-out",
            .continue_ => "continue",
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
            const availability = self.bindingAvailability(&binding, resolved_value);
            const previous_value = self.previousBindingValue(binding.name);
            const changed = current_value != previous_value;
            const marker = if (changed) "*" else " ";
            const row_text = if (binding.folded_value) |folded|
                self.scratchFmt("{s} {s} [{s}] = {s} [derived: folded]", .{ marker, binding.name, binding_label, folded }) catch binding.name
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
                    .numeric => |numeric| if (availability == .derived)
                        self.scratchFmt("{s} {s} [{s}] = {d} [derived]", .{ marker, binding.name, binding_label, numeric }) catch binding.name
                    else
                        self.scratchFmt("{s} {s} [{s}] = {d}", .{ marker, binding.name, binding_label, numeric }) catch binding.name,
                    .text => |text| if (availability == .derived)
                        self.scratchFmt("{s} {s} [{s}] = {s} [derived]", .{ marker, binding.name, binding_label, text }) catch binding.name
                    else
                        self.scratchFmt("{s} {s} [{s}] = {s}", .{ marker, binding.name, binding_label, text }) catch binding.name,
                }
            else if (availability == .optimized_away)
                self.scratchFmt("{s} {s} [{s}] = optimized away", .{ marker, binding.name, binding_label }) catch binding.name
            else if (availability == .out_of_scope)
                self.scratchFmt("{s} {s} [{s}] = out of scope", .{ marker, binding.name, binding_label }) catch binding.name
            else if (availability == .not_initialized)
                self.scratchFmt("{s} {s} [{s}] = not initialized", .{ marker, binding.name, binding_label }) catch binding.name
            else if (availability == .unavailable)
                if (std.mem.eql(u8, binding.runtime_kind, "ssa"))
                    self.scratchFmt("{s} {s} [{s}] = unavailable [not recoverable at this stop]", .{ marker, binding.name, binding_label }) catch binding.name
                else
                    self.scratchFmt("{s} {s} [{s}] = unavailable", .{ marker, binding.name, binding_label }) catch binding.name
            else
                self.scratchFmt("{s} {s} [{s}]", .{ marker, binding.name, binding_label }) catch binding.name;
            drawSegments(root, 1, @intCast(4 + i), &.{seg(row_text, self.bindingAvailabilityStyle(availability, changed))});
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
        const current_entry = self.session.debugger.currentEntry();
        const mapping = self.currentMappingWindow();
        const synthetic_path = self.currentSyntheticPath();
        const provenance_line = if (current_entry) |entry|
            if (mapping) |m|
                if (entry.statement_id) |stmt_id|
                    if (entry.origin_statement_id) |origin_stmt_id|
                        if (m.execution_region_id) |region_id|
                            if (m.statement_run_index) |run_index|
                                if (synthetic_path) |path|
                                    self.scratchFmt("provenance={s}  stmt={d}  origin={d}  region={d}.{d}  copies={s}", .{
                                        self.currentProvenanceLabel(), stmt_id, origin_stmt_id, region_id, run_index, self.scratchSyntheticPathSummary(path),
                                    }) catch "prov"
                                else
                                    self.scratchFmt("provenance={s}  stmt={d}  origin={d}  region={d}.{d}", .{
                                        self.currentProvenanceLabel(), stmt_id, origin_stmt_id, region_id, run_index,
                                    }) catch "prov"
                            else
                                self.scratchFmt("provenance={s}  stmt={d}  origin={d}  region={d}", .{
                                    self.currentProvenanceLabel(), stmt_id, origin_stmt_id, region_id,
                                }) catch "prov"
                        else
                            self.scratchFmt("provenance={s}  stmt={d}  origin={d}", .{
                                self.currentProvenanceLabel(), stmt_id, origin_stmt_id,
                            }) catch "prov"
                    else
                        self.scratchFmt("provenance={s}  stmt={d}", .{
                            self.currentProvenanceLabel(), stmt_id,
                        }) catch "prov"
                else if (entry.origin_statement_id) |origin_stmt_id|
                    self.scratchFmt("provenance={s}  origin={d}", .{
                        self.currentProvenanceLabel(), origin_stmt_id,
                    }) catch "prov"
                else
                    self.scratchFmt("provenance={s}", .{
                        self.currentProvenanceLabel(),
                    }) catch "prov"
            else
                if (entry.statement_id) |stmt_id|
                    if (entry.origin_statement_id) |origin_stmt_id|
                        self.scratchFmt("provenance={s}  stmt={d}  origin={d}", .{
                            self.currentProvenanceLabel(), stmt_id, origin_stmt_id,
                        }) catch "prov"
                    else
                        self.scratchFmt("provenance={s}  stmt={d}", .{
                            self.currentProvenanceLabel(), stmt_id,
                        }) catch "prov"
                else if (entry.origin_statement_id) |origin_stmt_id|
                    self.scratchFmt("provenance={s}  origin={d}", .{
                        self.currentProvenanceLabel(), origin_stmt_id,
                    }) catch "prov"
                else
                    self.scratchFmt("provenance={s}", .{
                        self.currentProvenanceLabel(),
                    }) catch "prov"
        else
            self.scratchFmt("provenance={s}", .{
                self.currentProvenanceLabel(),
            }) catch "prov";
        drawSegments(win, 1, row, &.{seg(provenance_line, style_muted())});
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

    fn drawTracePane(self: *Ui, root: Window) void {
        var row: u16 = 0;
        const current_line = self.focus_line orelse self.session.debugger.currentSourceLine() orelse 0;
        const origin_line = self.currentOriginLine();
        const entry = self.session.debugger.currentEntry();
        const prov = self.currentProvenanceLabel();

        if (entry) |e| {
            const stmt_id = e.statement_id;
            const origin_stmt_id = e.origin_statement_id;
            const region_id = e.execution_region_id;
            const run_index = e.statement_run_index;

            const top_line = if (origin_line) |origin|
                if (origin != current_line)
                    if (stmt_id) |stmt|
                        if (region_id) |region|
                            if (run_index) |run|
                                self.scratchFmt("{s} => {s} stmt {d} at line {d}, origin line {d}, region {d}.{d}", .{
                                    self.commandActionLabel(), prov, stmt, current_line, origin, region, run,
                                }) catch "trace"
                            else
                                self.scratchFmt("{s} => {s} stmt {d} at line {d}, origin line {d}, region {d}", .{
                                    self.commandActionLabel(), prov, stmt, current_line, origin, region,
                                }) catch "trace"
                        else
                            self.scratchFmt("{s} => {s} stmt {d} at line {d}, origin line {d}", .{
                                self.commandActionLabel(), prov, stmt, current_line, origin,
                            }) catch "trace"
                    else if (origin_stmt_id) |origin_stmt|
                        self.scratchFmt("{s} => {s} region from stmt {d}, line {d}, origin line {d}", .{
                            self.commandActionLabel(), prov, origin_stmt, current_line, origin,
                        }) catch "trace"
                    else
                        self.scratchFmt("{s} => {s} line {d}, origin line {d}", .{
                            self.commandActionLabel(), prov, current_line, origin,
                        }) catch "trace"
                else
                    self.scratchFmt("{s} => {s} at line {d}", .{ self.commandActionLabel(), prov, current_line }) catch "trace"
            else
                self.scratchFmt("{s} => {s} at line {d}", .{ self.commandActionLabel(), prov, current_line }) catch "trace";
            drawSegments(root, 1, row, &.{seg(top_line, style_emphasis())});
            row += 1;

            if (row < root.height) {
                const synthetic_path = self.currentSyntheticPath();
                const meta_line = if (stmt_id) |stmt|
                    if (origin_stmt_id) |origin_stmt|
                        if (region_id) |region|
                            if (run_index) |run|
                                if (synthetic_path) |path|
                                    self.scratchFmt("stmt={d}  origin={d}  region={d}.{d}  copies={s}  kind={s}/{s}", .{
                                        stmt, origin_stmt, region, run, self.scratchSyntheticPathSummary(path), self.statementKindLabel(current_line), self.lineProvenanceLabel(current_line),
                                    }) catch "meta"
                                else
                                    self.scratchFmt("stmt={d}  origin={d}  region={d}.{d}  kind={s}/{s}", .{
                                        stmt, origin_stmt, region, run, self.statementKindLabel(current_line), self.lineProvenanceLabel(current_line),
                                    }) catch "meta"
                            else
                                self.scratchFmt("stmt={d}  origin={d}  region={d}  kind={s}/{s}", .{
                                    stmt, origin_stmt, region, self.statementKindLabel(current_line), self.lineProvenanceLabel(current_line),
                                }) catch "meta"
                        else
                            self.scratchFmt("stmt={d}  origin={d}  kind={s}/{s}", .{
                                stmt, origin_stmt, self.statementKindLabel(current_line), self.lineProvenanceLabel(current_line),
                            }) catch "meta"
                    else
                        self.scratchFmt("stmt={d}  kind={s}/{s}", .{
                            stmt, self.statementKindLabel(current_line), self.lineProvenanceLabel(current_line),
                        }) catch "meta"
                else if (origin_stmt_id) |origin_stmt|
                    self.scratchFmt("origin={d}  kind={s}/{s}", .{
                        origin_stmt, self.statementKindLabel(current_line), self.lineProvenanceLabel(current_line),
                    }) catch "meta"
                    else
                        self.scratchFmt("kind={s}/{s}", .{
                            self.statementKindLabel(current_line), self.lineProvenanceLabel(current_line),
                        }) catch "meta";
                drawSegments(root, 1, row, &.{seg(meta_line, style_muted())});
                row += 1;
            }

            if (row < root.height) {
                const explain_line = if (self.isFoldedSourceLine(current_line))
                    self.foldedLineExplanation(current_line)
                else if (self.isRemovedSourceLine(current_line))
                    self.removedLineExplanation(current_line)
                else
                    self.scratchFmt("{s}; {s}", .{
                        self.statementKindExplanation(self.statementKindForLine(current_line)),
                        self.lineProvenanceExplanation(self.session.debugger.src_map.getLineProvenance(self.config.source_path, current_line)),
                    }) catch "explain";
                drawSegments(root, 1, row, &.{seg(explain_line, style_footer_note())});
                row += 2;
            }
        } else {
            drawSegments(root, 1, row, &.{seg(self.command_status, style_emphasis())});
            row += 2;
        }

        if (row < root.height) {
            drawSegments(root, 1, row, &.{seg("recent", style_muted())});
            row += 1;
        }

        const remaining_height: usize = if (root.height > row) root.height - row else 0;
        if (remaining_height == 0) return;
        const visible = @min(remaining_height, self.command_log.items.len);
        const start = self.command_log.items.len - visible;
        var i: usize = start;
        while (i < self.command_log.items.len and row < root.height) : ({
            i += 1;
            row += 1;
        }) {
            drawSegments(root, 1, row, &.{seg(self.command_log.items[i], style_footer_note())});
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
                const line = self.scratchFmt("{s}  value {s}", .{
                    self.describeTransientSlot(entry.key_ptr.slot) catch "slot",
                    self.scratchShortU256(entry.value_ptr.*),
                }) catch "slot";
                drawSegments(win, 1, row, &.{seg(line, if (changed) style_changed() else style_text())});
                row += 1;
                count += 1;
            }
            if (count == 0) {
                var any_it = self.session.evm.storage.transient.iterator();
                while (any_it.next()) |entry| {
                    if (row >= win.height) break;
                    const line = self.scratchFmt("{s}  value {s}", .{
                        self.describeTransientSlotWithAddress(entry.key_ptr.address, entry.key_ptr.slot) catch "slot",
                        self.scratchShortU256(entry.value_ptr.*),
                    }) catch "slot";
                    drawSegments(win, 1, row, &.{seg(line, style_text())});
                    row += 1;
                    count += 1;
                }
                if (count > 0) {
                    drawSegments(win, 1, 0, &.{seg("transient storage view | showing all tx transient slots", style_muted())});
                }
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

    fn describeTransientSlot(self: *Ui, slot: u256) ![]const u8 {
        if (!isLockSlot(slot)) {
            return self.scratchFmt("slot {s}", .{self.scratchFullU256(slot)});
        }

        const base_slot = slot - lock_prefix;
        if (self.lookupRuntimeRootName(base_slot)) |root_name| {
            return self.scratchFmt("lock({s})  slot {s}", .{
                root_name,
                self.scratchFullU256(slot),
            });
        }
        return self.scratchFmt("lock(slot {s})", .{self.scratchFullU256(base_slot)});
    }

    fn describeTransientSlotWithAddress(self: *Ui, address_bytes: [20]u8, slot: u256) ![]const u8 {
        if (isLockSlot(slot)) {
            const base_slot = slot - lock_prefix;
            if (self.lookupRuntimeRootName(base_slot)) |root_name| {
                return self.scratchFmt("lock({s})  addr 0x{s}  slot {s}", .{
                    root_name,
                    self.scratchShortAddressBytes(address_bytes),
                    self.scratchFullU256(slot),
                });
            }
            return self.scratchFmt("lock(slot {s})  addr 0x{s}", .{
                self.scratchFullU256(base_slot),
                self.scratchShortAddressBytes(address_bytes),
            });
        }
        return self.scratchFmt("slot {s}  addr 0x{s}", .{
            self.scratchFullU256(slot),
            self.scratchShortAddressBytes(address_bytes),
        });
    }

    fn scratchShortAddressBytes(self: *Ui, bytes: [20]u8) []const u8 {
        var raw: [40]u8 = undefined;
        _ = std.fmt.bufPrint(&raw, "{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}", .{
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4],
            bytes[5], bytes[6], bytes[7], bytes[8], bytes[9],
            bytes[10], bytes[11], bytes[12], bytes[13], bytes[14],
            bytes[15], bytes[16], bytes[17], bytes[18], bytes[19],
        }) catch return "????????";
        return self.scratchFmt("{s}..{s}", .{ raw[0..8], raw[36..40] }) catch "????????";
    }

    fn lookupRuntimeRootName(self: *Ui, slot: u256) ?[]const u8 {
        const current_idx = if (self.session.debugger.currentEntry()) |entry| entry.idx else return null;
        const bindings = self.session.debugger.getVisibleBindings(self.allocator) catch return null;
        defer self.allocator.free(bindings);
        _ = current_idx;

        for (bindings) |binding| {
            const location_slot = binding.runtime_location_slot orelse continue;
            const location_kind = binding.runtime_location_kind orelse continue;
            if (location_slot != slot) continue;
            if (std.mem.eql(u8, location_kind, "storage_root") or
                std.mem.eql(u8, location_kind, "tstore_root"))
            {
                return binding.runtime_location_root orelse binding.name;
            }
        }
        return null;
    }

    fn isLockSlot(slot: u256) bool {
        return (slot & lock_prefix) != 0;
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

    const BindingAvailability = enum {
        live,
        derived,
        not_initialized,
        out_of_scope,
        optimized_away,
        unavailable,
    };

    fn numericBindingValue(self: *Ui, binding: *const DebugInfo.VisibleBinding) !?u256 {
        return try self.session.debugger.getVisibleBindingValueByName(self.allocator, binding.name);
    }

    /// Return the AbiDoc that applies to a given frame.
    ///
    /// Resolution order:
    ///   1. Primary contract — when `frame.address == seed.contract`,
    ///      return the ABI loaded via `--abi <path>`.
    ///   2. Secondary registry — `--abi <hex-address>=<path>` entries
    ///      keyed by 20-byte address. Useful when the user is
    ///      stepping into an external callee whose ABI is known.
    ///   3. Otherwise null — `:bt` falls back to a raw selector hex.
    fn abiDocForFrame(self: *Ui, frame: *const Frame) ?*const AbiDoc {
        if (std.mem.eql(u8, &frame.address.bytes, &self.seed.contract.bytes)) {
            if (self.abi_doc) |*doc| return doc;
        }
        if (self.secondary_abi.getPtr(frame.address.bytes)) |doc| return doc;
        return null;
    }

    fn evalResolveBinding(ctx: *anyopaque, name: []const u8) ora_evm.debug_eval.EvalError!?ora_evm.debug_eval.Value {
        const self: *Ui = @alignCast(@ptrCast(ctx));
        const binding_opt = self.session.debugger.findVisibleBindingByName(self.allocator, name) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
        };
        const binding = binding_opt orelse return null;

        if (binding.folded_value) |folded_text| {
            const parsed = std.fmt.parseUnsigned(u256, folded_text, 0) catch return error.BindingUnavailable;
            return ora_evm.debug_eval.Value{ .num = parsed };
        }

        // numericBindingValue currently only returns OutOfMemory on the
        // error path; propagate it explicitly so the evaluator
        // surfaces it instead of mapping it to BindingUnavailable.
        const numeric_opt = try self.numericBindingValue(&binding);
        if (numeric_opt) |value| {
            return ora_evm.debug_eval.Value{ .num = value };
        }
        if (self.resolveAbiParamValue(&binding)) |resolved| {
            return switch (resolved) {
                .numeric => |n| ora_evm.debug_eval.Value{ .num = n },
                .text => error.BindingUnavailable,
            };
        }
        if (self.resolveIntrinsicLocalValue(&binding)) |resolved| {
            return switch (resolved) {
                .numeric => |n| ora_evm.debug_eval.Value{ .num = n },
                .text => error.BindingUnavailable,
            };
        }
        return error.BindingUnavailable;
    }

    fn evaluateExpr(self: *Ui, expr: []const u8) ora_evm.debug_eval.EvalError!ora_evm.debug_eval.Value {
        const resolver = ora_evm.debug_eval.Resolver{
            .ctx = @ptrCast(self),
            .resolveFn = evalResolveBinding,
        };
        return ora_evm.debug_eval.evaluate(expr, resolver);
    }

    fn resolvedBindingValue(self: *Ui, binding: *const DebugInfo.VisibleBinding) !?ResolvedBindingValue {
        if (try self.numericBindingValue(binding)) |value| {
            return .{ .numeric = value };
        }
        if (self.resolveAbiParamValue(binding)) |value| return value;
        return self.resolveIntrinsicLocalValue(binding);
    }

    fn bindingAvailability(
        self: *Ui,
        binding: *const DebugInfo.VisibleBinding,
        resolved_value: ?ResolvedBindingValue,
    ) BindingAvailability {
        if (self.bindingBeforeDeclaration(binding)) return .not_initialized;
        if (self.bindingPastLiveRange(binding)) return .out_of_scope;
        if (binding.is_folded or binding.folded_value != null) return .derived;
        if (std.mem.eql(u8, binding.runtime_kind, "optimized_out")) return .optimized_away;
        if (resolved_value != null) {
            if (std.mem.eql(u8, binding.runtime_kind, "ssa")) return .derived;
            return .live;
        }
        return .unavailable;
    }

    fn bindingAvailabilityStyle(self: *Ui, availability: BindingAvailability, changed: bool) Style {
        _ = self;
        if (changed) return style_changed();
        return switch (availability) {
            .live => style_text(),
            .derived => style_hint(),
            .not_initialized => style_hint(),
            .out_of_scope => style_dead(),
            .optimized_away => style_dead(),
            .unavailable => style_muted(),
        };
    }

    fn bindingBeforeDeclaration(self: *Ui, binding: *const DebugInfo.VisibleBinding) bool {
        const current_line = self.session.debugger.currentSourceLine() orelse return false;
        const decl = binding.decl orelse return false;
        return current_line < decl.start.line;
    }

    fn bindingPastLiveRange(self: *Ui, binding: *const DebugInfo.VisibleBinding) bool {
        const current_line = self.session.debugger.currentSourceLine() orelse return false;
        const live = binding.live orelse return false;
        return current_line > live.end.line;
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

    fn resolveIntrinsicLocalValue(self: *Ui, binding: *const DebugInfo.VisibleBinding) ?ResolvedBindingValue {
        if (!std.mem.eql(u8, binding.kind, "local")) return null;
        if (!std.mem.eql(u8, binding.runtime_kind, "ssa")) return null;

        const decl = binding.decl orelse return null;
        const decl_text = self.session.debugger.getSourceLineText(decl.start.line) orelse return null;
        const frame = self.session.evm.getCurrentFrame() orelse return null;

        if (std.mem.indexOf(u8, decl_text, "std.msg.sender()") != null or
            std.mem.indexOf(u8, decl_text, "msg.sender()") != null)
        {
            const start = self.render_scratch.items.len;
            self.render_scratch.writer(self.allocator).print("0x{x}", .{frame.caller.bytes}) catch return null;
            return .{ .text = self.render_scratch.items[start..] };
        }

        if (std.mem.indexOf(u8, decl_text, "std.msg.value()") != null or
            std.mem.indexOf(u8, decl_text, "msg.value()") != null)
        {
            return .{ .numeric = frame.value };
        }

        if (std.mem.indexOf(u8, decl_text, "std.block.timestamp()") != null or
            std.mem.indexOf(u8, decl_text, "std.block.timestamp") != null or
            std.mem.indexOf(u8, decl_text, "block.timestamp") != null)
        {
            return .{ .numeric = self.session.evm.block_context.block_timestamp };
        }

        return null;
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


    /// Try to decode a revert payload (as left in `frame.output` after a
    /// REVERT) against the loaded ABI. Returns `null` when the payload
    /// doesn't match a known custom error or string-revert; the caller
    /// should fall back to plain "execution_reverted" status. The returned
    /// slice is valid until `render_scratch` is reset.
    fn formatDecodedRevert(self: *Ui, payload: []const u8) ?[]const u8 {
        if (payload.len == 0) return null;
        const start = self.render_scratch.items.len;
        var writer = self.render_scratch.writer(self.allocator);

        if (payload.len >= 4 + 32 and std.mem.eql(u8, payload[0..4], &.{ 0x08, 0xc3, 0x79, 0xa0 })) {
            // Solidity-style Error(string) revert.
            const args = payload[4..];
            const offset = readU256BE(args[0..32]);
            if (offset == 32 and args.len >= 64) {
                const length: usize = @intCast(readU256BE(args[32..64]) & std.math.maxInt(u32));
                const bytes_start: usize = 64;
                const end = bytes_start + length;
                if (end <= args.len) {
                    writer.print("Error(\"{s}\")", .{args[bytes_start..end]}) catch return null;
                    return self.render_scratch.items[start..];
                }
            }
        }

        if (payload.len < 4) return null;
        const abi_doc = self.abi_doc orelse return null;
        var selector_buf: [4]u8 = undefined;
        @memcpy(&selector_buf, payload[0..4]);
        const error_callable = abi_doc.findErrorBySelector(selector_buf) orelse return null;

        const name = blk: {
            const n = error_callable.object.get("name") orelse break :blk @as([]const u8, "<error>");
            if (n != .string) break :blk @as([]const u8, "<error>");
            break :blk n.string;
        };
        writer.print("{s}(", .{name}) catch return null;

        const args = payload[4..];
        const inputs_value = error_callable.object.get("inputs");
        const inputs_array: []const std.json.Value = blk: {
            if (inputs_value) |iv| {
                if (iv == .array) break :blk iv.array.items;
            }
            break :blk &.{};
        };

        var i: usize = 0;
        while (i < inputs_array.len) : (i += 1) {
            if (i > 0) writer.writeAll(", ") catch return null;
            const input = inputs_array[i];
            const arg_name: []const u8 = blk: {
                if (input == .object) {
                    if (input.object.get("name")) |n| {
                        if (n == .string) break :blk n.string;
                    }
                }
                break :blk "_";
            };
            writer.print("{s}=", .{arg_name}) catch return null;
            const word_start = i * 32;
            const word_end = word_start + 32;
            if (word_end > args.len) {
                writer.writeAll("?") catch return null;
                continue;
            }
            const wire_type = abi_doc.formatInputType(input) orelse "uint256";
            writeAbiWord(writer, wire_type, args[word_start..word_end]) catch return null;
        }

        writer.writeAll(")") catch return null;
        return self.render_scratch.items[start..];
    }

    /// Decode a single emitted log against the loaded ABI. Returns a
    /// short human-readable string like `Transfer(from=0x..., to=0x...,
    /// amount=42)`. Returns `null` when no event in the ABI matches
    /// `topic[0]`. The returned slice is valid until `render_scratch`
    /// is reset.
    fn formatDecodedLog(self: *Ui, log_entry: ora_evm.Log) ?[]const u8 {
        if (log_entry.topics.len == 0) return null;
        const abi_doc = self.abi_doc orelse return null;
        const event = abi_doc.findEventByTopic0(log_entry.topics[0]) orelse return null;

        const name = blk: {
            const n = event.object.get("name") orelse break :blk @as([]const u8, "<event>");
            if (n != .string) break :blk @as([]const u8, "<event>");
            break :blk n.string;
        };

        const start = self.render_scratch.items.len;
        var writer = self.render_scratch.writer(self.allocator);
        writer.print("{s}(", .{name}) catch return null;

        const inputs_value = event.object.get("inputs");
        const inputs_array: []const std.json.Value = blk: {
            if (inputs_value) |iv| {
                if (iv == .array) break :blk iv.array.items;
            }
            break :blk &.{};
        };

        var topic_idx: usize = 1; // topic[0] is the selector
        var data_idx: usize = 0;
        var first = true;
        for (inputs_array) |input| {
            if (!first) writer.writeAll(", ") catch return null;
            first = false;

            const arg_name: []const u8 = blk: {
                if (input == .object) {
                    if (input.object.get("name")) |n| {
                        if (n == .string) break :blk n.string;
                    }
                }
                break :blk "_";
            };
            writer.print("{s}=", .{arg_name}) catch return null;
            const wire_type = abi_doc.formatInputType(input) orelse "uint256";

            if (AbiDoc.isInputIndexed(input)) {
                if (topic_idx >= log_entry.topics.len) {
                    writer.writeAll("?") catch return null;
                    continue;
                }
                var word: [32]u8 = undefined;
                writeU256BE(&word, log_entry.topics[topic_idx]);
                topic_idx += 1;
                writeAbiWord(writer, wire_type, &word) catch return null;
            } else {
                if (data_idx + 32 > log_entry.data.len) {
                    writer.writeAll("?") catch return null;
                    continue;
                }
                writeAbiWord(writer, wire_type, log_entry.data[data_idx .. data_idx + 32]) catch return null;
                data_idx += 32;
            }
        }
        writer.writeAll(")") catch return null;
        return self.render_scratch.items[start..];
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
            const bp = self.breakpoints.items[i];
            if (self.session.debugger.src_map.getLineProvenance(self.seed.source_path, bp.line)) |prov|
                try self.render_scratch.writer(self.allocator).print("  L{d}[{s}]", .{ bp.line, prov.label() })
            else
                try self.render_scratch.writer(self.allocator).print("  L{d}", .{bp.line});
            if (bp.condition != null) try self.render_scratch.writer(self.allocator).writeAll("?");
            if (bp.hit_target != null) try self.render_scratch.writer(self.allocator).print("@{d}/{d}", .{ bp.hit_count, bp.hit_target.? });
        }
        return self.render_scratch.items[start..];
    }

    fn drawSourceGutter(self: *Ui, win: Window, current_line: u32) void {
        const origin_line = self.currentOriginLine();
        var visible_row: u16 = 0;
        while (visible_row < win.height) : (visible_row += 1) {
            const line = self.scroll_line + visible_row;
            if (line > self.session.debugger.totalSourceLines()) break;
            const has_break = self.hasBreakpointLine(line);
            const is_current = line == current_line;
            const is_origin = if (origin_line) |origin| origin == line and origin != current_line else false;
            const kind = self.statementKindForLine(line);
            const provenance = self.session.debugger.src_map.getLineProvenance(self.config.source_path, line);
            const folded = self.isFoldedSourceLine(line);
            const removed = self.isRemovedSourceLine(line);
            const marker = if (has_break and is_current)
                ">"
            else if (has_break)
                "*"
            else if (is_origin)
                "^"
            else if (folded)
                "="
            else if (removed)
                "-"
            else if (provenance == .mixed)
                "+"
            else if (provenance == .synthetic)
                "~"
            else if (kind == .runtime_guard)
                "!"
            else if (kind != null)
                "."
            else
                " ";
            const style = if (has_break)
                style_changed()
            else if (is_origin)
                style_emphasis()
            else if (folded)
                style_hint()
            else if (removed)
                style_dead()
            else if (provenance == .mixed)
                style_changed()
            else if (provenance == .synthetic)
                style_guard()
            else if (kind == .runtime_guard)
                style_guard()
            else if (kind != null)
                style_muted()
            else
                style_muted();
            const line_cell = switch (self.overlay_mode) {
                .none => self.scratchFmt("{s}{d:>6} |", .{ marker, line }) catch marker,
                .coverage => blk: {
                    const hit = self.session.debugger.getLineHits(line);
                    if (hit) |h| {
                        break :blk self.scratchFmt("{s}{d:>6} {d:>4}|", .{ marker, line, h }) catch marker;
                    }
                    break :blk self.scratchFmt("{s}{d:>6}    .|", .{ marker, line }) catch marker;
                },
                .gas => blk: {
                    const gas = self.session.debugger.getLineGas(line);
                    if (gas) |g| {
                        break :blk self.scratchFmt("{s}{d:>6} {d:>7}|", .{ marker, line, g }) catch marker;
                    }
                    break :blk self.scratchFmt("{s}{d:>6}       .|", .{ marker, line }) catch marker;
                },
                .folded => blk: {
                    if (self.foldedValueForLine(line)) |fv| {
                        break :blk self.scratchFmt("{s}{d:>6} ={s}|", .{ marker, line, fv }) catch marker;
                    }
                    break :blk self.scratchFmt("{s}{d:>6}  |", .{ marker, line }) catch marker;
                },
                .hoist => blk: {
                    if (self.hoistOriginForLine(line)) |origin| {
                        break :blk self.scratchFmt("{s}{d:>6} <-{d}|", .{ marker, line, origin }) catch marker;
                    }
                    break :blk self.scratchFmt("{s}{d:>6}    |", .{ marker, line }) catch marker;
                },
            };
            drawSegments(win, 0, visible_row, &.{seg(line_cell, style)});
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
            if (bp.line == line) return true;
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

    fn exportTrace(self: *Ui, path: []const u8) !usize {
        return @import("debug_tui_session.zig").exportTrace(self, path);
    }

    fn writeSession(self: *Ui, path: []const u8) !void {
        return @import("debug_tui_session.zig").writeSession(self, path);
    }

    fn loadSession(self: *Ui, path: []const u8) !void {
        return @import("debug_tui_session.zig").loadSession(self, path);
    }
};

// seg, drawSegments, style_*, ascii_border_glyphs live in
// debug_tui_draw.zig; bring them back into the file's namespace
// via simple aliases so existing call sites read unchanged.
const draw = @import("debug_tui_draw.zig");
const seg = draw.seg;
const drawSegments = draw.drawSegments;
const style_header_title = draw.style_header_title;
const style_header_meta = draw.style_header_meta;
const style_footer_note = draw.style_footer_note;
const style_border = draw.style_border;
const style_title = draw.style_title;
const style_text = draw.style_text;
const style_emphasis = draw.style_emphasis;
const style_changed = draw.style_changed;
const style_guard = draw.style_guard;
const style_hint = draw.style_hint;
const style_muted = draw.style_muted;
const style_dead = draw.style_dead;
const style_tab_active = draw.style_tab_active;
const style_tab_inactive = draw.style_tab_inactive;
const style_error = draw.style_error;
const style_command_bg = draw.style_command_bg;
const style_command = draw.style_command;

fn tabLabel(tab: EvmTabKind) []const u8 {
    return switch (tab) {
        .stack => "Stack",
        .memory => "Memory",
        .storage => "Storage",
        .tstore => "TStore",
        .calldata => "Calldata",
    };
}

pub fn tabName(tab: EvmTabKind) []const u8 {
    return switch (tab) {
        .stack => "stack",
        .memory => "memory",
        .storage => "storage",
        .tstore => "tstore",
        .calldata => "calldata",
    };
}

pub fn parseTabName(name: []const u8) ?EvmTabKind {
    if (std.mem.eql(u8, name, "stack")) return .stack;
    if (std.mem.eql(u8, name, "memory")) return .memory;
    if (std.mem.eql(u8, name, "storage")) return .storage;
    if (std.mem.eql(u8, name, "tstore")) return .tstore;
    if (std.mem.eql(u8, name, "calldata")) return .calldata;
    return null;
}

pub fn stepModeName(mode: StepMode) []const u8 {
    return switch (mode) {
        .in => "in",
        .opcode => "opcode",
        .over => "over",
        .out => "out",
        .continue_ => "continue",
    };
}

pub fn parseStepMode(name: []const u8) ?StepMode {
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
    try configureTtyReadTimeout(&tty);
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
                const should_quit = try ui.handleKey(key);
                if (should_quit) {
                    running = false;
                    break;
                }
                ui.render(&vx);
                try vx.render(tty.writer());
            },
            .paste => |bytes| allocator.free(bytes),
            else => {},
        }
    }
}

fn configureTtyReadTimeout(tty: *vaxis.Tty) !void {
    if (builtin.os.tag == .windows) return;

    var termios = try std.posix.tcgetattr(tty.fd);
    // Vaxis stops its input thread by setting a flag and waking the read with
    // a terminal status query. Some terminals/PTYs do not answer that query;
    // a short read timeout lets the thread observe the flag without relying on
    // terminal-specific DSR behavior.
    termios.cc[@intFromEnum(std.posix.V.MIN)] = 0;
    termios.cc[@intFromEnum(std.posix.V.TIME)] = 1;
    try std.posix.tcsetattr(tty.fd, .NOW, termios);
}

pub fn loadSeedFromConfig(allocator: std.mem.Allocator, config: *const AppConfig) !SessionSeed {
    const limits = config.limits;

    const bytecode_hex = try ora_evm.loadDebuggerArtifactWithCap(allocator, config.bytecode_path, limits.artifact_max_bytes);
    defer allocator.free(bytecode_hex);
    const bytecode = try decodeHexAlloc(allocator, bytecode_hex);
    defer allocator.free(bytecode);

    const source_map_json = try ora_evm.loadDebuggerArtifactWithCap(allocator, config.source_map_path, limits.artifact_max_bytes);
    defer allocator.free(source_map_json);
    var source_map = try SourceMap.loadFromJson(allocator, source_map_json);
    errdefer source_map.deinit();

    const source_text = try ora_evm.loadDebuggerArtifactWithCap(allocator, config.source_path, limits.artifact_max_bytes);
    errdefer allocator.free(source_text);

    var debug_info_json: ?[]u8 = null;
    errdefer if (debug_info_json) |bytes| allocator.free(bytes);
    if (config.debug_info_path) |path| {
        debug_info_json = try ora_evm.loadDebuggerArtifactWithCap(allocator, path, limits.artifact_max_bytes);
    }

    var sir_text: ?[]u8 = null;
    errdefer if (sir_text) |bytes| allocator.free(bytes);
    if (config.sir_path) |path| {
        sir_text = ora_evm.loadDebuggerArtifactWithCap(allocator, path, limits.artifact_max_bytes) catch null;
    }

    const caller = primitives.Address.fromU256(0x100);
    const contract = primitives.Address.fromU256(0x200);

    var evm: Evm = undefined;
    try evm.init(allocator, null, null, ora_evm.deterministicBlockContext(), primitives.ZERO_ADDRESS, 0, null);
    defer evm.deinit();
    try evm.initTransactionState(null);
    try evm.preWarmTransaction(contract);

    const runtime_bytecode = SessionHelpers.deployRuntimeBytecode(allocator, &evm, .{
        .caller = caller,
        .contract = contract,
        .deployment_bytecode = bytecode,
        .init_calldata = config.init_calldata,
        .gas_limit = limits.gas_limit,
        .step_cap = limits.deploy_step_cap,
        .strict = true,
    }) catch |err| blk: {
        if (err != error.DeploymentRevertedWithNoRuntime or config.init_calldata_fallback.len == 0) {
            return err;
        }
        evm.deinit();
        try evm.init(allocator, null, null, ora_evm.deterministicBlockContext(), primitives.ZERO_ADDRESS, 0, null);
        try evm.initTransactionState(null);
        try evm.preWarmTransaction(contract);
        break :blk try SessionHelpers.deployRuntimeBytecode(allocator, &evm, .{
            .caller = caller,
            .contract = contract,
            .deployment_bytecode = bytecode,
            .init_calldata = config.init_calldata_fallback,
            .gas_limit = limits.gas_limit,
            .step_cap = limits.deploy_step_cap,
            .strict = true,
        });
    };
    errdefer allocator.free(runtime_bytecode);

    if (source_map.runtime_start_pc) |_| {
        const runtime_source_map = try SessionHelpers.rebaseSourceMapForRuntime(allocator, &source_map, runtime_bytecode);
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
        .limits = limits,
    };
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
            // Two forms accepted:
            //   --abi <abi.json>                     → primary contract
            //   --abi <hex-address>=<abi.json>       → external callee
            if (parseSecondaryAbiSpec(allocator, args[i])) |spec_opt| {
                if (spec_opt) |spec| {
                    try config.secondary_abi_specs.append(allocator, spec);
                } else {
                    if (config.abi_path) |path| allocator.free(path);
                    config.abi_path = try allocator.dupe(u8, args[i]);
                }
            } else |err| return err;
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

    if (init_signature) |sig| {
        allocator.free(config.init_calldata);
        allocator.free(config.init_calldata_fallback);
        if (config.abi_path) |abi_path| {
            const abi_bytes = try ora_evm.loadDebuggerArtifact(allocator, abi_path);
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
            const abi_bytes = try ora_evm.loadDebuggerArtifact(allocator, abi_path);
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
        \\  ora-evm-debug-tui <bytecode.hex> <source-map.json> <source.ora> [options]
        \\
        \\options:
        \\  --debug-info <debug.json>    Load source-scope debug info
        \\  --abi <abi.json>             Load primary contract's ABI (signature-driven calldata + decoded reverts/events)
        \\  --abi <0x..>=<abi.json>      Bind another ABI to an external callee address (repeatable; used by `:bt` decoded names)
        \\  --init-signature <sig>       Constructor signature, e.g. init(u256)
        \\  --init-arg <value>           Constructor argument (repeatable, in order)
        \\  --init-calldata-hex <hex>    Raw constructor calldata as hex
        \\  --signature <sig>            Function signature, e.g. add(u256,u256)
        \\  --arg <value>                Function argument (repeatable, in order)
        \\  --calldata-hex <hex>         Raw runtime calldata as hex
        \\  --gas-limit <i64>            Frame gas budget (default 5000000)
        \\  --max-steps <u64>            Per-command opcode safety cap (default 10000000)
        \\  --deploy-step-cap <usize>    Deployment opcode cap (default 200000)
        \\  --artifact-max-bytes <usize> Per-file artifact load cap (default 16777216)
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

/// Try to parse an `--abi` argument as a secondary spec
/// (`<hex-address>=<path>`). Returns:
///   - `null` if `arg` has no `=`, meaning the caller should treat it
///     as the primary `--abi <path>`.
///   - A `SecondaryAbiSpec` when a `=` is present and the LHS parses
///     as a 20-byte hex address; the returned `path` is owned by
///     `allocator` and must be freed by the caller.
fn parseSecondaryAbiSpec(allocator: std.mem.Allocator, arg: []const u8) !?SecondaryAbiSpec {
    const eq_idx = std.mem.indexOfScalar(u8, arg, '=') orelse return null;
    const addr_text = arg[0..eq_idx];
    const path_text = arg[eq_idx + 1 ..];
    if (path_text.len == 0) return error.InvalidArguments;
    const address_full = try primitives.Address.fromHex(addr_text);
    const path_dup = try allocator.dupe(u8, path_text);
    errdefer allocator.free(path_dup);
    return SecondaryAbiSpec{
        .address = address_full.bytes,
        .path = path_dup,
    };
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

pub fn decodeHexAlloc(allocator: std.mem.Allocator, text: []const u8) ![]u8 {
    const trimmed = std.mem.trim(u8, text, " \t\r\n");
    const hex = if (std.mem.startsWith(u8, trimmed, "0x")) trimmed[2..] else trimmed;
    if (hex.len % 2 != 0) return error.InvalidHex;

    const out = try allocator.alloc(u8, hex.len / 2);
    errdefer allocator.free(out);
    _ = try std.fmt.hexToBytes(out, hex);
    return out;
}
