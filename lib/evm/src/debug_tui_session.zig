//! Session save / load and EIP-3155 trace export for the EVM debugger TUI.
//!
//! Extracted out of debug_tui.zig so the JSON shape (`SavedSession`)
//! and its three I/O paths live in one place rather than 250 lines
//! of serialization at the bottom of a 5000-LOC file.
//!
//! All three entry points take `*Ui`. The Ui struct's matching
//! methods (`writeSession`, `loadSession`, `exportTrace`) are
//! one-line delegators back to here.

const std = @import("std");
const ora_evm = @import("ora_evm");
const tui = @import("debug_tui.zig");

const Ui = tui.Ui;
const AppConfig = tui.AppConfig;
const Session = tui.Session;
const SessionSeed = tui.SessionSeed;

/// JSON shape for `:write-session` output and `:load-session` input.
///
/// Captures everything the user did interactively that the bare seed
/// (bytecode + source map + calldata) doesn't already imply:
/// scroll position, focus, breakpoints (with their conditions and
/// hit targets), checkpoints, and the step history. Replaying the
/// step history against a fresh seed reproduces the exact stop the
/// session was saved at.
pub const SavedSession = struct {
    pub const SavedCheckpoint = struct {
        id: u32,
        step_index: usize,
        scroll_line: u32,
        focus_line: ?u32 = null,
        active_evm_tab: []const u8 = "stack",
    };

    pub const SavedBreakpoint = struct {
        line: u32,
        condition: ?[]const u8 = null,
        hit_target: ?u32 = null,
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
    breakpoints: []const SavedBreakpoint = &.{},
    checkpoints: []const SavedCheckpoint = &.{},
};

/// Export an EIP-3155 trace of the entire run by replaying
/// `step_history` against a shadow session with a `Tracer`
/// attached. Doesn't disturb the live session. Returns the number
/// of trace entries written.
pub fn exportTrace(ui: *Ui, path: []const u8) !usize {
    var shadow: Session = undefined;
    try Session.init(&shadow, ui.allocator, &ui.seed);
    defer shadow.deinit();

    var tracer = ora_evm.trace.Tracer.init(ui.allocator);
    defer tracer.deinit();
    tracer.enable();
    shadow.evm.setTracer(&tracer);

    // Apply the same breakpoints to the shadow session so a
    // `:continue_` in the user's history halts at the same place
    // it did live. Without this, replay diverges and the trace
    // wouldn't reflect what the user actually saw. Conditional /
    // hit-count gating is intentionally not replayed: the
    // debugger core doesn't know about it (it's a TUI-layer
    // concern), so a `:continue_` that was internally retried
    // until predicate-true is captured as a single user-step
    // here. The trace will land at the same final stop because
    // the EVM advances deterministically.
    for (ui.breakpoints.items) |bp| {
        _ = shadow.debugger.setBreakpoint(ui.seed.source_path, bp.line);
    }

    // Mirror primeInitialStop: advance the shadow session until the
    // first user-facing statement, so the trace's earliest entries
    // line up with what the user sees.
    var prime_attempts: usize = 0;
    while (!shadow.debugger.isHalted() and prime_attempts < 256) : (prime_attempts += 1) {
        shadow.debugger.stepIn() catch break;
        const entry = shadow.debugger.currentEntry() orelse continue;
        if (entry.is_statement) break;
    }

    for (ui.step_history.items) |mode| {
        const result = switch (mode) {
            .in => shadow.debugger.stepIn(),
            .opcode => shadow.debugger.stepOpcode(),
            .over => shadow.debugger.stepOver(),
            .out => shadow.debugger.stepOut(),
            .continue_ => shadow.debugger.continue_(),
        };
        result catch break;
        if (shadow.debugger.isHalted()) break;
    }

    try tracer.writeToFile(path);
    return tracer.entries.items.len;
}

/// Save the current session as JSON at `path`.
pub fn writeSession(ui: *Ui, path: []const u8) !void {
    var step_names = try ui.allocator.alloc([]const u8, ui.step_history.items.len);
    defer ui.allocator.free(step_names);
    for (ui.step_history.items, 0..) |mode, i| {
        step_names[i] = tui.stepModeName(mode);
    }
    const breakpoints = try ui.allocator.alloc(SavedSession.SavedBreakpoint, ui.breakpoints.items.len);
    defer ui.allocator.free(breakpoints);
    for (ui.breakpoints.items, 0..) |bp, i| {
        breakpoints[i] = .{
            .line = bp.line,
            .condition = bp.condition,
            .hit_target = bp.hit_target,
        };
    }
    var saved_checkpoints = try ui.allocator.alloc(SavedSession.SavedCheckpoint, ui.checkpoints.items.len);
    defer ui.allocator.free(saved_checkpoints);
    for (ui.checkpoints.items, 0..) |cp, i| {
        saved_checkpoints[i] = .{
            .id = cp.id,
            .step_index = cp.step_index,
            .scroll_line = cp.scroll_line,
            .focus_line = cp.focus_line,
            .active_evm_tab = tui.tabName(cp.active_evm_tab),
        };
    }

    const calldata_hex = try std.fmt.allocPrint(ui.allocator, "{x}", .{ui.seed.calldata});
    defer ui.allocator.free(calldata_hex);

    const session = SavedSession{
        .bytecode_path = ui.config.bytecode_path,
        .source_map_path = ui.config.source_map_path,
        .source_path = ui.config.source_path,
        .sir_path = ui.config.sir_path,
        .debug_info_path = ui.config.debug_info_path,
        .abi_path = ui.config.abi_path,
        .calldata_hex = calldata_hex,
        .scroll_line = ui.scroll_line,
        .sir_scroll_line = ui.sir_scroll_line,
        .sir_follow = ui.sir_follow,
        .focus_line = ui.focus_line,
        .active_evm_tab = tui.tabName(ui.active_evm_tab),
        .step_history = step_names,
        .breakpoints = breakpoints,
        .checkpoints = saved_checkpoints,
    };

    var json_buf: std.ArrayList(u8) = .{};
    defer json_buf.deinit(ui.allocator);
    var writer = json_buf.writer(ui.allocator);
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

/// Load a session JSON from `path`, replacing the current session
/// with one rebuilt from the saved seed and replayed step history.
pub fn loadSession(ui: *Ui, path: []const u8) !void {
    const json_bytes = try ora_evm.loadDebuggerArtifact(ui.allocator, path);
    defer ui.allocator.free(json_bytes);

    const parsed = try std.json.parseFromSlice(SavedSession, ui.allocator, json_bytes, .{
        .ignore_unknown_fields = true,
    });
    defer parsed.deinit();

    var new_config = AppConfig{
        .bytecode_path = try ui.allocator.dupe(u8, parsed.value.bytecode_path),
        .source_map_path = try ui.allocator.dupe(u8, parsed.value.source_map_path),
        .source_path = try ui.allocator.dupe(u8, parsed.value.source_path),
        .sir_path = if (parsed.value.sir_path) |p| try ui.allocator.dupe(u8, p) else null,
        .debug_info_path = if (parsed.value.debug_info_path) |p| try ui.allocator.dupe(u8, p) else null,
        .abi_path = if (parsed.value.abi_path) |p| try ui.allocator.dupe(u8, p) else null,
        .init_calldata = try ui.allocator.dupe(u8, &.{}),
        .init_calldata_fallback = try ui.allocator.dupe(u8, &.{}),
        .calldata = try tui.decodeHexAlloc(ui.allocator, parsed.value.calldata_hex),
    };
    errdefer new_config.deinit(ui.allocator);

    var new_seed = try tui.loadSeedFromConfig(ui.allocator, &new_config);
    errdefer new_seed.deinit(ui.allocator);

    ui.config.deinit(ui.allocator);
    ui.session.deinit();
    ui.seed.deinit(ui.allocator);
    ui.clearPreviousBindingsSnapshot();
    ui.step_history.clearRetainingCapacity();

    ui.config = new_config;
    ui.seed = new_seed;
    try Session.init(&ui.session, ui.allocator, &ui.seed);
    ui.controller = ora_evm.DebugController(.{}).init(ui.allocator, &ui.session.debugger);

    ui.command_mode = false;
    ui.command_buffer.clearRetainingCapacity();
    ui.active_evm_tab = tui.parseTabName(parsed.value.active_evm_tab) orelse .stack;
    for (ui.breakpoints.items) |*bp| bp.deinit(ui.allocator);
    ui.breakpoints.clearRetainingCapacity();
    for (parsed.value.breakpoints) |saved| {
        const condition_dup: ?[]u8 = if (saved.condition) |c| try ui.allocator.dupe(u8, c) else null;
        errdefer if (condition_dup) |c| ui.allocator.free(c);
        try ui.breakpoints.append(ui.allocator, .{
            .line = saved.line,
            .condition = condition_dup,
            .hit_target = saved.hit_target,
        });
    }
    ui.checkpoints.clearRetainingCapacity();
    ui.next_checkpoint_id = 1;
    for (parsed.value.checkpoints) |cp| {
        try ui.checkpoints.append(ui.allocator, .{
            .id = cp.id,
            .step_index = cp.step_index,
            .scroll_line = cp.scroll_line,
            .focus_line = cp.focus_line,
            .active_evm_tab = tui.parseTabName(cp.active_evm_tab) orelse .stack,
        });
        if (cp.id >= ui.next_checkpoint_id) ui.next_checkpoint_id = cp.id + 1;
    }

    ui.source_buffer.deinit(ui.allocator);
    ui.source_buffer = .{};
    try ui.source_buffer.update(ui.allocator, .{ .bytes = ui.seed.source_text });

    // loadSession just rebuilt the session and is about to replay
    // step_history; mirror rerunToHistory and start hit counts at 0.
    ui.resetBreakpointHitCounts();
    try ui.primeInitialStop();
    try ui.applyBreakpoints();
    for (parsed.value.step_history) |name| {
        const mode = tui.parseStepMode(name) orelse continue;
        ui.runStep(mode, true);
        if (std.mem.eql(u8, ui.status, "execution_error")) break;
    }
    ui.scroll_line = parsed.value.scroll_line;
    ui.sir_scroll_line = parsed.value.sir_scroll_line;
    ui.sir_follow = parsed.value.sir_follow;
    ui.focus_line = parsed.value.focus_line;
    ui.previous_snapshot = ui.captureSnapshot();
    try ui.refreshPreviousBindingsSnapshot();
    if (ui.focus_line == null) ui.syncFocusFromDebugger();
}
