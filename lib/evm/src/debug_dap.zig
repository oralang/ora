//! Ora EVM debugger — Debug Adapter Protocol (DAP) server.
//!
//! Speaks Content-Length-framed JSON-RPC over stdio. Reuses the
//! TUI's Session/SessionSeed/loadSeedFromConfig machinery via a
//! direct import of debug_tui.zig — cheaper than carving a separate
//! shared module while the typed boundary is still moving.
//!
//! Implemented requests:
//! - `initialize` — handshake with minimal capabilities.
//! - `launch` — takes pre-compiled artifact paths
//!   (bytecodePath, sourceMapPath, sourcePath, optional
//!   debugInfoPath, abiPath, sirPath, calldataHex). Builds a
//!   Session + DebugController, drains pending breakpoints,
//!   sends `initialized` event.
//! - `setBreakpoints` — stash before launch; replays into the
//!   debugger once a session exists.
//! - `threads` — single synthetic thread (the debugger has one
//!   logical thread of execution).
//! - `stackTrace` — frame chain from `Evm.frames`.
//! - `continue`, `next`, `stepIn`, `stepOut` — wired to
//!   `Session.debugger` methods.
//! - `pause` — sets the debugger state to paused.
//! - `evaluate` — side-effect-free expression evaluation via
//!   `DebugController.evaluateExpr`. No TUI dependency on this
//!   path.
//! - `disconnect` — ack and exit.
//!
//! NOT IMPLEMENTED:
//! - `scopes`, `variables` — would surface visible bindings as
//!   structured DAP variables. Needs ABI-param decoding lifted
//!   onto DebugController; tracked separately.
//! - `restart`, `setExceptionBreakpoints`,
//!   `configurationDone` — capabilities advertise as unsupported.
//! - Compile-from-source: launch arguments are pre-compiled
//!   artifact paths only. A future client can shell out to
//!   `ora debug --no-tui` to produce them.

const std = @import("std");
const ora_evm = @import("ora_evm");
const tui = @import("debug_tui.zig");

const Session = tui.Session;
const SessionSeed = tui.SessionSeed;
const AppConfig = tui.AppConfig;
const DebugController = ora_evm.DebugController(.{});

// JSON-RPC framing + escaping live in their own module so the
// helpers are testable from root.zig's test target. This file
// stays a thin handler dispatcher.
const jsonrpc = ora_evm.jsonrpc;
const writeJsonEscaped = jsonrpc.writeJsonEscaped;
const allocJsonString = jsonrpc.allocJsonString;
const writeFramed = jsonrpc.writeFramed;
const readDapMessage = jsonrpc.readDapMessage;

/// One breakpoint requested by the client before a session exists.
/// Owned strings: `path` is duped from the parsed JSON so it
/// outlives the request frame.
const PendingBreakpoint = struct {
    path: []u8,
    line: u32,

    fn deinit(self: *PendingBreakpoint, allocator: std.mem.Allocator) void {
        allocator.free(self.path);
    }
};

/// One breakpoint that's currently installed in the debugger,
/// tracked so we can diff against `pending_breakpoints` and
/// remove old entries when a setBreakpoints request implicitly
/// retires them.
const InstalledBreakpoint = struct {
    path: []u8,
    line: u32,

    fn deinit(self: *InstalledBreakpoint, allocator: std.mem.Allocator) void {
        allocator.free(self.path);
    }
};

/// All session-lifetime state lives here so handlers don't pass it
/// piecemeal. The session + seed are owned (deinit on shutdown);
/// pending breakpoints store the client's current intent and are
/// replayed into the debugger once `session_active` flips true.
const ServerState = struct {
    allocator: std.mem.Allocator,
    pending_breakpoints: std.ArrayList(PendingBreakpoint) = .{},
    session: Session = undefined,
    seed: SessionSeed = undefined,
    config: AppConfig = undefined,
    /// Front-end-agnostic controller layer. Built right after
    /// `Session.init` succeeds. `evaluate` requests delegate
    /// here instead of importing TUI helpers.
    controller: DebugController = undefined,
    /// Breakpoints that are *actually installed* in the debugger
    /// (as opposed to `pending_breakpoints`, which holds the
    /// client's last-known intent). On every setBreakpoints
    /// replay we diff `pending_breakpoints` against this set:
    /// installed-but-not-pending entries get `removeBreakpoint`,
    /// pending-but-not-installed entries get `setBreakpoint`.
    /// Without this diff, a client unsetting a breakpoint via
    /// `setBreakpoints` would still hit it because the prior
    /// replay had pushed it into the debugger and never lifted
    /// it back out.
    installed_breakpoints: std.ArrayList(InstalledBreakpoint) = .{},
    session_active: bool = false,
    seq: i64 = 0,

    fn deinit(self: *ServerState) void {
        for (self.pending_breakpoints.items) |*bp| bp.deinit(self.allocator);
        self.pending_breakpoints.deinit(self.allocator);
        for (self.installed_breakpoints.items) |*bp| bp.deinit(self.allocator);
        self.installed_breakpoints.deinit(self.allocator);
        if (self.session_active) {
            self.session.deinit();
            self.seed.deinit(self.allocator);
            self.config.deinit(self.allocator);
        }
    }

    fn nextSeq(self: *ServerState) i64 {
        self.seq += 1;
        return self.seq;
    }
};

pub fn main() !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const allocator = gpa_state.allocator();

    var state = ServerState{ .allocator = allocator };
    defer state.deinit();

    const stdin_file = std.fs.File.stdin();
    const stdout_file = std.fs.File.stdout();
    const stderr_file = std.fs.File.stderr();

    // Inbound message logging is opt-in. By default we don't log
    // request bodies because they may carry user paths, calldata,
    // and other things the operator hasn't consented to write to
    // a log file. Set ORA_DAP_LOG=1 to enable for debugging.
    const verbose_logging: bool = blk: {
        const val = std.process.getEnvVarOwned(allocator, "ORA_DAP_LOG") catch break :blk false;
        defer allocator.free(val);
        break :blk val.len > 0 and !std.mem.eql(u8, val, "0");
    };

    try writeAll(stderr_file, "ora-evm-debug-dap: ready (initialize+launch+setBreakpoints+threads+stackTrace+continue/next/stepIn/stepOut/pause+evaluate+disconnect; set ORA_DAP_LOG=1 for inbound trace)\n");

    while (true) {
        const message_bytes = readDapMessage(allocator, stdin_file) catch |err| switch (err) {
            error.EndOfStream => return,
            else => {
                writeAll(stderr_file, "ora-evm-debug-dap: read error\n") catch {};
                return err;
            },
        };
        defer allocator.free(message_bytes);

        if (verbose_logging) {
            writeAll(stderr_file, "ora-evm-debug-dap: <- ") catch {};
            writeAll(stderr_file, message_bytes) catch {};
            writeAll(stderr_file, "\n") catch {};
        }

        // Malformed JSON or wrong shape used to silently `continue`,
        // leaving the client waiting for a response. Now we send a
        // proper error response so clients can recover, and the
        // server keeps reading subsequent messages.
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, message_bytes, .{
            .ignore_unknown_fields = true,
        }) catch {
            try writeError(allocator, stdout_file, state.nextSeq(), 0, "unknown", "malformed_json");
            continue;
        };
        defer parsed.deinit();

        const root = parsed.value;
        if (root != .object) {
            try writeError(allocator, stdout_file, state.nextSeq(), 0, "unknown", "request_not_an_object");
            continue;
        }

        // Pull request_seq up-front so error responses below can
        // reference it even when the command field is unusable.
        const request_seq = blk: {
            const v = root.object.get("seq") orelse break :blk 0;
            break :blk switch (v) {
                .integer => |i| i,
                else => 0,
            };
        };

        const command = root.object.get("command") orelse {
            try writeError(allocator, stdout_file, state.nextSeq(), request_seq, "unknown", "missing_command");
            continue;
        };
        if (command != .string) {
            try writeError(allocator, stdout_file, state.nextSeq(), request_seq, "unknown", "command_not_a_string");
            continue;
        }

        const cmd = command.string;
        const seq = state.nextSeq();
        if (std.mem.eql(u8, cmd, "initialize")) {
            try writeInitializeResponse(allocator, stdout_file, seq, request_seq);
        } else if (std.mem.eql(u8, cmd, "launch")) {
            try handleLaunch(allocator, stdout_file, stderr_file, &state, seq, request_seq, root);
        } else if (std.mem.eql(u8, cmd, "setBreakpoints")) {
            try handleSetBreakpoints(allocator, stdout_file, stderr_file, seq, request_seq, root, &state);
        } else if (std.mem.eql(u8, cmd, "threads")) {
            try handleThreads(allocator, stdout_file, seq, request_seq);
        } else if (std.mem.eql(u8, cmd, "stackTrace")) {
            try handleStackTrace(allocator, stdout_file, seq, request_seq, &state);
        } else if (std.mem.eql(u8, cmd, "continue") or std.mem.eql(u8, cmd, "next") or
            std.mem.eql(u8, cmd, "stepIn") or std.mem.eql(u8, cmd, "stepOut"))
        {
            try handleStep(allocator, stdout_file, stderr_file, seq, request_seq, &state, cmd);
        } else if (std.mem.eql(u8, cmd, "pause")) {
            try handlePause(allocator, stdout_file, seq, request_seq, &state);
        } else if (std.mem.eql(u8, cmd, "evaluate")) {
            try handleEvaluate(allocator, stdout_file, seq, request_seq, &state, root);
        } else if (std.mem.eql(u8, cmd, "disconnect")) {
            try writeAck(allocator, stdout_file, seq, request_seq, cmd);
            return;
        } else {
            try writeNotImplemented(allocator, stdout_file, seq, request_seq, cmd);
        }
    }
}

/// DAP `launch` handler.
///
/// Expected arguments (all paths absolute or cwd-relative):
///   { "command": "launch",
///     "arguments": {
///       "bytecodePath": "...hex",
///       "sourceMapPath": "...sourcemap.json",
///       "sourcePath":   "...ora",
///       "debugInfoPath": "...debug.json",  // optional
///       "abiPath":      "...abi.json",     // optional
///       "sirPath":      "...sir",          // optional
///       "calldataHex":  "0x..." } }        // optional
///
/// Builds a Session, drains pending breakpoints into the
/// debugger, and replies with success. Sends an `initialized`
/// event after the response so the client knows it can issue
/// further requests.
fn handleLaunch(
    allocator: std.mem.Allocator,
    stdout_file: std.fs.File,
    stderr_file: std.fs.File,
    state: *ServerState,
    seq: i64,
    request_seq: i64,
    root: std.json.Value,
) !void {
    if (state.session_active) {
        try writeError(allocator, stdout_file, seq, request_seq, "launch", "session already active; restart not implemented");
        return;
    }

    const arguments = root.object.get("arguments") orelse {
        try writeError(allocator, stdout_file, seq, request_seq, "launch", "missing arguments");
        return;
    };
    if (arguments != .object) {
        try writeError(allocator, stdout_file, seq, request_seq, "launch", "arguments not an object");
        return;
    }

    var config = AppConfig{
        .bytecode_path = undefined,
        .source_map_path = undefined,
        .source_path = undefined,
    };
    config.bytecode_path = try dupRequiredString(allocator, arguments, "bytecodePath") orelse {
        try writeError(allocator, stdout_file, seq, request_seq, "launch", "missing bytecodePath");
        return;
    };
    errdefer allocator.free(config.bytecode_path);
    config.source_map_path = try dupRequiredString(allocator, arguments, "sourceMapPath") orelse {
        try writeError(allocator, stdout_file, seq, request_seq, "launch", "missing sourceMapPath");
        return;
    };
    errdefer allocator.free(config.source_map_path);
    config.source_path = try dupRequiredString(allocator, arguments, "sourcePath") orelse {
        try writeError(allocator, stdout_file, seq, request_seq, "launch", "missing sourcePath");
        return;
    };
    errdefer allocator.free(config.source_path);
    config.debug_info_path = try dupOptionalString(allocator, arguments, "debugInfoPath");
    errdefer if (config.debug_info_path) |p| allocator.free(p);
    config.abi_path = try dupOptionalString(allocator, arguments, "abiPath");
    errdefer if (config.abi_path) |p| allocator.free(p);
    config.sir_path = try dupOptionalString(allocator, arguments, "sirPath");
    errdefer if (config.sir_path) |p| allocator.free(p);
    config.init_calldata = try allocator.dupe(u8, &.{});
    errdefer allocator.free(config.init_calldata);
    config.init_calldata_fallback = try allocator.dupe(u8, &.{});
    errdefer allocator.free(config.init_calldata_fallback);

    if (arguments.object.get("calldataHex")) |cd| {
        if (cd == .string) {
            config.calldata = tui.decodeHexAlloc(allocator, cd.string) catch |err| {
                allocator.free(config.bytecode_path);
                allocator.free(config.source_map_path);
                allocator.free(config.source_path);
                if (config.debug_info_path) |p| allocator.free(p);
                if (config.abi_path) |p| allocator.free(p);
                if (config.sir_path) |p| allocator.free(p);
                allocator.free(config.init_calldata);
                allocator.free(config.init_calldata_fallback);
                try writeError(allocator, stdout_file, seq, request_seq, "launch", @errorName(err));
                return;
            };
        } else {
            config.calldata = try allocator.dupe(u8, &.{});
        }
    } else {
        config.calldata = try allocator.dupe(u8, &.{});
    }
    errdefer allocator.free(config.calldata);

    state.seed = tui.loadSeedFromConfig(allocator, &config) catch |err| {
        var dbg: [128]u8 = undefined;
        const msg = std.fmt.bufPrint(&dbg, "ora-evm-debug-dap: launch failed: {s}\n", .{@errorName(err)}) catch "";
        try writeAll(stderr_file, msg);
        config.deinit(allocator);
        try writeError(allocator, stdout_file, seq, request_seq, "launch", @errorName(err));
        return;
    };
    errdefer state.seed.deinit(allocator);

    Session.init(&state.session, allocator, &state.seed) catch |err| {
        config.deinit(allocator);
        state.seed.deinit(allocator);
        try writeError(allocator, stdout_file, seq, request_seq, "launch", @errorName(err));
        return;
    };
    state.config = config;
    state.controller = DebugController.init(allocator, &state.session.debugger);
    state.session_active = true;

    // Drain pending breakpoints into the now-existing debugger.
    try replayPendingBreakpoints(state);

    // Reply success, then send initialized event.
    try writeAck(allocator, stdout_file, seq, request_seq, "launch");
    const event = try std.fmt.allocPrint(allocator,
        \\{{"seq":{d},"type":"event","event":"initialized"}}
    , .{state.nextSeq()});
    defer allocator.free(event);
    try writeFramed(allocator, stdout_file, event);
}

/// Sync the debugger's installed breakpoints to match the
/// client's stated intent in `pending_breakpoints`. Diff against
/// `installed_breakpoints` so old entries that the client has
/// implicitly retired (by sending a new full setBreakpoints list
/// for their source) get a real `removeBreakpoint` call —
/// otherwise an "unset" action would leave the breakpoint live in
/// the debugger and the client would hit it on the next continue.
///
/// Best-effort on the install side: a `setBreakpoint` that fails
/// (e.g. line has no statement entry) is silently dropped — same
/// behavior as the TUI's `applyBreakpoints`. The pending entry
/// stays in `pending_breakpoints` so a subsequent re-send /
/// re-launch tries again.
fn replayPendingBreakpoints(state: *ServerState) !void {
    if (!state.session_active) return;

    // 1. Remove installed entries that aren't in pending.
    var i: usize = 0;
    while (i < state.installed_breakpoints.items.len) {
        const inst = state.installed_breakpoints.items[i];
        const still_wanted = blk: {
            for (state.pending_breakpoints.items) |bp| {
                if (bp.line == inst.line and std.mem.eql(u8, bp.path, inst.path)) break :blk true;
            }
            break :blk false;
        };
        if (still_wanted) {
            i += 1;
        } else {
            state.session.debugger.removeBreakpoint(inst.path, inst.line);
            var removed = state.installed_breakpoints.swapRemove(i);
            removed.deinit(state.allocator);
        }
    }

    // 2. Install pending entries that aren't already installed.
    for (state.pending_breakpoints.items) |bp| {
        const already = blk: {
            for (state.installed_breakpoints.items) |inst| {
                if (inst.line == bp.line and std.mem.eql(u8, inst.path, bp.path)) break :blk true;
            }
            break :blk false;
        };
        if (already) continue;
        if (state.session.debugger.setBreakpoint(bp.path, bp.line)) {
            const path_dup = try state.allocator.dupe(u8, bp.path);
            errdefer state.allocator.free(path_dup);
            try state.installed_breakpoints.append(state.allocator, .{
                .path = path_dup,
                .line = bp.line,
            });
        }
    }
}

/// DAP `threads` response — the debugger executes a single logical
/// thread, so we always return one synthetic entry with id 1.
fn handleThreads(
    allocator: std.mem.Allocator,
    stdout_file: std.fs.File,
    seq: i64,
    request_seq: i64,
) !void {
    const body = try std.fmt.allocPrint(allocator,
        \\{{"seq":{d},"type":"response","request_seq":{d},"success":true,"command":"threads","body":{{"threads":[{{"id":1,"name":"ora-evm"}}]}}}}
    , .{ seq, request_seq });
    defer allocator.free(body);
    try writeFramed(allocator, stdout_file, body);
}

/// DAP `stackTrace` response — walks `evm.frames` and emits one
/// stackFrame per frame. Innermost (top) frame is index 0.
fn handleStackTrace(
    allocator: std.mem.Allocator,
    stdout_file: std.fs.File,
    seq: i64,
    request_seq: i64,
    state: *ServerState,
) !void {
    if (!state.session_active) {
        try writeError(allocator, stdout_file, seq, request_seq, "stackTrace", "no session");
        return;
    }

    var frames_body: std.ArrayList(u8) = .{};
    defer frames_body.deinit(allocator);
    var w = frames_body.writer(allocator);
    try w.writeAll("[");

    const frames = state.session.evm.frames.items;
    var logical: usize = 0;
    var i: usize = frames.len;
    while (i > 0) {
        i -= 1;
        const frame = frames[i];
        if (logical > 0) try w.writeAll(",");
        const line: u32 = if (i == frames.len - 1)
            (state.session.debugger.currentSourceLine() orelse 0)
        else
            0;
        try w.print(
            \\{{"id":{d},"name":"frame#{d}","line":{d},"column":1,"source":{{"path":
        , .{ logical, logical, line });
        try w.writeAll("\"");
        try writeJsonEscaped(w, state.seed.source_path);
        try w.writeAll("\"}}");
        _ = frame;
        logical += 1;
    }
    try w.writeAll("]");

    const body = try std.fmt.allocPrint(allocator,
        \\{{"seq":{d},"type":"response","request_seq":{d},"success":true,"command":"stackTrace","body":{{"stackFrames":{s},"totalFrames":{d}}}}}
    , .{ seq, request_seq, frames_body.items, frames.len });
    defer allocator.free(body);
    try writeFramed(allocator, stdout_file, body);
}

/// `continue` / `next` / `stepIn` / `stepOut` all map onto the
/// same `Session.debugger` methods. After the step we send a
/// `stopped` event so the client refreshes its view.
fn handleStep(
    allocator: std.mem.Allocator,
    stdout_file: std.fs.File,
    stderr_file: std.fs.File,
    seq: i64,
    request_seq: i64,
    state: *ServerState,
    cmd: []const u8,
) !void {
    if (!state.session_active) {
        try writeError(allocator, stdout_file, seq, request_seq, cmd, "no session");
        return;
    }

    const result: anyerror!void = if (std.mem.eql(u8, cmd, "continue"))
        state.session.debugger.continue_()
    else if (std.mem.eql(u8, cmd, "next"))
        state.session.debugger.stepOver()
    else if (std.mem.eql(u8, cmd, "stepIn"))
        state.session.debugger.stepIn()
    else if (std.mem.eql(u8, cmd, "stepOut"))
        state.session.debugger.stepOut()
    else
        return error.UnknownStepCommand;

    // If the step itself errored, surface that to the client as a
    // failed response — previously we acked success and emitted a
    // `stopped` event regardless, which left the UI thinking
    // execution had halted cleanly when in fact the EVM blew up.
    // The client can re-issue or roll back from a failed step
    // response; it can't recover from a phantom success.
    if (result) {} else |err| {
        var dbg: [128]u8 = undefined;
        const msg = std.fmt.bufPrint(&dbg, "ora-evm-debug-dap: {s} error: {s}\n", .{ cmd, @errorName(err) }) catch "";
        writeAll(stderr_file, msg) catch {};
        try writeError(allocator, stdout_file, seq, request_seq, cmd, @errorName(err));
        return;
    }

    try writeAck(allocator, stdout_file, seq, request_seq, cmd);

    // `stopped` event so the client knows where we landed.
    // `@tagName` of a Zig enum is a compile-time constant string
    // with no quotes/backslashes — safe to interpolate without
    // escaping.
    const reason = @tagName(state.session.debugger.stop_reason);
    const event = try std.fmt.allocPrint(allocator,
        \\{{"seq":{d},"type":"event","event":"stopped","body":{{"reason":"{s}","threadId":1}}}}
    , .{ state.nextSeq(), reason });
    defer allocator.free(event);
    try writeFramed(allocator, stdout_file, event);
}

/// DAP `evaluate` handler. The request shape is:
///
///   { "command": "evaluate",
///     "arguments": { "expression": "n + 1", "context": "watch" } }
///
/// The expression is evaluated against the current stop's
/// visible bindings via the front-end-agnostic
/// `DebugController.evaluateExpr`. No TUI dependency on this
/// path — DAP doesn't import any TUI helpers for evaluate.
///
/// Reply body:
///   { "result": "<decimal>", "variablesReference": 0 }
fn handleEvaluate(
    allocator: std.mem.Allocator,
    stdout_file: std.fs.File,
    seq: i64,
    request_seq: i64,
    state: *ServerState,
    root: std.json.Value,
) !void {
    if (!state.session_active) {
        try writeError(allocator, stdout_file, seq, request_seq, "evaluate", "no session");
        return;
    }
    const arguments = root.object.get("arguments") orelse {
        try writeError(allocator, stdout_file, seq, request_seq, "evaluate", "missing arguments");
        return;
    };
    if (arguments != .object) {
        try writeError(allocator, stdout_file, seq, request_seq, "evaluate", "arguments not an object");
        return;
    }
    const expr_val = arguments.object.get("expression") orelse {
        try writeError(allocator, stdout_file, seq, request_seq, "evaluate", "missing expression");
        return;
    };
    if (expr_val != .string) {
        try writeError(allocator, stdout_file, seq, request_seq, "evaluate", "expression not a string");
        return;
    }

    const value = state.controller.evaluateExpr(expr_val.string) catch |err| {
        const msg: []const u8 = switch (err) {
            error.ParseError => "parse error",
            error.UnknownIdentifier => "unknown identifier",
            error.BindingUnavailable => "binding unavailable",
            error.DivisionByZero => "division by zero",
            error.Overflow => "literal overflow",
            error.OutOfMemory => "out of memory",
        };
        try writeError(allocator, stdout_file, seq, request_seq, "evaluate", msg);
        return;
    };

    var result_buf: std.ArrayList(u8) = .{};
    defer result_buf.deinit(allocator);
    var w = result_buf.writer(allocator);
    switch (value) {
        .num => |n| try w.print("{d}", .{n}),
        .bool_ => |b| try w.writeAll(if (b) "true" else "false"),
    }
    const result_quoted = try allocJsonString(allocator, result_buf.items);
    defer allocator.free(result_quoted);

    const body = try std.fmt.allocPrint(allocator,
        \\{{"seq":{d},"type":"response","request_seq":{d},"success":true,"command":"evaluate","body":{{"result":{s},"variablesReference":0}}}}
    , .{ seq, request_seq, result_quoted });
    defer allocator.free(body);
    try writeFramed(allocator, stdout_file, body);
}

/// `pause` is a no-op on a single-threaded synchronous debugger:
/// each step request returns synchronously, so by the time the
/// client could send `pause` the debugger is already paused.
/// We ack and emit a `stopped` event for symmetry.
fn handlePause(
    allocator: std.mem.Allocator,
    stdout_file: std.fs.File,
    seq: i64,
    request_seq: i64,
    state: *ServerState,
) !void {
    if (!state.session_active) {
        try writeError(allocator, stdout_file, seq, request_seq, "pause", "no session");
        return;
    }
    try writeAck(allocator, stdout_file, seq, request_seq, "pause");
    const event = try std.fmt.allocPrint(allocator,
        \\{{"seq":{d},"type":"event","event":"stopped","body":{{"reason":"pause","threadId":1}}}}
    , .{state.nextSeq()});
    defer allocator.free(event);
    try writeFramed(allocator, stdout_file, event);
}

fn dupRequiredString(allocator: std.mem.Allocator, args: std.json.Value, key: []const u8) !?[]u8 {
    const v = args.object.get(key) orelse return null;
    if (v != .string) return null;
    return try allocator.dupe(u8, v.string);
}

fn dupOptionalString(allocator: std.mem.Allocator, args: std.json.Value, key: []const u8) !?[]u8 {
    const v = args.object.get(key) orelse return null;
    if (v != .string) return null;
    return try allocator.dupe(u8, v.string);
}

fn writeError(allocator: std.mem.Allocator, file: std.fs.File, seq: i64, request_seq: i64, command: []const u8, message: []const u8) !void {
    const cmd_quoted = try allocJsonString(allocator, command);
    defer allocator.free(cmd_quoted);
    const msg_quoted = try allocJsonString(allocator, message);
    defer allocator.free(msg_quoted);
    const body = try std.fmt.allocPrint(allocator,
        \\{{"seq":{d},"type":"response","request_seq":{d},"success":false,"command":{s},"message":{s}}}
    , .{ seq, request_seq, cmd_quoted, msg_quoted });
    defer allocator.free(body);
    try writeFramed(allocator, file, body);
}

/// DAP `setBreakpoints` handler. The request shape is:
///
///   { "command": "setBreakpoints",
///     "arguments": {
///       "source": { "path": "/path/to/file.ora" },
///       "breakpoints": [ { "line": 12 }, { "line": 17 } ] } }
///
/// We replace any prior breakpoints for `source.path` with the new
/// list. If a session is active, replay before responding so the
/// returned `verified` flags describe the actual debugger state.
/// Before launch, entries remain pending and are reported as
/// unverified with a `no_session_yet` message.
fn handleSetBreakpoints(
    allocator: std.mem.Allocator,
    stdout_file: std.fs.File,
    stderr_file: std.fs.File,
    seq: i64,
    request_seq: i64,
    root: std.json.Value,
    state: *ServerState,
) !void {
    const arguments = root.object.get("arguments") orelse {
        try writeNotImplemented(allocator, stdout_file, seq, request_seq, "setBreakpoints");
        return;
    };
    if (arguments != .object) {
        try writeNotImplemented(allocator, stdout_file, seq, request_seq, "setBreakpoints");
        return;
    }
    const source = arguments.object.get("source") orelse {
        try writeNotImplemented(allocator, stdout_file, seq, request_seq, "setBreakpoints");
        return;
    };
    if (source != .object) {
        try writeNotImplemented(allocator, stdout_file, seq, request_seq, "setBreakpoints");
        return;
    }
    const path_val = source.object.get("path") orelse {
        try writeNotImplemented(allocator, stdout_file, seq, request_seq, "setBreakpoints");
        return;
    };
    if (path_val != .string) {
        try writeNotImplemented(allocator, stdout_file, seq, request_seq, "setBreakpoints");
        return;
    }
    const path = path_val.string;
    const pending_breakpoints = &state.pending_breakpoints;

    // Drop any prior breakpoints for this source path before
    // installing the new set. DAP semantics: the client always
    // sends the *full* breakpoint list per source, so we replace.
    var i: usize = 0;
    while (i < pending_breakpoints.items.len) {
        if (std.mem.eql(u8, pending_breakpoints.items[i].path, path)) {
            var removed = pending_breakpoints.swapRemove(i);
            removed.deinit(allocator);
        } else {
            i += 1;
        }
    }

    // Parse the new breakpoints and append. Keep the requested
    // lines in order so the response mirrors the request after
    // active-session replay.
    var requested_lines: std.ArrayList(u32) = .{};
    defer requested_lines.deinit(allocator);

    if (arguments.object.get("breakpoints")) |bps| {
        if (bps == .array) {
            for (bps.array.items) |bp| {
                if (bp != .object) continue;
                const line_val = bp.object.get("line") orelse continue;
                const line: u32 = switch (line_val) {
                    .integer => |n| if (n < 0) continue else @intCast(n),
                    else => continue,
                };
                const path_dup = try allocator.dupe(u8, path);
                errdefer allocator.free(path_dup);
                try pending_breakpoints.append(allocator, .{
                    .path = path_dup,
                    .line = line,
                });
                try requested_lines.append(allocator, line);
            }
        }
    }

    if (state.session_active) try replayPendingBreakpoints(state);

    var response_body: std.ArrayList(u8) = .{};
    defer response_body.deinit(allocator);
    var w = response_body.writer(allocator);
    try w.writeAll("[");
    for (requested_lines.items, 0..) |line, index| {
        if (index > 0) try w.writeAll(",");
        if (!state.session_active) {
            try w.print(
                \\{{"verified":false,"line":{d},"message":"no_session_yet"}}
            , .{line});
            continue;
        }
        if (state.session.debugger.hasBreakpoint(path, line)) {
            try w.print(
                \\{{"verified":true,"line":{d}}}
            , .{line});
        } else {
            try w.print(
                \\{{"verified":false,"line":{d},"message":"unresolved_source_line"}}
            , .{line});
        }
    }
    try w.writeAll("]");

    // Diagnostic: show pending count after this update.
    var dbg: [128]u8 = undefined;
    const dbg_msg = std.fmt.bufPrint(&dbg, "ora-evm-debug-dap: pending breakpoints now {d}\n", .{pending_breakpoints.items.len}) catch "";
    try writeAll(stderr_file, dbg_msg);

    const out = try std.fmt.allocPrint(allocator,
        \\{{"seq":{d},"type":"response","request_seq":{d},"success":true,"command":"setBreakpoints","body":{{"breakpoints":{s}}}}}
    , .{ seq, request_seq, response_body.items });
    defer allocator.free(out);
    try writeFramed(allocator, stdout_file, out);
}

fn writeAll(file: std.fs.File, bytes: []const u8) !void {
    try file.writeAll(bytes);
}

fn writeInitializeResponse(allocator: std.mem.Allocator, file: std.fs.File, seq: i64, request_seq: i64) !void {
    const body = try std.fmt.allocPrint(allocator,
        \\{{"seq":{d},"type":"response","request_seq":{d},"success":true,"command":"initialize","body":{{"supportsConfigurationDoneRequest":false,"supportsSetVariable":false,"supportsRestartRequest":false}}}}
    , .{ seq, request_seq });
    defer allocator.free(body);
    try writeFramed(allocator, file, body);
}

fn writeAck(allocator: std.mem.Allocator, file: std.fs.File, seq: i64, request_seq: i64, command: []const u8) !void {
    const cmd_quoted = try allocJsonString(allocator, command);
    defer allocator.free(cmd_quoted);
    const body = try std.fmt.allocPrint(allocator,
        \\{{"seq":{d},"type":"response","request_seq":{d},"success":true,"command":{s}}}
    , .{ seq, request_seq, cmd_quoted });
    defer allocator.free(body);
    try writeFramed(allocator, file, body);
}

fn writeNotImplemented(allocator: std.mem.Allocator, file: std.fs.File, seq: i64, request_seq: i64, command: []const u8) !void {
    const cmd_quoted = try allocJsonString(allocator, command);
    defer allocator.free(cmd_quoted);
    const body = try std.fmt.allocPrint(allocator,
        \\{{"seq":{d},"type":"response","request_seq":{d},"success":false,"command":{s},"message":"not_implemented"}}
    , .{ seq, request_seq, cmd_quoted });
    defer allocator.free(body);
    try writeFramed(allocator, file, body);
}
