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
//!   Session, drains pending breakpoints into it, sends
//!   `initialized` event.
//! - `setBreakpoints` — stash before launch; replays into the
//!   debugger once a session exists.
//! - `threads` — single synthetic thread (the debugger has one
//!   logical thread of execution).
//! - `stackTrace` — frame chain from `Evm.frames`.
//! - `continue`, `next`, `stepIn`, `stepOut` — wired to
//!   `Session.debugger` methods.
//! - `pause` — sets the debugger state to paused.
//! - `disconnect` — ack and exit.
//!
//! NOT IMPLEMENTED:
//! - `scopes`, `variables`, `evaluate` — would surface the
//!   binding-by-name lookup, but that lives on `Ui` (depends on
//!   render_scratch); needs a small refactor to lift onto Session.
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

/// All session-lifetime state lives here so handlers don't pass it
/// piecemeal. The session + seed are owned (deinit on shutdown);
/// pending breakpoints are drained once `session_active` flips
/// true.
const ServerState = struct {
    allocator: std.mem.Allocator,
    pending_breakpoints: std.ArrayList(PendingBreakpoint) = .{},
    session: Session = undefined,
    seed: SessionSeed = undefined,
    config: AppConfig = undefined,
    session_active: bool = false,
    seq: i64 = 0,

    fn deinit(self: *ServerState) void {
        for (self.pending_breakpoints.items) |*bp| bp.deinit(self.allocator);
        self.pending_breakpoints.deinit(self.allocator);
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

    try writeAll(stderr_file, "ora-evm-debug-dap: ready (initialize+launch+setBreakpoints+threads+stackTrace+continue/next/stepIn/stepOut/pause+disconnect)\n");

    while (true) {
        const message_bytes = readDapMessage(allocator, stdin_file) catch |err| switch (err) {
            error.EndOfStream => return,
            else => {
                try writeAll(stderr_file, "ora-evm-debug-dap: read error\n");
                return err;
            },
        };
        defer allocator.free(message_bytes);

        try writeAll(stderr_file, "ora-evm-debug-dap: <- ");
        try writeAll(stderr_file, message_bytes);
        try writeAll(stderr_file, "\n");

        const parsed = std.json.parseFromSlice(std.json.Value, allocator, message_bytes, .{
            .ignore_unknown_fields = true,
        }) catch continue;
        defer parsed.deinit();

        const root = parsed.value;
        if (root != .object) continue;

        const command = root.object.get("command") orelse continue;
        if (command != .string) continue;
        const request_seq = blk: {
            const v = root.object.get("seq") orelse break :blk 0;
            break :blk switch (v) {
                .integer => |i| i,
                else => 0,
            };
        };

        const cmd = command.string;
        const seq = state.nextSeq();
        if (std.mem.eql(u8, cmd, "initialize")) {
            try writeInitializeResponse(allocator, stdout_file, seq, request_seq);
        } else if (std.mem.eql(u8, cmd, "launch")) {
            try handleLaunch(allocator, stdout_file, stderr_file, &state, seq, request_seq, root);
        } else if (std.mem.eql(u8, cmd, "setBreakpoints")) {
            try handleSetBreakpoints(allocator, stdout_file, stderr_file, seq, request_seq, root, &state.pending_breakpoints);
            // If a session is already active, replay the new set
            // immediately so the client's view stays consistent.
            if (state.session_active) try replayPendingBreakpoints(&state);
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

/// Drain `pending_breakpoints` into the now-existing debugger.
/// Best-effort: a setBreakpoint call that fails (e.g. line has no
/// statement entry) is logged but doesn't abort the drain.
fn replayPendingBreakpoints(state: *ServerState) !void {
    if (!state.session_active) return;
    for (state.pending_breakpoints.items) |bp| {
        // The DAP client paths are usually absolute; the
        // sourcemap also stores absolute paths, so a direct
        // setBreakpoint(path, line) typically resolves. If the
        // path doesn't match what the sourcemap recorded, the
        // call fails silently — same behavior as the TUI's
        // applyBreakpoints.
        _ = state.session.debugger.setBreakpoint(bp.path, bp.line);
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
            \\{{"id":{d},"name":"frame#{d}","line":{d},"column":1,"source":{{"path":"{s}"}}}}
        , .{ logical, logical, line, state.seed.source_path });
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

    if (result) {} else |err| {
        var dbg: [128]u8 = undefined;
        const msg = std.fmt.bufPrint(&dbg, "ora-evm-debug-dap: {s} error: {s}\n", .{ cmd, @errorName(err) }) catch "";
        try writeAll(stderr_file, msg);
    }

    try writeAck(allocator, stdout_file, seq, request_seq, cmd);

    // `stopped` event so the client knows where we landed.
    const reason = @tagName(state.session.debugger.stop_reason);
    const event = try std.fmt.allocPrint(allocator,
        \\{{"seq":{d},"type":"event","event":"stopped","body":{{"reason":"{s}","threadId":1}}}}
    , .{ state.nextSeq(), reason });
    defer allocator.free(event);
    try writeFramed(allocator, stdout_file, event);
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
    const body = try std.fmt.allocPrint(allocator,
        \\{{"seq":{d},"type":"response","request_seq":{d},"success":false,"command":"{s}","message":"{s}"}}
    , .{ seq, request_seq, command, message });
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
/// list and reply with one entry per breakpoint, all `verified:
/// false` since no Session exists to actually arm them yet. The
/// future launch handler will drain `pending_breakpoints` into the
/// debugger core.
fn handleSetBreakpoints(
    allocator: std.mem.Allocator,
    stdout_file: std.fs.File,
    stderr_file: std.fs.File,
    seq: i64,
    request_seq: i64,
    root: std.json.Value,
    pending_breakpoints: *std.ArrayList(PendingBreakpoint),
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

    // Parse the new breakpoints and append. Build the response in
    // parallel — one body entry per request entry, in order.
    var response_body: std.ArrayList(u8) = .{};
    defer response_body.deinit(allocator);
    var w = response_body.writer(allocator);
    try w.writeAll("[");
    var first = true;

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
                if (!first) try w.writeAll(",");
                first = false;
                try w.print(
                    \\{{"verified":false,"line":{d},"message":"no_session_yet"}}
                , .{line});
            }
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

/// Read one Content-Length-framed JSON message from `file`.
/// Returns the message body bytes (caller frees).
fn readDapMessage(allocator: std.mem.Allocator, file: std.fs.File) ![]u8 {
    var content_length: ?usize = null;
    while (true) {
        var line_buf: [256]u8 = undefined;
        // Distinguish "saw any byte this line" (including the CRLF
        // of an empty separator) from "stream gave us nothing" so
        // an empty separator line terminates headers cleanly while
        // an immediate EOF still surfaces as EndOfStream.
        var saw_any_byte = false;
        var i: usize = 0;
        while (true) {
            var b: [1]u8 = undefined;
            const n = try file.read(&b);
            if (n == 0) {
                if (!saw_any_byte) return error.EndOfStream;
                break;
            }
            saw_any_byte = true;
            const byte = b[0];
            if (byte == '\r') {
                // Expect \n; consume but tolerate.
                var nl: [1]u8 = undefined;
                _ = file.read(&nl) catch {};
                break;
            }
            if (byte == '\n') break;
            if (i >= line_buf.len) return error.HeaderTooLong;
            line_buf[i] = byte;
            i += 1;
        }
        const line = line_buf[0..i];
        if (line.len == 0) {
            if (!saw_any_byte) return error.EndOfStream;
            break;
        }
        const prefix = "Content-Length:";
        if (std.mem.startsWith(u8, line, prefix)) {
            const value = std.mem.trim(u8, line[prefix.len..], " \t");
            content_length = try std.fmt.parseInt(usize, value, 10);
        }
        // Other headers (e.g. Content-Type) are ignored.
    }
    const len = content_length orelse return error.MissingContentLength;
    const body = try allocator.alloc(u8, len);
    errdefer allocator.free(body);
    var read_total: usize = 0;
    while (read_total < len) {
        const n = try file.read(body[read_total..]);
        if (n == 0) return error.UnexpectedEndOfStream;
        read_total += n;
    }
    return body;
}

fn writeFramed(allocator: std.mem.Allocator, file: std.fs.File, body: []const u8) !void {
    const header = try std.fmt.allocPrint(allocator, "Content-Length: {d}\r\n\r\n", .{body.len});
    defer allocator.free(header);
    try file.writeAll(header);
    try file.writeAll(body);
}

fn writeInitializeResponse(allocator: std.mem.Allocator, file: std.fs.File, seq: i64, request_seq: i64) !void {
    const body = try std.fmt.allocPrint(allocator,
        \\{{"seq":{d},"type":"response","request_seq":{d},"success":true,"command":"initialize","body":{{"supportsConfigurationDoneRequest":false,"supportsSetVariable":false,"supportsRestartRequest":false}}}}
    , .{ seq, request_seq });
    defer allocator.free(body);
    try writeFramed(allocator, file, body);
}

fn writeAck(allocator: std.mem.Allocator, file: std.fs.File, seq: i64, request_seq: i64, command: []const u8) !void {
    const body = try std.fmt.allocPrint(allocator,
        \\{{"seq":{d},"type":"response","request_seq":{d},"success":true,"command":"{s}"}}
    , .{ seq, request_seq, command });
    defer allocator.free(body);
    try writeFramed(allocator, file, body);
}

fn writeNotImplemented(allocator: std.mem.Allocator, file: std.fs.File, seq: i64, request_seq: i64, command: []const u8) !void {
    const body = try std.fmt.allocPrint(allocator,
        \\{{"seq":{d},"type":"response","request_seq":{d},"success":false,"command":"{s}","message":"not_implemented"}}
    , .{ seq, request_seq, command });
    defer allocator.free(body);
    try writeFramed(allocator, file, body);
}
