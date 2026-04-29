//! Ora EVM debugger — Debug Adapter Protocol (DAP) server.
//!
//! Skeleton entry point. Today's scope:
//! - Reads Content-Length-framed JSON-RPC messages from stdin.
//! - Acknowledges with a minimal initialize response so a DAP client
//!   can complete the handshake without crashing.
//! - Logs every received request to stderr for debugging.
//!
//! NOT YET IMPLEMENTED (incremental future work):
//! - launch / setBreakpoints / threads / stackTrace / scopes /
//!   variables / evaluate / continue / next / stepIn / stepOut /
//!   pause / restart / disconnect.
//! - VS Code package.json `debuggers` contribution + launch.json
//!   schema (lives under editors/vscode/).
//!
//! The intent of this skeleton is to bound the scaffolding work:
//! once the framing + dispatch loop compiles and runs, subsequent
//! daily increments add one DAP request handler at a time.

const std = @import("std");

pub fn main() !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const allocator = gpa_state.allocator();

    const stdin_file = std.fs.File.stdin();
    const stdout_file = std.fs.File.stdout();
    const stderr_file = std.fs.File.stderr();

    try writeAll(stderr_file, "ora-evm-debug-dap: ready (skeleton only — initialize handshake replied, all other requests ignored)\n");

    var seq: i64 = 0;
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

        seq += 1;
        if (std.mem.eql(u8, command.string, "initialize")) {
            try writeInitializeResponse(allocator, stdout_file, seq, request_seq);
        } else if (std.mem.eql(u8, command.string, "disconnect")) {
            try writeAck(allocator, stdout_file, seq, request_seq, command.string);
            return;
        } else {
            // Ack-with-not-implemented so the client doesn't hang on a
            // dropped request. Future increments replace these with
            // real handlers.
            try writeNotImplemented(allocator, stdout_file, seq, request_seq, command.string);
        }
    }
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
