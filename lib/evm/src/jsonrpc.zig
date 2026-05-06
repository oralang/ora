//! Content-Length-framed JSON-RPC helpers.
//!
//! Used by the DAP server (lib/evm/src/debug_dap.zig). Lives in
//! its own module so the framing + escaping primitives are
//! testable — `debug_dap.zig` is a binary, so tests defined
//! inside it never run; this module gets pulled in via
//! `root.zig`'s test target.
//!
//! Two responsibilities:
//!   1. Read / write Content-Length-framed messages over a
//!      file descriptor (stdin/stdout in production, an
//!      in-memory buffer in tests).
//!   2. Escape arbitrary byte strings as JSON string literals
//!      so paths, error messages, and command names with
//!      quotes / backslashes / control bytes don't corrupt
//!      the response stream.

const std = @import("std");

/// Escape `s` as the inside of a JSON string literal (no
/// surrounding quotes) into `writer`. Required wherever
/// user-controlled bytes (DAP commands, source paths, error
/// messages) flow into a JSON-RPC response — a stray quote or
/// backslash from an attacker-controlled path would otherwise
/// corrupt the framed stream.
pub fn writeJsonEscaped(writer: anytype, s: []const u8) !void {
    for (s) |c| {
        switch (c) {
            '"' => try writer.writeAll("\\\""),
            '\\' => try writer.writeAll("\\\\"),
            '\n' => try writer.writeAll("\\n"),
            '\r' => try writer.writeAll("\\r"),
            '\t' => try writer.writeAll("\\t"),
            0x08 => try writer.writeAll("\\b"),
            0x0c => try writer.writeAll("\\f"),
            0x00...0x07, 0x0b, 0x0e...0x1f => try writer.print("\\u{x:0>4}", .{c}),
            else => try writer.writeByte(c),
        }
    }
}

/// Allocator-returning convenience: produce a JSON-string-quoted
/// copy of `s` like `"foo\\nbar"` (with surrounding quotes
/// included). Caller frees.
pub fn allocJsonString(allocator: std.mem.Allocator, s: []const u8) ![]u8 {
    var buf: std.ArrayList(u8) = .{};
    errdefer buf.deinit(allocator);
    try buf.append(allocator, '"');
    try writeJsonEscaped(buf.writer(allocator), s);
    try buf.append(allocator, '"');
    return try buf.toOwnedSlice(allocator);
}

/// Write a Content-Length-framed body to `file`. Header is
/// allocated on `allocator` so we can control its lifetime; the
/// body slice is borrowed.
pub fn writeFramed(allocator: std.mem.Allocator, file: std.fs.File, body: []const u8) !void {
    const header = try std.fmt.allocPrint(allocator, "Content-Length: {d}\r\n\r\n", .{body.len});
    defer allocator.free(header);
    try file.writeAll(header);
    try file.writeAll(body);
}

/// Defensive cap on a single message body. 16 MiB matches the
/// debugger's artifact-load cap; a DAP request body larger than
/// that is almost certainly malformed or hostile. Without the
/// cap a remote client could announce `Content-Length:
/// 9999999999` and force an unbounded allocation.
pub const kMaxMessageBytes: usize = 16 * 1024 * 1024;

/// Read one Content-Length-framed message from `file`. Returns
/// the body bytes (caller frees). Errors:
///   - `error.EndOfStream` — clean EOF before any header byte
///   - `error.MissingContentLength` — headers ended without a
///     `Content-Length:` line
///   - `error.HeaderTooLong` — single header line longer than
///     256 bytes (defensive cap)
///   - `error.MessageTooLarge` — Content-Length exceeds
///     `kMaxMessageBytes`
///   - `error.UnexpectedEndOfStream` — header parsed OK but
///     body read short
pub fn readDapMessage(allocator: std.mem.Allocator, file: std.fs.File) ![]u8 {
    return readDapMessageReader(allocator, FileSource{ .file = file });
}

/// Read source abstraction so the framing logic can be tested
/// against in-memory buffers without a real file.
pub const Source = struct {
    ctx: *anyopaque,
    readFn: *const fn (ctx: *anyopaque, buf: []u8) anyerror!usize,
};

const FileSource = struct {
    file: std.fs.File,

    fn read(ctx: *anyopaque, buf: []u8) anyerror!usize {
        const self: *FileSource = @alignCast(@ptrCast(ctx));
        return self.file.read(buf);
    }
};

fn fileSourceAdapter(file_source: *FileSource) Source {
    return .{ .ctx = @ptrCast(file_source), .readFn = FileSource.read };
}

/// Generic-source variant. Used by the file-backed
/// `readDapMessage` and by tests with an in-memory `Source`.
pub fn readDapMessageFrom(allocator: std.mem.Allocator, source: Source) ![]u8 {
    return readDapMessageInner(allocator, source);
}

fn readDapMessageReader(allocator: std.mem.Allocator, file_source: FileSource) ![]u8 {
    var fs = file_source;
    return readDapMessageInner(allocator, fileSourceAdapter(&fs));
}

fn readDapMessageInner(allocator: std.mem.Allocator, source: Source) ![]u8 {
    var content_length: ?usize = null;
    while (true) {
        var line_buf: [256]u8 = undefined;
        // Distinguish "saw any byte this line" (including the
        // CRLF of an empty separator) from "stream gave us
        // nothing" so an empty separator line terminates
        // headers cleanly while an immediate EOF still
        // surfaces as EndOfStream.
        var saw_any_byte = false;
        var i: usize = 0;
        while (true) {
            var b: [1]u8 = undefined;
            const n = try source.readFn(source.ctx, &b);
            if (n == 0) {
                if (!saw_any_byte) return error.EndOfStream;
                break;
            }
            saw_any_byte = true;
            const byte = b[0];
            if (byte == '\r') {
                var nl: [1]u8 = undefined;
                _ = source.readFn(source.ctx, &nl) catch {};
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
    if (len > kMaxMessageBytes) return error.MessageTooLarge;
    const body = try allocator.alloc(u8, len);
    errdefer allocator.free(body);
    var read_total: usize = 0;
    while (read_total < len) {
        const n = try source.readFn(source.ctx, body[read_total..]);
        if (n == 0) return error.UnexpectedEndOfStream;
        read_total += n;
    }
    return body;
}

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;

fn escapedToOwned(allocator: std.mem.Allocator, s: []const u8) ![]u8 {
    var buf: std.ArrayList(u8) = .{};
    errdefer buf.deinit(allocator);
    try writeJsonEscaped(buf.writer(allocator), s);
    return try buf.toOwnedSlice(allocator);
}

test "writeJsonEscaped: plain ASCII passes through unchanged" {
    const out = try escapedToOwned(testing.allocator, "hello world");
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("hello world", out);
}

test "writeJsonEscaped: quotes and backslashes" {
    const out = try escapedToOwned(testing.allocator, "weird\"name\\with/quote");
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("weird\\\"name\\\\with/quote", out);
}

test "writeJsonEscaped: newlines, tabs, carriage returns" {
    const out = try escapedToOwned(testing.allocator, "a\nb\tc\rd");
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("a\\nb\\tc\\rd", out);
}

test "writeJsonEscaped: backspace and form feed" {
    const out = try escapedToOwned(testing.allocator, "a\x08b\x0cc");
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("a\\bb\\fc", out);
}

test "writeJsonEscaped: other control chars use \\uXXXX" {
    const out = try escapedToOwned(testing.allocator, "a\x01b\x1fc");
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("a\\u0001b\\u001fc", out);
}

test "writeJsonEscaped: NUL byte" {
    const out = try escapedToOwned(testing.allocator, "a\x00b");
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("a\\u0000b", out);
}

test "writeJsonEscaped: filesystem path with backslash on Windows-ish input" {
    const out = try escapedToOwned(testing.allocator, "C:\\Users\\name\\file.ora");
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("C:\\\\Users\\\\name\\\\file.ora", out);
}

test "allocJsonString: wraps with quotes and escapes" {
    const out = try allocJsonString(testing.allocator, "say \"hi\"");
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("\"say \\\"hi\\\"\"", out);
}

test "allocJsonString: empty string is just two quotes" {
    const out = try allocJsonString(testing.allocator, "");
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("\"\"", out);
}

test "writeJsonEscaped: round-trips through std.json parser" {
    const dirty = "field\"with\\quote and \nnewline and \x07bell";
    const wrapped = try allocJsonString(testing.allocator, dirty);
    defer testing.allocator.free(wrapped);

    const parsed = try std.json.parseFromSlice([]const u8, testing.allocator, wrapped, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings(dirty, parsed.value);
}

// =============================================================================
// Framing tests — drive `readDapMessage` against an in-memory
// byte buffer via the `Source` abstraction.
// =============================================================================

const BufferSource = struct {
    bytes: []const u8,
    cursor: usize = 0,

    fn read(ctx: *anyopaque, buf: []u8) anyerror!usize {
        const self: *BufferSource = @alignCast(@ptrCast(ctx));
        const remaining = self.bytes.len - self.cursor;
        const n = @min(buf.len, remaining);
        if (n == 0) return 0;
        @memcpy(buf[0..n], self.bytes[self.cursor .. self.cursor + n]);
        self.cursor += n;
        return n;
    }
};

fn readFromBytes(allocator: std.mem.Allocator, bytes: []const u8) ![]u8 {
    var src = BufferSource{ .bytes = bytes };
    return readDapMessageFrom(allocator, .{ .ctx = @ptrCast(&src), .readFn = BufferSource.read });
}

test "readDapMessage: simple framed body" {
    const bytes = "Content-Length: 12\r\n\r\n{\"hello\":1}\n";
    const body = try readFromBytes(testing.allocator, bytes);
    defer testing.allocator.free(body);
    try testing.expectEqualStrings("{\"hello\":1}\n", body);
}

test "readDapMessage: ignores non-Content-Length headers" {
    const bytes = "Content-Type: application/vscode-jsonrpc\r\nContent-Length: 4\r\n\r\nabcd";
    const body = try readFromBytes(testing.allocator, bytes);
    defer testing.allocator.free(body);
    try testing.expectEqualStrings("abcd", body);
}

test "readDapMessage: zero-length body is valid" {
    const bytes = "Content-Length: 0\r\n\r\n";
    const body = try readFromBytes(testing.allocator, bytes);
    defer testing.allocator.free(body);
    try testing.expectEqual(@as(usize, 0), body.len);
}

test "readDapMessage: missing Content-Length header errors" {
    const bytes = "Content-Type: x\r\n\r\nbody";
    try testing.expectError(error.MissingContentLength, readFromBytes(testing.allocator, bytes));
}

test "readDapMessage: clean EOF before any byte returns EndOfStream" {
    const bytes = "";
    try testing.expectError(error.EndOfStream, readFromBytes(testing.allocator, bytes));
}

test "readDapMessage: oversized Content-Length is rejected without allocating" {
    var buf: [128]u8 = undefined;
    const header = try std.fmt.bufPrint(&buf, "Content-Length: {d}\r\n\r\n", .{kMaxMessageBytes + 1});
    try testing.expectError(error.MessageTooLarge, readFromBytes(testing.allocator, header));
}

test "readDapMessage: header longer than 256 bytes errors" {
    var bytes: std.ArrayList(u8) = .{};
    defer bytes.deinit(testing.allocator);
    try bytes.appendSlice(testing.allocator, "X-Long: ");
    try bytes.appendNTimes(testing.allocator, 'a', 300);
    try bytes.appendSlice(testing.allocator, "\r\nContent-Length: 1\r\n\r\nx");
    try testing.expectError(error.HeaderTooLong, readFromBytes(testing.allocator, bytes.items));
}

test "readDapMessage: short body returns UnexpectedEndOfStream" {
    const bytes = "Content-Length: 10\r\n\r\nabc"; // header says 10, body is 3
    try testing.expectError(error.UnexpectedEndOfStream, readFromBytes(testing.allocator, bytes));
}

test "readDapMessage: tolerates LF-only line endings" {
    const bytes = "Content-Length: 4\n\nbody";
    const body = try readFromBytes(testing.allocator, bytes);
    defer testing.allocator.free(body);
    try testing.expectEqualStrings("body", body);
}
