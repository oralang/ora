const std = @import("std");
const Allocator = std.mem.Allocator;
const Ensip15 = @import("types.zig").Ensip15;
const OutputToken = @import("types.zig").OutputToken;

// === Label Operations ===

/// Split ENS name by '.' separator into labels
/// Returns slice of slices pointing into original name (caller must free outer slice)
pub fn split(allocator: Allocator, name: []const u8) ![][]const u8 {
    if (name.len == 0) {
        return &[_][]const u8{};
    }

    // Count dots to determine number of labels
    var count: usize = 1;
    for (name) |byte| {
        if (byte == '.') count += 1;
    }

    // Allocate array for label slices
    const labels = try allocator.alloc([]const u8, count);
    errdefer allocator.free(labels);

    var idx: usize = 0;
    var start: usize = 0;
    var i: usize = 0;

    while (i < name.len) : (i += 1) {
        if (name[i] == '.') {
            labels[idx] = name[start..i];
            idx += 1;
            start = i + 1;
        }
    }
    // Add final label
    labels[idx] = name[start..];

    return labels;
}

/// Join labels with '.' separator
/// Returns owned string (caller must free)
pub fn join(allocator: Allocator, labels: []const []const u8) ![]u8 {
    if (labels.len == 0) {
        return try allocator.alloc(u8, 0);
    }

    // Calculate total length
    var total_len: usize = 0;
    for (labels) |label| {
        total_len += label.len;
    }
    // Add separators (n-1 dots)
    if (labels.len > 1) {
        total_len += labels.len - 1;
    }

    // Allocate result
    const result = try allocator.alloc(u8, total_len);
    errdefer allocator.free(result);

    // Copy labels with separators
    var pos: usize = 0;
    for (labels, 0..) |label, i| {
        @memcpy(result[pos..][0..label.len], label);
        pos += label.len;
        if (i < labels.len - 1) {
            result[pos] = '.';
            pos += 1;
        }
    }

    return result;
}

// === Formatting Functions ===

/// Format codepoints as space-separated uppercase hex (e.g., "41 42 43")
pub fn toHexSequence(allocator: Allocator, cps: []const u21) ![]u8 {
    if (cps.len == 0) {
        return try allocator.alloc(u8, 0);
    }

    var list: std.ArrayListUnmanaged(u8) = .{};
    defer list.deinit(allocator);

    for (cps, 0..) |cp, i| {
        if (i > 0) {
            try list.append(allocator, ' ');
        }
        try appendHex(allocator, &list, cp);
    }

    return try list.toOwnedSlice(allocator);
}

/// Format single codepoint safely for display
/// Returns: "char" {HEX} or {HEX} depending on shouldEscape
pub fn safeCodepoint(self: *const Ensip15, allocator: Allocator, cp: u21) ![]u8 {
    var list: std.ArrayListUnmanaged(u8) = .{};
    defer list.deinit(allocator);

    if (!self.should_escape.contains(cp)) {
        try list.append(allocator, '"');
        try safeImplodeInternal(self, allocator, &list, &[_]u21{cp});
        try list.append(allocator, '"');
        try list.append(allocator, ' ');
    }
    try appendHexEscape(allocator, &list, cp);

    return try list.toOwnedSlice(allocator);
}

/// Format codepoint array safely for display
/// Handles combining marks, escaping, and bidi reset
pub fn safeImplode(self: *const Ensip15, allocator: Allocator, cps: []const u21) ![]u8 {
    var list: std.ArrayListUnmanaged(u8) = .{};
    defer list.deinit(allocator);

    try safeImplodeInternal(self, allocator, &list, cps);

    return try list.toOwnedSlice(allocator);
}

/// Internal helper for safeImplode that writes to ArrayList
fn safeImplodeInternal(self: *const Ensip15, allocator: Allocator, list: *std.ArrayListUnmanaged(u8), cps: []const u21) !void {
    if (cps.len == 0) {
        return;
    }

    // Prefix with U+25CC (◌) if first codepoint is combining mark
    if (self.combining_marks.contains(cps[0])) {
        try appendCodepoint(allocator, list, 0x25CC);
    }

    // Process each codepoint
    for (cps) |cp| {
        if (self.should_escape.contains(cp)) {
            try appendHexEscape(allocator, list, cp);
        } else {
            try appendCodepoint(allocator, list, cp);
        }
    }

    // Append U+200E (left-to-right mark) to reset bidi direction
    try appendCodepoint(allocator, list, 0x200E);
}

// === Rune Operations ===

/// Check if all codepoints are ASCII (< 0x80)
pub fn isAscii(cps: []const u21) bool {
    for (cps) |cp| {
        if (cp >= 0x80) {
            return false;
        }
    }
    return true;
}

/// Remove duplicate codepoints, preserving first occurrence order
pub fn uniqueRunes(allocator: Allocator, cps: []const u21) ![]u21 {
    var set = std.AutoHashMap(u21, void).init(allocator);
    defer set.deinit();

    var result: std.ArrayListUnmanaged(u21) = .{};
    defer result.deinit(allocator);

    for (cps) |cp| {
        const entry = try set.getOrPut(cp);
        if (!entry.found_existing) {
            try result.append(allocator, cp);
        }
    }

    return try result.toOwnedSlice(allocator);
}

/// Compare two codepoint slices lexicographically
/// Returns: negative if a < b, 0 if equal, positive if a > b
pub fn compareRunes(a: []const u21, b: []const u21) i32 {
    // First compare by length
    const len_diff: i32 = @as(i32, @intCast(a.len)) - @as(i32, @intCast(b.len));
    if (len_diff != 0) {
        return len_diff;
    }

    // Then compare lexicographically
    for (a, b) |a_cp, b_cp| {
        if (a_cp < b_cp) {
            return -1;
        } else if (a_cp > b_cp) {
            return 1;
        }
    }

    return 0;
}

// === Token Operations ===

/// Flatten OutputToken array into single codepoint array
pub fn flattenTokens(allocator: Allocator, tokens: []const OutputToken) ![]u21 {
    // Calculate total length
    var total_len: usize = 0;
    for (tokens) |token| {
        total_len += token.codepoints.len;
    }

    // Allocate result array
    const result = try allocator.alloc(u21, total_len);
    errdefer allocator.free(result);

    // Copy codepoints from each token
    var pos: usize = 0;
    for (tokens) |token| {
        @memcpy(result[pos..][0..token.codepoints.len], token.codepoints);
        pos += token.codepoints.len;
    }

    return result;
}

// === Internal Helpers ===

/// Helper: append single codepoint as hex to ArrayList (minimum 2 digits, uppercase)
fn appendHex(allocator: Allocator, list: *std.ArrayListUnmanaged(u8), cp: u21) !void {
    // Format as uppercase hex with minimum 2 digits
    const writer = list.writer(allocator);
    try std.fmt.format(writer, "{X:0>2}", .{cp});
}

/// Helper: append codepoint in {HEX} format to ArrayList
fn appendHexEscape(allocator: Allocator, list: *std.ArrayListUnmanaged(u8), cp: u21) !void {
    try list.append(allocator, '{');
    try appendHex(allocator, list, cp);
    try list.append(allocator, '}');
}

/// Helper: append a Unicode codepoint to ArrayList as UTF-8
fn appendCodepoint(allocator: Allocator, list: *std.ArrayListUnmanaged(u8), cp: u21) !void {
    var buf: [4]u8 = undefined;
    const len = std.unicode.utf8Encode(cp, &buf) catch {
        // If encoding fails, use replacement character
        const replacement_len = std.unicode.utf8Encode(0xFFFD, &buf) catch unreachable;
        try list.appendSlice(allocator, buf[0..replacement_len]);
        return;
    };
    try list.appendSlice(allocator, buf[0..len]);
}

// === Tests ===

test "split empty string" {
    const allocator = std.testing.allocator;
    const result = try split(allocator, "");
    defer allocator.free(result);
    try std.testing.expectEqual(@as(usize, 0), result.len);
}

test "split single label" {
    const allocator = std.testing.allocator;
    const result = try split(allocator, "eth");
    defer allocator.free(result);
    try std.testing.expectEqual(@as(usize, 1), result.len);
    try std.testing.expectEqualStrings("eth", result[0]);
}

test "split multiple labels" {
    const allocator = std.testing.allocator;
    const result = try split(allocator, "vitalik.eth");
    defer allocator.free(result);
    try std.testing.expectEqual(@as(usize, 2), result.len);
    try std.testing.expectEqualStrings("vitalik", result[0]);
    try std.testing.expectEqualStrings("eth", result[1]);
}

test "join empty array" {
    const allocator = std.testing.allocator;
    const result = try join(allocator, &[_][]const u8{});
    defer allocator.free(result);
    try std.testing.expectEqual(@as(usize, 0), result.len);
}

test "join single label" {
    const allocator = std.testing.allocator;
    const labels = [_][]const u8{"eth"};
    const result = try join(allocator, &labels);
    defer allocator.free(result);
    try std.testing.expectEqualStrings("eth", result);
}

test "join multiple labels" {
    const allocator = std.testing.allocator;
    const labels = [_][]const u8{ "vitalik", "eth" };
    const result = try join(allocator, &labels);
    defer allocator.free(result);
    try std.testing.expectEqualStrings("vitalik.eth", result);
}

test "toHexSequence empty" {
    const allocator = std.testing.allocator;
    const result = try toHexSequence(allocator, &[_]u21{});
    defer allocator.free(result);
    try std.testing.expectEqual(@as(usize, 0), result.len);
}

test "toHexSequence single codepoint" {
    const allocator = std.testing.allocator;
    const result = try toHexSequence(allocator, &[_]u21{0x41});
    defer allocator.free(result);
    try std.testing.expectEqualStrings("41", result);
}

test "toHexSequence multiple codepoints" {
    const allocator = std.testing.allocator;
    const result = try toHexSequence(allocator, &[_]u21{ 0x41, 0x42, 0x43 });
    defer allocator.free(result);
    try std.testing.expectEqualStrings("41 42 43", result);
}

test "toHexSequence with padding" {
    const allocator = std.testing.allocator;
    const result = try toHexSequence(allocator, &[_]u21{ 0x01, 0x0F, 0xFF });
    defer allocator.free(result);
    try std.testing.expectEqualStrings("01 0F FF", result);
}

test "isAscii empty" {
    try std.testing.expect(isAscii(&[_]u21{}));
}

test "isAscii all ascii" {
    try std.testing.expect(isAscii(&[_]u21{ 0x41, 0x42, 0x7F }));
}

test "isAscii with non-ascii" {
    try std.testing.expect(!isAscii(&[_]u21{ 0x41, 0x80 }));
}

test "uniqueRunes no duplicates" {
    const allocator = std.testing.allocator;
    const result = try uniqueRunes(allocator, &[_]u21{ 0x41, 0x42, 0x43 });
    defer allocator.free(result);
    try std.testing.expectEqual(@as(usize, 3), result.len);
}

test "uniqueRunes with duplicates" {
    const allocator = std.testing.allocator;
    const result = try uniqueRunes(allocator, &[_]u21{ 0x41, 0x42, 0x41, 0x43, 0x42 });
    defer allocator.free(result);
    try std.testing.expectEqual(@as(usize, 3), result.len);
    try std.testing.expectEqual(@as(u21, 0x41), result[0]);
    try std.testing.expectEqual(@as(u21, 0x42), result[1]);
    try std.testing.expectEqual(@as(u21, 0x43), result[2]);
}

test "compareRunes equal" {
    const a = [_]u21{ 0x41, 0x42 };
    const b = [_]u21{ 0x41, 0x42 };
    try std.testing.expectEqual(@as(i32, 0), compareRunes(&a, &b));
}

test "compareRunes different length" {
    const a = [_]u21{ 0x41, 0x42 };
    const b = [_]u21{0x41};
    try std.testing.expect(compareRunes(&a, &b) > 0);
    try std.testing.expect(compareRunes(&b, &a) < 0);
}

test "compareRunes lexicographic" {
    const a = [_]u21{ 0x41, 0x42 };
    const b = [_]u21{ 0x41, 0x43 };
    try std.testing.expect(compareRunes(&a, &b) < 0);
    try std.testing.expect(compareRunes(&b, &a) > 0);
}

test "flattenTokens empty" {
    const allocator = std.testing.allocator;
    const result = try flattenTokens(allocator, &[_]OutputToken{});
    defer allocator.free(result);
    try std.testing.expectEqual(@as(usize, 0), result.len);
}

test "flattenTokens single token" {
    const allocator = std.testing.allocator;
    const cps = [_]u21{ 0x41, 0x42 };
    const tokens = [_]OutputToken{.{ .codepoints = &cps, .emoji = null }};
    const result = try flattenTokens(allocator, &tokens);
    defer allocator.free(result);
    try std.testing.expectEqual(@as(usize, 2), result.len);
    try std.testing.expectEqual(@as(u21, 0x41), result[0]);
    try std.testing.expectEqual(@as(u21, 0x42), result[1]);
}

test "flattenTokens multiple tokens" {
    const allocator = std.testing.allocator;
    const cps1 = [_]u21{ 0x41, 0x42 };
    const cps2 = [_]u21{0x43};
    const tokens = [_]OutputToken{
        .{ .codepoints = &cps1, .emoji = null },
        .{ .codepoints = &cps2, .emoji = null },
    };
    const result = try flattenTokens(allocator, &tokens);
    defer allocator.free(result);
    try std.testing.expectEqual(@as(usize, 3), result.len);
    try std.testing.expectEqual(@as(u21, 0x41), result[0]);
    try std.testing.expectEqual(@as(u21, 0x42), result[1]);
    try std.testing.expectEqual(@as(u21, 0x43), result[2]);
}
