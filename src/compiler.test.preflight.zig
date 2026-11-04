const std = @import("std");
const testing = std.testing;
const compiler_test_options = @import("compiler_test_options");

/// Exact `ora` artifact produced by the build graph compiling this test suite.
/// Formal CLI tests use this path instead of trusting the install prefix.
pub const ora_binary_path = compiler_test_options.ora_binary_path;

const installed_ora_binary_path = "zig-out/bin/ora";

fn fileSha256(path: []const u8) ![std.crypto.hash.sha2.Sha256.digest_length]u8 {
    var file = try std.Io.Dir.cwd().openFile(testing.io, path, .{});
    defer file.close(testing.io);

    var read_buffer: [64 * 1024]u8 = undefined;
    var chunk: [64 * 1024]u8 = undefined;
    var reader = file.reader(testing.io, &read_buffer);
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    while (true) {
        const count = try reader.interface.readSliceShort(&chunk);
        hasher.update(chunk[0..count]);
        if (count != chunk.len) return hasher.finalResult();
    }
}

fn requireCurrentInstalledBinary(expected: [32]u8, installed: [32]u8) !void {
    if (!std.mem.eql(u8, &expected, &installed))
        return error.StaleInstalledOraBinary;
}

test "formal CLI binary matches the current build graph artifact" {
    if (std.mem.eql(u8, ora_binary_path, installed_ora_binary_path))
        return error.FormalCliBinaryPathNotBuildArtifact;
    const expected = try fileSha256(ora_binary_path);
    const installed = fileSha256(installed_ora_binary_path) catch |err| switch (err) {
        error.FileNotFound => return error.InstalledOraBinaryMissing,
        else => return err,
    };
    try requireCurrentInstalledBinary(expected, installed);
}

test "formal CLI preflight rejects a different installed artifact" {
    try testing.expectError(
        error.StaleInstalledOraBinary,
        requireCurrentInstalledBinary(@as([32]u8, @splat(0)), @as([32]u8, @splat(1))),
    );
}
