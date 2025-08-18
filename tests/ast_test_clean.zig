const std = @import("std");
const ora = @import("ora");

test "ast clean placeholder" {
    _ = ora;
    try std.testing.expect(true);
}
