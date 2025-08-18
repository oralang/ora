const std = @import("std");
const ora = @import("ora");

test "ast placeholder" {
    _ = ora; // ensure module links
    try std.testing.expect(true);
}
