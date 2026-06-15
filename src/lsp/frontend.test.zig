const std = @import("std");
const ora_root = @import("ora_root");

const frontend = ora_root.lsp.frontend;

test "lsp frontend: exposes shared range and diagnostic enums" {
    const range = frontend.Range{
        .start = .{ .line = 1, .character = 2 },
        .end = .{ .line = 3, .character = 4 },
    };

    try std.testing.expectEqual(@as(u32, 1), range.start.line);
    try std.testing.expectEqual(frontend.Severity.err, frontend.Severity.err);
    try std.testing.expectEqual(frontend.DiagnosticSource.sema, frontend.DiagnosticSource.sema);
}
