const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");

const frontend = ora_root.lsp.frontend;

test "lsp frontend: valid source has no diagnostics" {
    const source =
        \\contract Test {
        \\    pub fn value() -> u256 {
        \\        return 42;
        \\    }
        \\}
    ;

    var analysis = try frontend.analyzeDocument(testing.allocator, source);
    defer analysis.deinit(testing.allocator);

    try testing.expect(analysis.parse_succeeded);
    try testing.expectEqual(@as(usize, 0), analysis.diagnostics.len);
}

test "lsp frontend: lexer error produces lexer diagnostic" {
    const source = "contract Test { $ }";

    var analysis = try frontend.analyzeDocument(testing.allocator, source);
    defer analysis.deinit(testing.allocator);

    try testing.expect(analysis.diagnostics.len > 0);

    var found_lexer = false;
    for (analysis.diagnostics) |diagnostic| {
        if (diagnostic.source == .lexer) {
            found_lexer = true;
            break;
        }
    }

    try testing.expect(found_lexer);
}

test "lsp frontend: parser error produces parser diagnostic" {
    const source = "@import(\"std\");";

    var analysis = try frontend.analyzeDocument(testing.allocator, source);
    defer analysis.deinit(testing.allocator);

    try testing.expect(!analysis.parse_succeeded);

    var found_parser = false;
    for (analysis.diagnostics) |diagnostic| {
        if (diagnostic.source == .parser) {
            found_parser = true;
            break;
        }
    }

    try testing.expect(found_parser);
}

test "lsp frontend: reports resolution diagnostics for invalid result pseudo variable use" {
    const source =
        \\pub fn bad(value: u256) -> u256
        \\    requires(result >= value)
        \\{
        \\    return value;
        \\}
    ;

    var analysis = try frontend.analyzeDocument(testing.allocator, source);
    defer analysis.deinit(testing.allocator);

    var found_resolution = false;
    for (analysis.diagnostics) |diagnostic| {
        if (diagnostic.source == .sema and std.mem.indexOf(u8, diagnostic.message, "undefined name 'result'") != null) {
            found_resolution = true;
            break;
        }
    }

    try testing.expect(found_resolution);
}
