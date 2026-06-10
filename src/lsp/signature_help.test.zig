const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");

const line_index = ora_root.lsp.line_index;
const semantic_index = ora_root.lsp.semantic_index;
const signature_help = ora_root.lsp.signature_help;

test "lsp signature help: resolves callable through semantic name map" {
    const source =
        \\pub fn helper(first: u256, second: bool) -> u256 {
        \\    return first;
        \\}
        \\pub fn run() -> u256 {
        \\    return helper(1, true);
        \\}
    ;

    var index = try semantic_index.indexDocument(testing.allocator, source);
    defer index.deinit(testing.allocator);

    var lines = try line_index.LineIndex.init(testing.allocator, source);
    defer lines.deinit(testing.allocator);

    const comma = std.mem.indexOf(u8, source, "helper(1,") orelse return error.ExpectedCall;
    const position = lines.offsetToPosition(source, @intCast(comma + "helper(1,".len), .utf8);

    var response = (try signature_help.signatureAtIndex(
        testing.allocator,
        source,
        position,
        &index,
    )) orelse return error.ExpectedSignature;
    defer response.deinit(testing.allocator);

    try testing.expectEqualStrings("fn helper(first: u256, second: bool) -> u256", response.label);
    try testing.expectEqual(@as(u32, 1), response.active_parameter);
    try testing.expectEqual(@as(usize, 2), response.parameters.len);
    try testing.expectEqualStrings("first: u256", response.parameters[0].label);
    try testing.expectEqualStrings("second: bool", response.parameters[1].label);
}
