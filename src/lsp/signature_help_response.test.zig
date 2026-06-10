const std = @import("std");
const lsp = @import("lsp");
const ora_root = @import("ora_root");
const signature_help_response = @import("signature_help_response.zig");

const signature_help = ora_root.lsp.signature_help;
const types = lsp.types;

test "lsp signature help response: builds active signature and parameters" {
    var label = [_]u8{ 'f', 'o', 'o', '(', 'a', ':', ' ', 'u', '2', '5', '6', ')' };
    var doc = [_]u8{ 'F', 'o', 'o', ' ', 'd', 'o', 'c', 's' };
    var param_a = [_]u8{ 'a', ':', ' ', 'u', '2', '5', '6' };
    var param_b = [_]u8{ 'b', ':', ' ', 'b', 'o', 'o', 'l' };
    var params = [_]signature_help.ParameterInfo{
        .{ .label = param_a[0..] },
        .{ .label = param_b[0..] },
    };
    const signature = signature_help.SignatureInfo{
        .label = label[0..],
        .documentation = doc[0..],
        .parameters = params[0..],
        .active_parameter = 1,
    };

    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    const response = try signature_help_response.build(arena_state.allocator(), signature);

    try std.testing.expectEqual(@as(usize, 1), response.signatures.len);
    try std.testing.expectEqual(@as(u32, 0), response.activeSignature.?);
    try std.testing.expectEqual(@as(u32, 1), response.activeParameter.?);
    try std.testing.expectEqualStrings("foo(a: u256)", response.signatures[0].label);
    try std.testing.expectEqual(@as(u32, 1), response.signatures[0].activeParameter.?);
    try std.testing.expectEqual(@as(usize, 2), response.signatures[0].parameters.?.len);
    try std.testing.expectEqualStrings("b: bool", response.signatures[0].parameters.?[1].label.string);

    const documentation = response.signatures[0].documentation orelse return error.ExpectedDocumentation;
    switch (documentation) {
        .MarkupContent => |markup| {
            try std.testing.expectEqual(types.MarkupKind.markdown, markup.kind);
            try std.testing.expectEqualStrings("Foo docs", markup.value);
        },
        else => return error.ExpectedMarkupDocumentation,
    }

    const json = try std.json.Stringify.valueAlloc(std.testing.allocator, response, .{ .emit_null_optional_fields = false });
    defer std.testing.allocator.free(json);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"signatures\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "foo(a: u256)") != null);

    const RpcResponse = lsp.TypedJsonRPCResponse(?types.SignatureHelp);
    const rpc_json = try std.json.Stringify.valueAlloc(std.testing.allocator, RpcResponse{
        .id = .{ .number = 1 },
        .result_or_error = .{ .result = response },
    }, .{ .emit_null_optional_fields = false });
    defer std.testing.allocator.free(rpc_json);
    try std.testing.expect(std.mem.indexOf(u8, rpc_json, "\"result\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, rpc_json, "foo(a: u256)") != null);
}

test "lsp signature help response: omits documentation when absent" {
    var label = [_]u8{ 'f', 'o', 'o', '(', ')' };
    const signature = signature_help.SignatureInfo{
        .label = label[0..],
        .documentation = null,
        .parameters = &.{},
        .active_parameter = 0,
    };

    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    const response = try signature_help_response.build(arena_state.allocator(), signature);

    try std.testing.expectEqual(@as(usize, 1), response.signatures.len);
    try std.testing.expect(response.signatures[0].documentation == null);
    try std.testing.expectEqual(@as(usize, 0), response.signatures[0].parameters.?.len);
}
