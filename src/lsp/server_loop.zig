const std = @import("std");
const lsp = @import("lsp");

const types = lsp.types;
const parse_stack_bytes = 64 * 1024;

pub fn run(
    allocator: std.mem.Allocator,
    transport: *lsp.Transport,
    handler_ptr: anytype,
    comptime logErr: ?fn (comptime fmt: []const u8, args: anytype) void,
) !void {
    const Handler = @TypeOf(handler_ptr.*);
    const Message = messageType(Handler);

    comptime std.debug.assert(@hasDecl(Handler, "initialize") or @hasField(Handler, "initialize"));

    while (true) {
        const json_message = try transport.readJsonMessage(allocator);
        defer allocator.free(json_message);

        var stack_fallback = std.heap.stackFallback(parse_stack_bytes, allocator);
        var arena_allocator: std.heap.ArenaAllocator = .init(stack_fallback.get());
        defer arena_allocator.deinit();

        const arena = arena_allocator.allocator();

        const message = Message.parseFromSliceLeaky(
            arena,
            json_message,
            .{ .ignore_unknown_fields = true, .max_value_len = null },
        ) catch |err| {
            if (logErr) |log| log("Failed to handle message: {}", .{err});
            try transport.writeErrorResponse(
                allocator,
                null,
                .{ .code = .parse_error, .message = @errorName(err) },
                .{ .emit_null_optional_fields = false },
            );
            continue;
        };

        switch (message) {
            .request => |request| switch (request.params) {
                inline else => |params, method_tag| {
                    const method: []const u8 = @tagName(method_tag);
                    if (callHandler(handler_ptr, method, .{ arena, params })) |result| {
                        if (@TypeOf(result) != lsp.ResultType(method)) {
                            @compileError(std.fmt.comptimePrint(
                                \\The '{}.{f}' function has an unexpected result type:
                                \\
                                \\Expected: {}
                                \\Actual:   {}
                            , .{ Handler, std.zig.fmtId(method), lsp.ResultType(method), @TypeOf(result) }));
                        }
                        try transport.writeResponse(
                            allocator,
                            request.id,
                            lsp.ResultType(method),
                            result,
                            .{ .emit_null_optional_fields = false },
                        );
                    } else |err| {
                        if (logErr) |log| log("Failed to handle '{s}' request: {}", .{ method, err });
                        var code: lsp.JsonRPCMessage.Response.Error.Code = .internal_error;
                        for (
                            [_]lsp.basic_server.Error{
                                error.ParseError,
                                error.InvalidRequest,
                                error.MethodNotFound,
                                error.InvalidParams,
                                error.InternalError,
                                error.ServerNotInitialized,
                                error.RequestFailed,
                                error.ServerCancelled,
                                error.ContentModified,
                                error.RequestCancelled,
                            },
                            [_]lsp.JsonRPCMessage.Response.Error.Code{
                                .parse_error,
                                .invalid_request,
                                .method_not_found,
                                .invalid_params,
                                .internal_error,
                                @enumFromInt(@intFromEnum(types.ErrorCodes.ServerNotInitialized)),
                                @enumFromInt(@intFromEnum(types.LSPErrorCodes.RequestFailed)),
                                @enumFromInt(@intFromEnum(types.LSPErrorCodes.ServerCancelled)),
                                @enumFromInt(@intFromEnum(types.LSPErrorCodes.ContentModified)),
                                @enumFromInt(@intFromEnum(types.LSPErrorCodes.RequestCancelled)),
                            },
                        ) |zig_err, new_code| {
                            if (err == zig_err) {
                                code = new_code;
                                break;
                            }
                        }

                        try transport.writeErrorResponse(
                            allocator,
                            request.id,
                            .{ .code = code, .message = @errorName(err) },
                            .{ .emit_null_optional_fields = false },
                        );
                    }
                },
                .other => {
                    try transport.writeResponse(
                        allocator,
                        request.id,
                        ?void,
                        null,
                        .{},
                    );
                },
            },
            .notification => |notification| switch (notification.params) {
                inline else => |params, method_tag| {
                    const method: []const u8 = @tagName(method_tag);
                    if (callHandler(handler_ptr, method, .{ arena, params })) |result| {
                        if (@TypeOf(result) != void) {
                            @compileError(std.fmt.comptimePrint(
                                \\The '{}.{f}' function has an unexpected result type:
                                \\
                                \\Expected: void
                                \\Actual:   {}
                            , .{ Handler, std.zig.fmtId(method), @TypeOf(result) }));
                        }
                    } else |err| {
                        if (logErr) |log| log("Failed to handle '{s}' notification: {}", .{ method, err });
                    }
                    if (std.mem.eql(u8, method, "exit")) break;
                },
                .other => {},
            },
            .response => |response| try callHandler(handler_ptr, "onResponse", .{ arena, response }),
        }
    }
}

fn messageType(comptime Handler: type) type {
    @setEvalBranchQuota(100_000);
    var RequestParams: type = undefined;
    var NotificationParams: type = undefined;

    for (
        &.{ lsp.isRequestMethod, lsp.isNotificationMethod },
        &.{ &RequestParams, &NotificationParams },
    ) |isMethod, Params| {
        var methods: []const [:0]const u8 = &.{};

        for (std.meta.declarations(Handler)) |decl| {
            if (isMethod(decl.name)) {
                methods = methods ++ [1][:0]const u8{decl.name};
            }
        }

        var enum_fields: [methods.len + 1]std.builtin.Type.EnumField = undefined;
        for (enum_fields[0 .. enum_fields.len - 1], methods, 0..) |*field, method, i| {
            field.* = .{ .name = method, .value = i };
        }
        enum_fields[methods.len] = .{ .name = "other", .value = methods.len };

        const MethodEnum = @Type(.{ .@"enum" = .{
            .tag_type = std.math.IntFittingRange(0, methods.len),
            .fields = &enum_fields,
            .decls = &.{},
            .is_exhaustive = true,
        } });

        var union_fields: [methods.len + 1]std.builtin.Type.UnionField = undefined;
        for (union_fields[0 .. union_fields.len - 1], methods) |*field, method| {
            const field_type = lsp.ParamsType(method);
            field.* = .{ .name = method, .type = field_type, .alignment = @alignOf(field_type) };
        }
        union_fields[methods.len] = .{ .name = "other", .type = lsp.MethodWithParams, .alignment = @alignOf(lsp.MethodWithParams) };

        Params.* = @Type(.{ .@"union" = .{
            .layout = .auto,
            .tag_type = MethodEnum,
            .fields = &union_fields,
            .decls = &.{},
        } });
    }

    return lsp.Message(RequestParams, NotificationParams, .{});
}

fn CallHandlerReturnType(comptime Handler: type, comptime fn_name: []const u8) type {
    if (!@hasDecl(Handler, fn_name) and !@hasField(Handler, fn_name)) {
        @compileError(std.fmt.comptimePrint("Could not find '{}.{f}'", .{ Handler, std.zig.fmtId(fn_name) }));
    }
    const func = @field(Handler, fn_name);
    const FuncUnwrapped = switch (@typeInfo(@TypeOf(func))) {
        .pointer => |info| switch (info.size) {
            .one => info.child,
            .many, .slice, .c => @TypeOf(func),
        },
        else => @TypeOf(func),
    };
    const ReturnType = switch (@typeInfo(FuncUnwrapped)) {
        .@"fn" => |info| info.return_type.?,
        else => @compileError(std.fmt.comptimePrint("Expected '{}.{f}' to be a function but was '{}'", .{ Handler, std.zig.fmtId(fn_name), FuncUnwrapped })),
    };
    return switch (@typeInfo(ReturnType)) {
        .error_union, .error_set => ReturnType,
        else => error{}!ReturnType,
    };
}

fn callHandler(
    handler_ptr: anytype,
    comptime fn_name: []const u8,
    args: anytype,
) CallHandlerReturnType(@TypeOf(handler_ptr.*), fn_name) {
    const Handler = @TypeOf(handler_ptr.*);
    const func = @field(Handler, fn_name);

    const fn_info = switch (@typeInfo(@TypeOf(func))) {
        .pointer => |info| switch (info.size) {
            .one => @typeInfo(info.child).@"fn",
            .many, .slice, .c => comptime unreachable,
        },
        .@"fn" => |info| info,
        else => comptime unreachable,
    };

    if (fn_info.params.len >= 1) switch (fn_info.params[0].type.?) {
        Handler, *Handler, *const Handler => {},
        else => return @call(.auto, func, args),
    } else return @call(.auto, func, args);

    return switch (fn_info.params[0].type.?) {
        Handler => @call(.auto, func, .{handler_ptr.*} ++ args),
        *Handler, *const Handler => @call(.auto, func, .{handler_ptr} ++ args),
        else => comptime unreachable,
    };
}
