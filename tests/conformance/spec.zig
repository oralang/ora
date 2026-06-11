const std = @import("std");
const abi = @import("abi.zig");
const slots = @import("slots.zig");
const types = @import("types.zig");

pub const ParsedSpec = struct {
    arena: std.heap.ArenaAllocator,
    value: types.Spec,

    pub fn deinit(self: *ParsedSpec) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

const DeployBuilder = struct {
    caller: ?types.Address = null,
    value: ?u256 = null,
    args: ?[]types.ArgValue = null,

    fn toSpec(self: *DeployBuilder) !types.DeploySpec {
        const caller = self.caller orelse return error.MissingRequiredField;
        const value = self.value orelse return error.MissingRequiredField;
        const args = self.args orelse return error.MissingRequiredField;
        self.args = null;
        return .{ .caller = caller, .value = value, .args = args };
    }
};

const StorageBuilder = struct {
    slot: ?u256 = null,
    value: ?u256 = null,

    fn toAssertion(self: StorageBuilder) !types.StorageAssertion {
        return .{
            .slot = self.slot orelse return error.MissingRequiredField,
            .value = self.value orelse return error.MissingRequiredField,
        };
    }
};

const LogBuilder = struct {
    topics: ?[]u256 = null,
    data: ?[]u8 = null,

    fn toAssertion(self: *LogBuilder) !types.LogAssertion {
        const topics = self.topics orelse return error.MissingRequiredField;
        const data = self.data orelse return error.MissingRequiredField;
        self.topics = null;
        self.data = null;
        return .{ .topics = topics, .data = data };
    }
};

const CallBuilder = struct {
    @"fn": ?[]const u8 = null,
    caller: ?types.Address = null,
    value: ?u256 = null,
    args: ?[]types.ArgValue = null,
    returns: ?types.ExpectedOutcome = null,
    reverts: ?types.ExpectedOutcome = null,
    storage: std.ArrayList(types.StorageAssertion) = .{},
    logs: std.ArrayList(types.LogAssertion) = .{},

    fn toSpec(self: *CallBuilder, allocator: std.mem.Allocator) !types.CallSpec {
        const expected_outcome = try self.outcome();
        const storage = try self.storage.toOwnedSlice(allocator);
        errdefer allocator.free(storage);
        const logs = try self.logs.toOwnedSlice(allocator);
        self.storage = .{};
        self.logs = .{};
        return .{
            .@"fn" = self.@"fn" orelse return error.MissingRequiredField,
            .caller = self.caller orelse return error.MissingRequiredField,
            .value = self.value orelse return error.MissingRequiredField,
            .args = self.args orelse return error.MissingRequiredField,
            .outcome = expected_outcome,
            .storage = storage,
            .logs = logs,
        };
    }

    fn outcome(self: *const CallBuilder) !types.ExpectedOutcome {
        if (self.returns != null and self.reverts != null) return error.MultipleOutcomes;
        return self.returns orelse self.reverts orelse error.MissingOutcome;
    }
};

const Section = enum {
    none,
    deploy,
    call,
    call_storage,
    call_log,
};

const KeyPresence = enum { present };
const single_section_map = std.StaticStringMap(Section).initComptime(.{.{ "deploy", .deploy }});
const double_section_map = std.StaticStringMap(Section).initComptime(.{
    .{ "call", .call },
    .{ "call.storage", .call_storage },
    .{ "call.log", .call_log },
});
const deploy_key_map = std.StaticStringMap(KeyPresence).initComptime(.{
    .{ "caller", .present },
    .{ "value", .present },
    .{ "args", .present },
});
const call_key_map = std.StaticStringMap(KeyPresence).initComptime(.{
    .{ "fn", .present },
    .{ "caller", .present },
    .{ "value", .present },
    .{ "args", .present },
    .{ "returns", .present },
    .{ "reverts", .present },
});
const storage_key_map = std.StaticStringMap(KeyPresence).initComptime(.{
    .{ "slot", .present },
    .{ "value", .present },
});
const log_key_map = std.StaticStringMap(KeyPresence).initComptime(.{
    .{ "topics", .present },
    .{ "data", .present },
});

pub fn parse(allocator: std.mem.Allocator, source: []const u8) !ParsedSpec {
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const value = try parseInto(arena.allocator(), source);
    return .{ .arena = arena, .value = value };
}

fn parseInto(allocator: std.mem.Allocator, source: []const u8) !types.Spec {
    var deploy = DeployBuilder{};
    var deploy_seen = false;
    var calls = std.ArrayList(types.CallSpec){};
    errdefer calls.deinit(allocator);

    var current_call: ?CallBuilder = null;
    var pending_storage: ?StorageBuilder = null;
    var pending_log: ?LogBuilder = null;

    var section: Section = .none;
    var lines = std.mem.splitScalar(u8, source, '\n');
    while (lines.next()) |raw_line| {
        const line = std.mem.trim(u8, stripComment(raw_line), " \t\r");
        if (line.len == 0) continue;

        if (std.mem.startsWith(u8, line, "[[")) {
            try flushPendingChild(allocator, &current_call, &pending_storage, &pending_log);
            const name = parseDoubleSectionName(line);
            section = double_section_map.get(name) orelse return error.UnknownSection;
            switch (section) {
                .call => {
                    try flushCurrentCall(allocator, &calls, &current_call);
                    current_call = .{};
                },
                .call_storage => {
                    if (current_call == null or pending_storage != null) return error.ExpectedCall;
                    pending_storage = .{};
                },
                .call_log => {
                    if (current_call == null or pending_log != null) return error.ExpectedCall;
                    pending_log = .{};
                },
                else => return error.UnknownSection,
            }
            continue;
        }

        if (std.mem.startsWith(u8, line, "[")) {
            try flushPendingChild(allocator, &current_call, &pending_storage, &pending_log);
            const name = parseSingleSectionName(line);
            section = single_section_map.get(name) orelse return error.UnknownSection;
            if (section == .deploy) {
                if (deploy_seen or current_call != null or calls.items.len != 0) return error.InvalidSectionOrder;
                deploy_seen = true;
            }
            continue;
        }

        const kv = try splitKeyValue(line);
        switch (section) {
            .deploy => try parseBuilderKey(DeployBuilder, allocator, &deploy, kv.key, kv.value),
            .call => {
                if (current_call == null) return error.ExpectedCall;
                try parseBuilderKey(CallBuilder, allocator, &current_call.?, kv.key, kv.value);
            },
            .call_storage => {
                if (pending_storage == null) return error.ExpectedCall;
                try parseBuilderKey(StorageBuilder, allocator, &pending_storage.?, kv.key, kv.value);
            },
            .call_log => {
                if (pending_log == null) return error.ExpectedCall;
                try parseBuilderKey(LogBuilder, allocator, &pending_log.?, kv.key, kv.value);
            },
            .none => return error.ExpectedSection,
        }
    }

    try flushPendingChild(allocator, &current_call, &pending_storage, &pending_log);
    try flushCurrentCall(allocator, &calls, &current_call);

    if (!deploy_seen) return error.MissingRequiredField;
    return .{
        .deploy = try deploy.toSpec(),
        .calls = try calls.toOwnedSlice(allocator),
    };
}

fn keyMap(comptime Builder: type) *const std.StaticStringMap(KeyPresence) {
    return if (Builder == DeployBuilder)
        &deploy_key_map
    else if (Builder == CallBuilder)
        &call_key_map
    else if (Builder == StorageBuilder)
        &storage_key_map
    else if (Builder == LogBuilder)
        &log_key_map
    else
        @compileError("unsupported conformance spec builder");
}

fn parseBuilderKey(comptime Builder: type, allocator: std.mem.Allocator, builder: *Builder, key: []const u8, value: []const u8) !void {
    comptime assertBuilderFieldsSupported(Builder);
    if (keyMap(Builder).get(key) == null) return error.UnknownKey;

    inline for (@typeInfo(Builder).@"struct".fields) |field| {
        if (comptime (std.mem.eql(u8, field.name, "storage") or std.mem.eql(u8, field.name, "logs"))) continue;
        if (std.mem.eql(u8, key, field.name)) {
            return assignField(Builder, allocator, builder, field.name, value);
        }
    }
    return error.UnknownKey;
}

fn assertBuilderFieldsSupported(comptime Builder: type) void {
    inline for (@typeInfo(Builder).@"struct".fields) |field| {
        if (comptime (std.mem.eql(u8, field.name, "storage") or std.mem.eql(u8, field.name, "logs"))) continue;
        _ = ParsedField(field.name, field.type);
    }
}

fn ParsedField(comptime field_name: []const u8, comptime FieldType: type) type {
    const child = switch (@typeInfo(FieldType)) {
        .optional => |opt| opt.child,
        else => @compileError("conformance spec field must be optional: " ++ field_name),
    };
    if (child == types.Address or child == u256 or child == []types.ArgValue or child == []const u8 or child == []u256 or child == []u8 or child == types.ExpectedOutcome) {
        return child;
    }
    @compileError("missing conformance spec parser for field: " ++ field_name);
}

fn assignField(comptime Builder: type, allocator: std.mem.Allocator, builder: *Builder, comptime field_name: []const u8, value: []const u8) !void {
    const FieldType = @TypeOf(@field(builder.*, field_name));
    const Parsed = ParsedField(field_name, FieldType);
    if (@field(builder.*, field_name) != null) {
        if (Parsed == types.ExpectedOutcome) return error.MultipleOutcomes;
        return error.DuplicateKey;
    }

    if (Parsed == types.Address) {
        @field(builder.*, field_name) = try types.Address.fromHex(try abi.parseString(value));
        return;
    }
    if (Parsed == u256) {
        @field(builder.*, field_name) = if (comptime std.mem.eql(u8, field_name, "slot"))
            try slots.parseSlotExpressionValue(value)
        else
            try abi.parseU256(value);
        return;
    }
    if (Parsed == []types.ArgValue) {
        @field(builder.*, field_name) = try abi.parseArgArray(allocator, value);
        return;
    }
    if (Parsed == []const u8) {
        @field(builder.*, field_name) = try abi.parseString(value);
        return;
    }
    if (Parsed == []u256) {
        @field(builder.*, field_name) = try parseTopicArray(allocator, value);
        return;
    }
    if (Parsed == []u8) {
        @field(builder.*, field_name) = try abi.parseHexBytes(allocator, try abi.parseString(value));
        return;
    }
    if (Parsed == types.ExpectedOutcome) {
        @field(builder.*, field_name) = if (comptime std.mem.eql(u8, field_name, "returns"))
            try parseReturns(value)
        else if (comptime std.mem.eql(u8, field_name, "reverts"))
            try parseReverts(allocator, value)
        else
            @compileError("ExpectedOutcome field must be returns or reverts");
        return;
    }
    @compileError("unhandled conformance spec field type in " ++ @typeName(Builder) ++ "." ++ field_name);
}

fn flushCurrentCall(
    allocator: std.mem.Allocator,
    calls: *std.ArrayList(types.CallSpec),
    current_call: *?CallBuilder,
) !void {
    if (current_call.*) |*call| {
        try calls.append(allocator, try call.toSpec(allocator));
        current_call.* = null;
    }
}

fn flushPendingChild(
    allocator: std.mem.Allocator,
    current_call: *?CallBuilder,
    pending_storage: *?StorageBuilder,
    pending_log: *?LogBuilder,
) !void {
    if (pending_storage.*) |storage| {
        if (current_call.* == null) return error.ExpectedCall;
        try current_call.*.?.storage.append(allocator, try storage.toAssertion());
        pending_storage.* = null;
    }
    if (pending_log.*) |*log| {
        if (current_call.* == null) return error.ExpectedCall;
        try current_call.*.?.logs.append(allocator, try log.toAssertion());
        pending_log.* = null;
    }
}

fn stripComment(line: []const u8) []const u8 {
    var in_string = false;
    var escaped = false;
    for (line, 0..) |c, i| {
        if (escaped) {
            escaped = false;
            continue;
        }
        if (in_string and c == '\\') {
            escaped = true;
            continue;
        }
        if (c == '"') {
            in_string = !in_string;
            continue;
        }
        if (!in_string and c == '#') return line[0..i];
    }
    return line;
}

fn parseDoubleSectionName(line: []const u8) []const u8 {
    if (!std.mem.endsWith(u8, line, "]]")) return "";
    return std.mem.trim(u8, line[2 .. line.len - 2], " \t");
}

fn parseSingleSectionName(line: []const u8) []const u8 {
    if (!std.mem.endsWith(u8, line, "]")) return "";
    return std.mem.trim(u8, line[1 .. line.len - 1], " \t");
}

const KeyValue = struct { key: []const u8, value: []const u8 };

fn splitKeyValue(line: []const u8) !KeyValue {
    const eq = std.mem.indexOfScalar(u8, line, '=') orelse return error.InvalidSpec;
    const key = std.mem.trim(u8, line[0..eq], " \t");
    const value = std.mem.trim(u8, line[eq + 1 ..], " \t");
    if (key.len == 0 or value.len == 0) return error.InvalidSpec;
    return .{ .key = key, .value = value };
}

fn parseInlineObject(value: []const u8) !KeyValue {
    const trimmed = std.mem.trim(u8, value, " \t");
    if (trimmed.len < 2 or trimmed[0] != '{' or trimmed[trimmed.len - 1] != '}') return error.InvalidSpec;
    const inner = std.mem.trim(u8, trimmed[1 .. trimmed.len - 1], " \t");
    if (hasTopLevelComma(inner)) return error.UnsupportedInlineObject;
    return try splitKeyValue(inner);
}

fn hasTopLevelComma(value: []const u8) bool {
    var paren_depth: usize = 0;
    var bracket_depth: usize = 0;
    var in_string = false;
    var escaped = false;

    for (value) |c| {
        if (escaped) {
            escaped = false;
            continue;
        }
        if (in_string and c == '\\') {
            escaped = true;
            continue;
        }
        if (c == '"') {
            in_string = !in_string;
            continue;
        }
        if (in_string) continue;

        switch (c) {
            '(' => paren_depth += 1,
            ')' => {
                if (paren_depth != 0) paren_depth -= 1;
            },
            '[' => bracket_depth += 1,
            ']' => {
                if (bracket_depth != 0) bracket_depth -= 1;
            },
            ',' => if (paren_depth == 0 and bracket_depth == 0) return true,
            else => {},
        }
    }

    return false;
}

fn parseReturns(value: []const u8) !types.ExpectedOutcome {
    const trimmed = std.mem.trim(u8, value, " \t");
    if (std.mem.eql(u8, trimmed, "{}")) return .returns_empty;
    const kv = try parseInlineObject(value);
    return .{ .returns_static = .{
        .spec_type = kv.key,
        .value = try abi.parseArgValue(kv.value),
    } };
}

fn parseReverts(allocator: std.mem.Allocator, value: []const u8) !types.ExpectedOutcome {
    const trimmed = std.mem.trim(u8, value, " \t");
    if (std.mem.eql(u8, trimmed, "{}")) return .reverts_empty;
    const kv = try parseInlineObject(value);
    if (std.mem.eql(u8, kv.key, "selector")) {
        return .{ .reverts_selector = try abi.parseSelector(try abi.parseString(kv.value)) };
    }
    if (std.mem.eql(u8, kv.key, "data")) {
        return .{ .reverts_data = try abi.parseHexBytes(allocator, try abi.parseString(kv.value)) };
    }
    return error.UnsupportedRevertExpectation;
}

fn parseTopicArray(allocator: std.mem.Allocator, value: []const u8) ![]u256 {
    const trimmed = std.mem.trim(u8, value, " \t");
    if (trimmed.len < 2 or trimmed[0] != '[' or trimmed[trimmed.len - 1] != ']') return error.InvalidSpec;
    const inner = std.mem.trim(u8, trimmed[1 .. trimmed.len - 1], " \t");
    if (inner.len == 0) return allocator.alloc(u256, 0);

    var topics = std.ArrayList(u256){};
    errdefer topics.deinit(allocator);
    var parts = std.mem.splitScalar(u8, inner, ',');
    while (parts.next()) |part| {
        const string = try abi.parseString(std.mem.trim(u8, part, " \t"));
        try topics.append(allocator, try abi.parseU256(string));
    }
    return try topics.toOwnedSlice(allocator);
}
