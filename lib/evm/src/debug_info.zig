const std = @import("std");

pub const DebugInfo = struct {
    parsed: std.json.Parsed(JsonDebugInfo),

    pub const Position = struct {
        line: u32,
        col: u32,
        offset: ?u32 = null,
    };

    pub const Range = struct {
        start: Position,
        end: Position,
    };

    pub const Runtime = struct {
        pub const Location = struct {
            kind: []const u8,
            root: ?[]const u8 = null,
            slot: ?u64 = null,
        };

        kind: []const u8,
        name: ?[]const u8 = null,
        location: ?Location = null,
        editable: bool = false,
    };

    pub const Local = struct {
        id: u32,
        name: []const u8,
        kind: []const u8,
        binding_kind: ?[]const u8 = null,
        storage_class: ?[]const u8 = null,
        runtime: Runtime,
        folded_value: ?[]const u8 = null,
        decl: ?Range = null,
        live: ?Range = null,
    };

    pub const SourceScope = struct {
        id: u32,
        parent: ?u32 = null,
        file: []const u8,
        function: []const u8,
        contract: ?[]const u8 = null,
        kind: []const u8,
        label: ?[]const u8 = null,
        range: ?Range = null,
        locals: []const Local = &.{},
    };

    pub const OpVisibility = struct {
        idx: u32,
        scope_ids: []const u32 = &.{},
        visible_local_ids: []const u32 = &.{},
    };

    pub const OpMeta = struct {
        idx: u32,
        op: []const u8,
        function: []const u8,
        block: []const u8,
        result_names: []const []const u8 = &.{},
        is_terminator: bool = false,
        is_synthetic: bool = false,
        synthetic_index: ?u32 = null,
        synthetic_count: ?u32 = null,
    };

    pub const VisibleLocal = struct {
        local: *const Local,
        scope: ?*const SourceScope,
    };

    pub const VisibleBinding = struct {
        local_id: u32,
        name: []const u8,
        kind: []const u8,
        binding_kind: ?[]const u8 = null,
        storage_class: ?[]const u8 = null,
        runtime_kind: []const u8,
        runtime_name: ?[]const u8 = null,
        runtime_location_kind: ?[]const u8 = null,
        runtime_location_root: ?[]const u8 = null,
        runtime_location_slot: ?u64 = null,
        editable: bool,
        folded_value: ?[]const u8 = null,
        scope_id: ?u32 = null,
        scope_kind: ?[]const u8 = null,
        function: ?[]const u8 = null,
        contract: ?[]const u8 = null,
        file: ?[]const u8 = null,
    };

    pub fn loadFromJson(allocator: std.mem.Allocator, json_bytes: []const u8) !DebugInfo {
        const parsed = try std.json.parseFromSlice(JsonDebugInfo, allocator, json_bytes, .{
            .ignore_unknown_fields = true,
        });
        return .{ .parsed = parsed };
    }

    pub fn deinit(self: *DebugInfo) void {
        self.parsed.deinit();
    }

    pub fn version(self: *const DebugInfo) u32 {
        return self.parsed.value.version;
    }

    pub fn getVisibilityForIdx(self: *const DebugInfo, idx: u32) ?*const OpVisibility {
        for (self.parsed.value.op_visibility) |*op| {
            if (op.idx == idx) return op;
        }
        return null;
    }

    pub fn getOpMetaForIdx(self: *const DebugInfo, idx: u32) ?*const OpMeta {
        for (self.parsed.value.ops) |*op| {
            if (op.idx == idx) return op;
        }
        return null;
    }

    pub fn getScopeById(self: *const DebugInfo, id: u32) ?*const SourceScope {
        for (self.parsed.value.source_scopes) |*scope| {
            if (scope.id == id) return scope;
        }
        return null;
    }

    pub fn getLocalById(self: *const DebugInfo, id: u32) ?*const Local {
        for (self.parsed.value.source_scopes) |*scope| {
            for (scope.locals) |*local| {
                if (local.id == id) return local;
            }
        }
        return null;
    }

    pub fn collectVisibleLocals(self: *const DebugInfo, allocator: std.mem.Allocator, idx: u32) ![]VisibleLocal {
        const visibility = self.getVisibilityForIdx(idx) orelse return &.{};
        var locals: std.ArrayList(VisibleLocal) = .{};
        defer locals.deinit(allocator);

        for (visibility.visible_local_ids) |local_id| {
            const local = self.getLocalById(local_id) orelse continue;
            var owner_scope: ?*const SourceScope = null;
            for (visibility.scope_ids) |scope_id| {
                const scope = self.getScopeById(scope_id) orelse continue;
                for (scope.locals) |*scope_local| {
                    if (scope_local.id == local_id) {
                        owner_scope = scope;
                        break;
                    }
                }
                if (owner_scope != null) break;
            }
            try locals.append(allocator, .{ .local = local, .scope = owner_scope });
        }

        return try locals.toOwnedSlice(allocator);
    }

    pub fn collectVisibleScopes(self: *const DebugInfo, allocator: std.mem.Allocator, idx: u32) ![]*const SourceScope {
        const visibility = self.getVisibilityForIdx(idx) orelse return &.{};
        var scopes: std.ArrayList(*const SourceScope) = .{};
        defer scopes.deinit(allocator);

        for (visibility.scope_ids) |scope_id| {
            const scope = self.getScopeById(scope_id) orelse continue;
            try scopes.append(allocator, scope);
        }

        return try scopes.toOwnedSlice(allocator);
    }

    pub fn collectVisibleBindings(self: *const DebugInfo, allocator: std.mem.Allocator, idx: u32) ![]VisibleBinding {
        const visible_locals = try self.collectVisibleLocals(allocator, idx);
        defer allocator.free(visible_locals);

        var bindings: std.ArrayList(VisibleBinding) = .{};
        defer bindings.deinit(allocator);

        for (visible_locals) |visible| {
            const scope = visible.scope;
            try bindings.append(allocator, .{
                .local_id = visible.local.id,
                .name = visible.local.name,
                .kind = visible.local.kind,
                .binding_kind = visible.local.binding_kind,
                .storage_class = visible.local.storage_class,
                .runtime_kind = visible.local.runtime.kind,
                .runtime_name = visible.local.runtime.name,
                .runtime_location_kind = if (visible.local.runtime.location) |location| location.kind else null,
                .runtime_location_root = if (visible.local.runtime.location) |location| location.root else null,
                .runtime_location_slot = if (visible.local.runtime.location) |location| location.slot else null,
                .editable = visible.local.runtime.editable,
                .folded_value = visible.local.folded_value,
                .scope_id = if (scope) |s| s.id else null,
                .scope_kind = if (scope) |s| s.kind else null,
                .function = if (scope) |s| s.function else null,
                .contract = if (scope) |s| s.contract else null,
                .file = if (scope) |s| s.file else null,
            });
        }

        return try bindings.toOwnedSlice(allocator);
    }

    const JsonDebugInfo = struct {
        version: u32 = 1,
        ops: []const OpMeta = &.{},
        source_scopes: []const SourceScope = &.{},
        op_visibility: []const OpVisibility = &.{},
    };
};

test "DebugInfo parses scope and visibility metadata" {
    const allocator = std.testing.allocator;
    const json =
        \\{
        \\  "version": 2,
        \\  "source_scopes": [
        \\    {
        \\      "id": 1,
        \\      "parent": null,
        \\      "file": "main.ora",
        \\      "function": "foo",
        \\      "contract": "Main",
        \\      "kind": "function",
        \\      "label": null,
        \\      "range": {"start":{"line":1,"col":1},"end":{"line":5,"col":1}},
        \\      "locals": [
        \\        {
        \\          "id": 10,
        \\          "name": "counter",
        \\          "kind": "field",
        \\          "binding_kind": null,
        \\          "storage_class": "storage",
        \\          "runtime": {"kind":"storage_field","name":"counter","location":{"kind":"storage_root","root":"counter","slot":0},"editable":true},
        \\          "folded_value": null,
        \\          "decl": {"start":{"line":1,"col":1},"end":{"line":1,"col":8}},
        \\          "live": {"start":{"line":1,"col":1},"end":{"line":5,"col":1}}
        \\        }
        \\      ]
        \\    }
        \\  ],
        \\  "op_visibility": [
        \\    {"idx": 7, "scope_ids": [1], "visible_local_ids": [10]}
        \\  ]
        \\}
    ;

    var info = try DebugInfo.loadFromJson(allocator, json);
    defer info.deinit();

    try std.testing.expectEqual(@as(u32, 2), info.version());
    const visibility = info.getVisibilityForIdx(7).?;
    try std.testing.expectEqual(@as(usize, 1), visibility.visible_local_ids.len);
    const local = info.getLocalById(10).?;
    try std.testing.expectEqualStrings("counter", local.name);
    try std.testing.expectEqualStrings("storage_field", local.runtime.kind);
    try std.testing.expect(local.runtime.editable);

    const visible = try info.collectVisibleLocals(allocator, 7);
    defer allocator.free(visible);
    try std.testing.expectEqual(@as(usize, 1), visible.len);
    try std.testing.expectEqualStrings("counter", visible[0].local.name);
    try std.testing.expect(visible[0].scope != null);

    const bindings = try info.collectVisibleBindings(allocator, 7);
    defer allocator.free(bindings);
    try std.testing.expectEqual(@as(usize, 1), bindings.len);
    try std.testing.expectEqualStrings("counter", bindings[0].name);
    try std.testing.expectEqualStrings("storage_field", bindings[0].runtime_kind);
    try std.testing.expectEqualStrings("storage_root", bindings[0].runtime_location_kind.?);
    try std.testing.expectEqualStrings("counter", bindings[0].runtime_location_root.?);
    try std.testing.expectEqual(@as(?u64, 0), bindings[0].runtime_location_slot);
    try std.testing.expect(bindings[0].editable);
}

test "DebugInfo parses concrete memory and transient root payloads" {
    const allocator = std.testing.allocator;
    const json =
        \\{
        \\  "version": 2,
        \\  "source_scopes": [
        \\    {
        \\      "id": 1,
        \\      "parent": null,
        \\      "file": "main.ora",
        \\      "function": "tick",
        \\      "contract": "Main",
        \\      "kind": "function",
        \\      "label": null,
        \\      "range": {"start":{"line":1,"col":1},"end":{"line":8,"col":1}},
        \\      "locals": [
        \\        {
        \\          "id": 10,
        \\          "name": "scratch",
        \\          "kind": "field",
        \\          "binding_kind": null,
        \\          "storage_class": "memory",
        \\          "runtime": {"kind":"memory_field","name":"scratch","location":{"kind":"memory_root","root":"scratch","slot":0},"editable":true},
        \\          "folded_value": "30"
        \\        },
        \\        {
        \\          "id": 11,
        \\          "name": "temp_counter",
        \\          "kind": "field",
        \\          "binding_kind": null,
        \\          "storage_class": "tstore",
        \\          "runtime": {"kind":"tstore_field","name":"temp_counter","location":{"kind":"tstore_root","root":"temp_counter","slot":1},"editable":true},
        \\          "folded_value": null
        \\        }
        \\      ]
        \\    }
        \\  ],
        \\  "op_visibility": [
        \\    {"idx": 3, "scope_ids": [1], "visible_local_ids": [10, 11]}
        \\  ]
        \\}
    ;

    var info = try DebugInfo.loadFromJson(allocator, json);
    defer info.deinit();

    const bindings = try info.collectVisibleBindings(allocator, 3);
    defer allocator.free(bindings);

    try std.testing.expectEqual(@as(usize, 2), bindings.len);

    try std.testing.expectEqualStrings("scratch", bindings[0].name);
    try std.testing.expectEqualStrings("memory_field", bindings[0].runtime_kind);
    try std.testing.expectEqualStrings("memory_root", bindings[0].runtime_location_kind.?);
    try std.testing.expectEqualStrings("scratch", bindings[0].runtime_location_root.?);
    try std.testing.expectEqual(@as(?u64, 0), bindings[0].runtime_location_slot);
    try std.testing.expect(bindings[0].editable);
    try std.testing.expectEqualStrings("30", bindings[0].folded_value.?);

    try std.testing.expectEqualStrings("temp_counter", bindings[1].name);
    try std.testing.expectEqualStrings("tstore_field", bindings[1].runtime_kind);
    try std.testing.expectEqualStrings("tstore_root", bindings[1].runtime_location_kind.?);
    try std.testing.expectEqualStrings("temp_counter", bindings[1].runtime_location_root.?);
    try std.testing.expectEqual(@as(?u64, 1), bindings[1].runtime_location_slot);
    try std.testing.expect(bindings[1].editable);
}
