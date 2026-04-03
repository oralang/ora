const std = @import("std");
const mlir = @import("mlir_c_api").c;
const ast = @import("../ast/mod.zig");

pub const LocalId = ast.PatternId;

pub const LocalIdList = std.ArrayList(LocalId);
pub const LocalIdSet = std.AutoHashMap(LocalId, void);

pub const LocalEnv = struct {
    allocator: std.mem.Allocator,
    visible_names: std.StringHashMap(LocalId),
    values: std.AutoHashMap(LocalId, mlir.MlirValue),
    known_ints: std.AutoHashMap(LocalId, i64),

    pub fn init(allocator: std.mem.Allocator) LocalEnv {
        return .{
            .allocator = allocator,
            .visible_names = std.StringHashMap(LocalId).init(allocator),
            .values = std.AutoHashMap(LocalId, mlir.MlirValue).init(allocator),
            .known_ints = std.AutoHashMap(LocalId, i64).init(allocator),
        };
    }

    pub fn clone(self: *const LocalEnv) !LocalEnv {
        var env_clone = LocalEnv.init(self.allocator);

        var name_it = self.visible_names.iterator();
        while (name_it.next()) |entry| {
            try env_clone.visible_names.put(entry.key_ptr.*, entry.value_ptr.*);
        }

        var value_it = self.values.iterator();
        while (value_it.next()) |entry| {
            try env_clone.values.put(entry.key_ptr.*, entry.value_ptr.*);
        }

        var known_it = self.known_ints.iterator();
        while (known_it.next()) |entry| {
            try env_clone.known_ints.put(entry.key_ptr.*, entry.value_ptr.*);
        }

        return env_clone;
    }

    pub fn bindPattern(self: *LocalEnv, file: *const ast.AstFile, pattern_id: ast.PatternId, value: mlir.MlirValue) !void {
        switch (file.pattern(pattern_id).*) {
            .StructDestructure => |destructure| {
                for (destructure.fields) |field| {
                    try self.bindPattern(file, field.binding, value);
                }
            },
            else => {
                const binding = bindingRefFromPattern(file, pattern_id) orelse return;
                try self.visible_names.put(binding.name, binding.id);
                try self.values.put(binding.id, value);
                _ = self.known_ints.remove(binding.id);
            },
        }
    }

    pub fn bindPatternWithoutValue(self: *LocalEnv, file: *const ast.AstFile, pattern_id: ast.PatternId) !void {
        switch (file.pattern(pattern_id).*) {
            .StructDestructure => |destructure| {
                for (destructure.fields) |field| {
                    try self.bindPatternWithoutValue(file, field.binding);
                }
            },
            else => {
                const binding = bindingRefFromPattern(file, pattern_id) orelse return;
                try self.visible_names.put(binding.name, binding.id);
            },
        }
    }

    pub fn lookupName(self: *const LocalEnv, name: []const u8) ?LocalId {
        return self.visible_names.get(name);
    }

    pub fn hasLocal(self: *const LocalEnv, local_id: LocalId) bool {
        return self.values.contains(local_id);
    }

    pub fn getValue(self: *const LocalEnv, local_id: LocalId) ?mlir.MlirValue {
        return self.values.get(local_id);
    }

    pub fn getKnownInt(self: *const LocalEnv, local_id: LocalId) ?i64 {
        return self.known_ints.get(local_id);
    }

    pub fn setKnownInt(self: *LocalEnv, local_id: LocalId, known_int: ?i64) !void {
        if (known_int) |integer| {
            try self.known_ints.put(local_id, integer);
        } else {
            _ = self.known_ints.remove(local_id);
        }
    }

    pub fn setValue(self: *LocalEnv, local_id: LocalId, value: mlir.MlirValue) !void {
        if (!self.values.contains(local_id)) return error.UnknownLocalId;
        try self.values.put(local_id, value);
        _ = self.known_ints.remove(local_id);
    }

    pub fn setValueWithKnownInt(self: *LocalEnv, local_id: LocalId, value: mlir.MlirValue, known_int: ?i64) !void {
        if (!self.values.contains(local_id)) return error.UnknownLocalId;
        try self.values.put(local_id, value);
        if (known_int) |integer| {
            try self.known_ints.put(local_id, integer);
        } else {
            _ = self.known_ints.remove(local_id);
        }
    }

    pub fn setPatternKnownInt(self: *LocalEnv, file: *const ast.AstFile, pattern_id: ast.PatternId, known_int: ?i64) !void {
        switch (file.pattern(pattern_id).*) {
            .StructDestructure => |destructure| {
                for (destructure.fields) |field| {
                    try self.setPatternKnownInt(file, field.binding, null);
                }
            },
            else => {
                const binding = bindingRefFromPattern(file, pattern_id) orelse return;
                try self.setKnownInt(binding.id, known_int);
            },
        }
    }

    pub fn resolvePatternTarget(self: *const LocalEnv, file: *const ast.AstFile, pattern_id: ast.PatternId) ?LocalId {
        return switch (file.pattern(pattern_id).*) {
            .Name => |name| self.lookupName(name.name),
            .Field => |field| self.resolvePatternTarget(file, field.base),
            .Index => |index| self.resolvePatternTarget(file, index.base),
            .StructDestructure => |destructure| if (destructure.fields.len > 0) self.resolvePatternTarget(file, destructure.fields[0].binding) else null,
            else => null,
        };
    }

    const BindingRef = struct {
        id: LocalId,
        name: []const u8,
    };

    fn bindingRefFromPattern(file: *const ast.AstFile, pattern_id: ast.PatternId) ?BindingRef {
        return switch (file.pattern(pattern_id).*) {
            .Name => |name| .{ .id = pattern_id, .name = name.name },
            else => null,
        };
    }
};
