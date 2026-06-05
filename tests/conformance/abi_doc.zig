const std = @import("std");
const abi = @import("abi.zig");

pub const AbiDoc = struct {
    allocator: std.mem.Allocator,
    bytes: []u8,
    parsed: std.json.Parsed(std.json.Value),

    pub fn load(allocator: std.mem.Allocator, path: []const u8) !AbiDoc {
        const bytes = try std.fs.cwd().readFileAlloc(allocator, path, 4 * 1024 * 1024);
        errdefer allocator.free(bytes);
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, bytes, .{});
        errdefer parsed.deinit();
        return .{ .allocator = allocator, .bytes = bytes, .parsed = parsed };
    }

    pub fn deinit(self: *AbiDoc) void {
        self.parsed.deinit();
        self.allocator.free(self.bytes);
    }

    pub fn findFunction(self: *const AbiDoc, signature: []const u8) !FunctionAbi {
        const callables = getObjectField(self.parsed.value, "callables");
        if (callables != .array) return error.InvalidAbi;
        for (callables.array.items) |callable| {
            if (callable != .object) continue;
            const kind = getOptionalString(callable, "kind") orelse continue;
            if (!std.mem.eql(u8, kind, "function")) continue;
            const callable_signature = getOptionalString(callable, "signature") orelse continue;
            if (!std.mem.eql(u8, callable_signature, signature)) continue;
            const inputs = try inputWires(self, callable);
            errdefer self.allocator.free(inputs);
            return .{
                .selector = try selectorFromCallable(callable),
                .inputs = inputs,
                .outputs = try outputWires(self, callable),
            };
        }
        return error.UnknownFunction;
    }

    pub fn findConstructor(self: *const AbiDoc) !ConstructorAbi {
        const callables = getObjectField(self.parsed.value, "callables");
        if (callables != .array) return error.InvalidAbi;
        for (callables.array.items) |callable| {
            if (callable != .object) continue;
            const kind = getOptionalString(callable, "kind") orelse continue;
            if (!std.mem.eql(u8, kind, "constructor")) continue;
            return .{ .inputs = try inputWires(self, callable) };
        }
        return .{ .inputs = try self.allocator.alloc([]const u8, 0) };
    }

    fn wireTypeForTypeId(self: *const AbiDoc, type_id: []const u8) ![]const u8 {
        const abi_types = getObjectField(self.parsed.value, "types");
        if (abi_types != .object) return error.InvalidAbi;
        const type_value = abi_types.object.get(type_id) orelse return error.InvalidAbi;
        if (type_value != .object) return error.InvalidAbi;
        const wire = getObjectField(type_value, "wire");
        const evm_default = getObjectField(wire, "evm-default");
        return getStringField(evm_default, "type");
    }
};

pub const FunctionAbi = struct {
    selector: [4]u8,
    inputs: []const []const u8,
    outputs: []const []const u8,

    pub fn deinit(self: FunctionAbi, allocator: std.mem.Allocator) void {
        allocator.free(self.inputs);
        allocator.free(self.outputs);
    }
};

pub const ConstructorAbi = struct {
    inputs: []const []const u8,

    pub fn deinit(self: ConstructorAbi, allocator: std.mem.Allocator) void {
        allocator.free(self.inputs);
    }
};

fn selectorFromCallable(callable: std.json.Value) ![4]u8 {
    const wire = getObjectField(callable, "wire");
    const evm_default = getObjectField(wire, "evm-default");
    return try abi.parseSelector(try getStringField(evm_default, "selector"));
}

fn outputWires(doc: *const AbiDoc, callable: std.json.Value) ![]const []const u8 {
    return try typeRefWires(doc, callable, "outputs");
}

fn inputWires(doc: *const AbiDoc, callable: std.json.Value) ![]const []const u8 {
    return try typeRefWires(doc, callable, "inputs");
}

fn typeRefWires(doc: *const AbiDoc, callable: std.json.Value, field: []const u8) ![]const []const u8 {
    const refs = getObjectField(callable, field);
    if (refs != .array) return error.InvalidAbi;
    var wires = std.ArrayList([]const u8){};
    errdefer wires.deinit(doc.allocator);
    for (refs.array.items) |ref| {
        if (ref != .object) return error.InvalidAbi;
        const type_id = try getStringField(ref, "typeId");
        try wires.append(doc.allocator, try doc.wireTypeForTypeId(type_id));
    }
    return try wires.toOwnedSlice(doc.allocator);
}

fn getObjectField(value: std.json.Value, field: []const u8) std.json.Value {
    if (value != .object) return .null;
    return value.object.get(field) orelse .null;
}

fn getStringField(value: std.json.Value, field: []const u8) ![]const u8 {
    const child = getObjectField(value, field);
    if (child != .string) return error.InvalidAbi;
    return child.string;
}

fn getOptionalString(value: std.json.Value, field: []const u8) ?[]const u8 {
    const child = getObjectField(value, field);
    return if (child == .string) child.string else null;
}
