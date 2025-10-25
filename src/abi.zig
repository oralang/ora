const std = @import("std");
const lib = @import("root.zig");

/// ABI Type - Ethereum ABI types
pub const AbiType = union(enum) {
    uint: u16, // uint8, uint16, ..., uint256
    int: u16, // int8, int16, ..., int256
    address,
    bool,
    bytes,
    string,
    array: struct {
        element_type: *AbiType,
        size: ?usize, // null for dynamic arrays
    },

    pub fn format(self: AbiType, allocator: std.mem.Allocator) ![]const u8 {
        return switch (self) {
            .uint => |bits| try std.fmt.allocPrint(allocator, "uint{d}", .{bits}),
            .int => |bits| try std.fmt.allocPrint(allocator, "int{d}", .{bits}),
            .address => try allocator.dupe(u8, "address"),
            .bool => try allocator.dupe(u8, "bool"),
            .bytes => try allocator.dupe(u8, "bytes"),
            .string => try allocator.dupe(u8, "string"),
            .array => |arr| blk: {
                const elem_str = try arr.element_type.format(allocator);
                defer allocator.free(elem_str);
                if (arr.size) |size| {
                    break :blk try std.fmt.allocPrint(allocator, "{s}[{d}]", .{ elem_str, size });
                } else {
                    break :blk try std.fmt.allocPrint(allocator, "{s}[]", .{elem_str});
                }
            },
        };
    }
};

/// ABI Parameter
pub const AbiParameter = struct {
    name: []const u8,
    type: AbiType,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *AbiParameter) void {
        _ = self;
        // Types are managed by the generator
    }
};

/// ABI Function
pub const AbiFunction = struct {
    name: []const u8,
    inputs: []AbiParameter,
    outputs: []AbiParameter,
    state_mutability: []const u8, // "pure", "view", "nonpayable", "payable"
    function_type: []const u8, // "function", "constructor", "fallback", "receive"
    allocator: std.mem.Allocator,

    pub fn deinit(self: *AbiFunction) void {
        for (self.inputs) |*input| {
            input.deinit();
        }
        self.allocator.free(self.inputs);
        for (self.outputs) |*output| {
            output.deinit();
        }
        self.allocator.free(self.outputs);
    }
};

/// Contract ABI
pub const ContractAbi = struct {
    functions: []AbiFunction,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *ContractAbi) void {
        for (self.functions) |*func| {
            func.deinit();
        }
        self.allocator.free(self.functions);
    }

    /// Generate JSON representation of the ABI
    pub fn toJson(self: *const ContractAbi, allocator: std.mem.Allocator) ![]const u8 {
        var buffer = try std.ArrayList(u8).initCapacity(allocator, 1024);
        defer buffer.deinit(allocator);
        const writer = buffer.writer(allocator);

        try writer.writeAll("[\n");

        for (self.functions, 0..) |func, i| {
            try writer.writeAll("  {\n");
            try writer.print("    \"type\": \"{s}\",\n", .{func.function_type});
            try writer.print("    \"name\": \"{s}\",\n", .{func.name});

            // Inputs
            try writer.writeAll("    \"inputs\": [\n");
            for (func.inputs, 0..) |input, j| {
                const type_str = try input.type.format(allocator);
                defer allocator.free(type_str);
                try writer.writeAll("      {\n");
                try writer.print("        \"name\": \"{s}\",\n", .{input.name});
                try writer.print("        \"type\": \"{s}\"\n", .{type_str});
                if (j < func.inputs.len - 1) {
                    try writer.writeAll("      },\n");
                } else {
                    try writer.writeAll("      }\n");
                }
            }
            try writer.writeAll("    ],\n");

            // Outputs
            try writer.writeAll("    \"outputs\": [\n");
            for (func.outputs, 0..) |output, j| {
                const type_str = try output.type.format(allocator);
                defer allocator.free(type_str);
                try writer.writeAll("      {\n");
                try writer.print("        \"type\": \"{s}\"\n", .{type_str});
                if (j < func.outputs.len - 1) {
                    try writer.writeAll("      },\n");
                } else {
                    try writer.writeAll("      }\n");
                }
            }
            try writer.writeAll("    ],\n");

            try writer.print("    \"stateMutability\": \"{s}\"\n", .{func.state_mutability});

            if (i < self.functions.len - 1) {
                try writer.writeAll("  },\n");
            } else {
                try writer.writeAll("  }\n");
            }
        }

        try writer.writeAll("]\n");
        return buffer.toOwnedSlice(allocator);
    }
};

/// ABI Generator - extracts ABI from AST
pub const AbiGenerator = struct {
    allocator: std.mem.Allocator,
    functions: std.ArrayList(AbiFunction),

    pub fn init(allocator: std.mem.Allocator) !AbiGenerator {
        const functions = std.ArrayList(AbiFunction).initCapacity(allocator, 0) catch |err| {
            std.debug.print("Failed to initialize functions list: {}\n", .{err});
            return err;
        };
        return .{
            .allocator = allocator,
            .functions = functions,
        };
    }

    pub fn deinit(self: *AbiGenerator) void {
        for (self.functions.items) |*func| {
            func.deinit();
        }
        self.functions.deinit(self.allocator);
    }

    /// Generate ABI from AST nodes
    pub fn generate(self: *AbiGenerator, ast_nodes: []lib.AstNode) !ContractAbi {
        for (ast_nodes) |node| {
            try self.processNode(node);
        }

        return ContractAbi{
            .functions = try self.functions.toOwnedSlice(self.allocator),
            .allocator = self.allocator,
        };
    }

    fn processNode(self: *AbiGenerator, node: lib.AstNode) !void {
        switch (node) {
            .Contract => |contract| {
                for (contract.body) |body_node| {
                    try self.processBodyNode(body_node);
                }
            },
            else => {},
        }
    }

    fn processBodyNode(self: *AbiGenerator, node: lib.AstNode) !void {
        switch (node) {
            .Function => |func| {
                // Only include public functions in ABI
                if (func.visibility == .Public) {
                    try self.addFunction(func);
                }
            },
            else => {},
        }
    }

    fn addFunction(self: *AbiGenerator, func: lib.FunctionNode) !void {
        // Convert parameters to ABI inputs
        var inputs = std.ArrayList(AbiParameter).initCapacity(self.allocator, func.parameters.len) catch |err| {
            std.debug.print("Failed to allocate inputs array: {}\n", .{err});
            return err;
        };
        defer inputs.deinit(self.allocator);

        for (func.parameters) |param| {
            const abi_type = try self.convertType(param.type_info);
            const param_abi = AbiParameter{
                .name = param.name,
                .type = abi_type,
                .allocator = self.allocator,
            };
            try inputs.append(self.allocator, param_abi);
        }

        // Convert return type to ABI outputs
        var outputs = std.ArrayList(AbiParameter).initCapacity(self.allocator, 1) catch |err| {
            std.debug.print("Failed to allocate outputs array: {}\n", .{err});
            return err;
        };
        defer outputs.deinit(self.allocator);

        if (func.return_type_info) |ret_type| {
            const abi_type = try self.convertType(ret_type);
            const output_abi = AbiParameter{
                .name = "",
                .type = abi_type,
                .allocator = self.allocator,
            };
            try outputs.append(self.allocator, output_abi);
        }

        // Determine state mutability (simplified for now)
        const state_mutability = "nonpayable"; // TODO: Analyze function body for state changes

        const func_abi = AbiFunction{
            .name = func.name,
            .inputs = try inputs.toOwnedSlice(self.allocator),
            .outputs = try outputs.toOwnedSlice(self.allocator),
            .state_mutability = state_mutability,
            .function_type = "function",
            .allocator = self.allocator,
        };
        try self.functions.append(self.allocator, func_abi);
    }

    fn convertType(self: *AbiGenerator, type_info: lib.ast.type_info.TypeInfo) !AbiType {
        _ = self;
        if (type_info.ora_type) |ora_type| {
            return switch (ora_type) {
                .u8 => AbiType{ .uint = 8 },
                .u16 => AbiType{ .uint = 16 },
                .u32 => AbiType{ .uint = 32 },
                .u64 => AbiType{ .uint = 64 },
                .u128 => AbiType{ .uint = 128 },
                .u256 => AbiType{ .uint = 256 },
                .i8 => AbiType{ .int = 8 },
                .i16 => AbiType{ .int = 16 },
                .i32 => AbiType{ .int = 32 },
                .i64 => AbiType{ .int = 64 },
                .i128 => AbiType{ .int = 128 },
                .i256 => AbiType{ .int = 256 },
                .bool => AbiType.bool,
                .address => AbiType.address,
                .string => AbiType.string,
                .bytes => AbiType.bytes,
                else => AbiType{ .uint = 256 }, // Default to uint256 for unknown types
            };
        }
        return AbiType{ .uint = 256 }; // Default fallback
    }
};
