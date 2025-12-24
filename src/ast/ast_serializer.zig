// ============================================================================
// AST Serializer
// ============================================================================
//
// Serializes AST nodes to JSON format with comprehensive customization options.
//
// FEATURES:
//   • Multiple output formats (JSON, compact, pretty-printed)
//   • Optional span/type/lexeme inclusion
//   • Depth limiting for large trees
//   • Streaming to writer for memory efficiency
//
// SECTIONS:
//   • Options & initialization
//   • Core serialization infrastructure
//   • Top-level node serialization
//   • Statement serialization
//   • Expression serialization
//   • Helper serializers
//   • Pattern serializers
//
// ============================================================================

const std = @import("std");
const ast = @import("../ast.zig");
const AstNode = ast.AstNode;
const ExprNode = ast.Expressions.ExprNode;
const StmtNode = ast.Statements.StmtNode;
const SourceSpan = ast.SourceSpan;

// Import serializer modules
const helpers = @import("serializer/helpers.zig");
const declarations = @import("serializer/declarations.zig");
const statements = @import("serializer/statements.zig");
const expressions = @import("serializer/expressions.zig");
const patterns = @import("serializer/patterns.zig");

/// Serialization options for customizing output
pub const SerializationOptions = struct {
    /// Include source span information in output
    include_spans: bool = true,
    /// Include lexeme information in spans (original source text)
    include_lexemes: bool = true,
    /// Include type information where available
    include_types: bool = true,
    /// Pretty-print with indentation and newlines
    pretty_print: bool = true,
    /// Maximum depth to serialize (null for unlimited)
    max_depth: ?u32 = null,
    /// Custom formatters for specific node types (disabled for now)
    // custom_formatters: ?std.HashMap([]const u8, FormatterFn, std.hash_map.StringContext, std.hash_map.default_max_load_percentage) = null,
    /// Include metadata and attributes
    include_metadata: bool = false,
    /// Compact mode for minimal output size
    compact_mode: bool = false,
    /// Include debug information
    include_debug_info: bool = false,
};

pub const FormatterFn = *const fn (node: *const AstNode, writer: anytype, options: SerializationOptions) anyerror!void;

pub const SerializationError = error{
    OutOfMemory,
    WriteError,
    MaxDepthExceeded,
    InvalidNode,
    FormatterError,
} || std.fmt.BufPrintError || std.fs.File.WriteError || std.ArrayList(u8).Writer.Error;

/// Enhanced AST serializer with comprehensive customization options
pub const AstSerializer = struct {
    options: SerializationOptions,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, options: SerializationOptions) AstSerializer {
        return AstSerializer{
            .allocator = allocator,
            .options = options,
        };
    }

    pub fn deinit(_: *AstSerializer) void {
        // Nothing to clean up at the moment
    }

    /// Serialize AST nodes to a writer
    pub fn serialize(self: *AstSerializer, nodes: []const AstNode, writer: anytype) SerializationError!void {
        try self.serializeWithDepth(nodes, writer, 0);
    }

    /// Static convenience method for compatibility with existing API
    pub fn serializeAST(nodes: []const AstNode, writer: anytype) SerializationError!void {
        var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer arena.deinit();
        const allocator = arena.allocator();

        var serializer = AstSerializer.init(allocator, .{});
        defer serializer.deinit();

        try serializer.serialize(nodes, writer);
    }

    /// Serialize AST nodes to a string
    pub fn serializeToString(self: *AstSerializer, nodes: []const AstNode) SerializationError![]u8 {
        var list = std.ArrayList(u8){};
        defer list.deinit(self.allocator);

        try self.serialize(nodes, list.writer(self.allocator));
        return try list.toOwnedSlice(self.allocator);
    }

    /// Serialize with streaming support for large ASTs
    pub fn serializeStreaming(self: *AstSerializer, nodes: []const AstNode, writer: anytype, chunk_size: usize) SerializationError!void {
        const pretty = self.options.pretty_print and !self.options.compact_mode;
        const stride = if (chunk_size == 0) nodes.len else chunk_size;
        var wrote_any = false;

        if (pretty) {
            try writer.print("{\n");
            try self.writeIndent(writer, 1);
            try writer.print("\"type\": \"AST\",\n");
            try self.writeIndent(writer, 1);
            try writer.print("\"nodes\": [\n");
        } else {
            try writer.print("{\"type\":\"AST\",\"nodes\":[");
        }

        var i: usize = 0;
        while (i < nodes.len) {
            const end = @min(i + stride, nodes.len);
            for (nodes[i..end]) |*node| {
                if (wrote_any) {
                    if (pretty) {
                        try writer.print(",\n");
                    } else {
                        try writer.print(",");
                    }
                }
                try self.serializeAstNode(node, writer, if (pretty) 2 else 0, 1);
                wrote_any = true;
            }
            i = end;
        }

        if (pretty) {
            if (nodes.len > 0) try writer.print("\n");
            try self.writeIndent(writer, 1);
            try writer.print("]\n");
            try writer.print("}\n");
        } else {
            try writer.print("]}");
        }
    }

    fn serializeWithDepth(self: *AstSerializer, nodes: []const AstNode, writer: anytype, depth: u32) SerializationError!void {
        if (self.options.max_depth) |max_depth| {
            if (depth >= max_depth) {
                return SerializationError.MaxDepthExceeded;
            }
        }

        if (self.options.pretty_print and !self.options.compact_mode) {
            try writer.print("{\n");
            try self.writeIndent(writer, 1);
            try writer.print("\"type\": \"AST\",\n");
            try self.writeIndent(writer, 1);
            try writer.print("\"nodes\": [\n");

            for (nodes, 0..) |*node, i| {
                if (i > 0) try writer.print(",\n");
                try self.serializeAstNode(node, writer, 2, depth + 1);
            }

            try writer.print("\n");
            try self.writeIndent(writer, 1);
            try writer.print("]\n");
            try writer.print("}\n");
        } else {
            // Compact mode
            try writer.print("{\"type\":\"AST\",\"nodes\":[");
            for (nodes, 0..) |*node, i| {
                if (i > 0) try writer.print(",");
                try self.serializeAstNode(node, writer, 0, depth + 1);
            }
            try writer.print("]}");
        }
    }

    fn serializeAstNode(self: *AstSerializer, node: *const AstNode, writer: anytype, indent: u32, depth: u32) SerializationError!void {
        // Check depth limit before serializing
        if (self.options.max_depth) |max_depth| {
            if (depth >= max_depth) {
                return SerializationError.MaxDepthExceeded;
            }
        }

        // Check for custom formatter (disabled for now)
        // if (self.options.custom_formatters) |formatters| {
        //     const node_type = @tagName(node.*);
        //     if (formatters.get(node_type)) |formatter| {
        //         return formatter(node, writer, self.options) catch SerializationError.FormatterError;
        //     }
        // }

        if (self.options.pretty_print and !self.options.compact_mode) {
            try self.writeIndent(writer, indent);
            try writer.print("{\n");
        } else {
            try writer.print("{");
        }

        switch (node.*) {
            .Module => |*module| try declarations.serializeModule(self, module, writer, indent, depth),
            .Contract => |*contract| try declarations.serializeContract(self, contract, writer, indent, depth),
            .Function => |*function| try declarations.serializeFunction(self, function, writer, indent, depth),
            .Constant => |*constant| try declarations.serializeConstant(self, constant, writer, indent, depth),
            .VariableDecl => |*var_decl| try declarations.serializeVariableDecl(self, var_decl, writer, indent, depth),
            .StructDecl => |*struct_decl| try declarations.serializeStructDecl(self, struct_decl, writer, indent, depth),
            .EnumDecl => |*enum_decl| try declarations.serializeEnumDecl(self, enum_decl, writer, indent, depth),
            .LogDecl => |*log_decl| try declarations.serializeLogDecl(self, log_decl, writer, indent, depth),
            .Import => |*import| try declarations.serializeImport(self, import, writer, indent, depth),
            .ErrorDecl => |*error_decl| try declarations.serializeErrorDecl(self, error_decl, writer, indent, depth),
            .Block => |*block| try statements.serializeBlock(self, block, writer, indent, depth),
            .Expression => |expr| try expressions.serializeExpression(self, expr, writer, indent, depth),
            .Statement => |stmt| try statements.serializeStatement(self, stmt, writer, indent, depth),
            .TryBlock => |*try_block| try statements.serializeTryBlock(self, try_block, writer, indent, depth),
        }

        if (self.options.pretty_print and !self.options.compact_mode) {
            try writer.print("\n");
            try self.writeIndent(writer, indent);
            try writer.print("}");
        } else {
            try writer.print("}");
        }
    }

    // ========================================================================
    // TYPE SERIALIZATION
    // ========================================================================
    // Pattern serializers moved to serializer/patterns.zig
    // Type serializers remain here
    // ========================================================================
    fn serializeParameter(self: *AstSerializer, param: *const ast.ParameterNode, writer: anytype, indent: u32) SerializationError!void {
        return helpers.serializeParameter(self, param, writer, indent);
    }

    fn serializeTypeInfo(self: *AstSerializer, type_info: ast.Types.TypeInfo, writer: anytype) SerializationError!void {
        return helpers.serializeTypeInfo(self, type_info, writer);
    }

    // Helper functions for writing formatted output - delegate to helpers module
    fn writeIndent(self: *AstSerializer, writer: anytype, level: u32) SerializationError!void {
        return helpers.writeIndent(self, writer, level);
    }

    fn writeField(self: *AstSerializer, writer: anytype, key: []const u8, value: []const u8, indent: u32, is_first: bool) SerializationError!void {
        return helpers.writeField(self, writer, key, value, indent, is_first);
    }

    fn writeBoolField(self: *AstSerializer, writer: anytype, key: []const u8, value: bool, indent: u32) SerializationError!void {
        return helpers.writeBoolField(self, writer, key, value, indent);
    }

    fn writeSpanField(self: *AstSerializer, writer: anytype, span: *const SourceSpan, indent: u32) SerializationError!void {
        return helpers.writeSpanField(self, writer, span, indent);
    }

    // Pattern serializers - delegate to patterns module
    fn serializeLoopPattern(self: *AstSerializer, pattern: *const ast.Statements.LoopPattern, writer: anytype, indent: u32) SerializationError!void {
        return patterns.serializeLoopPattern(self, pattern, writer, indent);
    }

    fn serializeSwitchCase(self: *AstSerializer, case: *const ast.Switch.Case, writer: anytype, indent: u32, depth: u32) SerializationError!void {
        return patterns.serializeSwitchCase(self, case, writer, indent, depth);
    }

    fn serializeSwitchPattern(self: *AstSerializer, pattern: *const ast.Switch.Pattern, writer: anytype, indent: u32) SerializationError!void {
        return patterns.serializeSwitchPattern(self, pattern, writer, indent);
    }

    fn serializeSwitchBody(self: *AstSerializer, body: *const ast.Switch.Body, writer: anytype, indent: u32, depth: u32) SerializationError!void {
        return patterns.serializeSwitchBody(self, body, writer, indent, depth);
    }

    fn serializeDestructuringPattern(self: *AstSerializer, pattern: *const ast.Expressions.DestructuringPattern, writer: anytype, indent: u32) SerializationError!void {
        return patterns.serializeDestructuringPattern(self, pattern, writer, indent);
    }

    // Custom formatter functionality has been removed as it's not needed in the current implementation
    // If custom formatters are needed in the future, they should be implemented using a proper
    // HashMap with node type as key and formatter function as value

    // Statistics functionality removed as it's not used in the current implementation

    // SerializationStats has been removed as it's not used in the current implementation
};

// Tests
const testing = std.testing;

test "AstSerializer basic functionality" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Create a simple AST node for testing
    const contract = AstNode{
        .Contract = ast.ContractNode{
            .name = "TestContract",
            .body = &[_]AstNode{},
            .span = SourceSpan{ .line = 1, .column = 1, .length = 10 },
        },
    };

    const nodes = [_]AstNode{contract};

    var serializer = AstSerializer.init(allocator, .{});
    defer serializer.deinit();

    const result = try serializer.serializeToString(&nodes);
    defer allocator.free(result);

    // Basic validation that JSON was generated
    try testing.expect(std.mem.indexOf(u8, result, "TestContract") != null);
    try testing.expect(std.mem.indexOf(u8, result, "Contract") != null);
}

test "AstSerializer compact mode" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const contract = AstNode{
        .Contract = ast.ContractNode{
            .name = "TestContract",
            .body = &[_]AstNode{},
            .span = SourceSpan{ .line = 1, .column = 1, .length = 10 },
        },
    };

    const nodes = [_]AstNode{contract};

    var serializer = AstSerializer.init(allocator, .{ .compact_mode = true, .pretty_print = false });
    defer serializer.deinit();

    const result = try serializer.serializeToString(&nodes);
    defer allocator.free(result);

    // Compact mode should not have newlines or extra spaces
    try testing.expect(std.mem.indexOf(u8, result, "\n") == null);
    try testing.expect(std.mem.indexOf(u8, result, "TestContract") != null);
}

test "AstSerializer without spans" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const contract = AstNode{
        .Contract = ast.ContractNode{
            .name = "TestContract",
            .body = &[_]AstNode{},
            .span = SourceSpan{ .line = 1, .column = 1, .length = 10 },
        },
    };

    const nodes = [_]AstNode{contract};

    var serializer = AstSerializer.init(allocator, .{ .include_spans = false });
    defer serializer.deinit();

    const result = try serializer.serializeToString(&nodes);
    defer allocator.free(result);

    // Should not include span information
    try testing.expect(std.mem.indexOf(u8, result, "span") == null);
    try testing.expect(std.mem.indexOf(u8, result, "TestContract") != null);
}

test "AstSerializer streaming output is valid JSON" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const contract_a = AstNode{
        .Contract = ast.ContractNode{
            .name = "Alpha",
            .body = &[_]AstNode{},
            .span = SourceSpan{ .line = 1, .column = 1, .length = 5 },
        },
    };
    const contract_b = AstNode{
        .Contract = ast.ContractNode{
            .name = "Beta",
            .body = &[_]AstNode{},
            .span = SourceSpan{ .line = 2, .column = 1, .length = 4 },
        },
    };
    const nodes = [_]AstNode{ contract_a, contract_b };

    var serializer = AstSerializer.init(allocator, .{});
    defer serializer.deinit();

    var list = std.ArrayList(u8){};
    defer list.deinit(allocator);

    try serializer.serializeStreaming(&nodes, list.writer(allocator), 1);
    const output = try list.toOwnedSlice(allocator);
    defer allocator.free(output);

    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, output, .{});
    defer parsed.deinit();

    const root = parsed.value;
    switch (root) {
        .object => |obj| {
            const nodes_value = obj.get("nodes") orelse return error.TestExpectedEqual;
            switch (nodes_value) {
                .array => |arr| try testing.expectEqual(@as(usize, 2), arr.items.len),
                else => return error.TestExpectedEqual,
            }
        },
        else => return error.TestExpectedEqual,
    }
}
