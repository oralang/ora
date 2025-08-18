const std = @import("std");
const ast = @import("../ast.zig");
const AstNode = ast.AstNode;
const TypeRef = @import("types.zig").TypeRef;
const ExprNode = ast.ExprNode;
const StmtNode = ast.StmtNode;
const SourceSpan = ast.SourceSpan;

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
} || std.fmt.BufPrintError || std.fs.File.Writer.Error || std.ArrayList(u8).Writer.Error;

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
        var list = std.ArrayList(u8).init(self.allocator);
        defer list.deinit();

        try self.serialize(nodes, list.writer());
        return try list.toOwnedSlice();
    }

    /// Serialize with streaming support for large ASTs
    pub fn serializeStreaming(self: *AstSerializer, nodes: []const AstNode, writer: anytype, chunk_size: usize) SerializationError!void {
        var i: usize = 0;
        while (i < nodes.len) {
            const end = @min(i + chunk_size, nodes.len);
            const chunk = nodes[i..end];
            try self.serialize(chunk, writer);
            i = end;
        }
    }

    fn serializeWithDepth(self: *AstSerializer, nodes: []const AstNode, writer: anytype, depth: u32) SerializationError!void {
        if (self.options.max_depth) |max_depth| {
            if (depth >= max_depth) {
                return SerializationError.MaxDepthExceeded;
            }
        }

        if (self.options.pretty_print and !self.options.compact_mode) {
            try writer.writeAll("{\n");
            try self.writeIndent(writer, 1);
            try writer.writeAll("\"type\": \"AST\",\n");
            try self.writeIndent(writer, 1);
            try writer.writeAll("\"nodes\": [\n");

            for (nodes, 0..) |*node, i| {
                if (i > 0) try writer.writeAll(",\n");
                try self.serializeAstNode(node, writer, 2, depth + 1);
            }

            try writer.writeAll("\n");
            try self.writeIndent(writer, 1);
            try writer.writeAll("]\n");
            try writer.writeAll("}\n");
        } else {
            // Compact mode
            try writer.writeAll("{\"type\":\"AST\",\"nodes\":[");
            for (nodes, 0..) |*node, i| {
                if (i > 0) try writer.writeAll(",");
                try self.serializeAstNode(node, writer, 0, depth + 1);
            }
            try writer.writeAll("]}");
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
            try writer.writeAll("{\n");
        } else {
            try writer.writeAll("{");
        }

        switch (node.*) {
            .Module => |*module| try self.serializeModule(module, writer, indent, depth),
            .Contract => |*contract| try self.serializeContract(contract, writer, indent, depth),
            .Function => |*function| try self.serializeFunction(function, writer, indent, depth),
            .Constant => |*constant| try self.serializeConstant(constant, writer, indent, depth),
            .VariableDecl => |*var_decl| try self.serializeVariableDecl(var_decl, writer, indent, depth),
            .StructDecl => |*struct_decl| try self.serializeStructDecl(struct_decl, writer, indent, depth),
            .EnumDecl => |*enum_decl| try self.serializeEnumDecl(enum_decl, writer, indent, depth),
            .LogDecl => |*log_decl| try self.serializeLogDecl(log_decl, writer, indent, depth),
            .Import => |*import| try self.serializeImport(import, writer, indent, depth),
            .ErrorDecl => |*error_decl| try self.serializeErrorDecl(error_decl, writer, indent, depth),
            .Block => |*block| try self.serializeBlock(block, writer, indent, depth),
            .Expression => |expr| try self.serializeExpression(expr, writer, indent, depth),
            .Statement => |stmt| try self.serializeStatement(stmt, writer, indent, depth),
            .TryBlock => |*try_block| try self.serializeTryBlock(try_block, writer, indent, depth),
        }

        if (self.options.pretty_print and !self.options.compact_mode) {
            try writer.writeAll("\n");
            try self.writeIndent(writer, indent);
            try writer.writeAll("}");
        } else {
            try writer.writeAll("}");
        }
    }

    fn serializeContract(self: *AstSerializer, contract: *const ast.ContractNode, writer: anytype, indent: u32, depth: u32) SerializationError!void {
        try self.writeField(writer, "type", "Contract", indent + 1, true);
        try self.writeField(writer, "name", contract.name, indent + 1, false);

        if (self.options.include_spans) {
            try self.writeSpanField(writer, &contract.span, indent + 1);
        }

        if (self.options.pretty_print and !self.options.compact_mode) {
            try writer.writeAll(",\n");
            try self.writeIndent(writer, indent + 1);
            try writer.writeAll("\"body\": [\n");
        } else {
            try writer.writeAll(",\"body\":[");
        }

        for (contract.body, 0..) |*member, i| {
            if (i > 0) try writer.writeAll(",");
            if (self.options.pretty_print and !self.options.compact_mode) {
                try writer.writeAll("\n");
            }
            try self.serializeAstNode(member, writer, indent + 2, depth + 1);
        }

        if (self.options.pretty_print and !self.options.compact_mode) {
            try writer.writeAll("\n");
            try self.writeIndent(writer, indent + 1);
            try writer.writeAll("]");
        } else {
            try writer.writeAll("]");
        }
    }

    fn serializeFunction(self: *AstSerializer, function: *const ast.FunctionNode, writer: anytype, indent: u32, depth: u32) SerializationError!void {
        try self.writeField(writer, "type", "Function", indent + 1, true);
        try self.writeField(writer, "name", function.name, indent + 1, false);
        try self.writeBoolField(writer, "public", function.visibility == .Public, indent + 1);

        if (self.options.include_spans) {
            try self.writeSpanField(writer, &function.span, indent + 1);
        }

        // Parameters
        if (self.options.pretty_print and !self.options.compact_mode) {
            try writer.writeAll(",\n");
            try self.writeIndent(writer, indent + 1);
            try writer.writeAll("\"parameters\": [\n");
        } else {
            try writer.writeAll(",\"parameters\":[");
        }

        for (function.parameters, 0..) |*param, i| {
            if (i > 0) try writer.writeAll(",");
            if (self.options.pretty_print and !self.options.compact_mode) {
                try writer.writeAll("\n");
            }
            try self.serializeParameter(param, writer, indent + 2);
        }

        if (self.options.pretty_print and !self.options.compact_mode) {
            try writer.writeAll("\n");
            try self.writeIndent(writer, indent + 1);
            try writer.writeAll("],\n");
            try self.writeIndent(writer, indent + 1);
            try writer.writeAll("\"return_type\": ");
        } else {
            try writer.writeAll("],\"return_type\":");
        }

        if (function.return_type_info) |ret_type_info| {
            try self.serializeTypeInfo(ret_type_info, writer);
        } else {
            try writer.writeAll("null");
        }

        // Requires clauses
        if (function.requires_clauses.len > 0) {
            if (self.options.pretty_print and !self.options.compact_mode) {
                try writer.writeAll(",\n");
                try self.writeIndent(writer, indent + 1);
                try writer.writeAll("\"requires\": [\n");
            } else {
                try writer.writeAll(",\"requires\":[");
            }

            for (function.requires_clauses, 0..) |clause, i| {
                if (i > 0) try writer.writeAll(",");
                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll("\n");
                }
                try self.serializeExpression(clause, writer, indent + 2, depth + 1);
            }

            if (self.options.pretty_print and !self.options.compact_mode) {
                try writer.writeAll("\n");
                try self.writeIndent(writer, indent + 1);
                try writer.writeAll("]");
            } else {
                try writer.writeAll("]");
            }
        }

        // Ensures clauses
        if (function.ensures_clauses.len > 0) {
            if (self.options.pretty_print and !self.options.compact_mode) {
                try writer.writeAll(",\n");
                try self.writeIndent(writer, indent + 1);
                try writer.writeAll("\"ensures\": [\n");
            } else {
                try writer.writeAll(",\"ensures\":[");
            }

            for (function.ensures_clauses, 0..) |clause, i| {
                if (i > 0) try writer.writeAll(",");
                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll("\n");
                }
                try self.serializeExpression(clause, writer, indent + 2, depth + 1);
            }

            if (self.options.pretty_print and !self.options.compact_mode) {
                try writer.writeAll("\n");
                try self.writeIndent(writer, indent + 1);
                try writer.writeAll("]");
            } else {
                try writer.writeAll("]");
            }
        }

        // Function body
        if (self.options.pretty_print and !self.options.compact_mode) {
            try writer.writeAll(",\n");
            try self.writeIndent(writer, indent + 1);
            try writer.writeAll("\"body\": ");
        } else {
            try writer.writeAll(",\"body\":");
        }
        try self.serializeBlock(&function.body, writer, indent + 1, depth + 1);
    }

    fn serializeVariableDecl(self: *AstSerializer, var_decl: *const ast.VariableDeclNode, writer: anytype, indent: u32, depth: u32) SerializationError!void {
        try self.writeField(writer, "type", "VariableDecl", indent + 1, true);
        try self.writeField(writer, "name", var_decl.name, indent + 1, false);
        try self.writeField(writer, "region", @tagName(var_decl.region), indent + 1, false);
        try self.writeField(writer, "kind", @tagName(var_decl.kind), indent + 1, false);
        try self.writeBoolField(writer, "locked", var_decl.locked, indent + 1);

        if (self.options.include_spans) {
            try self.writeSpanField(writer, &var_decl.span, indent + 1);
        }

        if (self.options.pretty_print and !self.options.compact_mode) {
            try writer.writeAll(",\n");
            try self.writeIndent(writer, indent + 1);
            try writer.writeAll("\"var_type\": ");
        } else {
            try writer.writeAll(",\"var_type\":");
        }
        try self.serializeTypeInfo(var_decl.type_info, writer);

        if (var_decl.value) |value| {
            if (self.options.pretty_print and !self.options.compact_mode) {
                try writer.writeAll(",\n");
                try self.writeIndent(writer, indent + 1);
                try writer.writeAll("\"value\": ");
            } else {
                try writer.writeAll(",\"value\":");
            }
            try self.serializeExpression(value, writer, indent + 1, depth + 1);
        }

        if (var_decl.tuple_names) |tuple_names| {
            if (self.options.pretty_print and !self.options.compact_mode) {
                try writer.writeAll(",\n");
                try self.writeIndent(writer, indent + 1);
                try writer.writeAll("\"tuple_names\": [");
            } else {
                try writer.writeAll(",\"tuple_names\":[");
            }

            for (tuple_names, 0..) |name, i| {
                if (i > 0) try writer.writeAll(",");
                try writer.print("\"{s}\"", .{name});
            }
            try writer.writeAll("]");
        }
    }
    fn serializeStructDecl(self: *AstSerializer, struct_decl: *const ast.StructDeclNode, writer: anytype, indent: u32, depth: u32) SerializationError!void {
        _ = depth; // unused
        try self.writeField(writer, "type", "StructDecl", indent + 1, true);
        try self.writeField(writer, "name", struct_decl.name, indent + 1, false);

        if (self.options.include_spans) {
            try self.writeSpanField(writer, &struct_decl.span, indent + 1);
        }

        if (self.options.pretty_print and !self.options.compact_mode) {
            try writer.writeAll(",\n");
            try self.writeIndent(writer, indent + 1);
            try writer.writeAll("\"fields\": [\n");
        } else {
            try writer.writeAll(",\"fields\":[");
        }

        for (struct_decl.fields, 0..) |*field, i| {
            if (i > 0) try writer.writeAll(",");
            if (self.options.pretty_print and !self.options.compact_mode) {
                try writer.writeAll("\n");
                try self.writeIndent(writer, indent + 2);
                try writer.writeAll("{\n");
                try self.writeField(writer, "name", field.name, indent + 3, true);
                try writer.writeAll(",\n");
                try self.writeIndent(writer, indent + 3);
                try writer.writeAll("\"field_type\": ");
                try self.serializeTypeInfo(field.type_info, writer);
                if (self.options.include_spans) {
                    try writer.writeAll(",\n");
                    try self.writeSpanField(writer, &field.span, indent + 3);
                }
                try writer.writeAll("\n");
                try self.writeIndent(writer, indent + 2);
                try writer.writeAll("}");
            } else {
                try writer.writeAll("{\"name\":\"");
                try writer.writeAll(field.name);
                try writer.writeAll("\",\"field_type\":");
                try self.serializeTypeInfo(field.type_info, writer);
                try writer.writeAll("}");
            }
        }

        if (self.options.pretty_print and !self.options.compact_mode) {
            try writer.writeAll("\n");
            try self.writeIndent(writer, indent + 1);
            try writer.writeAll("]");
        } else {
            try writer.writeAll("]");
        }
    }

    fn serializeEnumDecl(self: *AstSerializer, enum_decl: *const ast.EnumDeclNode, writer: anytype, indent: u32, depth: u32) SerializationError!void {
        try self.writeField(writer, "type", "EnumDecl", indent + 1, true);
        try self.writeField(writer, "name", enum_decl.name, indent + 1, false);
        // Check if any variants have explicit values
        var has_explicit_values = false;
        for (enum_decl.variants) |variant| {
            if (variant.value != null) {
                has_explicit_values = true;
                break;
            }
        }
        try self.writeBoolField(writer, "has_explicit_values", has_explicit_values, indent + 1);

        // Serialize the underlying type if present
        if (enum_decl.underlying_type_info) |underlying_type_info| {
            try writer.writeAll(",\n");
            try self.writeIndent(writer, indent + 1);
            try writer.writeAll("\"underlying_type\": ");
            try self.serializeTypeInfo(underlying_type_info, writer);
        }

        if (self.options.include_spans) {
            try self.writeSpanField(writer, &enum_decl.span, indent + 1);
        }

        if (self.options.pretty_print and !self.options.compact_mode) {
            try writer.writeAll(",\n");
            try self.writeIndent(writer, indent + 1);
            try writer.writeAll("\"variants\": [\n");
        } else {
            try writer.writeAll(",\"variants\":[");
        }

        for (enum_decl.variants, 0..) |*variant, i| {
            if (i > 0) try writer.writeAll(",");
            if (self.options.pretty_print and !self.options.compact_mode) {
                try writer.writeAll("\n");
                try self.writeIndent(writer, indent + 2);
                try writer.writeAll("{\n");
                try self.writeField(writer, "name", variant.name, indent + 3, true);
                if (variant.value) |*value| {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 3);
                    try writer.writeAll("\"value\": ");

                    // Special handling for integer literals in enum variants to include enum's underlying type
                    if (value.* == .Literal and value.Literal == .Integer and enum_decl.underlying_type_info != null) {
                        // Start object
                        try writer.writeAll("{\n");
                        try self.writeIndent(writer, indent + 4);
                        try writer.writeAll("\"type\": \"Literal\",\n");
                        try self.writeIndent(writer, indent + 4);

                        // Always use "Integer" for literal_type consistency
                        try writer.writeAll("\"literal_type\": \"Integer\",\n");

                        // Include type information
                        try self.writeIndent(writer, indent + 4);
                        try writer.writeAll("\"type_info\": ");
                        try self.serializeTypeInfo(value.Literal.Integer.type_info, writer);
                        try writer.writeAll(",\n");

                        // Write the value
                        try self.writeIndent(writer, indent + 4);
                        try writer.writeAll("\"value\": \"");
                        try writer.writeAll(value.Literal.Integer.value);
                        try writer.writeAll("\"");

                        // Include span if needed
                        if (self.options.include_spans) {
                            try writer.writeAll(",\n");
                            // Custom span field handling for enum variant integer literals
                            try self.writeIndent(writer, indent + 4);
                            try writer.writeAll("\"span\": {\n");
                            try self.writeIndent(writer, indent + 5);
                            try writer.print("\"line\": {d},\n", .{value.Literal.Integer.span.line});
                            try self.writeIndent(writer, indent + 5);
                            try writer.print("\"column\": {d},\n", .{value.Literal.Integer.span.column});
                            try self.writeIndent(writer, indent + 5);
                            try writer.print("\"length\": {d},\n", .{value.Literal.Integer.span.length});
                            try self.writeIndent(writer, indent + 5);
                            try writer.writeAll("\"lexeme\": \"");
                            if (value.Literal.Integer.span.lexeme) |lexeme| {
                                try writer.writeAll(lexeme);
                            }
                            try writer.writeAll("\"\n");
                            try self.writeIndent(writer, indent + 4);
                            try writer.writeAll("}");
                        }

                        // End object
                        try writer.writeAll("\n");
                        try self.writeIndent(writer, indent + 3);
                        try writer.writeAll("}");
                    } else {
                        // Regular expression serialization for non-integer or complex expressions
                        try self.serializeExpression(value, writer, indent + 3, depth + 1);
                    }
                }

                if (self.options.include_spans) {
                    try writer.writeAll(",\n");
                    try self.writeSpanField(writer, &variant.span, indent + 3);
                }
                try writer.writeAll("\n");
                try self.writeIndent(writer, indent + 2);
                try writer.writeAll("}");
            } else {
                try writer.writeAll("{\"name\":\"");
                try writer.writeAll(variant.name);
                try writer.writeAll("\"");
                if (variant.value) |*value| {
                    try writer.writeAll(",\"value\":");

                    // Special handling for integer literals in enum variants to include enum's underlying type
                    if (value.* == .Literal and value.Literal == .Integer and enum_decl.underlying_type_info != null) {
                        try writer.writeAll("{\"type\":\"Literal\",");

                        // Always use "Integer" for literal_type consistency
                        try writer.writeAll("\"literal_type\":\"Integer\",");

                        // Include type information
                        try writer.writeAll("\"type_info\":");
                        try self.serializeTypeInfo(value.Literal.Integer.type_info, writer);
                        try writer.writeAll(",");

                        // Write the value
                        try writer.writeAll("\"value\":\"");
                        try writer.writeAll(value.Literal.Integer.value);
                        try writer.writeAll("\"");

                        // Include span if needed
                        if (self.options.include_spans) {
                            const span = &value.Literal.Integer.span;
                            try writer.writeAll(",\"span\":{\"line\":");
                            try writer.print("{d}", .{span.line});
                            try writer.writeAll(",\"column\":");
                            try writer.print("{d}", .{span.column});
                            try writer.writeAll(",\"length\":");
                            try writer.print("{d}", .{span.length});
                            try writer.writeAll(",\"lexeme\":\"");
                            if (span.lexeme) |lexeme| {
                                try writer.writeAll(lexeme);
                            } else {
                                try writer.writeAll("");
                            }
                            try writer.writeAll("\"}");
                        }

                        try writer.writeAll("}");
                    } else {
                        // Regular expression serialization for non-integer or complex expressions
                        try self.serializeExpression(value, writer, 0, depth + 1);
                    }
                }

                // Include span for the variant itself
                if (self.options.include_spans) {
                    try writer.writeAll(",\"span\":{\"line\":");
                    try writer.print("{d}", .{variant.span.line});
                    try writer.writeAll(",\"column\":");
                    try writer.print("{d}", .{variant.span.column});
                    try writer.writeAll(",\"length\":");
                    try writer.print("{d}", .{variant.span.length});
                    try writer.writeAll(",\"lexeme\":\"");
                    if (variant.span.lexeme) |lexeme| {
                        try writer.writeAll(lexeme);
                    }
                    try writer.writeAll("\"}");
                }

                try writer.writeAll("}");
            }
        }

        if (self.options.pretty_print and !self.options.compact_mode) {
            try writer.writeAll("\n");
            try self.writeIndent(writer, indent + 1);
            try writer.writeAll("]");
        } else {
            try writer.writeAll("]");
        }
    }

    fn serializeLogDecl(self: *AstSerializer, log_decl: *const ast.LogDeclNode, writer: anytype, indent: u32, depth: u32) SerializationError!void {
        _ = depth; // unused
        try self.writeField(writer, "type", "LogDecl", indent + 1, true);
        try self.writeField(writer, "name", log_decl.name, indent + 1, false);

        if (self.options.include_spans) {
            try self.writeSpanField(writer, &log_decl.span, indent + 1);
        }

        if (self.options.pretty_print and !self.options.compact_mode) {
            try writer.writeAll(",\n");
            try self.writeIndent(writer, indent + 1);
            try writer.writeAll("\"fields\": [\n");
        } else {
            try writer.writeAll(",\"fields\":[");
        }

        for (log_decl.fields, 0..) |*field, i| {
            if (i > 0) try writer.writeAll(",");
            if (self.options.pretty_print and !self.options.compact_mode) {
                try writer.writeAll("\n");
                try self.writeIndent(writer, indent + 2);
                try writer.writeAll("{\n");
                try self.writeField(writer, "name", field.name, indent + 3, true);
                try writer.writeAll(",\n");
                try self.writeIndent(writer, indent + 3);
                try writer.writeAll("\"field_type\": ");
                try self.serializeTypeInfo(field.type_info, writer);
                if (self.options.include_spans) {
                    try writer.writeAll(",\n");
                    try self.writeSpanField(writer, &field.span, indent + 3);
                }
                try writer.writeAll("\n");
                try self.writeIndent(writer, indent + 2);
                try writer.writeAll("}");
            } else {
                try writer.writeAll("{\"name\":\"");
                try writer.writeAll(field.name);
                try writer.writeAll("\",\"field_type\":");
                try self.serializeTypeInfo(field.type_info, writer);
                try writer.writeAll("}");
            }
        }

        if (self.options.pretty_print and !self.options.compact_mode) {
            try writer.writeAll("\n");
            try self.writeIndent(writer, indent + 1);
            try writer.writeAll("]");
        } else {
            try writer.writeAll("]");
        }
    }

    fn serializeImport(self: *AstSerializer, import: *const ast.ImportNode, writer: anytype, indent: u32, depth: u32) SerializationError!void {
        _ = depth; // unused
        try self.writeField(writer, "type", "Import", indent + 1, true);
        try self.writeField(writer, "name", import.alias orelse import.path, indent + 1, false);
        try self.writeField(writer, "path", import.path, indent + 1, false);

        if (self.options.include_spans) {
            try self.writeSpanField(writer, &import.span, indent + 1);
        }
    }

    fn serializeErrorDecl(self: *AstSerializer, error_decl: *const ast.ErrorDeclNode, writer: anytype, indent: u32, depth: u32) SerializationError!void {
        _ = depth; // unused
        try self.writeField(writer, "type", "ErrorDecl", indent + 1, true);
        try self.writeField(writer, "name", error_decl.name, indent + 1, false);

        if (self.options.include_spans) {
            try self.writeSpanField(writer, &error_decl.span, indent + 1);
        }
    }

    fn serializeModule(self: *AstSerializer, module: *const ast.ModuleNode, writer: anytype, indent: u32, depth: u32) SerializationError!void {
        try self.writeField(writer, "type", "Module", indent + 1, true);

        if (module.name) |name| {
            try self.writeField(writer, "name", name, indent + 1, false);
        } else {
            try self.writeField(writer, "name", "", indent + 1, false);
        }

        try writer.writeAll(",\n");
        try self.writeIndent(writer, indent + 1);
        try writer.writeAll("\"imports\": [");
        if (module.imports.len > 0) {
            try writer.writeAll("\n");
            for (module.imports, 0..) |import, i| {
                try self.writeIndent(writer, indent + 2);
                try writer.writeAll("{\n");
                try self.serializeImport(&import, writer, indent + 2, depth + 1);
                try writer.writeAll("\n");
                try self.writeIndent(writer, indent + 2);
                try writer.writeAll("}");
                if (i < module.imports.len - 1) {
                    try writer.writeAll(",");
                }
                try writer.writeAll("\n");
            }
            try self.writeIndent(writer, indent + 1);
        }
        try writer.writeAll("],\n");

        try self.writeIndent(writer, indent + 1);
        try writer.writeAll("\"declarations\": [");
        if (module.declarations.len > 0) {
            try writer.writeAll("\n");
            for (module.declarations, 0..) |decl, i| {
                try self.writeIndent(writer, indent + 2);
                try writer.writeAll("{\n");
                try self.serializeAstNode(&decl, writer, indent + 2, depth + 1);
                try writer.writeAll("\n");
                try self.writeIndent(writer, indent + 2);
                try writer.writeAll("}");
                if (i < module.declarations.len - 1) {
                    try writer.writeAll(",");
                }
                try writer.writeAll("\n");
            }
            try self.writeIndent(writer, indent + 1);
        }
        try writer.writeAll("]");

        if (self.options.include_spans) {
            try writer.writeAll(",\n");
            try self.writeSpanField(writer, &module.span, indent + 1);
        }
    }

    fn serializeConstant(self: *AstSerializer, constant: *const ast.ConstantNode, writer: anytype, indent: u32, depth: u32) SerializationError!void {
        try self.writeField(writer, "type", "Constant", indent + 1, true);
        try self.writeField(writer, "name", constant.name, indent + 1, false);

        try writer.writeAll(",\n");
        try self.writeIndent(writer, indent + 1);
        try writer.writeAll("\"typ\": ");
        try self.serializeTypeInfo(constant.typ, writer);

        try writer.writeAll(",\n");
        try self.writeIndent(writer, indent + 1);
        try writer.writeAll("\"value\": {\n");
        try self.serializeExpression(constant.value, writer, indent + 1, depth + 1);
        try writer.writeAll("\n");
        try self.writeIndent(writer, indent + 1);
        try writer.writeAll("}");

        try writer.writeAll(",\n");
        try self.writeIndent(writer, indent + 1);
        try writer.writeAll("\"visibility\": \"");
        switch (constant.visibility) {
            .Public => try writer.writeAll("Public"),
            .Private => try writer.writeAll("Private"),
        }
        try writer.writeAll("\"");

        if (self.options.include_spans) {
            try writer.writeAll(",\n");
            try self.writeSpanField(writer, &constant.span, indent + 1);
        }
    }

    fn serializeBlock(self: *AstSerializer, block: *const ast.BlockNode, writer: anytype, indent: u32, depth: u32) SerializationError!void {
        if (self.options.pretty_print and !self.options.compact_mode) {
            try writer.writeAll("{\n");
            try self.writeField(writer, "type", "Block", indent + 1, true);
            if (self.options.include_spans) {
                try writer.writeAll(",\n");
                try self.writeSpanField(writer, &block.span, indent + 1);
            }
            try writer.writeAll(",\n");
            try self.writeIndent(writer, indent + 1);
            try writer.writeAll("\"statements\": [\n");
        } else {
            try writer.writeAll("{\"type\":\"Block\",\"statements\":[");
        }

        for (block.statements, 0..) |*stmt, i| {
            if (i > 0) try writer.writeAll(",");
            if (self.options.pretty_print and !self.options.compact_mode) {
                try writer.writeAll("\n");
            }
            try self.serializeStatement(stmt, writer, indent + 2, depth + 1);
        }

        if (self.options.pretty_print and !self.options.compact_mode) {
            try writer.writeAll("\n");
            try self.writeIndent(writer, indent + 1);
            try writer.writeAll("]\n");
            try self.writeIndent(writer, indent);
            try writer.writeAll("}");
        } else {
            try writer.writeAll("]}");
        }
    }

    fn serializeTryBlock(self: *AstSerializer, try_block: *const ast.TryBlockNode, writer: anytype, indent: u32, depth: u32) SerializationError!void {
        try self.writeField(writer, "type", "TryBlock", indent + 1, true);

        if (self.options.include_spans) {
            try self.writeSpanField(writer, &try_block.span, indent + 1);
        }

        if (self.options.pretty_print and !self.options.compact_mode) {
            try writer.writeAll(",\n");
            try self.writeIndent(writer, indent + 1);
            try writer.writeAll("\"try_block\": ");
        } else {
            try writer.writeAll(",\"try_block\":");
        }
        try self.serializeBlock(&try_block.try_block, writer, indent + 1, depth + 1);

        if (try_block.catch_block) |*catch_block| {
            if (self.options.pretty_print and !self.options.compact_mode) {
                try writer.writeAll(",\n");
                try self.writeIndent(writer, indent + 1);
                try writer.writeAll("\"catch_block\": {\n");
                if (catch_block.error_variable) |error_var| {
                    try self.writeField(writer, "error_variable", error_var, indent + 2, true);
                    try writer.writeAll(",\n");
                }
                try self.writeIndent(writer, indent + 2);
                try writer.writeAll("\"block\": ");
                try self.serializeBlock(&catch_block.block, writer, indent + 2, depth + 1);
                if (self.options.include_spans) {
                    try writer.writeAll(",\n");
                    try self.writeSpanField(writer, &catch_block.span, indent + 2);
                }
                try writer.writeAll("\n");
                try self.writeIndent(writer, indent + 1);
                try writer.writeAll("}");
            } else {
                try writer.writeAll(",\"catch_block\":{");
                if (catch_block.error_variable) |error_var| {
                    try writer.print("\"error_variable\":\"{s}\",", .{error_var});
                }
                try writer.writeAll("\"block\":");
                try self.serializeBlock(&catch_block.block, writer, 0, depth + 1);
                try writer.writeAll("}");
            }
        }
    }
    fn serializeStatement(self: *AstSerializer, stmt: *const StmtNode, writer: anytype, indent: u32, depth: u32) SerializationError!void {
        if (self.options.pretty_print and !self.options.compact_mode) {
            try self.writeIndent(writer, indent);
            try writer.writeAll("{\n");
        } else {
            try writer.writeAll("{");
        }

        switch (stmt.*) {
            .Expr => |*expr| {
                try self.writeField(writer, "type", "ExprStatement", indent + 1, true);
                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"expression\": ");
                } else {
                    try writer.writeAll(",\"expression\":");
                }
                try self.serializeExpression(expr, writer, indent + 1, depth + 1);
            },
            .VariableDecl => |*var_decl| {
                try self.serializeVariableDecl(var_decl, writer, indent, depth);
            },
            .Return => |*ret| {
                try self.writeField(writer, "type", "Return", indent + 1, true);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &ret.span, indent + 1);
                }
                if (ret.value) |*value| {
                    if (self.options.pretty_print and !self.options.compact_mode) {
                        try writer.writeAll(",\n");
                        try self.writeIndent(writer, indent + 1);
                        try writer.writeAll("\"value\": ");
                    } else {
                        try writer.writeAll(",\"value\":");
                    }
                    try self.serializeExpression(value, writer, indent + 1, depth + 1);
                }
            },
            .If => |*if_stmt| {
                try self.writeField(writer, "type", "If", indent + 1, true);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &if_stmt.span, indent + 1);
                }
                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"condition\": ");
                } else {
                    try writer.writeAll(",\"condition\":");
                }
                try self.serializeExpression(&if_stmt.condition, writer, indent + 1, depth + 1);

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"then_branch\": ");
                } else {
                    try writer.writeAll(",\"then_branch\":");
                }
                try self.serializeBlock(&if_stmt.then_branch, writer, indent + 1, depth + 1);

                if (if_stmt.else_branch) |*else_branch| {
                    if (self.options.pretty_print and !self.options.compact_mode) {
                        try writer.writeAll(",\n");
                        try self.writeIndent(writer, indent + 1);
                        try writer.writeAll("\"else_branch\": ");
                    } else {
                        try writer.writeAll(",\"else_branch\":");
                    }
                    try self.serializeBlock(else_branch, writer, indent + 1, depth + 1);
                }
            },
            .While => |*while_stmt| {
                try self.writeField(writer, "type", "While", indent + 1, true);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &while_stmt.span, indent + 1);
                }
                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"condition\": ");
                } else {
                    try writer.writeAll(",\"condition\":");
                }
                try self.serializeExpression(&while_stmt.condition, writer, indent + 1, depth + 1);

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"body\": ");
                } else {
                    try writer.writeAll(",\"body\":");
                }
                try self.serializeBlock(&while_stmt.body, writer, indent + 1, depth + 1);

                if (while_stmt.invariants.len > 0) {
                    if (self.options.pretty_print and !self.options.compact_mode) {
                        try writer.writeAll(",\n");
                        try self.writeIndent(writer, indent + 1);
                        try writer.writeAll("\"invariants\": [\n");
                    } else {
                        try writer.writeAll(",\"invariants\":[");
                    }

                    for (while_stmt.invariants, 0..) |*inv, i| {
                        if (i > 0) try writer.writeAll(",");
                        if (self.options.pretty_print and !self.options.compact_mode) {
                            try writer.writeAll("\n");
                        }
                        try self.serializeExpression(inv, writer, indent + 2, depth + 1);
                    }

                    if (self.options.pretty_print and !self.options.compact_mode) {
                        try writer.writeAll("\n");
                        try self.writeIndent(writer, indent + 1);
                        try writer.writeAll("]");
                    } else {
                        try writer.writeAll("]");
                    }
                }
            },
            .ForLoop => |*for_loop| {
                try self.writeField(writer, "type", "ForLoop", indent + 1, true);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &for_loop.span, indent + 1);
                }

                // Serialize the iterable expression
                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"iterable\": ");
                } else {
                    try writer.writeAll(",\"iterable\":");
                }
                try self.serializeExpression(&for_loop.iterable, writer, indent + 1, depth + 1);

                // Serialize the loop pattern
                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"pattern\": ");
                } else {
                    try writer.writeAll(",\"pattern\":");
                }
                try self.serializeLoopPattern(&for_loop.pattern, writer, indent + 1);

                // Serialize the loop body
                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"body\": ");
                } else {
                    try writer.writeAll(",\"body\":");
                }
                try self.serializeBlock(&for_loop.body, writer, indent + 1, depth + 1);
            },
            .Switch => |*switch_stmt| {
                try self.writeField(writer, "type", "Switch", indent + 1, true);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &switch_stmt.span, indent + 1);
                }

                // Serialize the switch condition
                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"condition\": ");
                } else {
                    try writer.writeAll(",\"condition\":");
                }
                try self.serializeExpression(&switch_stmt.condition, writer, indent + 1, depth + 1);

                // Serialize switch cases
                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"cases\": [\n");
                } else {
                    try writer.writeAll(",\"cases\":[");
                }

                for (switch_stmt.cases, 0..) |*case, i| {
                    if (i > 0) try writer.writeAll(",");
                    if (self.options.pretty_print and !self.options.compact_mode) {
                        try writer.writeAll("\n");
                        try self.writeIndent(writer, indent + 2);
                        try writer.writeAll("{\n");
                    } else {
                        try writer.writeAll("{");
                    }

                    try self.serializeSwitchCase(case, writer, indent + 2, depth + 1);

                    if (self.options.pretty_print and !self.options.compact_mode) {
                        try writer.writeAll("\n");
                        try self.writeIndent(writer, indent + 2);
                        try writer.writeAll("}");
                    } else {
                        try writer.writeAll("}");
                    }
                }

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll("\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("]");
                } else {
                    try writer.writeAll("]");
                }

                // Serialize default case if present
                if (switch_stmt.default_case) |*default_case| {
                    if (self.options.pretty_print and !self.options.compact_mode) {
                        try writer.writeAll(",\n");
                        try self.writeIndent(writer, indent + 1);
                        try writer.writeAll("\"default_case\": ");
                    } else {
                        try writer.writeAll(",\"default_case\":");
                    }
                    try self.serializeBlock(default_case, writer, indent + 1, depth + 1);
                }
            },
            .Break => |*break_node| {
                try self.writeField(writer, "type", "Break", indent + 1, true);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &break_node.span, indent + 1);
                }

                // Serialize optional label
                if (break_node.label) |label| {
                    try self.writeField(writer, "label", label, indent + 1, false);
                }

                // Serialize optional value
                if (break_node.value) |value| {
                    if (self.options.pretty_print and !self.options.compact_mode) {
                        try writer.writeAll(",\n");
                        try self.writeIndent(writer, indent + 1);
                        try writer.writeAll("\"value\": ");
                    } else {
                        try writer.writeAll(",\"value\":");
                    }
                    try self.serializeExpression(value, writer, indent + 1, depth + 1);
                }
            },
            .Continue => |*continue_node| {
                try self.writeField(writer, "type", "Continue", indent + 1, true);
                if (continue_node.label) |label| {
                    try self.writeField(writer, "label", label, indent + 1, false);
                }
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &continue_node.span, indent + 1);
                }
            },
            .Log => |*log| {
                try self.writeField(writer, "type", "Log", indent + 1, true);
                try self.writeField(writer, "event_name", log.event_name, indent + 1, false);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &log.span, indent + 1);
                }

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"args\": [\n");
                } else {
                    try writer.writeAll(",\"args\":[");
                }

                for (log.args, 0..) |*arg, i| {
                    if (i > 0) try writer.writeAll(",");
                    if (self.options.pretty_print and !self.options.compact_mode) {
                        try writer.writeAll("\n");
                    }
                    try self.serializeExpression(arg, writer, indent + 2, depth + 1);
                }

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll("\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("]");
                } else {
                    try writer.writeAll("]");
                }
            },
            .Lock => |*lock| {
                try self.writeField(writer, "type", "Lock", indent + 1, true);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &lock.span, indent + 1);
                }
                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"path\": ");
                } else {
                    try writer.writeAll(",\"path\":");
                }
                try self.serializeExpression(&lock.path, writer, indent + 1, depth + 1);
            },
            .Invariant => |*inv| {
                try self.writeField(writer, "type", "Invariant", indent + 1, true);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &inv.span, indent + 1);
                }
                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"condition\": ");
                } else {
                    try writer.writeAll(",\"condition\":");
                }
                try self.serializeExpression(&inv.condition, writer, indent + 1, depth + 1);
            },
            .Requires => |*req| {
                try self.writeField(writer, "type", "Requires", indent + 1, true);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &req.span, indent + 1);
                }
                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"condition\": ");
                } else {
                    try writer.writeAll(",\"condition\":");
                }
                try self.serializeExpression(&req.condition, writer, indent + 1, depth + 1);
            },
            .Ensures => |*ens| {
                try self.writeField(writer, "type", "Ensures", indent + 1, true);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &ens.span, indent + 1);
                }
                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"condition\": ");
                } else {
                    try writer.writeAll(",\"condition\":");
                }
                try self.serializeExpression(&ens.condition, writer, indent + 1, depth + 1);
            },
            .ErrorDecl => |*error_decl| {
                try self.serializeErrorDecl(error_decl, writer, indent, depth);
            },
            .TryBlock => |*try_block| {
                try self.serializeTryBlock(try_block, writer, indent, depth);
            },
            .CompoundAssignment => |*compound| {
                try self.writeField(writer, "type", "CompoundAssignment", indent + 1, true);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &compound.span, indent + 1);
                }

                // Serialize the target expression
                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"target\": ");
                } else {
                    try writer.writeAll(",\"target\":");
                }
                try self.serializeExpression(compound.target, writer, indent + 1, depth + 1);

                // Serialize the operator
                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"operator\": \"");
                    try writer.writeAll(@tagName(compound.operator));
                    try writer.writeAll("\"");
                } else {
                    try writer.writeAll(",\"operator\":\"");
                    try writer.writeAll(@tagName(compound.operator));
                    try writer.writeAll("\"");
                }

                // Serialize the value expression
                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"value\": ");
                } else {
                    try writer.writeAll(",\"value\":");
                }
                try self.serializeExpression(compound.value, writer, indent + 1, depth + 1);
            },
            .DestructuringAssignment => |*dest_assign| {
                try self.writeField(writer, "type", "DestructuringAssignment", indent + 1, true);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &dest_assign.span, indent + 1);
                }
                // TODO: Serialize pattern and value
            },
            .Unlock => |*unlock| {
                try self.writeField(writer, "type", "Unlock", indent + 1, true);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &unlock.span, indent + 1);
                }
                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"path\": ");
                } else {
                    try writer.writeAll(",\"path\":");
                }
                try self.serializeExpression(&unlock.path, writer, indent + 1, depth + 1);
            },
            .Move => |*move_stmt| {
                try self.writeField(writer, "type", "Move", indent + 1, true);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &move_stmt.span, indent + 1);
                }
                // TODO: Serialize move statement fields
            },
            .LabeledBlock => |*labeled| {
                try self.writeField(writer, "type", "LabeledBlock", indent + 1, true);
                try self.writeField(writer, "label", labeled.label, indent + 1, false);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &labeled.span, indent + 1);
                }
                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"block\": ");
                } else {
                    try writer.writeAll(",\"block\":");
                }
                try self.serializeBlock(&labeled.block, writer, indent + 1, depth + 1);
            },
        }

        if (self.options.pretty_print and !self.options.compact_mode) {
            try writer.writeAll("\n");
            try self.writeIndent(writer, indent);
            try writer.writeAll("}");
        } else {
            try writer.writeAll("}");
        }
    }
    fn serializeExpression(self: *AstSerializer, expr: *const ExprNode, writer: anytype, indent: u32, depth: u32) SerializationError!void {
        if (self.options.pretty_print and !self.options.compact_mode) {
            try self.writeIndent(writer, indent);
            try writer.writeAll("{\n");
        } else {
            try writer.writeAll("{");
        }

        switch (expr.*) {
            .Identifier => |*ident| {
                try self.writeField(writer, "type", "Identifier", indent + 1, true);
                try self.writeField(writer, "name", ident.name, indent + 1, false);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &ident.span, indent + 1);
                }
            },
            .Literal => |*literal| {
                try self.writeField(writer, "type", "Literal", indent + 1, true);
                try self.serializeLiteral(literal, writer, indent + 1);
            },
            .Binary => |*binary| {
                try self.writeField(writer, "type", "Binary", indent + 1, true);
                try self.writeField(writer, "operator", @tagName(binary.operator), indent + 1, false);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &binary.span, indent + 1);
                }

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"lhs\": ");
                } else {
                    try writer.writeAll(",\"lhs\":");
                }
                try self.serializeExpression(binary.lhs, writer, indent + 1, depth + 1);

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"rhs\": ");
                } else {
                    try writer.writeAll(",\"rhs\":");
                }
                try self.serializeExpression(binary.rhs, writer, indent + 1, depth + 1);
            },
            .Unary => |*unary| {
                try self.writeField(writer, "type", "Unary", indent + 1, true);
                try self.writeField(writer, "operator", @tagName(unary.operator), indent + 1, false);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &unary.span, indent + 1);
                }

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"operand\": ");
                } else {
                    try writer.writeAll(",\"operand\":");
                }
                try self.serializeExpression(unary.operand, writer, indent + 1, depth + 1);
            },
            .Assignment => |*assign| {
                try self.writeField(writer, "type", "Assignment", indent + 1, true);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &assign.span, indent + 1);
                }

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"target\": ");
                } else {
                    try writer.writeAll(",\"target\":");
                }
                try self.serializeExpression(assign.target, writer, indent + 1, depth + 1);

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"value\": ");
                } else {
                    try writer.writeAll(",\"value\":");
                }
                try self.serializeExpression(assign.value, writer, indent + 1, depth + 1);
            },
            .CompoundAssignment => |*comp_assign| {
                try self.writeField(writer, "type", "CompoundAssignment", indent + 1, true);
                try self.writeField(writer, "operator", @tagName(comp_assign.operator), indent + 1, false);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &comp_assign.span, indent + 1);
                }

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"target\": ");
                } else {
                    try writer.writeAll(",\"target\":");
                }
                try self.serializeExpression(comp_assign.target, writer, indent + 1, depth + 1);

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"value\": ");
                } else {
                    try writer.writeAll(",\"value\":");
                }
                try self.serializeExpression(comp_assign.value, writer, indent + 1, depth + 1);
            },
            .Call => |*call| {
                try self.writeField(writer, "type", "Call", indent + 1, true);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &call.span, indent + 1);
                }

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"callee\": ");
                } else {
                    try writer.writeAll(",\"callee\":");
                }
                try self.serializeExpression(call.callee, writer, indent + 1, depth + 1);

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"arguments\": [\n");
                } else {
                    try writer.writeAll(",\"arguments\":[");
                }

                for (call.arguments, 0..) |arg, i| {
                    if (i > 0) try writer.writeAll(",");
                    if (self.options.pretty_print and !self.options.compact_mode) {
                        try writer.writeAll("\n");
                    }
                    try self.serializeExpression(arg, writer, indent + 2, depth + 1);
                }

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll("\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("]");
                } else {
                    try writer.writeAll("]");
                }
            },
            .Index => |*index| {
                try self.writeField(writer, "type", "Index", indent + 1, true);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &index.span, indent + 1);
                }

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"target\": ");
                } else {
                    try writer.writeAll(",\"target\":");
                }
                try self.serializeExpression(index.target, writer, indent + 1, depth + 1);

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"index\": ");
                } else {
                    try writer.writeAll(",\"index\":");
                }
                try self.serializeExpression(index.index, writer, indent + 1, depth + 1);
            },
            .FieldAccess => |*field_access| {
                try self.writeField(writer, "type", "FieldAccess", indent + 1, true);
                try self.writeField(writer, "field", field_access.field, indent + 1, false);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &field_access.span, indent + 1);
                }

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"target\": ");
                } else {
                    try writer.writeAll(",\"target\":");
                }
                try self.serializeExpression(field_access.target, writer, indent + 1, depth + 1);
            },
            .Cast => |*cast| {
                try self.writeField(writer, "type", "Cast", indent + 1, true);
                try self.writeField(writer, "cast_type", @tagName(cast.cast_type), indent + 1, false);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &cast.span, indent + 1);
                }

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"operand\": ");
                } else {
                    try writer.writeAll(",\"operand\":");
                }
                try self.serializeExpression(cast.operand, writer, indent + 1, depth + 1);

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"target_type\": ");
                } else {
                    try writer.writeAll(",\"target_type\":");
                }
                try self.serializeTypeInfo(cast.target_type, writer);
            },
            .Comptime => |*comptime_expr| {
                try self.writeField(writer, "type", "Comptime", indent + 1, true);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &comptime_expr.span, indent + 1);
                }

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"block\": ");
                } else {
                    try writer.writeAll(",\"block\":");
                }
                try self.serializeBlock(&comptime_expr.block, writer, indent + 1, depth + 1);
            },
            .Old => |*old| {
                try self.writeField(writer, "type", "Old", indent + 1, true);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &old.span, indent + 1);
                }

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"expr\": ");
                } else {
                    try writer.writeAll(",\"expr\":");
                }
                try self.serializeExpression(old.expr, writer, indent + 1, depth + 1);
            },
            .Tuple => |*tuple| {
                try self.writeField(writer, "type", "Tuple", indent + 1, true);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &tuple.span, indent + 1);
                }

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"elements\": [\n");
                } else {
                    try writer.writeAll(",\"elements\":[");
                }

                for (tuple.elements, 0..) |element, i| {
                    if (i > 0) try writer.writeAll(",");
                    if (self.options.pretty_print and !self.options.compact_mode) {
                        try writer.writeAll("\n");
                    }
                    try self.serializeExpression(element, writer, indent + 2, depth + 1);
                }

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll("\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("]");
                } else {
                    try writer.writeAll("]");
                }
            },
            .Try => |*try_expr| {
                try self.writeField(writer, "type", "Try", indent + 1, true);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &try_expr.span, indent + 1);
                }

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"expr\": ");
                } else {
                    try writer.writeAll(",\"expr\":");
                }
                try self.serializeExpression(try_expr.expr, writer, indent + 1, depth + 1);
            },
            .ErrorReturn => |*error_return| {
                try self.writeField(writer, "type", "ErrorReturn", indent + 1, true);
                try self.writeField(writer, "error_name", error_return.error_name, indent + 1, false);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &error_return.span, indent + 1);
                }
            },
            .ErrorCast => |*error_cast| {
                try self.writeField(writer, "type", "ErrorCast", indent + 1, true);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &error_cast.span, indent + 1);
                }

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"operand\": ");
                } else {
                    try writer.writeAll(",\"operand\":");
                }
                try self.serializeExpression(error_cast.operand, writer, indent + 1, depth + 1);

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"target_type\": ");
                } else {
                    try writer.writeAll(",\"target_type\":");
                }
                try self.serializeTypeInfo(error_cast.target_type, writer);
            },
            .Shift => |*shift| {
                try self.writeField(writer, "type", "Shift", indent + 1, true);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &shift.span, indent + 1);
                }

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"mapping\": ");
                } else {
                    try writer.writeAll(",\"mapping\":");
                }
                try self.serializeExpression(shift.mapping, writer, indent + 1, depth + 1);

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"source\": ");
                } else {
                    try writer.writeAll(",\"source\":");
                }
                try self.serializeExpression(shift.source, writer, indent + 1, depth + 1);

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"dest\": ");
                } else {
                    try writer.writeAll(",\"dest\":");
                }
                try self.serializeExpression(shift.dest, writer, indent + 1, depth + 1);

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"amount\": ");
                } else {
                    try writer.writeAll(",\"amount\":");
                }
                try self.serializeExpression(shift.amount, writer, indent + 1, depth + 1);
            },
            .StructInstantiation => |*struct_inst| {
                try self.writeField(writer, "type", "StructInstantiation", indent + 1, true);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &struct_inst.span, indent + 1);
                }

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"struct_name\": ");
                } else {
                    try writer.writeAll(",\"struct_name\":");
                }
                try self.serializeExpression(struct_inst.struct_name, writer, indent + 1, depth + 1);

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"fields\": [\n");
                } else {
                    try writer.writeAll(",\"fields\":[");
                }

                for (struct_inst.fields, 0..) |*field, i| {
                    if (i > 0) try writer.writeAll(",");
                    if (self.options.pretty_print and !self.options.compact_mode) {
                        try writer.writeAll("\n");
                        try self.writeIndent(writer, indent + 2);
                        try writer.writeAll("{\n");
                        try self.writeField(writer, "name", field.name, indent + 3, true);
                        try writer.writeAll(",\n");
                        try self.writeIndent(writer, indent + 3);
                        try writer.writeAll("\"value\": ");
                        try self.serializeExpression(field.value, writer, indent + 3, depth + 1);
                        if (self.options.include_spans) {
                            try writer.writeAll(",\n");
                            try self.writeSpanField(writer, &field.span, indent + 3);
                        }
                        try writer.writeAll("\n");
                        try self.writeIndent(writer, indent + 2);
                        try writer.writeAll("}");
                    } else {
                        try writer.writeAll("{\"name\":\"");
                        try writer.writeAll(field.name);
                        try writer.writeAll("\",\"value\":");
                        try self.serializeExpression(field.value, writer, 0, depth + 1);
                        try writer.writeAll("}");
                    }
                }

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll("\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("]");
                } else {
                    try writer.writeAll("]");
                }
            },
            .EnumLiteral => |*enum_literal| {
                try self.writeField(writer, "type", "EnumLiteral", indent + 1, true);
                try self.writeField(writer, "enum_name", enum_literal.enum_name, indent + 1, false);
                try self.writeField(writer, "variant_name", enum_literal.variant_name, indent + 1, false);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &enum_literal.span, indent + 1);
                }
            },
            .SwitchExpression => |*switch_expr| {
                try self.writeField(writer, "type", "SwitchExpression", indent + 1, true);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &switch_expr.span, indent + 1);
                }

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"target\": ");
                } else {
                    try writer.writeAll(",\"target\":");
                }
                try self.serializeExpression(switch_expr.condition, writer, indent + 1, depth + 1);

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"cases\": [\n");
                } else {
                    try writer.writeAll(",\"cases\":[");
                }

                for (switch_expr.cases, 0..) |*case, i| {
                    if (i > 0) try writer.writeAll(",");
                    if (self.options.pretty_print and !self.options.compact_mode) {
                        try writer.writeAll("\n");
                        try self.writeIndent(writer, indent + 2);
                        try writer.writeAll("{\n");
                    } else {
                        try writer.writeAll("{");
                    }

                    try self.serializeSwitchCase(case, writer, indent + 2, depth + 1);

                    if (self.options.pretty_print and !self.options.compact_mode) {
                        try writer.writeAll("\n");
                        try self.writeIndent(writer, indent + 2);
                        try writer.writeAll("}");
                    } else {
                        try writer.writeAll("}");
                    }
                }

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll("\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("]");
                } else {
                    try writer.writeAll("]");
                }
            },
            .Quantified => |*quantified| {
                try self.writeField(writer, "type", "Quantified", indent + 1, true);
                try self.writeField(writer, "quantifier", @tagName(quantified.quantifier), indent + 1, false);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &quantified.span, indent + 1);
                }

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"variable\": \"");
                    try writer.writeAll(quantified.variable);
                    try writer.writeAll("\",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"variable_type\": ");
                } else {
                    try writer.writeAll(",\"variable\":\"");
                    try writer.writeAll(quantified.variable);
                    try writer.writeAll("\",\"variable_type\":");
                }
                try self.serializeTypeInfo(quantified.variable_type, writer);

                // Serialize optional condition
                if (quantified.condition) |condition| {
                    if (self.options.pretty_print and !self.options.compact_mode) {
                        try writer.writeAll(",\n");
                        try self.writeIndent(writer, indent + 1);
                        try writer.writeAll("\"condition\": ");
                    } else {
                        try writer.writeAll(",\"condition\":");
                    }
                    try self.serializeExpression(condition, writer, indent + 1, depth + 1);
                }

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"body\": ");
                } else {
                    try writer.writeAll(",\"body\":");
                }
                try self.serializeExpression(quantified.body, writer, indent + 1, depth + 1);
            },
            .AnonymousStruct => |*anon_struct| {
                try self.writeField(writer, "type", "AnonymousStruct", indent + 1, true);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &anon_struct.span, indent + 1);
                }

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"fields\": [\n");
                } else {
                    try writer.writeAll(",\"fields\":[");
                }

                for (anon_struct.fields, 0..) |*field, i| {
                    if (i > 0) try writer.writeAll(",");
                    if (self.options.pretty_print and !self.options.compact_mode) {
                        try writer.writeAll("\n");
                        try self.writeIndent(writer, indent + 2);
                        try writer.writeAll("{\n");
                        try self.writeField(writer, "name", field.name, indent + 3, true);
                        try writer.writeAll(",\n");
                        try self.writeIndent(writer, indent + 3);
                        try writer.writeAll("\"value\": ");
                        try self.serializeExpression(field.value, writer, indent + 3, depth + 1);
                        if (self.options.include_spans) {
                            try writer.writeAll(",\n");
                            try self.writeSpanField(writer, &field.span, indent + 3);
                        }
                        try writer.writeAll("\n");
                        try self.writeIndent(writer, indent + 2);
                        try writer.writeAll("}");
                    } else {
                        try writer.writeAll("{\"name\":\"");
                        try writer.writeAll(field.name);
                        try writer.writeAll("\",\"value\":");
                        try self.serializeExpression(field.value, writer, 0, depth + 1);
                        try writer.writeAll("}");
                    }
                }

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll("\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("]");
                } else {
                    try writer.writeAll("]");
                }
            },
            .Range => |*range| {
                try self.writeField(writer, "type", "Range", indent + 1, true);
                try self.writeBoolField(writer, "inclusive", range.inclusive, indent + 1);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &range.span, indent + 1);
                }

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"start\": ");
                } else {
                    try writer.writeAll(",\"start\":");
                }
                try self.serializeExpression(range.start, writer, indent + 1, depth + 1);

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"end\": ");
                } else {
                    try writer.writeAll(",\"end\":");
                }
                try self.serializeExpression(range.end, writer, indent + 1, depth + 1);
            },
            .LabeledBlock => |*labeled_block| {
                try self.writeField(writer, "type", "LabeledBlock", indent + 1, true);
                try self.writeField(writer, "label", labeled_block.label, indent + 1, false);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &labeled_block.span, indent + 1);
                }

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"block\": ");
                } else {
                    try writer.writeAll(",\"block\":");
                }
                try self.serializeBlock(&labeled_block.block, writer, indent + 1, depth + 1);
            },
            .Destructuring => |*destructuring| {
                try self.writeField(writer, "type", "Destructuring", indent + 1, true);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &destructuring.span, indent + 1);
                }

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"pattern\": ");
                } else {
                    try writer.writeAll(",\"pattern\":");
                }
                try self.serializeDestructuringPattern(&destructuring.pattern, writer, indent + 1);

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"value\": ");
                } else {
                    try writer.writeAll(",\"value\":");
                }
                try self.serializeExpression(destructuring.value, writer, indent + 1, depth + 1);
            },
            .ArrayLiteral => |*array_literal| {
                try self.writeField(writer, "type", "ArrayLiteral", indent + 1, true);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &array_literal.span, indent + 1);
                }

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"elements\": [\n");
                } else {
                    try writer.writeAll(",\"elements\":[");
                }

                for (array_literal.elements, 0..) |element, i| {
                    if (i > 0) try writer.writeAll(",");
                    if (self.options.pretty_print and !self.options.compact_mode) {
                        try writer.writeAll("\n");
                    }
                    try self.serializeExpression(element, writer, indent + 2, depth + 1);
                }

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll("\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("]");
                } else {
                    try writer.writeAll("]");
                }
            },
        }

        if (self.options.pretty_print and !self.options.compact_mode) {
            try writer.writeAll("\n");
            try self.writeIndent(writer, indent);
            try writer.writeAll("}");
        } else {
            try writer.writeAll("}");
        }
    }
    fn serializeLiteral(self: *AstSerializer, literal: *const ast.LiteralExpr, writer: anytype, indent: u32) SerializationError!void {
        switch (literal.*) {
            .Integer => |*int_lit| {
                try self.writeField(writer, "literal_type", "Integer", indent, false);
                try self.writeField(writer, "value", int_lit.value, indent, false);
                // Include the integer type information
                try writer.writeAll(",\n");
                try self.writeIndent(writer, indent);
                try writer.writeAll("\"type_info\": ");
                try self.serializeTypeInfo(int_lit.type_info, writer);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &int_lit.span, indent);
                }
            },
            .String => |*str_lit| {
                try self.writeField(writer, "literal_type", "String", indent, false);
                try writer.writeAll(",\n");
                try self.writeIndent(writer, indent);
                try writer.writeAll("\"type_info\": ");
                try self.serializeTypeInfo(str_lit.type_info, writer);
                try writer.writeAll(",\n");
                try self.writeField(writer, "value", str_lit.value, indent, false);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &str_lit.span, indent);
                }
            },
            .Bool => |*bool_lit| {
                try self.writeField(writer, "literal_type", "Bool", indent, false);
                try writer.writeAll(",\n");
                try self.writeIndent(writer, indent);
                try writer.writeAll("\"type_info\": ");
                try self.serializeTypeInfo(bool_lit.type_info, writer);
                try writer.writeAll(",\n");
                try self.writeBoolField(writer, "value", bool_lit.value, indent);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &bool_lit.span, indent);
                }
            },
            .Address => |*addr_lit| {
                try self.writeField(writer, "literal_type", "Address", indent, false);
                try writer.writeAll(",\n");
                try self.writeIndent(writer, indent);
                try writer.writeAll("\"type_info\": ");
                try self.serializeTypeInfo(addr_lit.type_info, writer);
                try writer.writeAll(",\n");
                try self.writeField(writer, "value", addr_lit.value, indent, false);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &addr_lit.span, indent);
                }
            },
            .Hex => |*hex_lit| {
                try self.writeField(writer, "literal_type", "Hex", indent, false);
                try writer.writeAll(",\n");
                try self.writeIndent(writer, indent);
                try writer.writeAll("\"type_info\": ");
                try self.serializeTypeInfo(hex_lit.type_info, writer);
                try writer.writeAll(",\n");
                try self.writeField(writer, "value", hex_lit.value, indent, false);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &hex_lit.span, indent);
                }
            },
            .Binary => |*bin_lit| {
                try self.writeField(writer, "literal_type", "Binary", indent, false);
                try writer.writeAll(",\n");
                try self.writeIndent(writer, indent);
                try writer.writeAll("\"type_info\": ");
                try self.serializeTypeInfo(bin_lit.type_info, writer);
                try writer.writeAll(",\n");
                try self.writeField(writer, "value", bin_lit.value, indent, false);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &bin_lit.span, indent);
                }
            },
        }
    }

    fn serializeParameter(self: *AstSerializer, param: *const ast.ParameterNode, writer: anytype, indent: u32) SerializationError!void {
        if (self.options.pretty_print and !self.options.compact_mode) {
            try self.writeIndent(writer, indent);
            try writer.writeAll("{\n");
            try self.writeField(writer, "name", param.name, indent + 1, true);
            try writer.writeAll(",\n");
            try self.writeIndent(writer, indent + 1);
            try writer.writeAll("\"param_type\": ");
            try self.serializeTypeInfo(param.type_info, writer);
            if (self.options.include_spans) {
                try writer.writeAll(",\n");
                try self.writeSpanField(writer, &param.span, indent + 1);
            }
            try writer.writeAll("\n");
            try self.writeIndent(writer, indent);
            try writer.writeAll("}");
        } else {
            try writer.writeAll("{\"name\":\"");
            try writer.writeAll(param.name);
            try writer.writeAll("\",\"param_type\":");
            try self.serializeTypeInfo(param.type_info, writer);
            try writer.writeAll("}");
        }
    }

    fn serializeTypeRef(self: *AstSerializer, type_ref: *const TypeRef, writer: anytype) SerializationError!void {
        switch (type_ref.*) {
            .Bool => try writer.writeAll("\"Bool\""),
            .Address => try writer.writeAll("\"Address\""),
            .U8 => try writer.writeAll("\"U8\""),
            .U16 => try writer.writeAll("\"U16\""),
            .U32 => try writer.writeAll("\"U32\""),
            .U64 => try writer.writeAll("\"U64\""),
            .U128 => try writer.writeAll("\"U128\""),
            .U256 => try writer.writeAll("\"U256\""),
            .I8 => try writer.writeAll("\"I8\""),
            .I16 => try writer.writeAll("\"I16\""),
            .I32 => try writer.writeAll("\"I32\""),
            .I64 => try writer.writeAll("\"I64\""),
            .I128 => try writer.writeAll("\"I128\""),
            .I256 => try writer.writeAll("\"I256\""),
            .String => try writer.writeAll("\"String\""),
            .Bytes => try writer.writeAll("\"Bytes\""),
            .Unknown => try writer.writeAll("\"Unknown\""),
            .Identifier => |name| {
                try writer.writeAll("{\"type\":\"Identifier\",\"name\":\"");
                try writer.writeAll(name);
                try writer.writeAll("\"}");
            },
            .Slice => |elem_type| {
                try writer.writeAll("{\"type\":\"Slice\",\"element_type\":");
                try self.serializeTypeRef(elem_type, writer);
                try writer.writeAll("}");
            },
            .Mapping => |mapping| {
                try writer.writeAll("{\"type\":\"Mapping\",\"key\":");
                try self.serializeTypeRef(mapping.key, writer);
                try writer.writeAll(",\"value\":");
                try self.serializeTypeRef(mapping.value, writer);
                try writer.writeAll("}");
            },
            .DoubleMap => |doublemap| {
                try writer.writeAll("{\"type\":\"DoubleMap\",\"key1\":");
                try self.serializeTypeRef(doublemap.key1, writer);
                try writer.writeAll(",\"key2\":");
                try self.serializeTypeRef(doublemap.key2, writer);
                try writer.writeAll(",\"value\":");
                try self.serializeTypeRef(doublemap.value, writer);
                try writer.writeAll("}");
            },
            .Tuple => |tuple| {
                try writer.writeAll("{\"type\":\"Tuple\",\"types\":[");
                for (tuple.types, 0..) |*elem_type, i| {
                    if (i > 0) try writer.writeAll(",");
                    try self.serializeTypeRef(elem_type, writer);
                }
                try writer.writeAll("]}");
            },
            .ErrorUnion => |error_union| {
                try writer.writeAll("{\"type\":\"ErrorUnion\",\"success_type\":");
                try self.serializeTypeRef(error_union.success_type, writer);
                try writer.writeAll("}");
            },
            // Result removed
            .Struct => |struct_name| {
                try writer.writeAll("{\"type\":\"Struct\",\"name\":\"");
                try writer.writeAll(struct_name);
                try writer.writeAll("\"}");
            },
            .Enum => |enum_name| {
                try writer.writeAll("{\"type\":\"Enum\",\"name\":\"");
                try writer.writeAll(enum_name);
                try writer.writeAll("\"}");
            },
            .Contract => |contract_name| {
                try writer.writeAll("{\"type\":\"Contract\",\"name\":\"");
                try writer.writeAll(contract_name);
                try writer.writeAll("\"}");
            },
            .Function => |func_type| {
                try writer.writeAll("{\"type\":\"Function\",\"params\":[");
                for (func_type.params, 0..) |*param, i| {
                    if (i > 0) try writer.writeAll(",");
                    try self.serializeTypeRef(param, writer);
                }
                try writer.writeAll("],\"return_type\":");
                if (func_type.return_type) |ret_type| {
                    try self.serializeTypeRef(ret_type, writer);
                } else {
                    try writer.writeAll("null");
                }
                try writer.writeAll("}");
            },
            .Void => {
                try writer.writeAll("\"Void\"");
            },
            .Error => {
                try writer.writeAll("\"Error\"");
            },
            .Module => |module_name| {
                if (module_name) |name| {
                    try writer.writeAll("{\"type\":\"Module\",\"name\":\"");
                    try writer.writeAll(name);
                    try writer.writeAll("\"}");
                } else {
                    try writer.writeAll("\"Module\"");
                }
            },
        }
    }

    fn serializeTypeInfo(self: *AstSerializer, type_info: ast.TypeInfo, writer: anytype) SerializationError!void {
        try writer.writeAll("{");

        // Write category
        try writer.writeAll("\"category\":\"");
        try writer.writeAll(@tagName(type_info.category));
        try writer.writeAll("\"");

        // Write source
        try writer.writeAll(",\"source\":\"");
        try writer.writeAll(@tagName(type_info.source));
        try writer.writeAll("\"");

        // Write AST type if present
        if (type_info.ast_type) |ast_type| {
            try writer.writeAll(",\"ast_type\":");
            try self.serializeTypeRef(&ast_type, writer);
        }

        // Write ORA type if present
        if (type_info.ora_type) |ora_type| {
            try writer.writeAll(",\"ora_type\":");

            switch (ora_type) {
                .struct_type => |name| {
                    try writer.writeAll("{\"type\":\"struct_type\",\"name\":\"");
                    try writer.writeAll(name);
                    try writer.writeAll("\"}");
                },
                .enum_type => |name| {
                    try writer.writeAll("{\"type\":\"enum_type\",\"name\":\"");
                    try writer.writeAll(name);
                    try writer.writeAll("\"}");
                },
                .contract_type => |name| {
                    try writer.writeAll("{\"type\":\"contract_type\",\"name\":\"");
                    try writer.writeAll(name);
                    try writer.writeAll("\"}");
                },
                else => {
                    // For simple types just use the tag name
                    try writer.writeAll("\"");
                    try writer.writeAll(@tagName(ora_type));
                    try writer.writeAll("\"");
                },
            }
        }

        try writer.writeAll("}");
    }

    // Helper functions for writing formatted output
    fn writeIndent(self: *AstSerializer, writer: anytype, level: u32) SerializationError!void {
        if (!self.options.pretty_print or self.options.compact_mode) return;

        var i: u32 = 0;
        while (i < level) : (i += 1) {
            try writer.writeAll("  ");
        }
    }

    fn writeField(self: *AstSerializer, writer: anytype, key: []const u8, value: []const u8, indent: u32, is_first: bool) SerializationError!void {
        if (self.options.pretty_print and !self.options.compact_mode) {
            if (!is_first) try writer.writeAll(",\n");
            try self.writeIndent(writer, indent);
            try writer.print("\"{s}\": \"{s}\"", .{ key, value });
        } else {
            if (!is_first) try writer.writeAll(",");
            try writer.print("\"{s}\":\"{s}\"", .{ key, value });
        }
    }

    fn writeBoolField(self: *AstSerializer, writer: anytype, key: []const u8, value: bool, indent: u32) SerializationError!void {
        if (self.options.pretty_print and !self.options.compact_mode) {
            try writer.writeAll(",\n");
            try self.writeIndent(writer, indent);
            try writer.print("\"{s}\": {}", .{ key, value });
        } else {
            try writer.print(",\"{s}\":{}", .{ key, value });
        }
    }

    fn writeSpanField(self: *AstSerializer, writer: anytype, span: *const SourceSpan, indent: u32) SerializationError!void {
        if (self.options.pretty_print and !self.options.compact_mode) {
            try writer.writeAll(",\n");
            try self.writeIndent(writer, indent);

            // If lexeme is available and include_lexemes option is enabled, include it in the output
            if (self.options.include_lexemes and span.lexeme != null) {
                try writer.print("\"span\": {{\"line\": {}, \"column\": {}, \"length\": {}, \"lexeme\": \"{s}\"}}", .{ span.line, span.column, span.length, span.lexeme.? });
            } else {
                try writer.print("\"span\": {{\"line\": {}, \"column\": {}, \"length\": {}}}", .{ span.line, span.column, span.length });
            }
        } else {
            // Compact mode
            if (self.options.include_lexemes and span.lexeme != null) {
                try writer.print(",\"span\":{{\"line\":{},\"column\":{},\"length\":{},\"lexeme\":\"{s}\"}}", .{ span.line, span.column, span.length, span.lexeme.? });
            } else {
                try writer.print(",\"span\":{{\"line\":{},\"column\":{},\"length\":{}}}", .{ span.line, span.column, span.length });
            }
        }
    }

    /// Serialize a loop pattern (for ForLoop statements)
    fn serializeLoopPattern(self: *AstSerializer, pattern: *const ast.LoopPattern, writer: anytype, indent: u32) SerializationError!void {
        if (self.options.pretty_print and !self.options.compact_mode) {
            try self.writeIndent(writer, indent);
            try writer.writeAll("{\n");
        } else {
            try writer.writeAll("{");
        }

        switch (pattern.*) {
            .Single => |single| {
                try self.writeField(writer, "type", "Single", indent + 1, true);
                try self.writeField(writer, "variable", single.name, indent + 1, false);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &single.span, indent + 1);
                }
            },
            .IndexPair => |pair| {
                try self.writeField(writer, "type", "IndexPair", indent + 1, true);
                try self.writeField(writer, "item", pair.item, indent + 1, false);
                try self.writeField(writer, "index", pair.index, indent + 1, false);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &pair.span, indent + 1);
                }
            },
            .Destructured => |destructured| {
                try self.writeField(writer, "type", "Destructured", indent + 1, true);
                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"pattern\": ");
                } else {
                    try writer.writeAll(",\"pattern\":");
                }
                try self.serializeDestructuringPattern(&destructured.pattern, writer, indent + 1);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &destructured.span, indent + 1);
                }
            },
        }

        if (self.options.pretty_print and !self.options.compact_mode) {
            try writer.writeAll("\n");
            try self.writeIndent(writer, indent);
            try writer.writeAll("}");
        } else {
            try writer.writeAll("}");
        }
    }

    /// Serialize a switch case
    fn serializeSwitchCase(self: *AstSerializer, case: *const ast.SwitchCase, writer: anytype, indent: u32, depth: u32) SerializationError!void {
        try self.writeField(writer, "type", "SwitchCase", indent + 1, true);
        if (self.options.include_spans) {
            try self.writeSpanField(writer, &case.span, indent + 1);
        }

        // Serialize the pattern
        if (self.options.pretty_print and !self.options.compact_mode) {
            try writer.writeAll(",\n");
            try self.writeIndent(writer, indent + 1);
            try writer.writeAll("\"pattern\": ");
        } else {
            try writer.writeAll(",\"pattern\":");
        }
        try self.serializeSwitchPattern(&case.pattern, writer, indent + 1);

        // Serialize the body
        if (self.options.pretty_print and !self.options.compact_mode) {
            try writer.writeAll(",\n");
            try self.writeIndent(writer, indent + 1);
            try writer.writeAll("\"body\": ");
        } else {
            try writer.writeAll(",\"body\":");
        }
        try self.serializeSwitchBody(&case.body, writer, indent + 1, depth + 1);
    }

    /// Serialize a switch pattern
    fn serializeSwitchPattern(self: *AstSerializer, pattern: *const ast.SwitchPattern, writer: anytype, indent: u32) SerializationError!void {
        if (self.options.pretty_print and !self.options.compact_mode) {
            try self.writeIndent(writer, indent);
            try writer.writeAll("{\n");
        } else {
            try writer.writeAll("{");
        }

        switch (pattern.*) {
            .Literal => |*literal| {
                try self.writeField(writer, "type", "Literal", indent + 1, true);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &literal.span, indent + 1);
                }
                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"value\": ");
                } else {
                    try writer.writeAll(",\"value\":");
                }
                try self.serializeLiteral(&literal.value, writer, indent + 1);
            },
            .Range => |*range| {
                try self.writeField(writer, "type", "Range", indent + 1, true);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &range.span, indent + 1);
                }
                try self.writeBoolField(writer, "inclusive", range.inclusive, indent + 1);

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"start\": ");
                } else {
                    try writer.writeAll(",\"start\":");
                }
                try self.serializeExpression(range.start, writer, indent + 1, 0);

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"end\": ");
                } else {
                    try writer.writeAll(",\"end\":");
                }
                try self.serializeExpression(range.end, writer, indent + 1, 0);
            },
            .EnumValue => |*enum_value| {
                try self.writeField(writer, "type", "EnumValue", indent + 1, true);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &enum_value.span, indent + 1);
                }
                try self.writeField(writer, "enum_name", enum_value.enum_name, indent + 1, false);
                try self.writeField(writer, "variant_name", enum_value.variant_name, indent + 1, false);
            },
            .Else => |*span| {
                try self.writeField(writer, "type", "Else", indent + 1, true);
                if (self.options.include_spans) {
                    try self.writeSpanField(writer, span, indent + 1);
                }
            },
        }

        if (self.options.pretty_print and !self.options.compact_mode) {
            try writer.writeAll("\n");
            try self.writeIndent(writer, indent);
            try writer.writeAll("}");
        } else {
            try writer.writeAll("}");
        }
    }

    /// Serialize a switch body
    fn serializeSwitchBody(self: *AstSerializer, body: *const ast.SwitchBody, writer: anytype, indent: u32, depth: u32) SerializationError!void {
        switch (body.*) {
            .Expression => |expr| {
                try self.serializeExpression(expr, writer, indent, depth);
            },
            .Block => |*block| {
                try self.serializeBlock(block, writer, indent, depth);
            },
            .LabeledBlock => |*labeled| {
                if (self.options.pretty_print and !self.options.compact_mode) {
                    try self.writeIndent(writer, indent);
                    try writer.writeAll("{\n");
                } else {
                    try writer.writeAll("{");
                }

                try self.writeField(writer, "type", "LabeledBlock", indent + 1, true);
                try self.writeField(writer, "label", labeled.label, indent + 1, false);

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"block\": ");
                } else {
                    try writer.writeAll(",\"block\":");
                }
                try self.serializeBlock(&labeled.block, writer, indent + 1, depth + 1);

                if (self.options.include_spans) {
                    try self.writeSpanField(writer, &labeled.span, indent + 1);
                }

                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll("\n");
                    try self.writeIndent(writer, indent);
                    try writer.writeAll("}");
                } else {
                    try writer.writeAll("}");
                }
            },
        }
    }

    /// Serialize a destructuring pattern
    fn serializeDestructuringPattern(self: *AstSerializer, pattern: *const ast.DestructuringPattern, writer: anytype, indent: u32) SerializationError!void {
        if (self.options.pretty_print and !self.options.compact_mode) {
            try self.writeIndent(writer, indent);
            try writer.writeAll("{\n");
        } else {
            try writer.writeAll("{");
        }

        switch (pattern.*) {
            .Struct => |fields| {
                try self.writeField(writer, "type", "Struct", indent + 1, true);
                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"fields\": [");
                } else {
                    try writer.writeAll(",\"fields\":[");
                }
                for (fields, 0..) |*field, i| {
                    if (i > 0) try writer.writeAll(",");
                    try writer.writeAll("\"");
                    try writer.writeAll(field.name);
                    try writer.writeAll("\"");
                }
                try writer.writeAll("]");
            },
            .Tuple => |names| {
                try self.writeField(writer, "type", "Tuple", indent + 1, true);
                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"elements\": [");
                } else {
                    try writer.writeAll(",\"elements\":[");
                }
                for (names, 0..) |name, i| {
                    if (i > 0) try writer.writeAll(",");
                    try writer.writeAll("\"");
                    try writer.writeAll(name);
                    try writer.writeAll("\"");
                }
                try writer.writeAll("]");
            },
            .Array => |names| {
                try self.writeField(writer, "type", "Array", indent + 1, true);
                if (self.options.pretty_print and !self.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try self.writeIndent(writer, indent + 1);
                    try writer.writeAll("\"elements\": [");
                } else {
                    try writer.writeAll(",\"elements\":[");
                }
                for (names, 0..) |name, i| {
                    if (i > 0) try writer.writeAll(",");
                    try writer.writeAll("\"");
                    try writer.writeAll(name);
                    try writer.writeAll("\"");
                }
                try writer.writeAll("]");
            },
        }

        if (self.options.pretty_print and !self.options.compact_mode) {
            try writer.writeAll("\n");
            try self.writeIndent(writer, indent);
            try writer.writeAll("}");
        } else {
            try writer.writeAll("}");
        }
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
