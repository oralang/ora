// ============================================================================
// Serializer Helper Functions
// ============================================================================
//
// Helper utilities for AST serialization:
//   • Formatting utilities (indent, fields)
//   • Type and parameter serialization
//   • Span serialization
//
// ============================================================================

const std = @import("std");
const ast = @import("../../ast.zig");
const SourceSpan = ast.SourceSpan;

// Forward declaration
const AstSerializer = @import("../ast_serializer.zig").AstSerializer;
const SerializationError = @import("../ast_serializer.zig").SerializationError;

/// Write indentation based on options
pub fn writeIndent(serializer: *AstSerializer, writer: anytype, level: u32) SerializationError!void {
    if (!serializer.options.pretty_print or serializer.options.compact_mode) return;

    var i: u32 = 0;
    while (i < level) : (i += 1) {
        try writer.print("  ");
    }
}

/// Write a string field
pub fn writeField(serializer: *AstSerializer, writer: anytype, key: []const u8, value: []const u8, indent: u32, is_first: bool) SerializationError!void {
    if (serializer.options.pretty_print and !serializer.options.compact_mode) {
        if (!is_first) try writer.print(",\n");
        try writeIndent(serializer, writer, indent);
        try writer.print("\"{s}\": \"{s}\"", .{ key, value });
    } else {
        if (!is_first) try writer.print(",");
        try writer.print("\"{s}\":\"{s}\"", .{ key, value });
    }
}

/// Write a boolean field
pub fn writeBoolField(serializer: *AstSerializer, writer: anytype, key: []const u8, value: bool, indent: u32) SerializationError!void {
    if (serializer.options.pretty_print and !serializer.options.compact_mode) {
        try writer.print(",\n");
        try writeIndent(serializer, writer, indent);
        try writer.print("\"{s}\": {any}", .{ key, value });
    } else {
        try writer.print(",\"{s}\":{any}", .{ key, value });
    }
}

/// Write a span field
pub fn writeSpanField(serializer: *AstSerializer, writer: anytype, span: *const SourceSpan, indent: u32) SerializationError!void {
    if (serializer.options.pretty_print and !serializer.options.compact_mode) {
        try writer.print(",\n");
        try writeIndent(serializer, writer, indent);

        // If lexeme is available and include_lexemes option is enabled, include it in the output
        if (serializer.options.include_lexemes and span.lexeme != null) {
            try writer.print("\"span\": {{\"line\": {d}, \"column\": {d}, \"length\": {d}, \"lexeme\": \"{s}\"}}", .{ span.line, span.column, span.length, span.lexeme.? });
        } else {
            try writer.print("\"span\": {{\"line\": {d}, \"column\": {d}, \"length\": {d}}}", .{ span.line, span.column, span.length });
        }
    } else {
        // Compact mode
        if (serializer.options.include_lexemes and span.lexeme != null) {
            try writer.print(",\"span\":{{\"line\":{d},\"column\":{d},\"length\":{d},\"lexeme\":\"{s}\"}}", .{ span.line, span.column, span.length, span.lexeme.? });
        } else {
            try writer.print(",\"span\":{{\"line\":{d},\"column\":{d},\"length\":{d}}}", .{ span.line, span.column, span.length });
        }
    }
}

/// Serialize a parameter
pub fn serializeParameter(serializer: *AstSerializer, param: *const ast.ParameterNode, writer: anytype, indent: u32) SerializationError!void {
    if (serializer.options.pretty_print and !serializer.options.compact_mode) {
        try writeIndent(serializer, writer, indent);
        try writer.print("{\n");
        try writeField(serializer, writer, "name", param.name, indent + 1, true);
        try writer.print(",\n");
        try writeIndent(serializer, writer, indent + 1);
        try writer.print("\"param_type\": ");
        try serializeTypeInfo(serializer, param.type_info, writer);
        if (serializer.options.include_spans) {
            try writer.print(",\n");
            try writeSpanField(serializer, writer, &param.span, indent + 1);
        }
        try writer.print("\n");
        try writeIndent(serializer, writer, indent);
        try writer.print("}");
    } else {
        try writer.print("{\"name\":\"");
        try writer.print(param.name);
        try writer.print("\",\"param_type\":");
        try serializeTypeInfo(serializer, param.type_info, writer);
        try writer.print("}");
    }
}

/// Serialize type information
pub fn serializeTypeInfo(_: *AstSerializer, type_info: ast.Types.TypeInfo, writer: anytype) SerializationError!void {
    try writer.print("{");

    // Write category
    try writer.print("\"category\":\"");
    try writer.print(@tagName(type_info.category));
    try writer.print("\"");

    // Write source
    try writer.print(",\"source\":\"");
    try writer.print(@tagName(type_info.source));
    try writer.print("\"");

    // Write ORA type if present
    if (type_info.ora_type) |ora_type| {
        try writer.print(",\"ora_type\":");

        switch (ora_type) {
            .struct_type => |name| {
                try writer.print("{\"type\":\"struct_type\",\"name\":\"");
                try writer.print(name);
                try writer.print("\"}");
            },
            .enum_type => |name| {
                try writer.print("{\"type\":\"enum_type\",\"name\":\"");
                try writer.print(name);
                try writer.print("\"}");
            },
            .contract_type => |name| {
                try writer.print("{\"type\":\"contract_type\",\"name\":\"");
                try writer.print(name);
                try writer.print("\"}");
            },
            else => {
                // For simple types just use the tag name
                try writer.print("\"");
                try writer.print(@tagName(ora_type));
                try writer.print("\"");
            },
        }
    }

    try writer.print("}");
}

