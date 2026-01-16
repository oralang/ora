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
        try writer.writeAll("  ");
    }
}

/// Write a string field
pub fn writeField(serializer: *AstSerializer, writer: anytype, key: []const u8, value: []const u8, indent: u32, is_first: bool) SerializationError!void {
    if (serializer.options.pretty_print and !serializer.options.compact_mode) {
        if (!is_first) try writer.writeAll(",\n");
        try writeIndent(serializer, writer, indent);
        try writer.print("\"{s}\": \"{s}\"", .{ key, value });
    } else {
        if (!is_first) try writer.writeAll(",");
        try writer.print("\"{s}\":\"{s}\"", .{ key, value });
    }
}

/// Write a boolean field
pub fn writeBoolField(serializer: *AstSerializer, writer: anytype, key: []const u8, value: bool, indent: u32) SerializationError!void {
    if (serializer.options.pretty_print and !serializer.options.compact_mode) {
        try writer.writeAll(",\n");
        try writeIndent(serializer, writer, indent);
        try writer.print("\"{s}\": {any}", .{ key, value });
    } else {
        try writer.print(",\"{s}\":{any}", .{ key, value });
    }
}

/// Write a span field
pub fn writeSpanField(serializer: *AstSerializer, writer: anytype, span: *const SourceSpan, indent: u32) SerializationError!void {
    if (serializer.options.pretty_print and !serializer.options.compact_mode) {
        try writer.writeAll(",\n");
        try writeIndent(serializer, writer, indent);

        // if lexeme is available and include_lexemes option is enabled, include it in the output
        if (serializer.options.include_lexemes and span.lexeme != null) {
            try writer.print("\"span\": {{\"line\": {d}, \"column\": {d}, \"length\": {d}, \"lexeme\": \"{s}\"}}", .{ span.line, span.column, span.length, span.lexeme.? });
        } else {
            try writer.print("\"span\": {{\"line\": {d}, \"column\": {d}, \"length\": {d}}}", .{ span.line, span.column, span.length });
        }
    } else {
        // compact mode
        if (serializer.options.include_lexemes and span.lexeme != null) {
            try writer.print(",\"span\":{{\"line\":{d},\"column\":{d},\"length\":{d},\"lexeme\":\"{s}\"}}", .{ span.line, span.column, span.length, span.lexeme.? });
        } else {
            try writer.print(",\"span\":{{\"line\":{d},\"column\":{d},\"length\":{d}}}", .{ span.line, span.column, span.length });
        }
    }
}

/// Write a span field without adding a leading comma
pub fn writeSpanFieldNoComma(serializer: *AstSerializer, writer: anytype, span: *const SourceSpan, indent: u32) SerializationError!void {
    if (serializer.options.pretty_print and !serializer.options.compact_mode) {
        try writeIndent(serializer, writer, indent);

        if (serializer.options.include_lexemes and span.lexeme != null) {
            try writer.print("\"span\": {{\"line\": {d}, \"column\": {d}, \"length\": {d}, \"lexeme\": \"{s}\"}}", .{ span.line, span.column, span.length, span.lexeme.? });
        } else {
            try writer.print("\"span\": {{\"line\": {d}, \"column\": {d}, \"length\": {d}}}", .{ span.line, span.column, span.length });
        }
    } else {
        if (serializer.options.include_lexemes and span.lexeme != null) {
            try writer.print("\"span\":{{\"line\":{d},\"column\":{d},\"length\":{d},\"lexeme\":\"{s}\"}}", .{ span.line, span.column, span.length, span.lexeme.? });
        } else {
            try writer.print("\"span\":{{\"line\":{d},\"column\":{d},\"length\":{d}}}", .{ span.line, span.column, span.length });
        }
    }
}

/// Serialize a parameter
pub fn serializeParameter(serializer: *AstSerializer, param: *const ast.ParameterNode, writer: anytype, indent: u32) SerializationError!void {
    if (serializer.options.pretty_print and !serializer.options.compact_mode) {
        try writeIndent(serializer, writer, indent);
        try writer.writeAll("{\n");
        try writeField(serializer, writer, "name", param.name, indent + 1, true);
        try writer.writeAll(",\n");
        try writeIndent(serializer, writer, indent + 1);
        try writer.writeAll("\"param_type\": ");
        try serializeTypeInfo(serializer, param.type_info, writer);
        if (serializer.options.include_spans) {
            try writer.writeAll(",\n");
            try writeSpanFieldNoComma(serializer, writer, &param.span, indent + 1);
        }
        try writer.writeAll("\n");
        try writeIndent(serializer, writer, indent);
        try writer.writeAll("}");
    } else {
        try writer.writeAll("{\"name\":\"");
        try writer.writeAll(param.name);
        try writer.writeAll("\",\"param_type\":");
        try serializeTypeInfo(serializer, param.type_info, writer);
        try writer.writeAll("}");
    }
}

/// Write a type_info field
pub fn writeTypeInfoField(serializer: *AstSerializer, writer: anytype, type_info: *const ast.Types.TypeInfo, indent: u32) SerializationError!void {
    if (serializer.options.pretty_print and !serializer.options.compact_mode) {
        try writer.writeAll(",\n");
        try writeIndent(serializer, writer, indent);
        try writer.writeAll("\"type_info\": ");
    } else {
        try writer.writeAll(",\"type_info\":");
    }
    try serializeTypeInfo(serializer, type_info.*, writer);
}

/// Serialize type information
pub fn serializeTypeInfo(_: *AstSerializer, type_info: ast.Types.TypeInfo, writer: anytype) SerializationError!void {
    try writer.writeAll("{");

    // write category
    try writer.writeAll("\"category\":\"");
    try writer.writeAll(@tagName(type_info.category));
    try writer.writeAll("\"");

    // write source
    try writer.writeAll(",\"source\":\"");
    try writer.writeAll(@tagName(type_info.source));
    try writer.writeAll("\"");

    // write ORA type if present
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
                // for simple types just use the tag name
                try writer.writeAll("\"");
                try writer.writeAll(@tagName(ora_type));
                try writer.writeAll("\"");
            },
        }
    }

    try writer.writeAll("}");
}
