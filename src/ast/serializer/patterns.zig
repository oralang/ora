// ============================================================================
// Pattern Serializers
// ============================================================================
//
// Handles serialization of patterns used in various AST nodes:
//   • Loop patterns (for loops)
//   • Switch patterns and cases
//   • Switch bodies
//   • Destructuring patterns
//
// ============================================================================

const std = @import("std");
const ast = @import("../../ast.zig");
const helpers = @import("helpers.zig");
const expressions = @import("expressions.zig");
const statements = @import("statements.zig");

// Forward declarations
const AstSerializer = @import("../ast_serializer.zig").AstSerializer;
const SerializationError = @import("../ast_serializer.zig").SerializationError;

/// Serialize a loop pattern (for ForLoop statements)
pub fn serializeLoopPattern(serializer: *AstSerializer, pattern: *const ast.Statements.LoopPattern, writer: anytype, indent: u32) SerializationError!void {
    if (serializer.options.pretty_print and !serializer.options.compact_mode) {
        try helpers.writeIndent(serializer, writer, indent);
        try writer.writeAll("{\n");
    } else {
        try writer.writeAll("{");
    }

    switch (pattern.*) {
        .Single => |single| {
            try helpers.writeField(serializer, writer, "type", "Single", indent + 1, true);
            try helpers.writeField(serializer, writer, "variable", single.name, indent + 1, false);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &single.span, indent + 1);
            }
        },
        .IndexPair => |pair| {
            try helpers.writeField(serializer, writer, "type", "IndexPair", indent + 1, true);
            try helpers.writeField(serializer, writer, "item", pair.item, indent + 1, false);
            try helpers.writeField(serializer, writer, "index", pair.index, indent + 1, false);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &pair.span, indent + 1);
            }
        },
        .Destructured => |destructured| {
            try helpers.writeField(serializer, writer, "type", "Destructured", indent + 1, true);
            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("\"pattern\": ");
            } else {
                try writer.writeAll(",\"pattern\":");
            }
            try serializeDestructuringPattern(serializer, &destructured.pattern, writer, indent + 1);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &destructured.span, indent + 1);
            }
        },
    }

    if (serializer.options.pretty_print and !serializer.options.compact_mode) {
        try writer.writeAll("\n");
        try helpers.writeIndent(serializer, writer, indent);
        try writer.writeAll("}");
    } else {
        try writer.writeAll("}");
    }
}

/// Serialize a switch case
pub fn serializeSwitchCase(serializer: *AstSerializer, case: *const ast.Switch.Case, writer: anytype, indent: u32, depth: u32) SerializationError!void {
    try helpers.writeField(serializer, writer, "type", "SwitchCase", indent + 1, true);
    if (serializer.options.include_spans) {
        try helpers.writeSpanField(serializer, writer, &case.span, indent + 1);
    }

    // serialize the pattern
    if (serializer.options.pretty_print and !serializer.options.compact_mode) {
        try writer.writeAll(",\n");
        try helpers.writeIndent(serializer, writer, indent + 1);
        try writer.writeAll("\"pattern\": ");
    } else {
        try writer.writeAll(",\"pattern\":");
    }
    try serializeSwitchPattern(serializer, &case.pattern, writer, indent + 1);

    // serialize the body
    if (serializer.options.pretty_print and !serializer.options.compact_mode) {
        try writer.writeAll(",\n");
        try helpers.writeIndent(serializer, writer, indent + 1);
        try writer.writeAll("\"body\": ");
    } else {
        try writer.writeAll(",\"body\":");
    }
    try serializeSwitchBody(serializer, &case.body, writer, indent + 1, depth + 1);
}

/// Serialize a switch pattern
pub fn serializeSwitchPattern(serializer: *AstSerializer, pattern: *const ast.Switch.Pattern, writer: anytype, indent: u32) SerializationError!void {
    if (serializer.options.pretty_print and !serializer.options.compact_mode) {
        try helpers.writeIndent(serializer, writer, indent);
        try writer.writeAll("{\n");
    } else {
        try writer.writeAll("{");
    }

    switch (pattern.*) {
        .Literal => |*literal| {
            try helpers.writeField(serializer, writer, "type", "Literal", indent + 1, true);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &literal.span, indent + 1);
            }
            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("\"value\": ");
            } else {
                try writer.writeAll(",\"value\":");
            }
            try expressions.serializeLiteral(serializer, &literal.value, writer, indent + 1);
        },
        .Range => |*range| {
            try helpers.writeField(serializer, writer, "type", "Range", indent + 1, true);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &range.span, indent + 1);
            }
            try helpers.writeBoolField(serializer, writer, "inclusive", range.inclusive, indent + 1);

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("\"start\": ");
            } else {
                try writer.writeAll(",\"start\":");
            }
            try expressions.serializeExpression(serializer, range.start, writer, indent + 1, 0);

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("\"end\": ");
            } else {
                try writer.writeAll(",\"end\":");
            }
            try expressions.serializeExpression(serializer, range.end, writer, indent + 1, 0);
        },
        .EnumValue => |*enum_value| {
            try helpers.writeField(serializer, writer, "type", "EnumValue", indent + 1, true);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &enum_value.span, indent + 1);
            }
            try helpers.writeField(serializer, writer, "enum_name", enum_value.enum_name, indent + 1, false);
            try helpers.writeField(serializer, writer, "variant_name", enum_value.variant_name, indent + 1, false);
        },
        .Else => |*span| {
            try helpers.writeField(serializer, writer, "type", "Else", indent + 1, true);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, span, indent + 1);
            }
        },
    }

    if (serializer.options.pretty_print and !serializer.options.compact_mode) {
        try writer.writeAll("\n");
        try helpers.writeIndent(serializer, writer, indent);
        try writer.writeAll("}");
    } else {
        try writer.writeAll("}");
    }
}

/// Serialize a switch body
pub fn serializeSwitchBody(serializer: *AstSerializer, body: *const ast.Switch.Body, writer: anytype, indent: u32, depth: u32) SerializationError!void {
    switch (body.*) {
        .Expression => |expr| {
            try expressions.serializeExpression(serializer, expr, writer, indent, depth);
        },
        .Block => |*block| {
            try statements.serializeBlock(serializer, block, writer, indent, depth);
        },
        .LabeledBlock => |*labeled| {
            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try helpers.writeIndent(serializer, writer, indent);
                try writer.writeAll("{\n");
            } else {
                try writer.writeAll("{");
            }

            try helpers.writeField(serializer, writer, "type", "LabeledBlock", indent + 1, true);
            try helpers.writeField(serializer, writer, "label", labeled.label, indent + 1, false);

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("\"block\": ");
            } else {
                try writer.writeAll(",\"block\":");
            }
            try statements.serializeBlock(serializer, &labeled.block, writer, indent + 1, depth + 1);

            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &labeled.span, indent + 1);
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll("\n");
                try helpers.writeIndent(serializer, writer, indent);
                try writer.writeAll("}");
            } else {
                try writer.writeAll("}");
            }
        },
    }
}

/// Serialize a destructuring pattern
pub fn serializeDestructuringPattern(serializer: *AstSerializer, pattern: *const ast.Expressions.DestructuringPattern, writer: anytype, indent: u32) SerializationError!void {
    if (serializer.options.pretty_print and !serializer.options.compact_mode) {
        try helpers.writeIndent(serializer, writer, indent);
        try writer.writeAll("{\n");
    } else {
        try writer.writeAll("{");
    }

    switch (pattern.*) {
        .Struct => |fields| {
            try helpers.writeField(serializer, writer, "type", "Struct", indent + 1, true);
            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
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
            try helpers.writeField(serializer, writer, "type", "Tuple", indent + 1, true);
            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
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
            try helpers.writeField(serializer, writer, "type", "Array", indent + 1, true);
            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
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

    if (serializer.options.pretty_print and !serializer.options.compact_mode) {
        try writer.writeAll("\n");
        try helpers.writeIndent(serializer, writer, indent);
        try writer.writeAll("}");
    } else {
        try writer.writeAll("}");
    }
}
