// ============================================================================
// Expression Serializers
// ============================================================================
//
// Handles serialization of expressions and literals:
//   • All expression types (binary, unary, call, index, field access, etc.)
//   • Literal expressions (integer, string, bool, address, hex, binary, etc.)
//
// ============================================================================

const std = @import("std");
const ast = @import("../../ast.zig");
const helpers = @import("helpers.zig");
const patterns = @import("patterns.zig");

// Forward declarations
const AstSerializer = @import("../ast_serializer.zig").AstSerializer;
const SerializationError = @import("../ast_serializer.zig").SerializationError;
const ExprNode = ast.Expressions.ExprNode;

/// Serialize expression
pub fn serializeExpression(serializer: *AstSerializer, expr: *const ExprNode, writer: anytype, indent: u32, depth: u32) SerializationError!void {
    if (serializer.options.pretty_print and !serializer.options.compact_mode) {
        try helpers.writeIndent(serializer, writer, indent);
        try writer.writeAll("{\n");
    } else {
        try writer.writeAll("{");
    }

    switch (expr.*) {
        .Identifier => |*ident| {
            try helpers.writeField(serializer, writer, "type", "Identifier", indent + 1, true);
            try helpers.writeField(serializer, writer, "name", ident.name, indent + 1, false);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &ident.span, indent + 1);
            }
        },
        .Literal => |*literal| {
            try helpers.writeField(serializer, writer, "type", "Literal", indent + 1, true);
            try serializeLiteral(serializer, literal, writer, indent + 1);
        },
        .Binary => |*binary| {
            try helpers.writeField(serializer, writer, "type", "Binary", indent + 1, true);
            try helpers.writeField(serializer, writer, "operator", @tagName(binary.operator), indent + 1, false);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &binary.span, indent + 1);
            }
            if (serializer.options.include_types) {
                try helpers.writeTypeInfoField(serializer, writer, &binary.type_info, indent + 1);
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("\"lhs\": ");
            } else {
                try writer.writeAll(",\"lhs\":");
            }
            try serializeExpression(serializer, binary.lhs, writer, indent + 1, depth + 1);

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("\"rhs\": ");
            } else {
                try writer.writeAll(",\"rhs\":");
            }
            try serializeExpression(serializer, binary.rhs, writer, indent + 1, depth + 1);
        },
        .Unary => |*unary| {
            try helpers.writeField(serializer, writer, "type", "Unary", indent + 1, true);
            try helpers.writeField(serializer, writer, "operator", @tagName(unary.operator), indent + 1, false);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &unary.span, indent + 1);
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("\"operand\": ");
            } else {
                try writer.writeAll(",\"operand\":");
            }
            try serializeExpression(serializer, unary.operand, writer, indent + 1, depth + 1);
        },
        .Assignment => |*assign| {
            try helpers.writeField(serializer, writer, "type", "Assignment", indent + 1, true);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &assign.span, indent + 1);
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("\"target\": ");
            } else {
                try writer.writeAll(",\"target\":");
            }
            try serializeExpression(serializer, assign.target, writer, indent + 1, depth + 1);

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("\"value\": ");
            } else {
                try writer.writeAll(",\"value\":");
            }
            try serializeExpression(serializer, assign.value, writer, indent + 1, depth + 1);
        },
        .CompoundAssignment => |*comp_assign| {
            try helpers.writeField(serializer, writer, "type", "CompoundAssignment", indent + 1, true);
            try helpers.writeField(serializer, writer, "operator", @tagName(comp_assign.operator), indent + 1, false);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &comp_assign.span, indent + 1);
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("\"target\": ");
            } else {
                try writer.writeAll(",\"target\":");
            }
            try serializeExpression(serializer, comp_assign.target, writer, indent + 1, depth + 1);

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("\"value\": ");
            } else {
                try writer.writeAll(",\"value\":");
            }
            try serializeExpression(serializer, comp_assign.value, writer, indent + 1, depth + 1);
        },
        .Call => |*call| {
            try helpers.writeField(serializer, writer, "type", "Call", indent + 1, true);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &call.span, indent + 1);
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("\"callee\": ");
            } else {
                try writer.writeAll(",\"callee\":");
            }
            try serializeExpression(serializer, call.callee, writer, indent + 1, depth + 1);

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("\"arguments\": [\n");
            } else {
                try writer.writeAll(",\"arguments\":[");
            }

            for (call.arguments, 0..) |arg, i| {
                if (i > 0) try writer.writeAll(",");
                if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                    try writer.writeAll("\n");
                }
                try serializeExpression(serializer, arg, writer, indent + 2, depth + 1);
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll("\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("]");
            } else {
                try writer.writeAll("]");
            }
        },
        .Index => |*index| {
            try helpers.writeField(serializer, writer, "type", "Index", indent + 1, true);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &index.span, indent + 1);
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("\"target\": ");
            } else {
                try writer.writeAll(",\"target\":");
            }
            try serializeExpression(serializer, index.target, writer, indent + 1, depth + 1);

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("\"index\": ");
            } else {
                try writer.writeAll(",\"index\":");
            }
            try serializeExpression(serializer, index.index, writer, indent + 1, depth + 1);
        },
        .FieldAccess => |*field_access| {
            try helpers.writeField(serializer, writer, "type", "FieldAccess", indent + 1, true);
            try helpers.writeField(serializer, writer, "field", field_access.field, indent + 1, false);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &field_access.span, indent + 1);
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("\"target\": ");
            } else {
                try writer.writeAll(",\"target\":");
            }
            try serializeExpression(serializer, field_access.target, writer, indent + 1, depth + 1);
        },
        .Cast => |*cast| {
            try helpers.writeField(serializer, writer, "type", "Cast", indent + 1, true);
            try helpers.writeField(serializer, writer, "cast_type", @tagName(cast.cast_type), indent + 1, false);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &cast.span, indent + 1);
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("\"operand\": ");
            } else {
                try writer.writeAll(",\"operand\":");
            }
            try serializeExpression(serializer, cast.operand, writer, indent + 1, depth + 1);

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("\"target_type\": ");
            } else {
                try writer.writeAll(",\"target_type\":");
            }
            try serializer.serializeTypeInfo(cast.target_type, writer);
        },
        .Comptime => |*comptime_expr| {
            try helpers.writeField(serializer, writer, "type", "Comptime", indent + 1, true);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &comptime_expr.span, indent + 1);
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("\"block\": ");
            } else {
                try writer.writeAll(",\"block\":");
            }
            const statements = @import("statements.zig");
            try statements.serializeBlock(serializer, &comptime_expr.block, writer, indent + 1, depth + 1);
        },
        .Old => |*old| {
            try helpers.writeField(serializer, writer, "type", "Old", indent + 1, true);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &old.span, indent + 1);
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("\"expr\": ");
            } else {
                try writer.writeAll(",\"expr\":");
            }
            try serializeExpression(serializer, old.expr, writer, indent + 1, depth + 1);
        },
        .Tuple => |*tuple| {
            try helpers.writeField(serializer, writer, "type", "Tuple", indent + 1, true);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &tuple.span, indent + 1);
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("\"elements\": [\n");
            } else {
                try writer.writeAll(",\"elements\":[");
            }

            for (tuple.elements, 0..) |element, i| {
                if (i > 0) try writer.writeAll(",");
                if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                    try writer.writeAll("\n");
                }
                try serializeExpression(serializer, element, writer, indent + 2, depth + 1);
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll("\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("]");
            } else {
                try writer.writeAll("]");
            }
        },
        .Try => |*try_expr| {
            try helpers.writeField(serializer, writer, "type", "Try", indent + 1, true);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &try_expr.span, indent + 1);
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("\"expr\": ");
            } else {
                try writer.writeAll(",\"expr\":");
            }
            try serializeExpression(serializer, try_expr.expr, writer, indent + 1, depth + 1);
        },
        .ErrorReturn => |*error_return| {
            try helpers.writeField(serializer, writer, "type", "ErrorReturn", indent + 1, true);
            try helpers.writeField(serializer, writer, "error_name", error_return.error_name, indent + 1, false);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &error_return.span, indent + 1);
            }
        },
        .ErrorCast => |*error_cast| {
            try helpers.writeField(serializer, writer, "type", "ErrorCast", indent + 1, true);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &error_cast.span, indent + 1);
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("\"operand\": ");
            } else {
                try writer.writeAll(",\"operand\":");
            }
            try serializeExpression(serializer, error_cast.operand, writer, indent + 1, depth + 1);

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("\"target_type\": ");
            } else {
                try writer.writeAll(",\"target_type\":");
            }
            try serializer.serializeTypeInfo(error_cast.target_type, writer);
        },
        .Shift => |*shift| {
            try helpers.writeField(serializer, writer, "type", "Shift", indent + 1, true);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &shift.span, indent + 1);
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("\"mapping\": ");
            } else {
                try writer.writeAll(",\"mapping\":");
            }
            try serializeExpression(serializer, shift.mapping, writer, indent + 1, depth + 1);

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("\"source\": ");
            } else {
                try writer.writeAll(",\"source\":");
            }
            try serializeExpression(serializer, shift.source, writer, indent + 1, depth + 1);

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("\"dest\": ");
            } else {
                try writer.writeAll(",\"dest\":");
            }
            try serializeExpression(serializer, shift.dest, writer, indent + 1, depth + 1);

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("\"amount\": ");
            } else {
                try writer.writeAll(",\"amount\":");
            }
            try serializeExpression(serializer, shift.amount, writer, indent + 1, depth + 1);
        },
        .StructInstantiation => |*struct_inst| {
            try helpers.writeField(serializer, writer, "type", "StructInstantiation", indent + 1, true);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &struct_inst.span, indent + 1);
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("\"struct_name\": ");
            } else {
                try writer.writeAll(",\"struct_name\":");
            }
            try serializeExpression(serializer, struct_inst.struct_name, writer, indent + 1, depth + 1);

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("\"fields\": [\n");
            } else {
                try writer.writeAll(",\"fields\":[");
            }

            for (struct_inst.fields, 0..) |*field, i| {
                if (i > 0) try writer.writeAll(",");
                if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                    try writer.writeAll("\n");
                    try helpers.writeIndent(serializer, writer, indent + 2);
                    try writer.writeAll("{\n");
                    try helpers.writeField(serializer, writer, "name", field.name, indent + 3, true);
                    try writer.writeAll(",\n");
                    try helpers.writeIndent(serializer, writer, indent + 3);
                    try writer.writeAll("\"value\": ");
                    try serializeExpression(serializer, field.value, writer, indent + 3, depth + 1);
                    if (serializer.options.include_spans) {
                        try writer.writeAll(",\n");
                        try helpers.writeSpanFieldNoComma(serializer, writer, &field.span, indent + 3);
                    }
                    try writer.writeAll("\n");
                    try helpers.writeIndent(serializer, writer, indent + 2);
                    try writer.writeAll("}");
                } else {
                    try writer.writeAll("{\"name\":\"");
                    try writer.writeAll(field.name);
                    try writer.writeAll("\",\"value\":");
                    try serializeExpression(serializer, field.value, writer, 0, depth + 1);
                    try writer.writeAll("}");
                }
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll("\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("]");
            } else {
                try writer.writeAll("]");
            }
        },
        .EnumLiteral => |*enum_literal| {
            try helpers.writeField(serializer, writer, "type", "EnumLiteral", indent + 1, true);
            try helpers.writeField(serializer, writer, "enum_name", enum_literal.enum_name, indent + 1, false);
            try helpers.writeField(serializer, writer, "variant_name", enum_literal.variant_name, indent + 1, false);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &enum_literal.span, indent + 1);
            }
        },
        .SwitchExpression => |*switch_expr| {
            try helpers.writeField(serializer, writer, "type", "SwitchExpression", indent + 1, true);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &switch_expr.span, indent + 1);
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("\"target\": ");
            } else {
                try writer.writeAll(",\"target\":");
            }
            try serializeExpression(serializer, switch_expr.condition, writer, indent + 1, depth + 1);

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("\"cases\": [\n");
            } else {
                try writer.writeAll(",\"cases\":[");
            }

            for (switch_expr.cases, 0..) |*case, i| {
                if (i > 0) try writer.writeAll(",");
                if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                    try writer.writeAll("\n");
                    try helpers.writeIndent(serializer, writer, indent + 2);
                    try writer.writeAll("{\n");
                } else {
                    try writer.writeAll("{");
                }

                try patterns.serializeSwitchCase(serializer, case, writer, indent + 2, depth + 1);

                if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                    try writer.writeAll("\n");
                    try helpers.writeIndent(serializer, writer, indent + 2);
                    try writer.writeAll("}");
                } else {
                    try writer.writeAll("}");
                }
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll("\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("]");
            } else {
                try writer.writeAll("]");
            }
        },
        .Quantified => |*quantified| {
            try helpers.writeField(serializer, writer, "type", "Quantified", indent + 1, true);
            try helpers.writeField(serializer, writer, "quantifier", @tagName(quantified.quantifier), indent + 1, false);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &quantified.span, indent + 1);
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("\"variable\": \"");
                try writer.writeAll(quantified.variable);
                try writer.writeAll("\",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("\"variable_type\": ");
            } else {
                try writer.writeAll(",\"variable\":\"");
                try writer.writeAll(quantified.variable);
                try writer.writeAll("\",\"variable_type\":");
            }
            try serializer.serializeTypeInfo(quantified.variable_type, writer);

            // serialize optional condition
            if (quantified.condition) |condition| {
                if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                    try writer.writeAll(",\n");
                    try helpers.writeIndent(serializer, writer, indent + 1);
                    try writer.writeAll("\"condition\": ");
                } else {
                    try writer.writeAll(",\"condition\":");
                }
                try serializeExpression(serializer, condition, writer, indent + 1, depth + 1);
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("\"body\": ");
            } else {
                try writer.writeAll(",\"body\":");
            }
            try serializeExpression(serializer, quantified.body, writer, indent + 1, depth + 1);
        },
        .AnonymousStruct => |*anon_struct| {
            try helpers.writeField(serializer, writer, "type", "AnonymousStruct", indent + 1, true);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &anon_struct.span, indent + 1);
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("\"fields\": [\n");
            } else {
                try writer.writeAll(",\"fields\":[");
            }

            for (anon_struct.fields, 0..) |*field, i| {
                if (i > 0) try writer.writeAll(",");
                if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                    try writer.writeAll("\n");
                    try helpers.writeIndent(serializer, writer, indent + 2);
                    try writer.writeAll("{\n");
                    try helpers.writeField(serializer, writer, "name", field.name, indent + 3, true);
                    try writer.writeAll(",\n");
                    try helpers.writeIndent(serializer, writer, indent + 3);
                    try writer.writeAll("\"value\": ");
                    try serializeExpression(serializer, field.value, writer, indent + 3, depth + 1);
                    if (serializer.options.include_spans) {
                        try writer.writeAll(",\n");
                        try helpers.writeSpanFieldNoComma(serializer, writer, &field.span, indent + 3);
                    }
                    try writer.writeAll("\n");
                    try helpers.writeIndent(serializer, writer, indent + 2);
                    try writer.writeAll("}");
                } else {
                    try writer.writeAll("{\"name\":\"");
                    try writer.writeAll(field.name);
                    try writer.writeAll("\",\"value\":");
                    try serializeExpression(serializer, field.value, writer, 0, depth + 1);
                    try writer.writeAll("}");
                }
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll("\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("]");
            } else {
                try writer.writeAll("]");
            }
        },
        .Range => |*range| {
            try helpers.writeField(serializer, writer, "type", "Range", indent + 1, true);
            try helpers.writeBoolField(serializer, writer, "inclusive", range.inclusive, indent + 1);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &range.span, indent + 1);
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("\"start\": ");
            } else {
                try writer.writeAll(",\"start\":");
            }
            try serializeExpression(serializer, range.start, writer, indent + 1, depth + 1);

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("\"end\": ");
            } else {
                try writer.writeAll(",\"end\":");
            }
            try serializeExpression(serializer, range.end, writer, indent + 1, depth + 1);
        },
        .LabeledBlock => |*labeled_block| {
            try helpers.writeField(serializer, writer, "type", "LabeledBlock", indent + 1, true);
            try helpers.writeField(serializer, writer, "label", labeled_block.label, indent + 1, false);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &labeled_block.span, indent + 1);
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("\"block\": ");
            } else {
                try writer.writeAll(",\"block\":");
            }
            const statements = @import("statements.zig");
            try statements.serializeBlock(serializer, &labeled_block.block, writer, indent + 1, depth + 1);
        },
        .Destructuring => |*destructuring| {
            try helpers.writeField(serializer, writer, "type", "Destructuring", indent + 1, true);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &destructuring.span, indent + 1);
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("\"pattern\": ");
            } else {
                try writer.writeAll(",\"pattern\":");
            }
            try patterns.serializeDestructuringPattern(serializer, &destructuring.pattern, writer, indent + 1);

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("\"value\": ");
            } else {
                try writer.writeAll(",\"value\":");
            }
            try serializeExpression(serializer, destructuring.value, writer, indent + 1, depth + 1);
        },
        .ArrayLiteral => |*array_literal| {
            try helpers.writeField(serializer, writer, "type", "ArrayLiteral", indent + 1, true);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &array_literal.span, indent + 1);
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("\"elements\": [\n");
            } else {
                try writer.writeAll(",\"elements\":[");
            }

            for (array_literal.elements, 0..) |element, i| {
                if (i > 0) try writer.writeAll(",");
                if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                    try writer.writeAll("\n");
                }
                try serializeExpression(serializer, element, writer, indent + 2, depth + 1);
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.writeAll("\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.writeAll("]");
            } else {
                try writer.writeAll("]");
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

/// Serialize literal expression
pub fn serializeLiteral(serializer: *AstSerializer, literal: *const ast.Expressions.LiteralExpr, writer: anytype, indent: u32) SerializationError!void {
    switch (literal.*) {
        .Integer => |*int_lit| {
            try helpers.writeField(serializer, writer, "literal_type", "Integer", indent, false);
            try helpers.writeField(serializer, writer, "value", int_lit.value, indent, false);
            // include the integer type information
            try writer.writeAll(",\n");
            try helpers.writeIndent(serializer, writer, indent);
            try writer.writeAll("\"type_info\": ");
            try serializer.serializeTypeInfo(int_lit.type_info, writer);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &int_lit.span, indent);
            }
        },
        .String => |*str_lit| {
            try helpers.writeField(serializer, writer, "literal_type", "String", indent, false);
            try writer.writeAll(",\n");
            try helpers.writeIndent(serializer, writer, indent);
            try writer.writeAll("\"type_info\": ");
            try serializer.serializeTypeInfo(str_lit.type_info, writer);
            try writer.writeAll(",\n");
            try helpers.writeField(serializer, writer, "value", str_lit.value, indent, false);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &str_lit.span, indent);
            }
        },
        .Bool => |*bool_lit| {
            try helpers.writeField(serializer, writer, "literal_type", "Bool", indent, false);
            try writer.writeAll(",\n");
            try helpers.writeIndent(serializer, writer, indent);
            try writer.writeAll("\"type_info\": ");
            try serializer.serializeTypeInfo(bool_lit.type_info, writer);
            try writer.writeAll(",\n");
            try helpers.writeBoolField(serializer, writer, "value", bool_lit.value, indent);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &bool_lit.span, indent);
            }
        },
        .Address => |*addr_lit| {
            try helpers.writeField(serializer, writer, "literal_type", "Address", indent, false);
            try writer.writeAll(",\n");
            try helpers.writeIndent(serializer, writer, indent);
            try writer.writeAll("\"type_info\": ");
            try serializer.serializeTypeInfo(addr_lit.type_info, writer);
            try writer.writeAll(",\n");
            try helpers.writeField(serializer, writer, "value", addr_lit.value, indent, false);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &addr_lit.span, indent);
            }
        },
        .Hex => |*hex_lit| {
            try helpers.writeField(serializer, writer, "literal_type", "Hex", indent, false);
            try writer.writeAll(",\n");
            try helpers.writeIndent(serializer, writer, indent);
            try writer.writeAll("\"type_info\": ");
            try serializer.serializeTypeInfo(hex_lit.type_info, writer);
            try writer.writeAll(",\n");
            try helpers.writeField(serializer, writer, "value", hex_lit.value, indent, false);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &hex_lit.span, indent);
            }
        },
        .Binary => |*bin_lit| {
            try helpers.writeField(serializer, writer, "literal_type", "Binary", indent, false);
            try writer.writeAll(",\n");
            try helpers.writeIndent(serializer, writer, indent);
            try writer.writeAll("\"type_info\": ");
            try serializer.serializeTypeInfo(bin_lit.type_info, writer);
            try writer.writeAll(",\n");
            try helpers.writeField(serializer, writer, "value", bin_lit.value, indent, false);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &bin_lit.span, indent);
            }
        },
        .Character => |*char_lit| {
            try helpers.writeField(serializer, writer, "literal_type", "Character", indent, false);
            try writer.writeAll(",\n");
            try helpers.writeIndent(serializer, writer, indent);
            try writer.writeAll("\"type_info\": ");
            try serializer.serializeTypeInfo(char_lit.type_info, writer);
            try writer.writeAll(",\n");
            try helpers.writeField(serializer, writer, "value", try std.fmt.allocPrint(serializer.allocator, "{c}", .{char_lit.value}), indent, false);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &char_lit.span, indent);
            }
        },
        .Bytes => |*bytes_lit| {
            try helpers.writeField(serializer, writer, "literal_type", "Bytes", indent, false);
            try writer.writeAll(",\n");
            try helpers.writeIndent(serializer, writer, indent);
            try writer.writeAll("\"type_info\": ");
            try serializer.serializeTypeInfo(bytes_lit.type_info, writer);
            try writer.writeAll(",\n");
            try helpers.writeField(serializer, writer, "value", bytes_lit.value, indent, false);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &bytes_lit.span, indent);
            }
        },
    }
}
