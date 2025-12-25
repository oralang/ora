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
        try writer.print("{\n");
    } else {
        try writer.print("{");
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

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"lhs\": ");
            } else {
                try writer.print(",\"lhs\":");
            }
            try serializeExpression(serializer, binary.lhs, writer, indent + 1, depth + 1);

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"rhs\": ");
            } else {
                try writer.print(",\"rhs\":");
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
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"operand\": ");
            } else {
                try writer.print(",\"operand\":");
            }
            try serializeExpression(serializer, unary.operand, writer, indent + 1, depth + 1);
        },
        .Assignment => |*assign| {
            try helpers.writeField(serializer, writer, "type", "Assignment", indent + 1, true);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &assign.span, indent + 1);
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"target\": ");
            } else {
                try writer.print(",\"target\":");
            }
            try serializeExpression(serializer, assign.target, writer, indent + 1, depth + 1);

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"value\": ");
            } else {
                try writer.print(",\"value\":");
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
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"target\": ");
            } else {
                try writer.print(",\"target\":");
            }
            try serializeExpression(serializer, comp_assign.target, writer, indent + 1, depth + 1);

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"value\": ");
            } else {
                try writer.print(",\"value\":");
            }
            try serializeExpression(serializer, comp_assign.value, writer, indent + 1, depth + 1);
        },
        .Call => |*call| {
            try helpers.writeField(serializer, writer, "type", "Call", indent + 1, true);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &call.span, indent + 1);
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"callee\": ");
            } else {
                try writer.print(",\"callee\":");
            }
            try serializeExpression(serializer, call.callee, writer, indent + 1, depth + 1);

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"arguments\": [\n");
            } else {
                try writer.print(",\"arguments\":[");
            }

            for (call.arguments, 0..) |arg, i| {
                if (i > 0) try writer.print(",");
                if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                    try writer.print("\n");
                }
                try serializeExpression(serializer, arg, writer, indent + 2, depth + 1);
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print("\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("]");
            } else {
                try writer.print("]");
            }
        },
        .Index => |*index| {
            try helpers.writeField(serializer, writer, "type", "Index", indent + 1, true);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &index.span, indent + 1);
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"target\": ");
            } else {
                try writer.print(",\"target\":");
            }
            try serializeExpression(serializer, index.target, writer, indent + 1, depth + 1);

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"index\": ");
            } else {
                try writer.print(",\"index\":");
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
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"target\": ");
            } else {
                try writer.print(",\"target\":");
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
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"operand\": ");
            } else {
                try writer.print(",\"operand\":");
            }
            try serializeExpression(serializer, cast.operand, writer, indent + 1, depth + 1);

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"target_type\": ");
            } else {
                try writer.print(",\"target_type\":");
            }
            try serializer.serializeTypeInfo(cast.target_type, writer);
        },
        .Comptime => |*comptime_expr| {
            try helpers.writeField(serializer, writer, "type", "Comptime", indent + 1, true);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &comptime_expr.span, indent + 1);
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"block\": ");
            } else {
                try writer.print(",\"block\":");
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
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"expr\": ");
            } else {
                try writer.print(",\"expr\":");
            }
            try serializeExpression(serializer, old.expr, writer, indent + 1, depth + 1);
        },
        .Tuple => |*tuple| {
            try helpers.writeField(serializer, writer, "type", "Tuple", indent + 1, true);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &tuple.span, indent + 1);
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"elements\": [\n");
            } else {
                try writer.print(",\"elements\":[");
            }

            for (tuple.elements, 0..) |element, i| {
                if (i > 0) try writer.print(",");
                if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                    try writer.print("\n");
                }
                try serializeExpression(serializer, element, writer, indent + 2, depth + 1);
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print("\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("]");
            } else {
                try writer.print("]");
            }
        },
        .Try => |*try_expr| {
            try helpers.writeField(serializer, writer, "type", "Try", indent + 1, true);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &try_expr.span, indent + 1);
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"expr\": ");
            } else {
                try writer.print(",\"expr\":");
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
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"operand\": ");
            } else {
                try writer.print(",\"operand\":");
            }
            try serializeExpression(serializer, error_cast.operand, writer, indent + 1, depth + 1);

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"target_type\": ");
            } else {
                try writer.print(",\"target_type\":");
            }
            try serializer.serializeTypeInfo(error_cast.target_type, writer);
        },
        .Shift => |*shift| {
            try helpers.writeField(serializer, writer, "type", "Shift", indent + 1, true);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &shift.span, indent + 1);
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"mapping\": ");
            } else {
                try writer.print(",\"mapping\":");
            }
            try serializeExpression(serializer, shift.mapping, writer, indent + 1, depth + 1);

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"source\": ");
            } else {
                try writer.print(",\"source\":");
            }
            try serializeExpression(serializer, shift.source, writer, indent + 1, depth + 1);

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"dest\": ");
            } else {
                try writer.print(",\"dest\":");
            }
            try serializeExpression(serializer, shift.dest, writer, indent + 1, depth + 1);

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"amount\": ");
            } else {
                try writer.print(",\"amount\":");
            }
            try serializeExpression(serializer, shift.amount, writer, indent + 1, depth + 1);
        },
        .StructInstantiation => |*struct_inst| {
            try helpers.writeField(serializer, writer, "type", "StructInstantiation", indent + 1, true);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &struct_inst.span, indent + 1);
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"struct_name\": ");
            } else {
                try writer.print(",\"struct_name\":");
            }
            try serializeExpression(serializer, struct_inst.struct_name, writer, indent + 1, depth + 1);

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"fields\": [\n");
            } else {
                try writer.print(",\"fields\":[");
            }

            for (struct_inst.fields, 0..) |*field, i| {
                if (i > 0) try writer.print(",");
                if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                    try writer.print("\n");
                    try helpers.writeIndent(serializer, writer, indent + 2);
                    try writer.print("{\n");
                    try helpers.writeField(serializer, writer, "name", field.name, indent + 3, true);
                    try writer.print(",\n");
                    try helpers.writeIndent(serializer, writer, indent + 3);
                    try writer.print("\"value\": ");
                    try serializeExpression(serializer, field.value, writer, indent + 3, depth + 1);
                    if (serializer.options.include_spans) {
                        try writer.print(",\n");
                        try helpers.writeSpanField(serializer, writer, &field.span, indent + 3);
                    }
                    try writer.print("\n");
                    try helpers.writeIndent(serializer, writer, indent + 2);
                    try writer.print("}");
                } else {
                    try writer.print("{\"name\":\"");
                    try writer.print(field.name);
                    try writer.print("\",\"value\":");
                    try serializeExpression(serializer, field.value, writer, 0, depth + 1);
                    try writer.print("}");
                }
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print("\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("]");
            } else {
                try writer.print("]");
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
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"target\": ");
            } else {
                try writer.print(",\"target\":");
            }
            try serializeExpression(serializer, switch_expr.condition, writer, indent + 1, depth + 1);

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"cases\": [\n");
            } else {
                try writer.print(",\"cases\":[");
            }

            for (switch_expr.cases, 0..) |*case, i| {
                if (i > 0) try writer.print(",");
                if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                    try writer.print("\n");
                    try helpers.writeIndent(serializer, writer, indent + 2);
                    try writer.print("{\n");
                } else {
                    try writer.print("{");
                }

                try patterns.serializeSwitchCase(serializer, case, writer, indent + 2, depth + 1);

                if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                    try writer.print("\n");
                    try helpers.writeIndent(serializer, writer, indent + 2);
                    try writer.print("}");
                } else {
                    try writer.print("}");
                }
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print("\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("]");
            } else {
                try writer.print("]");
            }
        },
        .Quantified => |*quantified| {
            try helpers.writeField(serializer, writer, "type", "Quantified", indent + 1, true);
            try helpers.writeField(serializer, writer, "quantifier", @tagName(quantified.quantifier), indent + 1, false);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &quantified.span, indent + 1);
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"variable\": \"");
                try writer.print(quantified.variable);
                try writer.print("\",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"variable_type\": ");
            } else {
                try writer.print(",\"variable\":\"");
                try writer.print(quantified.variable);
                try writer.print("\",\"variable_type\":");
            }
            try serializer.serializeTypeInfo(quantified.variable_type, writer);

            // serialize optional condition
            if (quantified.condition) |condition| {
                if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                    try writer.print(",\n");
                    try helpers.writeIndent(serializer, writer, indent + 1);
                    try writer.print("\"condition\": ");
                } else {
                    try writer.print(",\"condition\":");
                }
                try serializeExpression(serializer, condition, writer, indent + 1, depth + 1);
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"body\": ");
            } else {
                try writer.print(",\"body\":");
            }
            try serializeExpression(serializer, quantified.body, writer, indent + 1, depth + 1);
        },
        .AnonymousStruct => |*anon_struct| {
            try helpers.writeField(serializer, writer, "type", "AnonymousStruct", indent + 1, true);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &anon_struct.span, indent + 1);
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"fields\": [\n");
            } else {
                try writer.print(",\"fields\":[");
            }

            for (anon_struct.fields, 0..) |*field, i| {
                if (i > 0) try writer.print(",");
                if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                    try writer.print("\n");
                    try helpers.writeIndent(serializer, writer, indent + 2);
                    try writer.print("{\n");
                    try helpers.writeField(serializer, writer, "name", field.name, indent + 3, true);
                    try writer.print(",\n");
                    try helpers.writeIndent(serializer, writer, indent + 3);
                    try writer.print("\"value\": ");
                    try serializeExpression(serializer, field.value, writer, indent + 3, depth + 1);
                    if (serializer.options.include_spans) {
                        try writer.print(",\n");
                        try helpers.writeSpanField(serializer, writer, &field.span, indent + 3);
                    }
                    try writer.print("\n");
                    try helpers.writeIndent(serializer, writer, indent + 2);
                    try writer.print("}");
                } else {
                    try writer.print("{\"name\":\"");
                    try writer.print(field.name);
                    try writer.print("\",\"value\":");
                    try serializeExpression(serializer, field.value, writer, 0, depth + 1);
                    try writer.print("}");
                }
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print("\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("]");
            } else {
                try writer.print("]");
            }
        },
        .Range => |*range| {
            try helpers.writeField(serializer, writer, "type", "Range", indent + 1, true);
            try helpers.writeBoolField(serializer, writer, "inclusive", range.inclusive, indent + 1);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &range.span, indent + 1);
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"start\": ");
            } else {
                try writer.print(",\"start\":");
            }
            try serializeExpression(serializer, range.start, writer, indent + 1, depth + 1);

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"end\": ");
            } else {
                try writer.print(",\"end\":");
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
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"block\": ");
            } else {
                try writer.print(",\"block\":");
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
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"pattern\": ");
            } else {
                try writer.print(",\"pattern\":");
            }
            try patterns.serializeDestructuringPattern(serializer, &destructuring.pattern, writer, indent + 1);

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"value\": ");
            } else {
                try writer.print(",\"value\":");
            }
            try serializeExpression(serializer, destructuring.value, writer, indent + 1, depth + 1);
        },
        .ArrayLiteral => |*array_literal| {
            try helpers.writeField(serializer, writer, "type", "ArrayLiteral", indent + 1, true);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &array_literal.span, indent + 1);
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"elements\": [\n");
            } else {
                try writer.print(",\"elements\":[");
            }

            for (array_literal.elements, 0..) |element, i| {
                if (i > 0) try writer.print(",");
                if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                    try writer.print("\n");
                }
                try serializeExpression(serializer, element, writer, indent + 2, depth + 1);
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print("\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("]");
            } else {
                try writer.print("]");
            }
        },
    }

    if (serializer.options.pretty_print and !serializer.options.compact_mode) {
        try writer.print("\n");
        try helpers.writeIndent(serializer, writer, indent);
        try writer.print("}");
    } else {
        try writer.print("}");
    }
}

/// Serialize literal expression
pub fn serializeLiteral(serializer: *AstSerializer, literal: *const ast.Expressions.LiteralExpr, writer: anytype, indent: u32) SerializationError!void {
    switch (literal.*) {
        .Integer => |*int_lit| {
            try helpers.writeField(serializer, writer, "literal_type", "Integer", indent, false);
            try helpers.writeField(serializer, writer, "value", int_lit.value, indent, false);
            // include the integer type information
            try writer.print(",\n");
            try helpers.writeIndent(serializer, writer, indent);
            try writer.print("\"type_info\": ");
            try serializer.serializeTypeInfo(int_lit.type_info, writer);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &int_lit.span, indent);
            }
        },
        .String => |*str_lit| {
            try helpers.writeField(serializer, writer, "literal_type", "String", indent, false);
            try writer.print(",\n");
            try helpers.writeIndent(serializer, writer, indent);
            try writer.print("\"type_info\": ");
            try serializer.serializeTypeInfo(str_lit.type_info, writer);
            try writer.print(",\n");
            try helpers.writeField(serializer, writer, "value", str_lit.value, indent, false);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &str_lit.span, indent);
            }
        },
        .Bool => |*bool_lit| {
            try helpers.writeField(serializer, writer, "literal_type", "Bool", indent, false);
            try writer.print(",\n");
            try helpers.writeIndent(serializer, writer, indent);
            try writer.print("\"type_info\": ");
            try serializer.serializeTypeInfo(bool_lit.type_info, writer);
            try writer.print(",\n");
            try helpers.writeBoolField(serializer, writer, "value", bool_lit.value, indent);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &bool_lit.span, indent);
            }
        },
        .Address => |*addr_lit| {
            try helpers.writeField(serializer, writer, "literal_type", "Address", indent, false);
            try writer.print(",\n");
            try helpers.writeIndent(serializer, writer, indent);
            try writer.print("\"type_info\": ");
            try serializer.serializeTypeInfo(addr_lit.type_info, writer);
            try writer.print(",\n");
            try helpers.writeField(serializer, writer, "value", addr_lit.value, indent, false);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &addr_lit.span, indent);
            }
        },
        .Hex => |*hex_lit| {
            try helpers.writeField(serializer, writer, "literal_type", "Hex", indent, false);
            try writer.print(",\n");
            try helpers.writeIndent(serializer, writer, indent);
            try writer.print("\"type_info\": ");
            try serializer.serializeTypeInfo(hex_lit.type_info, writer);
            try writer.print(",\n");
            try helpers.writeField(serializer, writer, "value", hex_lit.value, indent, false);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &hex_lit.span, indent);
            }
        },
        .Binary => |*bin_lit| {
            try helpers.writeField(serializer, writer, "literal_type", "Binary", indent, false);
            try writer.print(",\n");
            try helpers.writeIndent(serializer, writer, indent);
            try writer.print("\"type_info\": ");
            try serializer.serializeTypeInfo(bin_lit.type_info, writer);
            try writer.print(",\n");
            try helpers.writeField(serializer, writer, "value", bin_lit.value, indent, false);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &bin_lit.span, indent);
            }
        },
        .Character => |*char_lit| {
            try helpers.writeField(serializer, writer, "literal_type", "Character", indent, false);
            try writer.print(",\n");
            try helpers.writeIndent(serializer, writer, indent);
            try writer.print("\"type_info\": ");
            try serializer.serializeTypeInfo(char_lit.type_info, writer);
            try writer.print(",\n");
            try helpers.writeField(serializer, writer, "value", try std.fmt.allocPrint(serializer.allocator, "{c}", .{char_lit.value}), indent, false);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &char_lit.span, indent);
            }
        },
        .Bytes => |*bytes_lit| {
            try helpers.writeField(serializer, writer, "literal_type", "Bytes", indent, false);
            try writer.print(",\n");
            try helpers.writeIndent(serializer, writer, indent);
            try writer.print("\"type_info\": ");
            try serializer.serializeTypeInfo(bytes_lit.type_info, writer);
            try writer.print(",\n");
            try helpers.writeField(serializer, writer, "value", bytes_lit.value, indent, false);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &bytes_lit.span, indent);
            }
        },
    }
}
