// ============================================================================
// Statement Serializers
// ============================================================================
//
// Handles serialization of statements and blocks:
//   • Blocks, try-catch blocks
//   • All statement types (if, while, for, switch, break, continue, etc.)
//
// ============================================================================

const std = @import("std");
const ast = @import("../../ast.zig");
const helpers = @import("helpers.zig");
const patterns = @import("patterns.zig");
const expressions = @import("expressions.zig");

// Forward declarations
const AstSerializer = @import("../ast_serializer.zig").AstSerializer;
const SerializationError = @import("../ast_serializer.zig").SerializationError;
const StmtNode = ast.Statements.StmtNode;

/// Serialize block statement
pub fn serializeBlock(serializer: *AstSerializer, block: *const ast.Statements.BlockNode, writer: anytype, indent: u32, depth: u32) SerializationError!void {
    if (serializer.options.pretty_print and !serializer.options.compact_mode) {
        try writer.print("{\n");
        try helpers.writeField(serializer, writer, "type", "Block", indent + 1, true);
        if (serializer.options.include_spans) {
            try writer.print(",\n");
            try helpers.writeSpanField(serializer, writer, &block.span, indent + 1);
        }
        try writer.print(",\n");
        try helpers.writeIndent(serializer, writer, indent + 1);
        try writer.print("\"statements\": [\n");
    } else {
        try writer.print("{\"type\":\"Block\",\"statements\":[");
    }

    for (block.statements, 0..) |*stmt, i| {
        if (i > 0) try writer.print(",");
        if (serializer.options.pretty_print and !serializer.options.compact_mode) {
            try writer.print("\n");
        }
        try serializeStatement(serializer, stmt, writer, indent + 2, depth + 1);
    }

    if (serializer.options.pretty_print and !serializer.options.compact_mode) {
        try writer.print("\n");
        try helpers.writeIndent(serializer, writer, indent + 1);
        try writer.print("]\n");
        try helpers.writeIndent(serializer, writer, indent);
        try writer.print("}");
    } else {
        try writer.print("]}");
    }
}

/// Serialize try-catch block
pub fn serializeTryBlock(serializer: *AstSerializer, try_block: *const ast.Statements.TryBlockNode, writer: anytype, indent: u32, depth: u32) SerializationError!void {
    try helpers.writeField(serializer, writer, "type", "TryBlock", indent + 1, true);

    if (serializer.options.include_spans) {
        try helpers.writeSpanField(serializer, writer, &try_block.span, indent + 1);
    }

    if (serializer.options.pretty_print and !serializer.options.compact_mode) {
        try writer.print(",\n");
        try helpers.writeIndent(serializer, writer, indent + 1);
        try writer.print("\"try_block\": ");
    } else {
        try writer.print(",\"try_block\":");
    }
    try serializeBlock(serializer, &try_block.try_block, writer, indent + 1, depth + 1);

    if (try_block.catch_block) |*catch_block| {
        if (serializer.options.pretty_print and !serializer.options.compact_mode) {
            try writer.print(",\n");
            try helpers.writeIndent(serializer, writer, indent + 1);
            try writer.print("\"catch_block\": {\n");
            if (catch_block.error_variable) |error_var| {
                try helpers.writeField(serializer, writer, "error_variable", error_var, indent + 2, true);
                try writer.print(",\n");
            }
            try helpers.writeIndent(serializer, writer, indent + 2);
            try writer.print("\"block\": ");
            try serializeBlock(serializer, &catch_block.block, writer, indent + 2, depth + 1);
            if (serializer.options.include_spans) {
                try writer.print(",\n");
                try helpers.writeSpanField(serializer, writer, &catch_block.span, indent + 2);
            }
            try writer.print("\n");
            try helpers.writeIndent(serializer, writer, indent + 1);
            try writer.print("}");
        } else {
            try writer.print(",\"catch_block\":{");
            if (catch_block.error_variable) |error_var| {
                try writer.print("\"error_variable\":\"{s}\",", .{error_var});
            }
            try writer.print("\"block\":");
            try serializeBlock(serializer, &catch_block.block, writer, 0, depth + 1);
            try writer.print("}");
        }
    }
}

/// Serialize statement
pub fn serializeStatement(serializer: *AstSerializer, stmt: *const StmtNode, writer: anytype, indent: u32, depth: u32) SerializationError!void {
    if (serializer.options.pretty_print and !serializer.options.compact_mode) {
        try helpers.writeIndent(serializer, writer, indent);
        try writer.print("{\n");
    } else {
        try writer.print("{");
    }

    switch (stmt.*) {
        .Expr => |*expr| {
            try helpers.writeField(serializer, writer, "type", "ExprStatement", indent + 1, true);
            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"expression\": ");
            } else {
                try writer.print(",\"expression\":");
            }
            try expressions.serializeExpression(serializer, expr, writer, indent + 1, depth + 1);
        },
        .VariableDecl => |*var_decl| {
            const declarations = @import("declarations.zig");
            try declarations.serializeVariableDecl(serializer, var_decl, writer, indent, depth);
        },
        .Return => |*ret| {
            try helpers.writeField(serializer, writer, "type", "Return", indent + 1, true);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &ret.span, indent + 1);
            }
            if (ret.value) |*value| {
                if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                    try writer.print(",\n");
                    try helpers.writeIndent(serializer, writer, indent + 1);
                    try writer.print("\"value\": ");
                } else {
                    try writer.print(",\"value\":");
                }
                try expressions.serializeExpression(serializer, value, writer, indent + 1, depth + 1);
            }
        },
        .If => |*if_stmt| {
            try helpers.writeField(serializer, writer, "type", "If", indent + 1, true);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &if_stmt.span, indent + 1);
            }
            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"condition\": ");
            } else {
                try writer.print(",\"condition\":");
            }
            try expressions.serializeExpression(serializer, &if_stmt.condition, writer, indent + 1, depth + 1);

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"then_branch\": ");
            } else {
                try writer.print(",\"then_branch\":");
            }
            try serializeBlock(serializer, &if_stmt.then_branch, writer, indent + 1, depth + 1);

            if (if_stmt.else_branch) |*else_branch| {
                if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                    try writer.print(",\n");
                    try helpers.writeIndent(serializer, writer, indent + 1);
                    try writer.print("\"else_branch\": ");
                } else {
                    try writer.print(",\"else_branch\":");
                }
                try serializeBlock(serializer, else_branch, writer, indent + 1, depth + 1);
            }
        },
        .While => |*while_stmt| {
            try helpers.writeField(serializer, writer, "type", "While", indent + 1, true);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &while_stmt.span, indent + 1);
            }
            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"condition\": ");
            } else {
                try writer.print(",\"condition\":");
            }
            try expressions.serializeExpression(serializer, &while_stmt.condition, writer, indent + 1, depth + 1);

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"body\": ");
            } else {
                try writer.print(",\"body\":");
            }
            try serializeBlock(serializer, &while_stmt.body, writer, indent + 1, depth + 1);

            if (while_stmt.invariants.len > 0) {
                if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                    try writer.print(",\n");
                    try helpers.writeIndent(serializer, writer, indent + 1);
                    try writer.print("\"invariants\": [\n");
                } else {
                    try writer.print(",\"invariants\":[");
                }

                for (while_stmt.invariants, 0..) |*inv, i| {
                    if (i > 0) try writer.print(",");
                    if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                        try writer.print("\n");
                    }
                    try expressions.serializeExpression(serializer, inv, writer, indent + 2, depth + 1);
                }

                if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                    try writer.print("\n");
                    try helpers.writeIndent(serializer, writer, indent + 1);
                    try writer.print("]");
                } else {
                    try writer.print("]");
                }
            }
        },
        .ForLoop => |*for_loop| {
            try helpers.writeField(serializer, writer, "type", "ForLoop", indent + 1, true);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &for_loop.span, indent + 1);
            }

            // serialize the iterable expression
            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"iterable\": ");
            } else {
                try writer.print(",\"iterable\":");
            }
            try expressions.serializeExpression(serializer, &for_loop.iterable, writer, indent + 1, depth + 1);

            // serialize the loop pattern
            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"pattern\": ");
            } else {
                try writer.print(",\"pattern\":");
            }
            try patterns.serializeLoopPattern(serializer, &for_loop.pattern, writer, indent + 1);

            // serialize the loop body
            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"body\": ");
            } else {
                try writer.print(",\"body\":");
            }
            try serializeBlock(serializer, &for_loop.body, writer, indent + 1, depth + 1);
        },
        .Switch => |*switch_stmt| {
            try helpers.writeField(serializer, writer, "type", "Switch", indent + 1, true);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &switch_stmt.span, indent + 1);
            }

            // serialize the switch condition
            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"condition\": ");
            } else {
                try writer.print(",\"condition\":");
            }
            try expressions.serializeExpression(serializer, &switch_stmt.condition, writer, indent + 1, depth + 1);

            // serialize switch cases
            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"cases\": [\n");
            } else {
                try writer.print(",\"cases\":[");
            }

            for (switch_stmt.cases, 0..) |*case, i| {
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

            // serialize default case if present
            if (switch_stmt.default_case) |*default_case| {
                if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                    try writer.print(",\n");
                    try helpers.writeIndent(serializer, writer, indent + 1);
                    try writer.print("\"default_case\": ");
                } else {
                    try writer.print(",\"default_case\":");
                }
                try serializeBlock(serializer, default_case, writer, indent + 1, depth + 1);
            }
        },
        .Break => |*break_node| {
            try helpers.writeField(serializer, writer, "type", "Break", indent + 1, true);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &break_node.span, indent + 1);
            }

            // serialize optional label
            if (break_node.label) |label| {
                try helpers.writeField(serializer, writer, "label", label, indent + 1, false);
            }

            // serialize optional value
            if (break_node.value) |value| {
                if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                    try writer.print(",\n");
                    try helpers.writeIndent(serializer, writer, indent + 1);
                    try writer.print("\"value\": ");
                } else {
                    try writer.print(",\"value\":");
                }
                try expressions.serializeExpression(serializer, value, writer, indent + 1, depth + 1);
            }
        },
        .Continue => |*continue_node| {
            try helpers.writeField(serializer, writer, "type", "Continue", indent + 1, true);
            if (continue_node.label) |label| {
                try helpers.writeField(serializer, writer, "label", label, indent + 1, false);
            }
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &continue_node.span, indent + 1);
            }
        },
        .Log => |*log| {
            try helpers.writeField(serializer, writer, "type", "Log", indent + 1, true);
            try helpers.writeField(serializer, writer, "event_name", log.event_name, indent + 1, false);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &log.span, indent + 1);
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"args\": [\n");
            } else {
                try writer.print(",\"args\":[");
            }

            for (log.args, 0..) |*arg, i| {
                if (i > 0) try writer.print(",");
                if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                    try writer.print("\n");
                }
                try expressions.serializeExpression(serializer, arg, writer, indent + 2, depth + 1);
            }

            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print("\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("]");
            } else {
                try writer.print("]");
            }
        },
        .Lock => |*lock| {
            try helpers.writeField(serializer, writer, "type", "Lock", indent + 1, true);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &lock.span, indent + 1);
            }
            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"path\": ");
            } else {
                try writer.print(",\"path\":");
            }
            try expressions.serializeExpression(serializer, &lock.path, writer, indent + 1, depth + 1);
        },
        .Invariant => |*inv| {
            try helpers.writeField(serializer, writer, "type", "Invariant", indent + 1, true);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &inv.span, indent + 1);
            }
            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"condition\": ");
            } else {
                try writer.print(",\"condition\":");
            }
            try expressions.serializeExpression(serializer, &inv.condition, writer, indent + 1, depth + 1);
        },
        .Requires => |*req| {
            try helpers.writeField(serializer, writer, "type", "Requires", indent + 1, true);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &req.span, indent + 1);
            }
            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"condition\": ");
            } else {
                try writer.print(",\"condition\":");
            }
            try expressions.serializeExpression(serializer, &req.condition, writer, indent + 1, depth + 1);
        },
        .Ensures => |*ens| {
            try helpers.writeField(serializer, writer, "type", "Ensures", indent + 1, true);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &ens.span, indent + 1);
            }
            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"condition\": ");
            } else {
                try writer.print(",\"condition\":");
            }
            try expressions.serializeExpression(serializer, &ens.condition, writer, indent + 1, depth + 1);
        },
        .ErrorDecl => |*error_decl| {
            const declarations = @import("declarations.zig");
            try declarations.serializeErrorDecl(serializer, error_decl, writer, indent, depth);
        },
        .TryBlock => |*try_block| {
            try serializeTryBlock(serializer, try_block, writer, indent, depth);
        },
        .CompoundAssignment => |*compound| {
            try helpers.writeField(serializer, writer, "type", "CompoundAssignment", indent + 1, true);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &compound.span, indent + 1);
            }

            // serialize the target expression
            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"target\": ");
            } else {
                try writer.print(",\"target\":");
            }
            try expressions.serializeExpression(serializer, compound.target, writer, indent + 1, depth + 1);

            // serialize the operator
            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"operator\": \"");
                try writer.print(@tagName(compound.operator));
                try writer.print("\"");
            } else {
                try writer.print(",\"operator\":\"");
                try writer.print(@tagName(compound.operator));
                try writer.print("\"");
            }

            // serialize the value expression
            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"value\": ");
            } else {
                try writer.print(",\"value\":");
            }
            try expressions.serializeExpression(serializer, compound.value, writer, indent + 1, depth + 1);
        },
        .DestructuringAssignment => |*dest_assign| {
            try helpers.writeField(serializer, writer, "type", "DestructuringAssignment", indent + 1, true);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &dest_assign.span, indent + 1);
            }
            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"pattern\": ");
            } else {
                try writer.print(",\"pattern\":");
            }
            try patterns.serializeDestructuringPattern(serializer, &dest_assign.pattern, writer, indent + 1);
            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"value\": ");
            } else {
                try writer.print(",\"value\":");
            }
            try expressions.serializeExpression(serializer, dest_assign.value, writer, indent + 1, depth + 1);
        },
        .Unlock => |*unlock| {
            try helpers.writeField(serializer, writer, "type", "Unlock", indent + 1, true);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &unlock.span, indent + 1);
            }
            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"path\": ");
            } else {
                try writer.print(",\"path\":");
            }
            try expressions.serializeExpression(serializer, &unlock.path, writer, indent + 1, depth + 1);
        },
        .LabeledBlock => |*labeled| {
            try helpers.writeField(serializer, writer, "type", "LabeledBlock", indent + 1, true);
            try helpers.writeField(serializer, writer, "label", labeled.label, indent + 1, false);
            if (serializer.options.include_spans) {
                try helpers.writeSpanField(serializer, writer, &labeled.span, indent + 1);
            }
            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 1);
                try writer.print("\"block\": ");
            } else {
                try writer.print(",\"block\":");
            }
            try serializeBlock(serializer, &labeled.block, writer, indent + 1, depth + 1);
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
