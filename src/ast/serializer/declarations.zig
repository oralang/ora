// ============================================================================
// Declaration Serializers
// ============================================================================
//
// Handles serialization of top-level and contract-level declarations:
//   • Contracts, functions, structs, enums, logs, errors
//   • Constants, variables, imports, modules
//
// ============================================================================

const std = @import("std");
const ast = @import("../../ast.zig");
const helpers = @import("helpers.zig");

// Forward declarations
const AstSerializer = @import("../ast_serializer.zig").AstSerializer;
const SerializationError = @import("../ast_serializer.zig").SerializationError;

/// Serialize contract declaration
pub fn serializeContract(serializer: *AstSerializer, contract: *const ast.ContractNode, writer: anytype, indent: u32, depth: u32) SerializationError!void {
    try helpers.writeField(serializer, writer, "type", "Contract", indent + 1, true);
    try helpers.writeField(serializer, writer, "name", contract.name, indent + 1, false);

    if (serializer.options.include_spans) {
        try helpers.writeSpanField(serializer, writer, &contract.span, indent + 1);
    }

    if (serializer.options.pretty_print and !serializer.options.compact_mode) {
        try writer.print(",\n");
        try helpers.writeIndent(serializer, writer, indent + 1);
        try writer.print("\"body\": [\n");
    } else {
        try writer.print(",\"body\":[");
    }

    for (contract.body, 0..) |*member, i| {
        if (i > 0) try writer.print(",");
        if (serializer.options.pretty_print and !serializer.options.compact_mode) {
            try writer.print("\n");
        }
        try serializer.serializeAstNode(member, writer, indent + 2, depth + 1);
    }

    if (serializer.options.pretty_print and !serializer.options.compact_mode) {
        try writer.print("\n");
        try helpers.writeIndent(serializer, writer, indent + 1);
        try writer.print("]");
    } else {
        try writer.print("]");
    }
}

/// Serialize function declaration
pub fn serializeFunction(serializer: *AstSerializer, function: *const ast.FunctionNode, writer: anytype, indent: u32, depth: u32) SerializationError!void {
    try helpers.writeField(serializer, writer, "type", "Function", indent + 1, true);
    try helpers.writeField(serializer, writer, "name", function.name, indent + 1, false);
    try helpers.writeBoolField(serializer, writer, "public", function.visibility == .Public, indent + 1);

    if (serializer.options.include_spans) {
        try helpers.writeSpanField(serializer, writer, &function.span, indent + 1);
    }

    // parameters
    if (serializer.options.pretty_print and !serializer.options.compact_mode) {
        try writer.print(",\n");
        try helpers.writeIndent(serializer, writer, indent + 1);
        try writer.print("\"parameters\": [\n");
    } else {
        try writer.print(",\"parameters\":[");
    }

    for (function.parameters, 0..) |*param, i| {
        if (i > 0) try writer.print(",");
        if (serializer.options.pretty_print and !serializer.options.compact_mode) {
            try writer.print("\n");
        }
        try serializer.serializeParameter(param, writer, indent + 2);
    }

    if (serializer.options.pretty_print and !serializer.options.compact_mode) {
        try writer.print("\n");
        try helpers.writeIndent(serializer, writer, indent + 1);
        try writer.print("],\n");
        try helpers.writeIndent(serializer, writer, indent + 1);
        try writer.print("\"return_type\": ");
    } else {
        try writer.print("],\"return_type\":");
    }

    if (function.return_type_info) |ret_type_info| {
        try serializer.serializeTypeInfo(ret_type_info, writer);
    } else {
        try writer.print("null");
    }

    // requires clauses
    if (function.requires_clauses.len > 0) {
        if (serializer.options.pretty_print and !serializer.options.compact_mode) {
            try writer.print(",\n");
            try helpers.writeIndent(serializer, writer, indent + 1);
            try writer.print("\"requires\": [\n");
        } else {
            try writer.print(",\"requires\":[");
        }

        for (function.requires_clauses, 0..) |clause, i| {
            if (i > 0) try writer.print(",");
            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print("\n");
            }
            try serializer.serializeExpression(clause, writer, indent + 2, depth + 1);
        }

        if (serializer.options.pretty_print and !serializer.options.compact_mode) {
            try writer.print("\n");
            try helpers.writeIndent(serializer, writer, indent + 1);
            try writer.print("]");
        } else {
            try writer.print("]");
        }
    }

    // ensures clauses
    if (function.ensures_clauses.len > 0) {
        if (serializer.options.pretty_print and !serializer.options.compact_mode) {
            try writer.print(",\n");
            try helpers.writeIndent(serializer, writer, indent + 1);
            try writer.print("\"ensures\": [\n");
        } else {
            try writer.print(",\"ensures\":[");
        }

        for (function.ensures_clauses, 0..) |clause, i| {
            if (i > 0) try writer.print(",");
            if (serializer.options.pretty_print and !serializer.options.compact_mode) {
                try writer.print("\n");
            }
            try serializer.serializeExpression(clause, writer, indent + 2, depth + 1);
        }

        if (serializer.options.pretty_print and !serializer.options.compact_mode) {
            try writer.print("\n");
            try helpers.writeIndent(serializer, writer, indent + 1);
            try writer.print("]");
        } else {
            try writer.print("]");
        }
    }

    // function body
    if (serializer.options.pretty_print and !serializer.options.compact_mode) {
        try writer.print(",\n");
        try helpers.writeIndent(serializer, writer, indent + 1);
        try writer.print("\"body\": ");
    } else {
        try writer.print(",\"body\":");
    }
    try serializer.serializeBlock(&function.body, writer, indent + 1, depth + 1);
}

/// Serialize variable declaration
pub fn serializeVariableDecl(serializer: *AstSerializer, var_decl: *const ast.Statements.VariableDeclNode, writer: anytype, indent: u32, depth: u32) SerializationError!void {
    try helpers.writeField(serializer, writer, "type", "VariableDecl", indent + 1, true);
    try helpers.writeField(serializer, writer, "name", var_decl.name, indent + 1, false);
    try helpers.writeField(serializer, writer, "region", @tagName(var_decl.region), indent + 1, false);
    try helpers.writeField(serializer, writer, "kind", @tagName(var_decl.kind), indent + 1, false);
    try helpers.writeBoolField(serializer, writer, "locked", var_decl.locked, indent + 1);

    if (serializer.options.include_spans) {
        try helpers.writeSpanField(serializer, writer, &var_decl.span, indent + 1);
    }

    if (serializer.options.pretty_print and !serializer.options.compact_mode) {
        try writer.print(",\n");
        try helpers.writeIndent(serializer, writer, indent + 1);
        try writer.print("\"var_type\": ");
    } else {
        try writer.print(",\"var_type\":");
    }
    try serializer.serializeTypeInfo(var_decl.type_info, writer);

    if (var_decl.value) |value| {
        if (serializer.options.pretty_print and !serializer.options.compact_mode) {
            try writer.print(",\n");
            try helpers.writeIndent(serializer, writer, indent + 1);
            try writer.print("\"value\": ");
        } else {
            try writer.print(",\"value\":");
        }
        try serializer.serializeExpression(value, writer, indent + 1, depth + 1);
    }

    if (var_decl.tuple_names) |tuple_names| {
        if (serializer.options.pretty_print and !serializer.options.compact_mode) {
            try writer.print(",\n");
            try helpers.writeIndent(serializer, writer, indent + 1);
            try writer.print("\"tuple_names\": [");
        } else {
            try writer.print(",\"tuple_names\":[");
        }

        for (tuple_names, 0..) |name, i| {
            if (i > 0) try writer.print(",");
            try writer.print("\"{s}\"", .{name});
        }
        try writer.print("]");
    }
}

/// Serialize struct declaration
pub fn serializeStructDecl(serializer: *AstSerializer, struct_decl: *const ast.StructDeclNode, writer: anytype, indent: u32, depth: u32) SerializationError!void {
    _ = depth; // unused
    try helpers.writeField(serializer, writer, "type", "StructDecl", indent + 1, true);
    try helpers.writeField(serializer, writer, "name", struct_decl.name, indent + 1, false);

    if (serializer.options.include_spans) {
        try helpers.writeSpanField(serializer, writer, &struct_decl.span, indent + 1);
    }

    if (serializer.options.pretty_print and !serializer.options.compact_mode) {
        try writer.print(",\n");
        try helpers.writeIndent(serializer, writer, indent + 1);
        try writer.print("\"fields\": [\n");
    } else {
        try writer.print(",\"fields\":[");
    }

    for (struct_decl.fields, 0..) |*field, i| {
        if (i > 0) try writer.print(",");
        if (serializer.options.pretty_print and !serializer.options.compact_mode) {
            try writer.print("\n");
            try helpers.writeIndent(serializer, writer, indent + 2);
            try writer.print("{\n");
            try helpers.writeField(serializer, writer, "name", field.name, indent + 3, true);
            try writer.print(",\n");
            try helpers.writeIndent(serializer, writer, indent + 3);
            try writer.print("\"field_type\": ");
            try serializer.serializeTypeInfo(field.type_info, writer);
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
            try writer.print("\",\"field_type\":");
            try serializer.serializeTypeInfo(field.type_info, writer);
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
}

/// Serialize enum declaration
pub fn serializeEnumDecl(serializer: *AstSerializer, enum_decl: *const ast.EnumDeclNode, writer: anytype, indent: u32, depth: u32) SerializationError!void {
    try helpers.writeField(serializer, writer, "type", "EnumDecl", indent + 1, true);
    try helpers.writeField(serializer, writer, "name", enum_decl.name, indent + 1, false);
    // check if any variants have explicit values
    var has_explicit_values = false;
    for (enum_decl.variants) |variant| {
        if (variant.value != null) {
            has_explicit_values = true;
            break;
        }
    }
    try helpers.writeBoolField(serializer, writer, "has_explicit_values", has_explicit_values, indent + 1);

    // serialize the underlying type if present
    if (enum_decl.underlying_type_info) |underlying_type_info| {
        try writer.print(",\n");
        try helpers.writeIndent(serializer, writer, indent + 1);
        try writer.print("\"underlying_type\": ");
        try serializer.serializeTypeInfo(underlying_type_info, writer);
    }

    if (serializer.options.include_spans) {
        try helpers.writeSpanField(serializer, writer, &enum_decl.span, indent + 1);
    }

    if (serializer.options.pretty_print and !serializer.options.compact_mode) {
        try writer.print(",\n");
        try helpers.writeIndent(serializer, writer, indent + 1);
        try writer.print("\"variants\": [\n");
    } else {
        try writer.print(",\"variants\":[");
    }

    for (enum_decl.variants, 0..) |*variant, i| {
        if (i > 0) try writer.print(",");
        if (serializer.options.pretty_print and !serializer.options.compact_mode) {
            try writer.print("\n");
            try helpers.writeIndent(serializer, writer, indent + 2);
            try writer.print("{\n");
            try helpers.writeField(serializer, writer, "name", variant.name, indent + 3, true);
            if (variant.value) |*value| {
                try writer.print(",\n");
                try helpers.writeIndent(serializer, writer, indent + 3);
                try writer.print("\"value\": ");

                // special handling for integer literals in enum variants to include enum's underlying type
                if (value.* == .Literal and value.Literal == .Integer and enum_decl.underlying_type_info != null) {
                    // start object
                    try writer.print("{\n");
                    try helpers.writeIndent(serializer, writer, indent + 4);
                    try writer.print("\"type\": \"Literal\",\n");
                    try helpers.writeIndent(serializer, writer, indent + 4);

                    // always use "Integer" for literal_type consistency
                    try writer.print("\"literal_type\": \"Integer\",\n");

                    // include type information
                    try helpers.writeIndent(serializer, writer, indent + 4);
                    try writer.print("\"type_info\": ");
                    try serializer.serializeTypeInfo(value.Literal.Integer.type_info, writer);
                    try writer.print(",\n");

                    // write the value
                    try helpers.writeIndent(serializer, writer, indent + 4);
                    try writer.print("\"value\": \"");
                    try writer.print(value.Literal.Integer.value);
                    try writer.print("\"");

                    // include span if needed
                    if (serializer.options.include_spans) {
                        try writer.print(",\n");
                        // custom span field handling for enum variant integer literals
                        try helpers.writeIndent(serializer, writer, indent + 4);
                        try writer.print("\"span\": {\n");
                        try helpers.writeIndent(serializer, writer, indent + 5);
                        try writer.print("\"line\": {d},\n", .{value.Literal.Integer.span.line});
                        try helpers.writeIndent(serializer, writer, indent + 5);
                        try writer.print("\"column\": {d},\n", .{value.Literal.Integer.span.column});
                        try helpers.writeIndent(serializer, writer, indent + 5);
                        try writer.print("\"length\": {d},\n", .{value.Literal.Integer.span.length});
                        try helpers.writeIndent(serializer, writer, indent + 5);
                        try writer.print("\"lexeme\": \"");
                        if (value.Literal.Integer.span.lexeme) |lexeme| {
                            try writer.print(lexeme);
                        }
                        try writer.print("\"\n");
                        try helpers.writeIndent(serializer, writer, indent + 4);
                        try writer.print("}");
                    }

                    // end object
                    try writer.print("\n");
                    try helpers.writeIndent(serializer, writer, indent + 3);
                    try writer.print("}");
                } else {
                    // regular expression serialization for non-integer or complex expressions
                    try serializer.serializeExpression(value, writer, indent + 3, depth + 1);
                }
            }

            if (serializer.options.include_spans) {
                try writer.print(",\n");
                try helpers.writeSpanField(serializer, writer, &variant.span, indent + 3);
            }
            try writer.print("\n");
            try helpers.writeIndent(serializer, writer, indent + 2);
            try writer.print("}");
        } else {
            try writer.print("{\"name\":\"");
            try writer.print(variant.name);
            try writer.print("\"");
            if (variant.value) |*value| {
                try writer.print(",\"value\":");

                // special handling for integer literals in enum variants to include enum's underlying type
                if (value.* == .Literal and value.Literal == .Integer and enum_decl.underlying_type_info != null) {
                    try writer.print("{\"type\":\"Literal\",");

                    // always use "Integer" for literal_type consistency
                    try writer.print("\"literal_type\":\"Integer\",");

                    // include type information
                    try writer.print("\"type_info\":");
                    try serializer.serializeTypeInfo(value.Literal.Integer.type_info, writer);
                    try writer.print(",");

                    // write the value
                    try writer.print("\"value\":\"");
                    try writer.print(value.Literal.Integer.value);
                    try writer.print("\"");

                    // include span if needed
                    if (serializer.options.include_spans) {
                        const span = &value.Literal.Integer.span;
                        try writer.print(",\"span\":{\"line\":");
                        try writer.print("{d}", .{span.line});
                        try writer.print(",\"column\":");
                        try writer.print("{d}", .{span.column});
                        try writer.print(",\"length\":");
                        try writer.print("{d}", .{span.length});
                        try writer.print(",\"lexeme\":\"");
                        if (span.lexeme) |lexeme| {
                            try writer.print(lexeme);
                        } else {
                            try writer.print("");
                        }
                        try writer.print("\"}");
                    }

                    try writer.print("}");
                } else {
                    // regular expression serialization for non-integer or complex expressions
                    try serializer.serializeExpression(value, writer, 0, depth + 1);
                }
            }

            // include span for the variant itself
            if (serializer.options.include_spans) {
                try writer.print(",\"span\":{\"line\":");
                try writer.print("{d}", .{variant.span.line});
                try writer.print(",\"column\":");
                try writer.print("{d}", .{variant.span.column});
                try writer.print(",\"length\":");
                try writer.print("{d}", .{variant.span.length});
                try writer.print(",\"lexeme\":\"");
                if (variant.span.lexeme) |lexeme| {
                    try writer.print(lexeme);
                }
                try writer.print("\"}");
            }

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
}

/// Serialize log declaration
pub fn serializeLogDecl(serializer: *AstSerializer, log_decl: *const ast.LogDeclNode, writer: anytype, indent: u32, depth: u32) SerializationError!void {
    _ = depth; // unused
    try helpers.writeField(serializer, writer, "type", "LogDecl", indent + 1, true);
    try helpers.writeField(serializer, writer, "name", log_decl.name, indent + 1, false);

    if (serializer.options.include_spans) {
        try helpers.writeSpanField(serializer, writer, &log_decl.span, indent + 1);
    }

    if (serializer.options.pretty_print and !serializer.options.compact_mode) {
        try writer.print(",\n");
        try helpers.writeIndent(serializer, writer, indent + 1);
        try writer.print("\"fields\": [\n");
    } else {
        try writer.print(",\"fields\":[");
    }

    for (log_decl.fields, 0..) |*field, i| {
        if (i > 0) try writer.print(",");
        if (serializer.options.pretty_print and !serializer.options.compact_mode) {
            try writer.print("\n");
            try helpers.writeIndent(serializer, writer, indent + 2);
            try writer.print("{\n");
            try helpers.writeField(serializer, writer, "name", field.name, indent + 3, true);
            try writer.print(",\n");
            try helpers.writeIndent(serializer, writer, indent + 3);
            try writer.print("\"field_type\": ");
            try serializer.serializeTypeInfo(field.type_info, writer);
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
            try writer.print("\",\"field_type\":");
            try serializer.serializeTypeInfo(field.type_info, writer);
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
}

/// Serialize import declaration
pub fn serializeImport(serializer: *AstSerializer, import: *const ast.ImportNode, writer: anytype, indent: u32, depth: u32) SerializationError!void {
    _ = depth; // unused
    try helpers.writeField(serializer, writer, "type", "Import", indent + 1, true);
    try helpers.writeField(serializer, writer, "name", import.alias orelse import.path, indent + 1, false);
    try helpers.writeField(serializer, writer, "path", import.path, indent + 1, false);

    if (serializer.options.include_spans) {
        try helpers.writeSpanField(serializer, writer, &import.span, indent + 1);
    }
}

/// Serialize error declaration
pub fn serializeErrorDecl(serializer: *AstSerializer, error_decl: *const ast.Statements.ErrorDeclNode, writer: anytype, indent: u32, depth: u32) SerializationError!void {
    _ = depth; // unused
    try helpers.writeField(serializer, writer, "type", "ErrorDecl", indent + 1, true);
    try helpers.writeField(serializer, writer, "name", error_decl.name, indent + 1, false);

    if (serializer.options.include_spans) {
        try helpers.writeSpanField(serializer, writer, &error_decl.span, indent + 1);
    }
}

/// Serialize module declaration
pub fn serializeModule(serializer: *AstSerializer, module: *const ast.ModuleNode, writer: anytype, indent: u32, depth: u32) SerializationError!void {
    try helpers.writeField(serializer, writer, "type", "Module", indent + 1, true);

    if (module.name) |name| {
        try helpers.writeField(serializer, writer, "name", name, indent + 1, false);
    } else {
        try helpers.writeField(serializer, writer, "name", "", indent + 1, false);
    }

    try writer.print(",\n");
    try helpers.writeIndent(serializer, writer, indent + 1);
    try writer.print("\"imports\": [");
    if (module.imports.len > 0) {
        try writer.print("\n");
        for (module.imports, 0..) |import, i| {
            try helpers.writeIndent(serializer, writer, indent + 2);
            try writer.print("{\n");
            try serializeImport(serializer, &import, writer, indent + 2, depth + 1);
            try writer.print("\n");
            try helpers.writeIndent(serializer, writer, indent + 2);
            try writer.print("}");
            if (i < module.imports.len - 1) {
                try writer.print(",");
            }
            try writer.print("\n");
        }
        try helpers.writeIndent(serializer, writer, indent + 1);
    }
    try writer.print("],\n");

    try helpers.writeIndent(serializer, writer, indent + 1);
    try writer.print("\"declarations\": [");
    if (module.declarations.len > 0) {
        try writer.print("\n");
        for (module.declarations, 0..) |decl, i| {
            try helpers.writeIndent(serializer, writer, indent + 2);
            try writer.print("{\n");
            try serializer.serializeAstNode(&decl, writer, indent + 2, depth + 1);
            try writer.print("\n");
            try helpers.writeIndent(serializer, writer, indent + 2);
            try writer.print("}");
            if (i < module.declarations.len - 1) {
                try writer.print(",");
            }
            try writer.print("\n");
        }
        try helpers.writeIndent(serializer, writer, indent + 1);
    }
    try writer.print("]");

    if (serializer.options.include_spans) {
        try writer.print(",\n");
        try helpers.writeSpanField(serializer, writer, &module.span, indent + 1);
    }
}

/// Serialize constant declaration
pub fn serializeConstant(serializer: *AstSerializer, constant: *const ast.ConstantNode, writer: anytype, indent: u32, depth: u32) SerializationError!void {
    try helpers.writeField(serializer, writer, "type", "Constant", indent + 1, true);
    try helpers.writeField(serializer, writer, "name", constant.name, indent + 1, false);

    try writer.print(",\n");
    try helpers.writeIndent(serializer, writer, indent + 1);
    try writer.print("\"typ\": ");
    try serializer.serializeTypeInfo(constant.typ, writer);

    try writer.print(",\n");
    try helpers.writeIndent(serializer, writer, indent + 1);
    try writer.print("\"value\": {\n");
    try serializer.serializeExpression(constant.value, writer, indent + 1, depth + 1);
    try writer.print("\n");
    try helpers.writeIndent(serializer, writer, indent + 1);
    try writer.print("}");

    try writer.print(",\n");
    try helpers.writeIndent(serializer, writer, indent + 1);
    try writer.print("\"visibility\": \"");
    switch (constant.visibility) {
        .Public => try writer.print("Public"),
        .Private => try writer.print("Private"),
    }
    try writer.print("\"");

    if (serializer.options.include_spans) {
        try writer.print(",\n");
        try helpers.writeSpanField(serializer, writer, &constant.span, indent + 1);
    }
}
