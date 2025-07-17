const std = @import("std");
const ora = @import("ora_lib");
const lexer = ora.lexer;
const parser = ora.parser;
const ast = ora.ast;

const AstNode = ast.AstNode;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Read the test contract
    const file_path = "examples/simple_test.ora";
    const source = std.fs.cwd().readFileAlloc(allocator, file_path, 1024 * 1024) catch |err| {
        std.debug.print("Error reading file {s}: {}\n", .{ file_path, err });
        return;
    };
    defer allocator.free(source);

    std.debug.print("Parsing Ora Contract\n", .{});
    std.debug.print("=" ** 50 ++ "\n\n", .{});

    std.debug.print("Source Code:\n", .{});
    std.debug.print("{s}\n", .{source});
    std.debug.print("\n" ++ "=" ** 50 ++ "\n\n", .{});

    // Tokenize
    std.debug.print("Tokenizing...\n", .{});
    const tokens = lexer.scan(source, allocator) catch |err| {
        std.debug.print("Lexer error: {}\n", .{err});
        return;
    };
    defer allocator.free(tokens);

    std.debug.print("Generated {} tokens\n\n", .{tokens.len});

    // Show some tokens
    std.debug.print("Sample Tokens:\n", .{});
    for (tokens[0..@min(15, tokens.len)]) |token| {
        std.debug.print("  {}\n", .{token});
    }
    if (tokens.len > 15) {
        std.debug.print("  ... and {} more tokens\n", .{tokens.len - 15});
    }
    std.debug.print("\n" ++ "=" ** 50 ++ "\n\n", .{});

    // Parse
    std.debug.print("Parsing into AST...\n", .{});
    const nodes = parser.parse(allocator, tokens) catch |err| {
        std.debug.print("Parser error: {}\n", .{err});
        return;
    };
    defer {
        // Clean up AST nodes
        for (nodes) |node| {
            cleanupAstNode(allocator, node);
        }
        allocator.free(nodes);
    }

    std.debug.print("Generated {} top-level AST nodes\n\n", .{nodes.len});

    // Print AST structure
    std.debug.print("AST Structure:\n", .{});
    for (nodes, 0..) |node, i| {
        std.debug.print("\n[{}] ", .{i});
        try printAstNode(node, 0);
    }

    std.debug.print("\n" ++ "=" ** 50 ++ "\n", .{});
    std.debug.print("Parser demo completed successfully\n", .{});
}

/// Clean up dynamically allocated AST nodes
fn cleanupAstNode(allocator: std.mem.Allocator, node: AstNode) void {
    switch (node) {
        .Contract => |contract| {
            for (contract.body) |body_node| {
                cleanupAstNode(allocator, body_node);
            }
            allocator.free(contract.body);
        },
        .Function => |function| {
            allocator.free(function.parameters);
            allocator.free(function.requires_clauses);
            allocator.free(function.ensures_clauses);
            cleanupBlock(allocator, function.body);
        },
        .VariableDecl => |var_decl| {
            if (var_decl.value) |value| {
                cleanupExprNode(allocator, value);
            }
        },
        .StructDecl => |struct_decl| {
            allocator.free(struct_decl.fields);
        },
        .EnumDecl => |enum_decl| {
            allocator.free(enum_decl.variants);
        },
        .LogDecl => |log_decl| {
            allocator.free(log_decl.fields);
        },
        else => {
            // Most other nodes don't need special cleanup
        },
    }
}

fn cleanupBlock(allocator: std.mem.Allocator, block: ast.BlockNode) void {
    for (block.statements) |stmt| {
        cleanupStmtNode(allocator, stmt);
    }
    allocator.free(block.statements);
}

fn cleanupStmtNode(allocator: std.mem.Allocator, stmt: ast.StmtNode) void {
    switch (stmt) {
        .Expr => |expr| cleanupExprNode(allocator, expr),
        .VariableDecl => |var_decl| {
            if (var_decl.value) |value| {
                cleanupExprNode(allocator, value);
            }
        },
        .Return => |ret| {
            if (ret.value) |value| {
                cleanupExprNode(allocator, value);
            }
        },
        .Log => |log_stmt| {
            for (log_stmt.args) |arg| {
                cleanupExprNode(allocator, arg);
            }
            allocator.free(log_stmt.args);
        },
        else => {},
    }
}

fn cleanupExprNode(allocator: std.mem.Allocator, expr: ast.ExprNode) void {
    switch (expr) {
        .Binary => |binary| {
            cleanupExprNode(allocator, binary.lhs.*);
            cleanupExprNode(allocator, binary.rhs.*);
            allocator.destroy(binary.lhs);
            allocator.destroy(binary.rhs);
        },
        .Unary => |unary| {
            cleanupExprNode(allocator, unary.operand.*);
            allocator.destroy(unary.operand);
        },
        .Assignment => |assignment| {
            cleanupExprNode(allocator, assignment.target.*);
            cleanupExprNode(allocator, assignment.value.*);
            allocator.destroy(assignment.target);
            allocator.destroy(assignment.value);
        },
        .CompoundAssignment => |compound| {
            cleanupExprNode(allocator, compound.target.*);
            cleanupExprNode(allocator, compound.value.*);
            allocator.destroy(compound.target);
            allocator.destroy(compound.value);
        },
        .Call => |call| {
            cleanupExprNode(allocator, call.callee.*);
            for (call.arguments) |arg| {
                cleanupExprNode(allocator, arg);
            }
            allocator.destroy(call.callee);
            allocator.free(call.arguments);
        },
        .Index => |index| {
            cleanupExprNode(allocator, index.target.*);
            cleanupExprNode(allocator, index.index.*);
            allocator.destroy(index.target);
            allocator.destroy(index.index);
        },
        .FieldAccess => |field_access| {
            cleanupExprNode(allocator, field_access.target.*);
            allocator.destroy(field_access.target);
        },
        else => {
            // Literals and identifiers don't need cleanup
        },
    }
}

/// Print AST node structure for debugging
fn printAstNode(node: AstNode, indent: u32) !void {
    // Print indentation
    var idx: u32 = 0;
    while (idx < indent) : (idx += 1) {
        std.debug.print("  ", .{});
    }

    switch (node) {
        .Contract => |contract| {
            std.debug.print("Contract '{s}' with {} members:\n", .{ contract.name, contract.body.len });
            for (contract.body, 0..) |member, i| {
                std.debug.print("[{}] ", .{i});
                try printAstNode(member, indent + 1);
            }
        },

        .Function => |function| {
            const visibility = if (function.pub_) "pub " else "";
            std.debug.print("{s}Function '{s}' ({} params", .{ visibility, function.name, function.parameters.len });
            if (function.return_type) |ret_type| {
                std.debug.print(", returns ", .{});
                printTypeRef(ret_type);
            }
            std.debug.print("):\n", .{});

            if (function.requires_clauses.len > 0) {
                std.debug.print("  Requires: {} clause(s)\n", .{function.requires_clauses.len});
            }
            if (function.ensures_clauses.len > 0) {
                std.debug.print("  Ensures: {} clause(s)\n", .{function.ensures_clauses.len});
            }

            std.debug.print("  Body: {} statements\n", .{function.body.statements.len});
        },

        .VariableDecl => |var_decl| {
            const mutability = if (var_decl.mutable) "mut " else "";
            std.debug.print("Variable {s}{s} '{s}' : ", .{ @tagName(var_decl.region), mutability, var_decl.name });
            printTypeRef(var_decl.typ);
            if (var_decl.value != null) {
                std.debug.print(" = <expr>", .{});
            }
            std.debug.print("\n", .{});
        },

        .LogDecl => |log_decl| {
            std.debug.print("Log '{s}' ({} fields)\n", .{ log_decl.name, log_decl.fields.len });
            for (log_decl.fields) |field| {
                std.debug.print("  {s}: ", .{field.name});
                printTypeRef(field.typ);
                std.debug.print("\n", .{});
            }
        },

        .StructDecl => |struct_decl| {
            std.debug.print("Struct '{s}' ({} fields)\n", .{ struct_decl.name, struct_decl.fields.len });
        },

        .EnumDecl => |enum_decl| {
            std.debug.print("Enum '{s}' ({} variants)\n", .{ enum_decl.name, enum_decl.variants.len });
        },

        .Import => |import| {
            std.debug.print("Import '{s}'\n", .{import.path});
        },

        else => {
            std.debug.print("{s}\n", .{@tagName(node)});
        },
    }
}

/// Print type reference
fn printTypeRef(type_ref: ast.TypeRef) void {
    switch (type_ref) {
        .Bool => std.debug.print("bool", .{}),
        .Address => std.debug.print("address", .{}),
        .U8 => std.debug.print("u8", .{}),
        .U16 => std.debug.print("u16", .{}),
        .U32 => std.debug.print("u32", .{}),
        .U64 => std.debug.print("u64", .{}),
        .U128 => std.debug.print("u128", .{}),
        .U256 => std.debug.print("u256", .{}),
        .String => std.debug.print("string", .{}),
        .Slice => |slice| {
            std.debug.print("slice[", .{});
            printTypeRef(slice.*);
            std.debug.print("]", .{});
        },
        .Mapping => |mapping| {
            std.debug.print("map[", .{});
            printTypeRef(mapping.key.*);
            std.debug.print(", ", .{});
            printTypeRef(mapping.value.*);
            std.debug.print("]", .{});
        },
        .DoubleMap => |doublemap| {
            std.debug.print("doublemap[", .{});
            printTypeRef(doublemap.key1.*);
            std.debug.print(", ", .{});
            printTypeRef(doublemap.key2.*);
            std.debug.print(", ", .{});
            printTypeRef(doublemap.value.*);
            std.debug.print("]", .{});
        },
        .Identifier => |name| std.debug.print("{s}", .{name}),
    }
}
