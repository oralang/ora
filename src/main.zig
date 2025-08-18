//! Ora Language Compiler CLI
//!
//! Command-line interface for the Ora domain-specific language compiler frontend.
//! Provides commands for lexing, parsing, and AST generation.
//! Backend compilation to Yul/EVM bytecode happens internally.
//!
//! Available commands:
//! - lex: Tokenize source files
//! - parse: Generate Abstract Syntax Tree
//! - ast: Generate AST JSON
//! - compile: Full frontend pipeline (Ora -> AST)

const std = @import("std");
const lib = @import("ora_lib");

/// Ora CLI application
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        try printUsage();
        return;
    }

    // Parse arguments to find output directory option
    var output_dir: ?[]const u8 = null;
    var no_cst: bool = false;
    var command: ?[]const u8 = null;
    var input_file: ?[]const u8 = null;
    var i: usize = 1;

    while (i < args.len) {
        if (std.mem.eql(u8, args[i], "-o") or std.mem.eql(u8, args[i], "--output-dir")) {
            if (i + 1 >= args.len) {
                try printUsage();
                return;
            }
            output_dir = args[i + 1];
            i += 2;
        } else if (std.mem.eql(u8, args[i], "--no-cst")) {
            no_cst = true;
            i += 1;
        } else if (command == null) {
            command = args[i];
            i += 1;
        } else if (input_file == null) {
            input_file = args[i];
            i += 1;
        } else {
            try printUsage();
            return;
        }
    }

    if (command == null or input_file == null) {
        try printUsage();
        return;
    }

    const cmd = command.?;
    const file_path = input_file.?;

    if (std.mem.eql(u8, cmd, "lex")) {
        try runLexer(allocator, file_path);
    } else if (std.mem.eql(u8, cmd, "parse")) {
        try runParser(allocator, file_path, !no_cst);
    } else if (std.mem.eql(u8, cmd, "ast")) {
        try runASTGeneration(allocator, file_path, output_dir, !no_cst);
    } else if (std.mem.eql(u8, cmd, "compile")) {
        try runFullCompilation(allocator, file_path, !no_cst);
    } else {
        try printUsage();
    }
}

fn printUsage() !void {
    const stdout = std.io.getStdOut().writer();
    try stdout.print("Ora Compiler v0.1\n", .{});
    try stdout.print("Usage: ora [options] <command> <file>\n", .{});
    try stdout.print("\nOptions:\n", .{});
    try stdout.print("  -o, --output-dir <dir>  - Specify output directory for generated files\n", .{});
    try stdout.print("      --no-cst            - Disable CST building (enabled by default)\n", .{});
    try stdout.print("\nCommands:\n", .{});
    try stdout.print("  lex <file>     - Tokenize a .ora file\n", .{});
    try stdout.print("  parse <file>   - Parse a .ora file to AST\n", .{});
    try stdout.print("  ast <file>     - Generate AST and save to JSON file\n", .{});
    try stdout.print("  compile <file> - Full frontend pipeline (lex -> parse)\n", .{});
    try stdout.print("\nExample:\n", .{});
    try stdout.print("  ora -o build ast example.ora\n", .{});
}

/// Run lexer on file and display tokens
fn runLexer(allocator: std.mem.Allocator, file_path: []const u8) !void {
    const stdout = std.io.getStdOut().writer();

    // Read source file
    const source = std.fs.cwd().readFileAlloc(allocator, file_path, 1024 * 1024) catch |err| {
        try stdout.print("Error reading file {s}: {}\n", .{ file_path, err });
        return;
    };
    defer allocator.free(source);

    try stdout.print("Lexing {s}\n", .{file_path});
    try stdout.print("==================================================\n", .{});

    // Run lexer
    var lexer = lib.Lexer.init(allocator, source);
    defer lexer.deinit();

    const tokens = lexer.scanTokens() catch |err| {
        try stdout.print("Lexer error: {}\n", .{err});
        if (err == lib.lexer.LexerError.UnexpectedCharacter) {
            const error_details = try lexer.getErrorDetails(allocator);
            defer allocator.free(error_details);
            try stdout.print("   {s}\n", .{error_details});
        }
        return;
    };
    defer allocator.free(tokens);

    try stdout.print("Generated {} tokens\n\n", .{tokens.len});

    // Display all tokens without truncation
    for (tokens, 0..) |token, i| {
        try stdout.print("[{:3}] {}\n", .{ i, token });
    }
}

/// Run parser on file and display AST
fn runParser(allocator: std.mem.Allocator, file_path: []const u8, enable_cst: bool) !void {
    const stdout = std.io.getStdOut().writer();

    // Read source file
    const source = std.fs.cwd().readFileAlloc(allocator, file_path, 1024 * 1024) catch |err| {
        try stdout.print("Error reading file {s}: {}\n", .{ file_path, err });
        return;
    };
    defer allocator.free(source);

    try stdout.print("Parsing {s}\n", .{file_path});
    try stdout.print("==================================================\n", .{});

    // Run lexer
    var lexer = lib.Lexer.init(allocator, source);
    defer lexer.deinit();

    const tokens = lexer.scanTokens() catch |err| {
        try stdout.print("Lexer error: {}\n", .{err});
        return;
    };
    defer allocator.free(tokens);

    try stdout.print("Lexed {} tokens\n", .{tokens.len});

    // Run parser
    var arena = lib.ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var parser = lib.Parser.init(tokens, &arena);
    parser.setFileId(1);
    var cst_builder_storage: lib.cst.CstBuilder = undefined;
    var cst_builder_ptr: ?*lib.cst.CstBuilder = null;
    if (enable_cst) {
        cst_builder_storage = lib.cst.CstBuilder.init(allocator);
        cst_builder_ptr = &cst_builder_storage;
        parser.withCst(cst_builder_ptr.?);
    }
    const ast_nodes = parser.parse() catch |err| {
        try stdout.print("Parser error: {}\n", .{err});
        return;
    };
    // Note: AST nodes are allocated in arena, so they're automatically freed when arena is deinited

    try stdout.print("Generated {} AST nodes\n\n", .{ast_nodes.len});

    // Display AST summary
    for (ast_nodes, 0..) |*node, i| {
        try stdout.print("[{}] ", .{i});
        try printAstSummary(stdout, node, 0);
    }

    if (enable_cst) {
        if (cst_builder_ptr) |builder| {
            const cst_root = try builder.buildRoot(tokens);
            _ = cst_root; // TODO: optional dump in future flag
            builder.deinit();
        }
    }
}

/// Run full compilation pipeline
fn runFullCompilation(allocator: std.mem.Allocator, file_path: []const u8, enable_cst: bool) !void {
    const stdout = std.io.getStdOut().writer();

    try stdout.print("Compiling {s}\n", .{file_path});
    try stdout.print("============================================================\n", .{});

    // Read source file
    const source = std.fs.cwd().readFileAlloc(allocator, file_path, 1024 * 1024) catch |err| {
        try stdout.print("Error reading file {s}: {}\n", .{ file_path, err });
        return;
    };
    defer allocator.free(source);

    try stdout.print("Source ({} bytes):\n", .{source.len});
    try stdout.print("{s}\n\n", .{source});

    // Phase 1: Lexical Analysis
    try stdout.print("Phase 1: Lexical Analysis\n", .{});
    var lexer = lib.Lexer.init(allocator, source);
    defer lexer.deinit();

    const tokens = lexer.scanTokens() catch |err| {
        try stdout.print("Lexer failed: {}\n", .{err});
        return;
    };
    defer allocator.free(tokens);

    try stdout.print("Generated {} tokens\n\n", .{tokens.len});

    // Phase 2: Parsing
    try stdout.print("Phase 2: Syntax Analysis\n", .{});
    var arena = lib.ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var parser = lib.Parser.init(tokens, &arena);
    parser.setFileId(1);
    var cst_builder_storage: lib.cst.CstBuilder = undefined;
    var cst_builder_ptr: ?*lib.cst.CstBuilder = null;
    if (enable_cst) {
        cst_builder_storage = lib.cst.CstBuilder.init(allocator);
        cst_builder_ptr = &cst_builder_storage;
        parser.withCst(cst_builder_ptr.?);
    }
    const ast_nodes = parser.parse() catch |err| {
        try stdout.print("Parser failed: {}\n", .{err});
        return;
    };
    // Note: AST nodes are allocated in arena, so they're automatically freed when arena is deinited

    try stdout.print("Generated {} AST nodes\n", .{ast_nodes.len});

    // Display AST structure
    for (ast_nodes, 0..) |*node, i| {
        try stdout.print("  [{}] ", .{i});
        try printAstSummary(stdout, node, 1);
    }
    try stdout.print("\n", .{});

    try stdout.print("============================================================\n", .{});
    if (enable_cst) {
        if (cst_builder_ptr) |builder| {
            const cst_root = try builder.buildRoot(tokens);
            _ = cst_root; // TODO: optional dump in future flag
            builder.deinit();
        }
    }

    try stdout.print("Frontend compilation completed successfully!\n", .{});
    try stdout.print("Pipeline: {} tokens -> {} AST nodes\n", .{ tokens.len, ast_nodes.len });
}

/// Print a concise AST summary
fn printAstSummary(writer: anytype, node: *lib.AstNode, indent: u32) !void {
    // Print indentation
    var indent_count: u32 = 0;
    while (indent_count < indent) : (indent_count += 1) {
        try writer.print("  ", .{});
    }

    switch (node.*) {
        .Contract => |*contract| {
            try writer.print("Contract '{s}' ({} members)\n", .{ contract.name, contract.body.len });
        },
        .Function => |*function| {
            const visibility = if (function.visibility == .Public) "pub " else "";
            try writer.print("{s}Function '{s}' ({} params)\n", .{ visibility, function.name, function.parameters.len });
        },
        .VariableDecl => |*var_decl| {
            const mutability = switch (var_decl.kind) {
                .Var => "var ",
                .Let => "let ",
                .Const => "const ",
                .Immutable => "immutable ",
            };
            try writer.print("Variable {s}{s}'{s}'\n", .{ @tagName(var_decl.region), mutability, var_decl.name });
        },
        .LogDecl => |*log_decl| {
            try writer.print("Log '{s}' ({} fields)\n", .{ log_decl.name, log_decl.fields.len });
        },
        else => {
            try writer.print("AST Node\n", .{});
        },
    }
}

/// Generate AST and save to JSON file
fn runASTGeneration(allocator: std.mem.Allocator, file_path: []const u8, output_dir: ?[]const u8, enable_cst: bool) !void {
    const stdout = std.io.getStdOut().writer();

    // Read source file
    const source = std.fs.cwd().readFileAlloc(allocator, file_path, 1024 * 1024) catch |err| {
        try stdout.print("Error reading file {s}: {}\n", .{ file_path, err });
        return;
    };
    defer allocator.free(source);

    try stdout.print("Generating AST for {s}\n", .{file_path});
    try stdout.print("==================================================\n", .{});

    // Run lexer + parser
    var lexer = lib.Lexer.init(allocator, source);
    defer lexer.deinit();

    const tokens = lexer.scanTokens() catch |err| {
        try stdout.print("Lexer error: {}\n", .{err});
        return;
    };
    defer allocator.free(tokens);

    var arena = lib.ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var parser = lib.Parser.init(tokens, &arena);
    parser.setFileId(1);
    var cst_builder_storage: lib.cst.CstBuilder = undefined;
    var cst_builder_ptr: ?*lib.cst.CstBuilder = null;
    if (enable_cst) {
        cst_builder_storage = lib.cst.CstBuilder.init(allocator);
        cst_builder_ptr = &cst_builder_storage;
        parser.withCst(cst_builder_ptr.?);
    }
    const ast_nodes = parser.parse() catch |err| {
        try stdout.print("Parser error: {}\n", .{err});
        return;
    };
    if (enable_cst) {
        if (cst_builder_ptr) |builder| {
            const cst_root = try builder.buildRoot(tokens);
            _ = cst_root; // CST not emitted here yet
            builder.deinit();
        }
    }
    // Note: AST nodes are allocated in arena, so they're automatically freed when arena is deinited

    try stdout.print("Generated {} AST nodes\n", .{ast_nodes.len});

    // Generate output filename
    const output_file = if (output_dir) |dir| blk: {
        // Create output directory if it doesn't exist
        std.fs.cwd().makeDir(dir) catch |err| switch (err) {
            error.PathAlreadyExists => {},
            else => return err,
        };

        const basename = std.fs.path.stem(file_path);
        const filename = try std.mem.concat(allocator, u8, &[_][]const u8{ basename, ".ast.json" });
        defer allocator.free(filename);
        break :blk try std.fs.path.join(allocator, &[_][]const u8{ dir, filename });
    } else blk: {
        const basename = std.fs.path.stem(file_path);
        break :blk try std.mem.concat(allocator, u8, &[_][]const u8{ basename, ".ast.json" });
    };
    defer allocator.free(output_file);

    // Save AST to JSON file
    const file = std.fs.cwd().createFile(output_file, .{}) catch |err| {
        try stdout.print("Error creating output file {s}: {}\n", .{ output_file, err });
        return;
    };
    defer file.close();

    const writer = file.writer();
    lib.ast.AstSerializer.serializeAST(ast_nodes, writer) catch |err| {
        try stdout.print("Error serializing AST: {}\n", .{err});
        return;
    };

    try stdout.print("AST saved to {s}\n", .{output_file});
}

test "simple test" {
    var list = std.ArrayList(i32).init(std.testing.allocator);
    defer list.deinit(); // Try commenting this out and see if zig detects the memory leak!
    try list.append(42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}

test "use lexer module" {
    var lexer = lib.Lexer.init(std.testing.allocator, "contract Test {}");
    defer lexer.deinit();

    const tokens = try lexer.scanTokens();
    defer std.testing.allocator.free(tokens);

    // Should have at least: contract, Test, {, }, EOF = 5 tokens
    try std.testing.expect(tokens.len >= 5);
    try std.testing.expect(tokens[0].type == lib.TokenType.Contract);
    try std.testing.expect(tokens[1].type == lib.TokenType.Identifier);
    try std.testing.expect(tokens[tokens.len - 1].type == lib.TokenType.Eof);
}

test "fuzz example" {
    const Context = struct {
        fn testOne(context: @This(), input: []const u8) anyerror!void {
            _ = context;
            // Try passing `--fuzz` to `zig build test` and see if it manages to fail this test case!
            try std.testing.expect(!std.mem.eql(u8, "canyoufindme", input));
        }
    };
    try std.testing.fuzz(Context{}, Context.testOne, .{});
}
