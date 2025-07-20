//! Ora Language Compiler CLI
//!
//! Command-line interface for the Ora domain-specific language compiler.
//! Provides commands for each phase of compilation from source code to EVM bytecode.
//!
//! Available commands:
//! - lex: Tokenize source files
//! - parse: Generate Abstract Syntax Tree
//! - analyze: Perform semantic analysis
//! - ir/hir: Generate High-level Intermediate Representation
//! - yul: Generate Yul intermediate code
//! - bytecode: Generate EVM bytecode
//! - compile: Full compilation pipeline

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
        try runParser(allocator, file_path);
    } else if (std.mem.eql(u8, cmd, "analyze")) {
        try runSemanticAnalysis(allocator, file_path);
    } else if (std.mem.eql(u8, cmd, "ir")) {
        try runIRGeneration(allocator, file_path);
    } else if (std.mem.eql(u8, cmd, "compile")) {
        try runFullCompilation(allocator, file_path);
    } else if (std.mem.eql(u8, cmd, "ast")) {
        try runASTGeneration(allocator, file_path, output_dir);
    } else if (std.mem.eql(u8, cmd, "hir")) {
        try runHIRGeneration(allocator, file_path, output_dir);
    } else if (std.mem.eql(u8, cmd, "yul")) {
        try runYulGeneration(allocator, file_path, output_dir);
    } else if (std.mem.eql(u8, cmd, "bytecode")) {
        try runBytecodeGeneration(allocator, file_path, output_dir);
    } else {
        try printUsage();
    }
}

fn printUsage() !void {
    const stdout = std.io.getStdOut().writer();
    try stdout.print("Ora DSL Compiler v0.1\n", .{});
    try stdout.print("Usage: ora [options] <command> <file>\n", .{});
    try stdout.print("\nOptions:\n", .{});
    try stdout.print("  -o, --output-dir <dir>  - Specify output directory for generated files\n", .{});
    try stdout.print("\nCommands:\n", .{});
    try stdout.print("  lex <file>     - Tokenize a .ora file\n", .{});
    try stdout.print("  parse <file>   - Parse a .ora file to AST\n", .{});
    try stdout.print("  analyze <file> - Perform semantic analysis\n", .{});
    try stdout.print("  ir <file>      - Generate and validate IR from source\n", .{});
    try stdout.print("  compile <file> - Full compilation pipeline (lex -> parse -> analyze -> ir)\n", .{});
    try stdout.print("  ast <file>     - Generate AST and save to JSON file\n", .{});
    try stdout.print("  hir <file>     - Generate HIR and save to JSON file\n", .{});
    try stdout.print("  yul <file>     - Generate Yul code from HIR\n", .{});
    try stdout.print("  bytecode <file> - Generate EVM bytecode from HIR\n", .{});
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

    // Display tokens
    for (tokens, 0..) |token, i| {
        if (i < 20 or token.type == .Eof) { // Show first 20 tokens + EOF
            try stdout.print("[{:3}] {}\n", .{ i, token });
        } else if (i == 20) {
            try stdout.print("... ({} more tokens)\n", .{tokens.len - 21});
        }
    }
}

/// Run parser on file and display AST
fn runParser(allocator: std.mem.Allocator, file_path: []const u8) !void {
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
    var parser = lib.Parser.init(allocator, tokens);
    const ast_nodes = parser.parse() catch |err| {
        try stdout.print("Parser error: {}\n", .{err});
        return;
    };
    defer lib.ast.deinitAstNodes(allocator, ast_nodes);

    try stdout.print("Generated {} AST nodes\n\n", .{ast_nodes.len});

    // Display AST summary
    for (ast_nodes, 0..) |*node, i| {
        try stdout.print("[{}] ", .{i});
        try printAstSummary(stdout, node, 0);
    }
}

/// Run semantic analysis on file
fn runSemanticAnalysis(allocator: std.mem.Allocator, file_path: []const u8) !void {
    const stdout = std.io.getStdOut().writer();

    // Read source file
    const source = std.fs.cwd().readFileAlloc(allocator, file_path, 1024 * 1024) catch |err| {
        try stdout.print("Error reading file {s}: {}\n", .{ file_path, err });
        return;
    };
    defer allocator.free(source);

    try stdout.print("Analyzing {s}\n", .{file_path});
    try stdout.print("==================================================\n", .{});

    // Run lexer + parser
    var lexer = lib.Lexer.init(allocator, source);
    defer lexer.deinit();

    const tokens = lexer.scanTokens() catch |err| {
        try stdout.print("Lexer error: {}\n", .{err});
        return;
    };
    defer allocator.free(tokens);

    var parser = lib.Parser.init(allocator, tokens);
    const ast_nodes = parser.parse() catch |err| {
        try stdout.print("Parser error: {}\n", .{err});
        return;
    };
    defer lib.ast.deinitAstNodes(allocator, ast_nodes);

    try stdout.print("Parsed {} AST nodes\n", .{ast_nodes.len});

    // Run semantic analysis
    try stdout.print("Running semantic analysis...\n", .{});

    var semantic_analyzer = lib.SemanticAnalyzer.init(allocator);
    semantic_analyzer.initSelfReferences(); // Fix self-references after struct is in final location
    defer semantic_analyzer.deinit();

    const diagnostics = semantic_analyzer.analyze(ast_nodes) catch |err| {
        try stdout.print("Semantic analysis failed: {}\n", .{err});
        return;
    };
    defer {
        // Free each diagnostic message before freeing the array
        for (diagnostics) |diagnostic| {
            allocator.free(diagnostic.message);
        }
        allocator.free(diagnostics);
    }

    try stdout.print("Semantic analysis completed with {} diagnostics\n", .{diagnostics.len});
    for (diagnostics) |diagnostic| {
        try stdout.print("  {}\n", .{diagnostic});
    }
}

/// Run IR generation on file
fn runIRGeneration(allocator: std.mem.Allocator, file_path: []const u8) !void {
    const stdout = std.io.getStdOut().writer();

    // Read source file
    const source = std.fs.cwd().readFileAlloc(allocator, file_path, 1024 * 1024) catch |err| {
        try stdout.print("Error reading file {s}: {}\n", .{ file_path, err });
        return;
    };
    defer allocator.free(source);

    try stdout.print("Generating IR for {s}\n", .{file_path});
    try stdout.print("==================================================\n", .{});

    // Run lexer + parser
    var lexer = lib.Lexer.init(allocator, source);
    defer lexer.deinit();

    const tokens = lexer.scanTokens() catch |err| {
        try stdout.print("Lexer error: {}\n", .{err});
        return;
    };
    defer allocator.free(tokens);

    var parser = lib.Parser.init(allocator, tokens);
    const ast_nodes = parser.parse() catch |err| {
        try stdout.print("Parser error: {}\n", .{err});
        return;
    };
    defer lib.ast.deinitAstNodes(allocator, ast_nodes);

    try stdout.print("Parsed {} AST nodes\n", .{ast_nodes.len});

    // Create HIR
    var ir_builder = lib.IRBuilder.init(allocator);
    defer ir_builder.deinit();

    // Convert AST to HIR
    ir_builder.buildFromAST(ast_nodes) catch |err| {
        try stdout.print("AST to HIR conversion failed: {}\n", .{err});
        return;
    };

    const hir_program = ir_builder.getProgramPtr();
    try stdout.print("HIR program created (version: {s})\n", .{hir_program.version});
    try stdout.print("  {} contracts converted\n", .{hir_program.contracts.len});

    // Validate HIR
    var validator = lib.Validator.init(allocator);
    defer validator.deinit();

    const validation_result = validator.validateProgram(hir_program) catch |err| {
        try stdout.print("IR validation failed: {}\n", .{err});
        return;
    };

    if (validation_result.valid) {
        try stdout.print("IR validation passed\n", .{});
    } else {
        try stdout.print("IR validation failed with {} errors\n", .{validation_result.errors.len});
        for (validation_result.errors) |error_| {
            try stdout.print("  Error at line {}, column {}: {s}\n", .{ error_.location.line, error_.location.column, error_.message });
        }
    }

    // Optionally export to JSON
    try stdout.print("\nIR Generation complete\n", .{});
    try stdout.print("  {} contracts in HIR\n", .{hir_program.contracts.len});

    // Display HIR summary
    for (hir_program.contracts) |*contract| {
        try stdout.print("    Contract '{s}': {} storage, {} functions, {} events\n", .{
            contract.name,
            contract.storage.len,
            contract.functions.len,
            contract.events.len,
        });
    }
}

/// Run full compilation pipeline
fn runFullCompilation(allocator: std.mem.Allocator, file_path: []const u8) !void {
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
    var parser = lib.Parser.init(allocator, tokens);
    const ast_nodes = parser.parse() catch |err| {
        try stdout.print("Parser failed: {}\n", .{err});
        return;
    };
    defer lib.ast.deinitAstNodes(allocator, ast_nodes);

    try stdout.print("Generated {} AST nodes\n", .{ast_nodes.len});

    // Display AST structure
    for (ast_nodes, 0..) |*node, i| {
        try stdout.print("  [{}] ", .{i});
        try printAstSummary(stdout, node, 1);
    }
    try stdout.print("\n", .{});

    // Phase 3: Semantic Analysis
    try stdout.print("Phase 3: Semantic Analysis\n", .{});

    var semantic_analyzer = lib.SemanticAnalyzer.init(allocator);
    semantic_analyzer.initSelfReferences(); // Fix self-references after struct is in final location
    defer semantic_analyzer.deinit();

    const diagnostics = semantic_analyzer.analyze(ast_nodes) catch |err| {
        try stdout.print("Semantic analysis failed: {}\n", .{err});
        return;
    };
    defer {
        // Free each diagnostic message before freeing the array
        for (diagnostics) |diagnostic| {
            allocator.free(diagnostic.message);
        }
        allocator.free(diagnostics);
    }

    try stdout.print("Semantic analysis completed with {} diagnostics\n", .{diagnostics.len});
    for (diagnostics) |diagnostic| {
        try stdout.print("  {}\n", .{diagnostic});
    }

    try stdout.print("\n", .{});

    // Phase 4: IR Generation
    try stdout.print("Phase 4: IR Generation\n", .{});
    var ir_builder = lib.IRBuilder.init(allocator);
    defer ir_builder.deinit();

    // Convert AST to HIR
    ir_builder.buildFromAST(ast_nodes) catch |err| {
        try stdout.print("AST to HIR conversion failed: {}\n", .{err});
        return;
    };

    const hir_program = ir_builder.getProgramPtr();
    try stdout.print("HIR program created (version: {s})\n", .{hir_program.version});
    try stdout.print("  {} contracts converted to HIR\n", .{hir_program.contracts.len});

    // Phase 5: IR Validation
    try stdout.print("Phase 5: IR Validation\n", .{});
    var validator = lib.Validator.init(allocator);
    defer validator.deinit();

    const validation_result = validator.validateProgram(hir_program) catch |err| {
        try stdout.print("IR validation failed: {}\n", .{err});
        return;
    };

    if (validation_result.valid) {
        try stdout.print("IR validation passed\n", .{});
    } else {
        try stdout.print("IR validation failed with {} errors\n", .{validation_result.errors.len});
        for (validation_result.errors) |error_| {
            try stdout.print("  Error at line {}, column {}: {s}\n", .{ error_.location.line, error_.location.column, error_.message });
        }
    }

    try stdout.print("\n", .{});

    // Phase 6: Yul Code Generation
    try stdout.print("Phase 6: Yul Code Generation\n", .{});

    // Generate Yul code from actual HIR program
    var yul_codegen = lib.YulCodegen.init(allocator);
    defer yul_codegen.deinit();

    const yul_code = yul_codegen.generateYulFromProgram(hir_program) catch |err| {
        try stdout.print("Yul generation failed: {}\n", .{err});
        return;
    };
    defer allocator.free(yul_code);

    try stdout.print("Generated Yul code ({} bytes)\n", .{yul_code.len});

    // Phase 7: EVM Bytecode Generation
    try stdout.print("Phase 7: EVM Bytecode Generation\n", .{});

    var result = yul_codegen.compileYulToBytecode(yul_code) catch |err| {
        try stdout.print("Bytecode generation failed: {}\n", .{err});
        return;
    };
    defer result.deinit(allocator);

    if (result.success) {
        if (result.bytecode) |bytecode| {
            try stdout.print("✓ Bytecode generation successful! ({} bytes)\n", .{bytecode.len});
        }
    } else {
        try stdout.print("✗ Bytecode generation failed\n", .{});
        if (result.error_message) |error_msg| {
            try stdout.print("Error: {s}\n", .{error_msg});
        }
    }

    try stdout.print("\n", .{});

    try stdout.print("Compilation pipeline completed successfully\n", .{});
    try stdout.print("   {} tokens -> {} AST nodes -> {} HIR contracts -> Yul -> EVM bytecode\n", .{ tokens.len, ast_nodes.len, hir_program.contracts.len });
}

/// Format TypeRef for display
fn formatTypeRef(typ: lib.TypeRef) []const u8 {
    return switch (typ) {
        .Bool => "bool",
        .Address => "address",
        .U8 => "u8",
        .U16 => "u16",
        .U32 => "u32",
        .U64 => "u64",
        .U128 => "u128",
        .U256 => "u256",
        .String => "string",
        .Bytes => "bytes",
        .Slice => "slice[T]",
        .Mapping => "map[K,V]",
        .DoubleMap => "doublemap[K1,K2,V]",
        .Identifier => |name| name,
    };
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
            const visibility = if (function.pub_) "pub " else "";
            try writer.print("{s}Function '{s}' ({} params)\n", .{ visibility, function.name, function.parameters.len });
        },
        .VariableDecl => |*var_decl| {
            const mutability = if (var_decl.mutable) "var " else "";
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
fn runASTGeneration(allocator: std.mem.Allocator, file_path: []const u8, output_dir: ?[]const u8) !void {
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

    var parser = lib.Parser.init(allocator, tokens);
    const ast_nodes = parser.parse() catch |err| {
        try stdout.print("Parser error: {}\n", .{err});
        return;
    };
    defer lib.ast.deinitAstNodes(allocator, ast_nodes);

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
    lib.ast.ASTSerializer.serializeAST(ast_nodes, writer) catch |err| {
        try stdout.print("Error serializing AST: {}\n", .{err});
        return;
    };

    try stdout.print("AST saved to {s}\n", .{output_file});
}

/// Generate HIR and save to JSON file
fn runHIRGeneration(allocator: std.mem.Allocator, file_path: []const u8, output_dir: ?[]const u8) !void {
    const stdout = std.io.getStdOut().writer();

    // Read source file
    const source = std.fs.cwd().readFileAlloc(allocator, file_path, 1024 * 1024) catch |err| {
        try stdout.print("Error reading file {s}: {}\n", .{ file_path, err });
        return;
    };
    defer allocator.free(source);

    try stdout.print("Generating HIR for {s}\n", .{file_path});
    try stdout.print("==================================================\n", .{});

    // Run lexer + parser
    var lexer = lib.Lexer.init(allocator, source);
    defer lexer.deinit();

    const tokens = lexer.scanTokens() catch |err| {
        try stdout.print("Lexer error: {}\n", .{err});
        return;
    };
    defer allocator.free(tokens);

    var parser = lib.Parser.init(allocator, tokens);
    const ast_nodes = parser.parse() catch |err| {
        try stdout.print("Parser error: {}\n", .{err});
        return;
    };
    defer lib.ast.deinitAstNodes(allocator, ast_nodes);

    try stdout.print("Generated {} AST nodes\n", .{ast_nodes.len});

    // Skip semantic analysis for now - it's causing segfaults
    // var typer = lib.Typer.init(allocator);
    // defer typer.deinit();

    // typer.typeCheck(ast_nodes) catch |err| {
    //     try stdout.print("Semantic analysis failed: {}\n", .{err});
    //     return;
    // };

    try stdout.print("Skipping semantic analysis for now\n", .{});

    // Create HIR
    var ir_builder = lib.IRBuilder.init(allocator);
    defer ir_builder.deinit();

    ir_builder.buildFromAST(ast_nodes) catch |err| {
        try stdout.print("AST to HIR conversion failed: {}\n", .{err});
        return;
    };

    const hir_program = ir_builder.getProgramPtr();
    try stdout.print("HIR program created (version: {s})\n", .{hir_program.version});
    try stdout.print("  {} contracts converted to HIR\n", .{hir_program.contracts.len});

    // Generate output filename
    const output_file = if (output_dir) |dir| blk: {
        // Create output directory if it doesn't exist
        std.fs.cwd().makeDir(dir) catch |err| switch (err) {
            error.PathAlreadyExists => {},
            else => return err,
        };

        const basename = std.fs.path.stem(file_path);
        const filename = try std.mem.concat(allocator, u8, &[_][]const u8{ basename, ".hir.json" });
        defer allocator.free(filename);
        break :blk try std.fs.path.join(allocator, &[_][]const u8{ dir, filename });
    } else blk: {
        const basename = std.fs.path.stem(file_path);
        break :blk try std.mem.concat(allocator, u8, &[_][]const u8{ basename, ".hir.json" });
    };
    defer allocator.free(output_file);

    // Save HIR to JSON file
    const file = std.fs.cwd().createFile(output_file, .{}) catch |err| {
        try stdout.print("Error creating output file {s}: {}\n", .{ output_file, err });
        return;
    };
    defer file.close();

    const writer = file.writer();
    lib.JSONSerializer.serializeProgram(hir_program, writer) catch |err| {
        try stdout.print("Error serializing HIR: {}\n", .{err});
        return;
    };

    try stdout.print("HIR saved to {s}\n", .{output_file});
}

/// Generate Yul code from HIR
fn runYulGeneration(allocator: std.mem.Allocator, file_path: []const u8, output_dir: ?[]const u8) !void {
    const stdout = std.io.getStdOut().writer();

    // Read source file
    const source = std.fs.cwd().readFileAlloc(allocator, file_path, 1024 * 1024) catch |err| {
        try stdout.print("Error reading file {s}: {}\n", .{ file_path, err });
        return;
    };
    defer allocator.free(source);

    try stdout.print("Generating Yul code for {s}\n", .{file_path});
    try stdout.print("==================================================\n", .{});

    // Run lexer + parser
    var lexer = lib.Lexer.init(allocator, source);
    defer lexer.deinit();

    const tokens = lexer.scanTokens() catch |err| {
        try stdout.print("Lexer error: {}\n", .{err});
        return;
    };
    defer allocator.free(tokens);

    var parser = lib.Parser.init(allocator, tokens);
    const ast_nodes = parser.parse() catch |err| {
        try stdout.print("Parser error: {}\n", .{err});
        return;
    };
    defer lib.ast.deinitAstNodes(allocator, ast_nodes);

    // Create HIR
    var ir_builder = lib.IRBuilder.init(allocator);
    defer ir_builder.deinit();

    ir_builder.buildFromAST(ast_nodes) catch |err| {
        try stdout.print("AST to HIR conversion failed: {}\n", .{err});
        return;
    };

    const hir_program = ir_builder.getProgramPtr();
    try stdout.print("HIR program created with {} contracts\n", .{hir_program.contracts.len});

    // Generate Yul code from actual HIR program
    var yul_codegen = lib.YulCodegen.init(allocator);
    defer yul_codegen.deinit();

    const yul_code = yul_codegen.generateYulFromProgram(hir_program) catch |err| {
        try stdout.print("Yul generation failed: {}\n", .{err});
        return;
    };
    defer allocator.free(yul_code);

    try stdout.print("Generated Yul code:\n", .{});
    try stdout.print("{s}\n", .{yul_code});

    // Save to file
    const output_file = if (output_dir) |dir| blk: {
        // Create output directory if it doesn't exist
        std.fs.cwd().makeDir(dir) catch |err| switch (err) {
            error.PathAlreadyExists => {},
            else => return err,
        };

        const basename = std.fs.path.stem(file_path);
        const filename = try std.mem.concat(allocator, u8, &[_][]const u8{ basename, ".yul" });
        defer allocator.free(filename);
        break :blk try std.fs.path.join(allocator, &[_][]const u8{ dir, filename });
    } else blk: {
        const basename = std.fs.path.stem(file_path);
        break :blk try std.mem.concat(allocator, u8, &[_][]const u8{ basename, ".yul" });
    };
    defer allocator.free(output_file);

    const file = std.fs.cwd().createFile(output_file, .{}) catch |err| {
        try stdout.print("Error creating output file {s}: {}\n", .{ output_file, err });
        return;
    };
    defer file.close();

    try file.writeAll(yul_code);
    try stdout.print("Yul code saved to {s}\n", .{output_file});
}

/// Generate EVM bytecode from HIR
fn runBytecodeGeneration(allocator: std.mem.Allocator, file_path: []const u8, output_dir: ?[]const u8) !void {
    const stdout = std.io.getStdOut().writer();

    // Read source file
    const source = std.fs.cwd().readFileAlloc(allocator, file_path, 1024 * 1024) catch |err| {
        try stdout.print("Error reading file {s}: {}\n", .{ file_path, err });
        return;
    };
    defer allocator.free(source);

    try stdout.print("Generating EVM bytecode for {s}\n", .{file_path});
    try stdout.print("==================================================\n", .{});

    // Run lexer + parser
    var lexer = lib.Lexer.init(allocator, source);
    defer lexer.deinit();

    const tokens = lexer.scanTokens() catch |err| {
        try stdout.print("Lexer error: {}\n", .{err});
        return;
    };
    defer allocator.free(tokens);

    var parser = lib.Parser.init(allocator, tokens);
    const ast_nodes = parser.parse() catch |err| {
        try stdout.print("Parser error: {}\n", .{err});
        return;
    };
    defer lib.ast.deinitAstNodes(allocator, ast_nodes);

    // Create HIR
    var ir_builder = lib.IRBuilder.init(allocator);
    defer ir_builder.deinit();

    ir_builder.buildFromAST(ast_nodes) catch |err| {
        try stdout.print("AST to HIR conversion failed: {}\n", .{err});
        return;
    };

    const hir_program = ir_builder.getProgramPtr();
    try stdout.print("HIR program created with {} contracts\n", .{hir_program.contracts.len});

    // Generate bytecode from actual HIR program
    var yul_codegen = lib.YulCodegen.init(allocator);
    defer yul_codegen.deinit();

    const yul_code = yul_codegen.generateYulFromProgram(hir_program) catch |err| {
        try stdout.print("Yul generation failed: {}\n", .{err});
        return;
    };
    defer allocator.free(yul_code);

    var result = yul_codegen.compileYulToBytecode(yul_code) catch |err| {
        try stdout.print("Bytecode generation failed: {}\n", .{err});
        return;
    };
    defer result.deinit(allocator);

    if (result.success) {
        if (result.bytecode) |bytecode| {
            try stdout.print("✓ Bytecode generation successful!\n", .{});
            try stdout.print("Bytecode: {s}\n", .{bytecode});

            // Save to file
            const output_file = if (output_dir) |dir| blk: {
                // Create output directory if it doesn't exist
                std.fs.cwd().makeDir(dir) catch |err| switch (err) {
                    error.PathAlreadyExists => {},
                    else => return err,
                };

                const basename = std.fs.path.stem(file_path);
                const filename = try std.mem.concat(allocator, u8, &[_][]const u8{ basename, ".bin" });
                defer allocator.free(filename);
                break :blk try std.fs.path.join(allocator, &[_][]const u8{ dir, filename });
            } else blk: {
                const basename = std.fs.path.stem(file_path);
                break :blk try std.mem.concat(allocator, u8, &[_][]const u8{ basename, ".bin" });
            };
            defer allocator.free(output_file);

            const file = std.fs.cwd().createFile(output_file, .{}) catch |err| {
                try stdout.print("Error creating output file {s}: {}\n", .{ output_file, err });
                return;
            };
            defer file.close();

            try file.writeAll(bytecode);
            try stdout.print("Bytecode saved to {s}\n", .{output_file});
        }
    } else {
        try stdout.print("✗ Bytecode generation failed\n", .{});
        if (result.error_message) |error_msg| {
            try stdout.print("Error: {s}\n", .{error_msg});
        }
    }
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
