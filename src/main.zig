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
const build_options = @import("build_options");
const mlir_pipeline = @import("mlir/pipeline.zig");

/// Artifact saving options
const ArtifactOptions = struct {
    save_tokens: bool,
    save_ast: bool,
    save_mlir: bool,
    save_yul: bool,
    save_bytecode: bool,
    output_dir: ?[]const u8,

    fn createOutputDir(self: ArtifactOptions, allocator: std.mem.Allocator, base_name: []const u8) ![]const u8 {
        if (self.output_dir) |dir| {
            return try std.fmt.allocPrint(allocator, "{s}/{s}", .{ dir, base_name });
        } else {
            return try std.fmt.allocPrint(allocator, "{s}_artifacts", .{base_name});
        }
    }
};

/// MLIR-related command line options
const MlirOptions = struct {
    emit_mlir: bool,
    verify: bool,
    passes: ?[]const u8,
    opt_level: ?[]const u8,
    timing: bool,
    print_ir: bool,
    output_dir: ?[]const u8,
    use_pipeline: bool,

    fn getOptimizationLevel(self: MlirOptions) OptimizationLevel {
        if (self.opt_level) |level| {
            if (std.mem.eql(u8, level, "none")) return .None;
            if (std.mem.eql(u8, level, "basic")) return .Basic;
            if (std.mem.eql(u8, level, "aggressive")) return .Aggressive;
        }

        // Use build-time default if no command-line option provided
        const build_default = build_options.mlir_opt_level;
        if (std.mem.eql(u8, build_default, "none")) return .None;
        if (std.mem.eql(u8, build_default, "basic")) return .Basic;
        if (std.mem.eql(u8, build_default, "aggressive")) return .Aggressive;

        return .Basic; // Final fallback
    }

    fn shouldEnableVerification(self: MlirOptions) bool {
        return self.verify or build_options.mlir_debug;
    }

    fn shouldEnableTiming(self: MlirOptions) bool {
        return self.timing or build_options.mlir_timing;
    }

    fn getDefaultPasses(self: MlirOptions) ?[]const u8 {
        return self.passes orelse build_options.mlir_passes;
    }
};

const OptimizationLevel = enum {
    None,
    Basic,
    Aggressive,
};

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

    // Parse arguments with enhanced MLIR support
    var output_dir: ?[]const u8 = null;
    var no_cst: bool = false;
    var command: ?[]const u8 = null;
    var input_file: ?[]const u8 = null;
    var emit_mlir: bool = false;
    var mlir_verify: bool = false;
    var mlir_passes: ?[]const u8 = null;
    var mlir_opt_level: ?[]const u8 = null;
    var mlir_timing: bool = false;
    var mlir_print_ir: bool = false;
    var mlir_use_pipeline: bool = false;

    // Artifact saving options
    var save_tokens: bool = false;
    var save_ast: bool = false;
    var save_mlir: bool = false;
    var save_yul: bool = false;
    var save_bytecode: bool = false;
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
        } else if (std.mem.eql(u8, args[i], "--emit-mlir")) {
            emit_mlir = true;
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--mlir-verify")) {
            mlir_verify = true;
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--mlir-passes")) {
            if (i + 1 >= args.len) {
                try printUsage();
                return;
            }
            mlir_passes = args[i + 1];
            i += 2;
        } else if (std.mem.eql(u8, args[i], "--mlir-opt")) {
            if (i + 1 >= args.len) {
                try printUsage();
                return;
            }
            mlir_opt_level = args[i + 1];
            i += 2;
        } else if (std.mem.eql(u8, args[i], "--mlir-timing")) {
            mlir_timing = true;
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--mlir-print-ir")) {
            mlir_print_ir = true;
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--mlir-pipeline")) {
            mlir_use_pipeline = true;
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--save-tokens")) {
            save_tokens = true;
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--save-ast")) {
            save_ast = true;
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--save-mlir")) {
            save_mlir = true;
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--save-yul")) {
            save_yul = true;
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--save-bytecode")) {
            save_bytecode = true;
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--save-all")) {
            save_tokens = true;
            save_ast = true;
            save_mlir = true;
            save_yul = true;
            save_bytecode = true;
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

    // Create MLIR options structure
    const mlir_options = MlirOptions{
        .emit_mlir = emit_mlir,
        .verify = mlir_verify,
        .passes = mlir_passes,
        .opt_level = mlir_opt_level,
        .timing = mlir_timing,
        .print_ir = mlir_print_ir,
        .output_dir = output_dir,
        .use_pipeline = mlir_use_pipeline,
    };

    // Create artifact options structure
    const artifact_options = ArtifactOptions{
        .save_tokens = save_tokens,
        .save_ast = save_ast,
        .save_mlir = save_mlir,
        .save_yul = save_yul,
        .save_bytecode = save_bytecode,
        .output_dir = output_dir,
    };

    if (std.mem.eql(u8, cmd, "lex")) {
        try runLexer(allocator, file_path, artifact_options);
    } else if (std.mem.eql(u8, cmd, "parse")) {
        try runParser(allocator, file_path, !no_cst, artifact_options);
    } else if (std.mem.eql(u8, cmd, "ast")) {
        try runASTGeneration(allocator, file_path, output_dir, !no_cst, artifact_options);
    } else if (std.mem.eql(u8, cmd, "compile")) {
        if (mlir_options.emit_mlir or artifact_options.save_mlir or artifact_options.save_yul or artifact_options.save_bytecode) {
            // If MLIR, Yul, or bytecode is requested, use the advanced MLIR pipeline with Yul conversion
            try runMlirEmitAdvanced(allocator, file_path, mlir_options, artifact_options);
        } else {
            // Otherwise, use the basic frontend compilation
            try runFullCompilation(allocator, file_path, !no_cst, mlir_options, artifact_options);
        }
    } else if (std.mem.eql(u8, cmd, "mlir")) {
        try runMlirEmitAdvanced(allocator, file_path, mlir_options, artifact_options);
    } else {
        try printUsage();
    }
}

fn printUsage() !void {
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;
    try stdout.print("Ora Compiler v0.1 - Asuka\n", .{});
    try stdout.print("Usage: ora [options] <command> <file>\n", .{});
    try stdout.print("\nGeneral Options:\n", .{});
    try stdout.print("  -o, --output-dir <dir>  - Specify output directory for generated files\n", .{});
    try stdout.print("      --no-cst            - Disable CST building (enabled by default)\n", .{});
    try stdout.print("\nArtifact Saving Options:\n", .{});
    try stdout.print("      --save-tokens       - Save tokens from lexical analysis\n", .{});
    try stdout.print("      --save-ast          - Save AST from syntax analysis\n", .{});
    try stdout.print("      --save-mlir         - Save MLIR from generation\n", .{});
    try stdout.print("      --save-yul          - Save Yul from lowering\n", .{});
    try stdout.print("      --save-bytecode     - Save EVM bytecode\n", .{});
    try stdout.print("      --save-all          - Save all artifacts\n", .{});
    try stdout.print("\nMLIR Options:\n", .{});
    try stdout.print("      --emit-mlir         - Generate MLIR output in addition to normal compilation\n", .{});
    try stdout.print("      --mlir-verify       - Run MLIR verification passes\n", .{});
    try stdout.print("      --mlir-passes <str> - Custom MLIR pass pipeline (e.g., 'canonicalize,cse')\n", .{});
    try stdout.print("      --mlir-opt <level>  - Optimization level: none, basic, aggressive\n", .{});
    try stdout.print("      --mlir-timing       - Enable pass timing statistics\n", .{});
    try stdout.print("      --mlir-print-ir     - Print IR before and after passes\n", .{});
    try stdout.print("      --mlir-pipeline     - Use comprehensive MLIR optimization pipeline\n", .{});
    try stdout.print("\nCommands:\n", .{});
    try stdout.print("  lex <file>     - Tokenize a .ora file\n", .{});
    try stdout.print("  parse <file>   - Parse a .ora file to AST\n", .{});
    try stdout.print("  ast <file>     - Generate AST and save to JSON file\n", .{});
    try stdout.print("  compile <file> - Full frontend pipeline (lex -> parse -> [mlir])\n", .{});
    try stdout.print("  mlir <file>    - Run front-end and emit MLIR with advanced options\n", .{});
    try stdout.print("\nExamples:\n", .{});
    try stdout.print("  ora -o build ast example.ora\n", .{});
    try stdout.print("  ora --emit-mlir compile example.ora\n", .{});
    try stdout.print("  ora --save-all compile example.ora\n", .{});
    try stdout.print("  ora --save-tokens --save-ast lex example.ora\n", .{});
    try stdout.print("  ora --mlir-opt aggressive --mlir-verify mlir example.ora\n", .{});
    try stdout.print("  ora --mlir-passes 'canonicalize,cse,sccp' --mlir-timing mlir example.ora\n", .{});
    try stdout.flush();
}

/// Run lexer on file and display tokens
fn runLexer(allocator: std.mem.Allocator, file_path: []const u8, artifact_options: ArtifactOptions) !void {
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    // Read source file
    const source = std.fs.cwd().readFileAlloc(allocator, file_path, 1024 * 1024) catch |err| {
        try stdout.print("Error reading file {s}: {s}\n", .{ file_path, @errorName(err) });
        return;
    };
    defer allocator.free(source);

    try stdout.print("Lexing {s}\n", .{file_path});
    try stdout.print("==================================================\n", .{});

    // Run lexer
    var lexer = lib.Lexer.init(allocator, source);
    defer lexer.deinit();

    const tokens = lexer.scanTokens() catch |err| {
        try stdout.print("Lexer error: {s}\n", .{@errorName(err)});
        if (err == lib.lexer.LexerError.UnexpectedCharacter) {
            const error_details = try lexer.getErrorDetails(allocator);
            defer allocator.free(error_details);
            try stdout.print("   {s}\n", .{error_details});
        }
        return;
    };
    defer allocator.free(tokens);

    try stdout.print("Generated {d} tokens\n\n", .{tokens.len});

    // Display all tokens without truncation
    for (tokens, 0..) |token, i| {
        try stdout.print("[{d:3}] {any}\n", .{ i, token });
    }

    // Save tokens if requested
    if (artifact_options.save_tokens) {
        try saveTokens(allocator, file_path, tokens, artifact_options);
    }
    try stdout.flush();
}

/// Save tokens to file
fn saveTokens(allocator: std.mem.Allocator, file_path: []const u8, tokens: []const lib.Token, artifact_options: ArtifactOptions) !void {
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    // Create output directory
    const base_name = std.fs.path.stem(file_path);
    const output_dir = try artifact_options.createOutputDir(allocator, base_name);
    defer allocator.free(output_dir);

    // Create directory if it doesn't exist
    std.fs.cwd().makeDir(output_dir) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return err,
    };

    // Write tokens to file
    const tokens_file = try std.fmt.allocPrint(allocator, "{s}/tokens.txt", .{output_dir});
    defer allocator.free(tokens_file);

    var file = try std.fs.cwd().createFile(tokens_file, .{});
    defer file.close();

    try file.writeAll("Ora Compiler - Token Output\n");
    try file.writeAll("============================\n\n");

    for (tokens, 0..) |token, i| {
        var file_buffer: [1024]u8 = undefined;
        const formatted = try std.fmt.bufPrint(file_buffer[0..], "[{d:3}] {any}\n", .{ i, token });
        try file.writeAll(formatted);
    }

    try stdout.print("Saved tokens to: {s}\n", .{tokens_file});
    try stdout.flush();
}

/// Save AST to file
fn saveAST(allocator: std.mem.Allocator, file_path: []const u8, ast_nodes: []const lib.ast.AstNode, artifact_options: ArtifactOptions) !void {
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    // Create output directory
    const base_name = std.fs.path.stem(file_path);
    const output_dir = try artifact_options.createOutputDir(allocator, base_name);
    defer allocator.free(output_dir);

    // Create directory if it doesn't exist
    std.fs.cwd().makeDir(output_dir) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return err,
    };

    // Write AST to file
    const ast_file = try std.fmt.allocPrint(allocator, "{s}/ast.txt", .{output_dir});
    defer allocator.free(ast_file);

    var file = try std.fs.cwd().createFile(ast_file, .{});
    defer file.close();

    try file.writeAll("Ora Compiler - AST Output\n");
    try file.writeAll("=========================\n\n");

    for (ast_nodes, 0..) |node, i| {
        var file_buffer: [1024]u8 = undefined;
        const formatted = try std.fmt.bufPrint(file_buffer[0..], "[{d}] ", .{i});
        try file.writeAll(formatted);
        var writer_buffer: [1024]u8 = undefined;
        var file_writer = file.writer(&writer_buffer);
        try printAstSummary(&file_writer.interface, @constCast(&node), 1);
        try file.writeAll("\n");
    }

    try stdout.print("Saved AST to: {s}\n", .{ast_file});
    try stdout.flush();
}

/// Save MLIR to file
fn saveMLIR(allocator: std.mem.Allocator, file_path: []const u8, mlir_text: []const u8, artifact_options: ArtifactOptions) !void {
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    // Create output directory
    const base_name = std.fs.path.stem(file_path);
    const output_dir = try artifact_options.createOutputDir(allocator, base_name);
    defer allocator.free(output_dir);

    // Create directory if it doesn't exist
    std.fs.cwd().makeDir(output_dir) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return err,
    };

    // Write MLIR to file
    const mlir_file = try std.fmt.allocPrint(allocator, "{s}/mlir.mlir", .{output_dir});
    defer allocator.free(mlir_file);

    var file = try std.fs.cwd().createFile(mlir_file, .{});
    defer file.close();

    try file.writeAll(mlir_text);

    try stdout.print("Saved MLIR to: {s}\n", .{mlir_file});
    try stdout.flush();
}

/// Save Yul to file
fn saveYul(allocator: std.mem.Allocator, file_path: []const u8, yul_code: []const u8, artifact_options: ArtifactOptions) !void {
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    // Create output directory
    const base_name = std.fs.path.stem(file_path);
    const output_dir = try artifact_options.createOutputDir(allocator, base_name);
    defer allocator.free(output_dir);

    // Create directory if it doesn't exist
    std.fs.cwd().makeDir(output_dir) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return err,
    };

    // Write Yul to file
    const yul_file = try std.fmt.allocPrint(allocator, "{s}/yul.yul", .{output_dir});
    defer allocator.free(yul_file);

    var file = try std.fs.cwd().createFile(yul_file, .{});
    defer file.close();

    try file.writeAll(yul_code);

    try stdout.print("Saved Yul to: {s}\n", .{yul_file});
    try stdout.flush();
}

/// Save bytecode to file
fn saveBytecode(allocator: std.mem.Allocator, file_path: []const u8, bytecode: []const u8, artifact_options: ArtifactOptions) !void {
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    // Create output directory
    const base_name = std.fs.path.stem(file_path);
    const output_dir = try artifact_options.createOutputDir(allocator, base_name);
    defer allocator.free(output_dir);

    // Create directory if it doesn't exist
    std.fs.cwd().makeDir(output_dir) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return err,
    };

    // Write bytecode to file
    const bytecode_file = try std.fmt.allocPrint(allocator, "{s}/bytecode.hex", .{output_dir});
    defer allocator.free(bytecode_file);

    var file = try std.fs.cwd().createFile(bytecode_file, .{});
    defer file.close();

    try file.writeAll(bytecode);

    try stdout.print("Saved bytecode to: {s}\n", .{bytecode_file});
    try stdout.flush();
}

/// Run parser on file and display AST
fn runParser(allocator: std.mem.Allocator, file_path: []const u8, enable_cst: bool, artifact_options: ArtifactOptions) !void {
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    // Read source file
    const source = std.fs.cwd().readFileAlloc(allocator, file_path, 1024 * 1024) catch |err| {
        try stdout.print("Error reading file {s}: {s}\n", .{ file_path, @errorName(err) });
        return;
    };
    defer allocator.free(source);

    try stdout.print("Parsing {s}\n", .{file_path});
    try stdout.print("==================================================\n", .{});

    // Run lexer
    var lexer = lib.Lexer.init(allocator, source);
    defer lexer.deinit();

    const tokens = lexer.scanTokens() catch |err| {
        try stdout.print("Lexer error: {s}\n", .{@errorName(err)});
        return;
    };
    defer allocator.free(tokens);

    try stdout.print("Lexed {d} tokens\n", .{tokens.len});

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
        try stdout.print("Parser error: {s}\n", .{@errorName(err)});
        return;
    };
    // Note: AST nodes are allocated in arena, so they're automatically freed when arena is deinited

    try stdout.print("Generated {d} AST nodes\n\n", .{ast_nodes.len});

    // Display AST summary
    for (ast_nodes, 0..) |*node, i| {
        try stdout.print("[{d}] ", .{i});
        try printAstSummary(stdout, node, 0);
    }

    // Save AST if requested
    if (artifact_options.save_ast) {
        try saveAST(allocator, file_path, ast_nodes, artifact_options);
    }

    if (enable_cst) {
        if (cst_builder_ptr) |builder| {
            const cst_root = try builder.buildRoot(tokens);
            _ = cst_root; // TODO: optional dump in future flag
            builder.deinit();
        }
    }
}

/// Run full compilation pipeline with optional MLIR support
fn runFullCompilation(allocator: std.mem.Allocator, file_path: []const u8, enable_cst: bool, mlir_options: MlirOptions, artifact_options: ArtifactOptions) !void {
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    try stdout.print("Compiling {s}\n", .{file_path});
    try stdout.print("============================================================\n", .{});

    // Read source file
    const source = std.fs.cwd().readFileAlloc(allocator, file_path, 1024 * 1024) catch |err| {
        try stdout.print("Error reading file {s}: {s}\n", .{ file_path, @errorName(err) });
        return;
    };
    defer allocator.free(source);

    try stdout.print("Source ({d} bytes):\n", .{source.len});
    try stdout.print("{s}\n\n", .{source});

    // Phase 1: Lexical Analysis
    try stdout.print("Phase 1: Lexical Analysis\n", .{});
    var lexer = lib.Lexer.init(allocator, source);
    defer lexer.deinit();

    const tokens = lexer.scanTokens() catch |err| {
        try stdout.print("Lexer failed: {s}\n", .{@errorName(err)});
        return;
    };
    defer allocator.free(tokens);

    try stdout.print("Generated {d} tokens\n\n", .{tokens.len});

    // Save tokens if requested
    if (artifact_options.save_tokens) {
        try saveTokens(allocator, file_path, tokens, artifact_options);
    }

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
        try stdout.print("Parser failed: {s}\n", .{@errorName(err)});
        return;
    };
    // Note: AST nodes are allocated in arena, so they're automatically freed when arena is deinited

    try stdout.print("Generated {d} AST nodes\n", .{ast_nodes.len});

    // Display AST structure
    for (ast_nodes, 0..) |*node, i| {
        try stdout.print("  [{d}] ", .{i});
        try printAstSummary(stdout, node, 1);
    }
    try stdout.print("\n", .{});

    // Save AST if requested
    if (artifact_options.save_ast) {
        try saveAST(allocator, file_path, ast_nodes, artifact_options);
    }

    try stdout.print("============================================================\n", .{});
    if (enable_cst) {
        if (cst_builder_ptr) |builder| {
            const cst_root = try builder.buildRoot(tokens);
            _ = cst_root; // TODO: optional dump in future flag
            builder.deinit();
        }
    }

    // Phase 3: MLIR Generation (if requested)
    if (mlir_options.emit_mlir) {
        try stdout.print("Phase 3: MLIR Generation\n", .{});
        try generateMlirOutput(allocator, ast_nodes, file_path, mlir_options, artifact_options);
    }

    try stdout.print("Frontend compilation completed successfully!\n", .{});
    try stdout.print("Pipeline: {d} tokens -> {d} AST nodes", .{ tokens.len, ast_nodes.len });
    if (mlir_options.emit_mlir) {
        try stdout.print(" -> MLIR module", .{});
    }
    try stdout.print("\n", .{});
    try stdout.flush();
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
            try writer.print("Contract '{s}' ({d} members)\n", .{ contract.name, contract.body.len });
        },
        .Function => |*function| {
            const visibility = if (function.visibility == .Public) "pub " else "";
            try writer.print("{s}Function '{s}' ({d} params)\n", .{ visibility, function.name, function.parameters.len });
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
            try writer.print("Log '{s}' ({d} fields)\n", .{ log_decl.name, log_decl.fields.len });
        },
        else => {
            try writer.print("AST Node\n", .{});
        },
    }
}

/// Generate AST and save to JSON file
fn runASTGeneration(allocator: std.mem.Allocator, file_path: []const u8, output_dir: ?[]const u8, enable_cst: bool, artifact_options: ArtifactOptions) !void {
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    // Read source file
    const source = std.fs.cwd().readFileAlloc(allocator, file_path, 1024 * 1024) catch |err| {
        try stdout.print("Error reading file {s}: {s}\n", .{ file_path, @errorName(err) });
        return;
    };
    defer allocator.free(source);

    try stdout.print("Generating AST for {s}\n", .{file_path});
    try stdout.print("==================================================\n", .{});

    // Run lexer + parser
    var lexer = lib.Lexer.init(allocator, source);
    defer lexer.deinit();

    const tokens = lexer.scanTokens() catch |err| {
        try stdout.print("Lexer error: {s}\n", .{@errorName(err)});
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
        try stdout.print("Parser error: {s}\n", .{@errorName(err)});
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

    try stdout.print("Generated {d} AST nodes\n", .{ast_nodes.len});

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
        try stdout.print("Error creating output file {s}: {s}\n", .{ output_file, @errorName(err) });
        return;
    };
    defer file.close();

    // Use a simple direct approach for file writing
    // TODO: Fix AST serialization for Zig 0.15.1 compatibility
    try file.writeAll("{\"type\":\"AST\",\"nodes\":[]}");

    try stdout.print("AST saved to {s}\n", .{output_file});

    // Also save AST in artifact format if requested
    if (artifact_options.save_ast) {
        try saveAST(allocator, file_path, ast_nodes, artifact_options);
    }
    try stdout.flush();
}

/// Advanced MLIR emission with full pass pipeline support
fn runMlirEmitAdvanced(allocator: std.mem.Allocator, file_path: []const u8, mlir_options: MlirOptions, artifact_options: ArtifactOptions) !void {
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    // Read source file
    const source = std.fs.cwd().readFileAlloc(allocator, file_path, 1024 * 1024) catch |err| {
        try stdout.print("Error reading file {s}: {s}\n", .{ file_path, @errorName(err) });
        return;
    };
    defer allocator.free(source);

    // Front half: lex + parse (ensures we have a valid AST before MLIR)
    var lexer = lib.Lexer.init(allocator, source);
    defer lexer.deinit();

    const tokens = lexer.scanTokens() catch |err| {
        try stdout.print("Lexer error: {s}\n", .{@errorName(err)});
        return;
    };
    defer allocator.free(tokens);

    var arena = lib.ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var parser = lib.Parser.init(tokens, &arena);
    parser.setFileId(1);
    const ast_nodes = parser.parse() catch |err| {
        try stdout.print("Parser error: {s}\n", .{@errorName(err)});
        return;
    };

    // Generate MLIR with advanced options
    try generateMlirOutput(allocator, ast_nodes, file_path, mlir_options, artifact_options);
    try stdout.flush();
}

/// Generate MLIR output with comprehensive options
fn generateMlirOutput(allocator: std.mem.Allocator, ast_nodes: []lib.AstNode, file_path: []const u8, mlir_options: MlirOptions, artifact_options: ArtifactOptions) !void {
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    // Import MLIR modules
    const mlir = @import("mlir/mod.zig");
    const c = @import("mlir/c.zig").c;

    // Create MLIR context
    const h = mlir.ctx.createContext(allocator);
    defer mlir.ctx.destroyContext(h);

    // Choose lowering function based on options
    const lowering_result = if (mlir_options.passes != null or mlir_options.verify or mlir_options.timing or mlir_options.print_ir) blk: {
        // Use advanced lowering with passes
        const PassPipelineConfig = @import("mlir/pass_manager.zig").PassPipelineConfig;
        const PassOptimizationLevel = @import("mlir/pass_manager.zig").OptimizationLevel;
        const IRPrintingConfig = @import("mlir/pass_manager.zig").IRPrintingConfig;

        const opt_level: PassOptimizationLevel = switch (mlir_options.getOptimizationLevel()) {
            .None => .None,
            .Basic => .Basic,
            .Aggressive => .Aggressive,
        };

        const ir_config = IRPrintingConfig{
            .print_before_all = mlir_options.print_ir,
            .print_after_all = mlir_options.print_ir,
            .print_after_change = mlir_options.print_ir,
            .print_after_failure = true,
        };

        var custom_passes = std.ArrayList([]const u8){};
        defer custom_passes.deinit(allocator);

        // Parse custom passes if provided (command-line or build-time default)
        if (mlir_options.getDefaultPasses()) |passes_str| {
            var pass_iter = std.mem.splitSequence(u8, passes_str, ",");
            while (pass_iter.next()) |pass_name| {
                const trimmed = std.mem.trim(u8, pass_name, " \t");
                if (trimmed.len > 0) {
                    try custom_passes.append(allocator, trimmed);
                }
            }
        }

        const pass_config = PassPipelineConfig{
            .optimization_level = opt_level,
            .enable_verification = mlir_options.shouldEnableVerification(),
            .custom_passes = custom_passes.items,
            .enable_timing = mlir_options.shouldEnableTiming(),
            .ir_printing = ir_config,
        };

        if (mlir_options.getDefaultPasses()) |passes_str| {
            // Use pipeline string parsing
            break :blk try mlir.lower.lowerFunctionsToModuleWithPipelineString(h.ctx, ast_nodes, allocator, passes_str);
        } else {
            // Use configuration-based approach
            break :blk try mlir.lower.lowerFunctionsToModuleWithPasses(h.ctx, ast_nodes, allocator, pass_config);
        }
    } else blk: {
        // Use basic lowering
        break :blk try mlir.lower.lowerFunctionsToModuleWithErrors(h.ctx, ast_nodes, allocator);
    };

    // Apply MLIR pipeline if requested
    var final_module = lowering_result.module;
    if (mlir_options.use_pipeline) {
        const opt_level = mlir_options.getOptimizationLevel();
        const pipeline_config = switch (opt_level) {
            .None => mlir_pipeline.no_opt_config,
            .Basic => mlir_pipeline.basic_config,
            .Aggressive => mlir_pipeline.aggressive_config,
        };

        var pipeline_result = mlir_pipeline.runMLIRPipeline(h.ctx, lowering_result.module, pipeline_config, allocator) catch |err| {
            try stdout.print("MLIR pipeline failed: {s}\n", .{@errorName(err)});
            return;
        };
        defer pipeline_result.deinit();

        if (!pipeline_result.success) {
            try stdout.print("MLIR pipeline failed: {s}\n", .{pipeline_result.error_message orelse "Unknown error"});
            return;
        }

        // Pipeline optimization completed successfully
        // The module has been optimized in-place
        final_module = pipeline_result.optimized_module;

        // Report applied passes
        try stdout.print("MLIR pipeline applied {d} passes: ", .{pipeline_result.passes_applied.items.len});
        for (pipeline_result.passes_applied.items, 0..) |pass, i| {
            if (i > 0) try stdout.print(", ", .{});
            try stdout.print("{s}", .{pass});
        }
        try stdout.print("\n", .{});
    }

    // Convert MLIR to Yul
    try stdout.print("Converting MLIR to Yul...\n", .{});
    var yul_result = mlir.yul_lowering.lowerToYul(final_module, h.ctx, allocator) catch |err| {
        try stdout.print("MLIR to Yul conversion failed: {s}\n", .{@errorName(err)});
        return;
    };
    defer yul_result.deinit();

    if (!yul_result.success) {
        try stdout.print("MLIR to Yul conversion failed with {d} errors:\n", .{yul_result.errors.len});
        for (yul_result.errors) |err| {
            try stdout.print("  - {s}\n", .{err});
        }
        return;
    }

    try stdout.print("Generated Yul code ({d} bytes):\n", .{yul_result.yul_code.len});
    try stdout.print("{s}\n", .{yul_result.yul_code});

    // Save Yul if requested
    if (artifact_options.save_yul) {
        try saveYul(allocator, file_path, yul_result.yul_code, artifact_options);
    }

    // Compile Yul to bytecode using the existing Yul backend
    try stdout.print("Compiling Yul to EVM bytecode...\n", .{});
    var yul_compile_result = lib.yul_bindings.YulCompiler.compile(allocator, yul_result.yul_code) catch |err| {
        try stdout.print("Yul compilation failed: {s}\n", .{@errorName(err)});
        return;
    };
    defer yul_compile_result.deinit(allocator);

    if (!yul_compile_result.success) {
        try stdout.print("Yul compilation failed: {?s}\n", .{yul_compile_result.error_message});
        return;
    }

    try stdout.print("Successfully compiled to EVM bytecode!\n", .{});
    if (yul_compile_result.bytecode) |bytecode| {
        try stdout.print("Bytecode: {s}\n", .{bytecode});

        // Save bytecode if requested
        if (artifact_options.save_bytecode) {
            try saveBytecode(allocator, file_path, bytecode, artifact_options);
        }
    } else {
        try stdout.print("No bytecode generated\n", .{});
    }

    // Check for errors
    if (!lowering_result.success) {
        try stdout.print("MLIR lowering failed with {d} errors:\n", .{lowering_result.errors.len});
        for (lowering_result.errors) |err| {
            try stdout.print("  - {s}\n", .{err.message});
            if (err.suggestion) |suggestion| {
                try stdout.print("    Suggestion: {s}\n", .{suggestion});
            }
        }
        return;
    }

    // Print warnings if any
    if (lowering_result.warnings.len > 0) {
        try stdout.print("MLIR lowering completed with {d} warnings:\n", .{lowering_result.warnings.len});
        for (lowering_result.warnings) |warn| {
            try stdout.print("  - {s}\n", .{warn.message});
        }
    }

    // Print pass results if available
    if (lowering_result.pass_result) |pass_result| {
        if (pass_result.success) {
            try stdout.print("Pass pipeline executed successfully\n", .{});
        } else {
            try stdout.print("Pass pipeline failed: {s}\n", .{pass_result.error_message orelse "unknown error"});
        }
    }

    defer c.mlirModuleDestroy(lowering_result.module);

    // Output MLIR
    const callback = struct {
        fn cb(str: c.MlirStringRef, user: ?*anyopaque) callconv(.c) void {
            const w: *std.fs.File = @ptrCast(@alignCast(user.?));
            _ = w.writeAll(str.data[0..str.length]) catch {};
        }
    };

    // Determine output destination
    if (mlir_options.output_dir) |output_dir| {
        // Save to file
        std.fs.cwd().makeDir(output_dir) catch |err| switch (err) {
            error.PathAlreadyExists => {},
            else => return err,
        };

        const basename = std.fs.path.stem(file_path);
        const filename = try std.mem.concat(allocator, u8, &[_][]const u8{ basename, ".mlir" });
        defer allocator.free(filename);
        const output_file = try std.fs.path.join(allocator, &[_][]const u8{ output_dir, filename });
        defer allocator.free(output_file);

        const file = std.fs.cwd().createFile(output_file, .{}) catch |err| {
            try stdout.print("Error creating output file {s}: {s}\n", .{ output_file, @errorName(err) });
            return;
        };
        defer file.close();

        var file_buffer: [4096]u8 = undefined;
        const file_writer = file.writer(&file_buffer);
        const op = c.mlirModuleGetOperation(lowering_result.module);
        c.mlirOperationPrint(op, callback.cb, @constCast(&file_writer));

        try stdout.print("MLIR saved to {s}\n", .{output_file});
    } else {
        // Print to stdout
        var stdout_file = std.fs.File.stdout();

        const op = c.mlirModuleGetOperation(lowering_result.module);
        c.mlirOperationPrint(op, callback.cb, @constCast(&stdout_file));
        try stdout.print("\n", .{});
    }

    // Always save MLIR to artifact directory if requested
    if (artifact_options.save_mlir) {
        const base_name = std.fs.path.stem(file_path);
        const artifact_dir = try artifact_options.createOutputDir(allocator, base_name);
        defer allocator.free(artifact_dir);

        // Create directory if it doesn't exist
        std.fs.cwd().makeDir(artifact_dir) catch |err| switch (err) {
            error.PathAlreadyExists => {},
            else => return err,
        };

        // Write MLIR to file
        const mlir_file = try std.fmt.allocPrint(allocator, "{s}/mlir.mlir", .{artifact_dir});
        defer allocator.free(mlir_file);

        var file = try std.fs.cwd().createFile(mlir_file, .{});
        defer file.close();

        const op = c.mlirModuleGetOperation(lowering_result.module);
        c.mlirOperationPrint(op, callback.cb, @constCast(&file));

        try stdout.print("Saved MLIR to: {s}\n", .{mlir_file});
    }
    try stdout.flush();
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
