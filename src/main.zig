//! Ora Language Compiler CLI
//!
//! Command-line interface for the Ora compiler. Supports lexing, parsing,
//! AST generation, MLIR emission, and full compilation to Yul/bytecode.
//!
//! SECTIONS:
//!   • Main entry & argument parsing
//!   • Usage & help text
//!   • Command handlers (lex, parse, ast, compile)
//!   • Artifact saving functions
//!   • Parser & compilation workflows
//!   • MLIR integration & code generation

const std = @import("std");
const lib = @import("ora_lib");
const build_options = @import("build_options");
const mlir_pipeline = @import("mlir/pass_manager.zig");

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

// ============================================================================
// SECTION 1: Main Entry Point & Argument Parsing
// ============================================================================

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

    // Parse arguments with compiler-style CLI
    var output_dir: ?[]const u8 = null;
    var command: ?[]const u8 = null; // For legacy commands (lex, parse, etc.)
    var input_file: ?[]const u8 = null;

    // Compilation stage control (--emit-X flags)
    var emit_tokens: bool = false;
    var emit_ast: bool = false;
    var emit_mlir: bool = false;
    var emit_yul: bool = false;
    var emit_bytecode: bool = false;
    var analyze_complexity: bool = false;

    // MLIR options
    var mlir_verify: bool = false;
    var mlir_passes: ?[]const u8 = null;
    var mlir_opt_level: ?[]const u8 = null;
    var mlir_timing: bool = false;
    var mlir_print_ir: bool = false;
    var mlir_use_pipeline: bool = false;

    // Artifact saving options (for --save-all)
    var save_tokens: bool = false;
    var save_ast: bool = false;
    var save_mlir: bool = false;
    var save_yul: bool = false;
    var save_bytecode: bool = false;

    // var verbose: bool = false;  // TODO: implement verbose mode
    var i: usize = 1;

    while (i < args.len) {
        if (std.mem.eql(u8, args[i], "-o") or std.mem.eql(u8, args[i], "--output-dir")) {
            if (i + 1 >= args.len) {
                try printUsage();
                return;
            }
            output_dir = args[i + 1];
            i += 2;
            // New --emit-X flags
        } else if (std.mem.eql(u8, args[i], "--emit-tokens")) {
            emit_tokens = true;
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--emit-ast")) {
            emit_ast = true;
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--emit-mlir")) {
            emit_mlir = true;
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--emit-yul")) {
            emit_yul = true;
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--emit-bytecode")) {
            emit_bytecode = true;
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--analyze-complexity")) {
            analyze_complexity = true;
            i += 1;
            // Optimization level flags (-O0, -O1, -O2)
        } else if (std.mem.eql(u8, args[i], "-O0") or std.mem.eql(u8, args[i], "-Onone")) {
            mlir_opt_level = "none";
            i += 1;
        } else if (std.mem.eql(u8, args[i], "-O1") or std.mem.eql(u8, args[i], "-Obasic")) {
            mlir_opt_level = "basic";
            i += 1;
        } else if (std.mem.eql(u8, args[i], "-O2") or std.mem.eql(u8, args[i], "-Oaggressive")) {
            mlir_opt_level = "aggressive";
            i += 1;
            // MLIR options
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
            // Legacy --save-X flags
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
            // Debug/verbose
        } else if (std.mem.eql(u8, args[i], "--verbose") or std.mem.eql(u8, args[i], "-v")) {
            // verbose = true;  // TODO: implement verbose mode
            i += 1;
            // Legacy commands (lex, parse, ast, compile, mlir)
        } else if (std.mem.eql(u8, args[i], "lex") or
            std.mem.eql(u8, args[i], "parse") or
            std.mem.eql(u8, args[i], "ast") or
            std.mem.eql(u8, args[i], "compile") or
            std.mem.eql(u8, args[i], "mlir"))
        {
            command = args[i];
            i += 1;
            // Input file
        } else if (input_file == null and !std.mem.startsWith(u8, args[i], "-")) {
            input_file = args[i];
            i += 1;
        } else {
            try printUsage();
            return;
        }
    }

    // Require input file
    if (input_file == null) {
        try printUsage();
        return;
    }

    const file_path = input_file.?;

    // Handle complexity analysis first (it's a special mode)
    if (analyze_complexity) {
        try runComplexityAnalysis(allocator, file_path);
        return;
    }

    // Determine compilation mode
    // If no --emit-X flag is set and no legacy command, default to bytecode
    if (!emit_tokens and !emit_ast and !emit_mlir and !emit_yul and !emit_bytecode and command == null) {
        emit_bytecode = true; // Default: compile to bytecode
    }

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

    // Handle legacy commands first
    if (command) |cmd| {
        if (std.mem.eql(u8, cmd, "lex")) {
            try runLexer(allocator, file_path, artifact_options);
        } else if (std.mem.eql(u8, cmd, "parse")) {
            try runParser(allocator, file_path, artifact_options);
        } else if (std.mem.eql(u8, cmd, "ast")) {
            try runASTGeneration(allocator, file_path, output_dir, artifact_options);
        } else if (std.mem.eql(u8, cmd, "compile")) {
            if (mlir_options.emit_mlir or artifact_options.save_mlir or artifact_options.save_yul or artifact_options.save_bytecode) {
                try runMlirEmitAdvanced(allocator, file_path, mlir_options, artifact_options);
            } else {
                try runFullCompilation(allocator, file_path, mlir_options, artifact_options);
            }
        } else if (std.mem.eql(u8, cmd, "mlir")) {
            try runMlirEmitAdvancedWithYul(allocator, file_path, mlir_options, artifact_options, false);
        } else {
            try printUsage();
        }
        return;
    }

    // New compiler-style behavior: process --emit-X flags
    // Stop at the earliest stage specified, but save later stages if --save-X is set
    // TODO: Use verbose flag for detailed output

    if (emit_tokens) {
        // Stop after lexer
        try runLexer(allocator, file_path, artifact_options);
    } else if (emit_ast) {
        // Stop after parser
        try runParser(allocator, file_path, artifact_options);
    } else if (emit_mlir or emit_yul or emit_bytecode) {
        // Run full MLIR pipeline (includes Yul and bytecode if needed)
        try runMlirEmitAdvancedWithYul(allocator, file_path, mlir_options, artifact_options, emit_yul or emit_bytecode);
    } else {
        // Default: full compilation
        try runMlirEmitAdvancedWithYul(allocator, file_path, mlir_options, artifact_options, false);
    }
}

// ============================================================================
// SECTION 2: Usage & Help Text
// ============================================================================

fn printUsage() !void {
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;
    try stdout.print("Ora Compiler v0.1 - Asuka\n", .{});
    try stdout.print("Usage: ora [options] <file.ora>\n", .{});
    try stdout.print("\nCompilation Control:\n", .{});
    try stdout.print("  (default)              - Compile to EVM bytecode\n", .{});
    try stdout.print("  --emit-tokens          - Stop after lexical analysis (emit tokens)\n", .{});
    try stdout.print("  --emit-ast             - Stop after parsing (emit AST)\n", .{});
    try stdout.print("  --emit-mlir            - Stop after MLIR generation\n", .{});
    try stdout.print("  --emit-yul             - Stop after Yul lowering\n", .{});
    try stdout.print("  --emit-bytecode        - Generate EVM bytecode (default)\n", .{});
    try stdout.print("\nOutput Options:\n", .{});
    try stdout.print("  -o <file>              - Write output to <file> (e.g., -o out.hex, -o out.mlir)\n", .{});
    try stdout.print("  -o <dir>/              - Write artifacts to <dir>/ (e.g., -o build/)\n", .{});
    try stdout.print("                           Default: ./<basename>_artifacts/ or current dir\n", .{});
    try stdout.print("\nOptimization Options:\n", .{});
    try stdout.print("  -O0, -Onone            - No optimization (default)\n", .{});
    try stdout.print("  -O1, -Obasic           - Basic optimizations\n", .{});
    try stdout.print("  -O2, -Oaggressive      - Aggressive optimizations\n", .{});
    try stdout.print("\nMLIR Options:\n", .{});
    try stdout.print("  --mlir-verify          - Run MLIR verification passes\n", .{});
    try stdout.print("  --mlir-passes <passes> - Custom MLIR pass pipeline (comma-separated)\n", .{});
    try stdout.print("  --mlir-timing          - Enable pass timing statistics\n", .{});
    try stdout.print("  --mlir-print-ir        - Print IR before and after passes\n", .{});
    try stdout.print("  --mlir-pipeline        - Use comprehensive MLIR optimization pipeline\n", .{});
    try stdout.print("\nAnalysis Options:\n", .{});
    try stdout.print("  --analyze-complexity   - Analyze function complexity metrics\n", .{});
    try stdout.print("\nDevelopment/Debug Options:\n", .{});
    try stdout.print("  --save-all             - Save all intermediate artifacts\n", .{});
    try stdout.print("  --verbose              - Verbose output (show each compilation stage)\n", .{});
    try stdout.print("\nLegacy Commands (for step-by-step debugging):\n", .{});
    try stdout.print("  ora lex <file>         - Only tokenize (legacy: use --emit-tokens)\n", .{});
    try stdout.print("  ora parse <file>       - Only parse (legacy: use --emit-ast)\n", .{});
    try stdout.print("  ora ast <file>         - Generate AST JSON (legacy)\n", .{});
    try stdout.print("  ora compile <file>     - Full compilation (legacy: just use 'ora <file>')\n", .{});
    try stdout.print("\nExamples:\n", .{});
    try stdout.print("  ora contract.ora                      # Compile to bytecode → contract_artifacts/\n", .{});
    try stdout.print("  ora contract.ora -o build/            # Compile to bytecode → build/\n", .{});
    try stdout.print("  ora contract.ora -o contract.hex      # Compile to specific file\n", .{});
    try stdout.print("  ora --emit-ast contract.ora           # Stop at AST → contract_artifacts/ast.json\n", .{});
    try stdout.print("  ora --emit-mlir -O2 contract.ora      # Generate optimized MLIR\n", .{});
    try stdout.print("  ora --emit-yul -o out.yul contract.ora  # Compile to Yul\n", .{});
    try stdout.print("  ora --save-all contract.ora           # Save all stages (tokens, AST, MLIR, Yul, bytecode)\n", .{});
    try stdout.print("  ora --mlir-verify --mlir-timing -O2 contract.ora  # Compile with verification and timing\n", .{});
    try stdout.print("  ora --analyze-complexity contract.ora  # Analyze function complexity\n", .{});
    try stdout.flush();
}

// ============================================================================
// SECTION 3: Command Handlers (lex, parse, ast, compile)
// ============================================================================

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

// ============================================================================
// SECTION 4: Artifact Saving Functions
// ============================================================================

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

// ============================================================================
// SECTION 5: Parser & Compilation Workflows
// ============================================================================

/// Run parser on file and display AST
fn runParser(allocator: std.mem.Allocator, file_path: []const u8, artifact_options: ArtifactOptions) !void {
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
        try stdout.flush();
        std.process.exit(1);
    };
    defer allocator.free(tokens);

    try stdout.print("Lexed {d} tokens\n", .{tokens.len});

    // Run parser
    var arena = lib.ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var parser = lib.Parser.init(tokens, &arena);
    parser.setFileId(1);
    const ast_nodes = parser.parse() catch |err| {
        try stdout.print("Parser error: {s}\n", .{@errorName(err)});
        try stdout.flush();
        std.process.exit(1);
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

    try stdout.flush();
}

/// Run complexity analysis on all functions in the contract
fn runComplexityAnalysis(allocator: std.mem.Allocator, file_path: []const u8) !void {
    const complexity = lib.complexity;

    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    // Read source file
    const source = std.fs.cwd().readFileAlloc(allocator, file_path, 1024 * 1024) catch |err| {
        try stdout.print("Error reading file {s}: {s}\n", .{ file_path, @errorName(err) });
        return;
    };
    defer allocator.free(source);

    try stdout.print("Analyzing complexity for {s}\n", .{file_path});
    try stdout.print("==================================================\n", .{});

    // Run lexer
    var lexer = lib.Lexer.init(allocator, source);
    defer lexer.deinit();

    const tokens = lexer.scanTokens() catch |err| {
        try stdout.print("Lexer error: {s}\n", .{@errorName(err)});
        try stdout.flush();
        std.process.exit(1);
    };
    defer allocator.free(tokens);

    // Run parser
    var arena = lib.ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var parser = lib.Parser.init(tokens, &arena);
    parser.setFileId(1);
    const ast_nodes = parser.parse() catch |err| {
        try stdout.print("Parser error: {s}\n", .{@errorName(err)});
        try stdout.flush();
        std.process.exit(1);
    };

    try stdout.print("\n", .{});

    // Analyze each function
    var analyzer = complexity.ComplexityAnalyzer.init(allocator);
    var function_count: u32 = 0;
    var total_complexity: u32 = 0;
    var simple_count: u32 = 0;
    var moderate_count: u32 = 0;
    var complex_count: u32 = 0;

    for (ast_nodes) |*node| {
        switch (node.*) {
            .Contract => |*contract| {
                try stdout.print("Contract: {s}\n", .{contract.name});
                try stdout.print("{s}\n", .{"─" ** 50});

                for (contract.body) |*body_node| {
                    if (body_node.* == .Function) {
                        const func = &body_node.Function;
                        function_count += 1;

                        const metrics = analyzer.analyzeFunction(func);
                        total_complexity += metrics.node_count;

                        // Categorize
                        if (metrics.isSimple()) {
                            simple_count += 1;
                        } else if (metrics.isComplex()) {
                            complex_count += 1;
                        } else {
                            moderate_count += 1;
                        }

                        // Format visibility
                        const visibility = if (func.visibility == .Public) "pub " else "";
                        const inline_marker = if (func.is_inline) "inline " else "";

                        // Complexity rating
                        const rating = if (metrics.isSimple())
                            "✓ Simple"
                        else if (metrics.isComplex())
                            "✗ Complex"
                        else
                            "○ Moderate";

                        try stdout.print("\n{s}{s}fn {s}()\n", .{ visibility, inline_marker, func.name });
                        try stdout.print("  Complexity: {s}\n", .{rating});
                        try stdout.print("  Nodes:      {d}\n", .{metrics.node_count});
                        try stdout.print("  Statements: {d}\n", .{metrics.statement_count});
                        try stdout.print("  Max Depth:  {d}\n", .{metrics.max_depth});

                        // Warning for inline functions
                        if (func.is_inline and metrics.isComplex()) {
                            try stdout.print("  ⚠️  WARNING: Function marked 'inline' but has high complexity!\n", .{});
                            try stdout.print("      Consider removing 'inline' or refactoring.\n", .{});
                        }

                        // Recommendations
                        if (metrics.isComplex()) {
                            try stdout.print("  💡 Recommendation: Consider breaking into smaller functions\n", .{});
                        } else if (metrics.isSimple() and !func.is_inline and func.visibility == .Private) {
                            try stdout.print("  💡 Tip: This function is a good candidate for 'inline'\n", .{});
                        }
                    }
                }
            },
            .Function => |*func| {
                // Top-level function (rare, but handle it)
                function_count += 1;
                const metrics = analyzer.analyzeFunction(func);
                total_complexity += metrics.node_count;

                if (metrics.isSimple()) {
                    simple_count += 1;
                } else if (metrics.isComplex()) {
                    complex_count += 1;
                } else {
                    moderate_count += 1;
                }

                try stdout.print("\nFunction: {s}\n", .{func.name});
                try stdout.print("  Nodes:      {d}\n", .{metrics.node_count});
                try stdout.print("  Statements: {d}\n", .{metrics.statement_count});
                try stdout.print("  Max Depth:  {d}\n", .{metrics.max_depth});
            },
            else => {},
        }
    }

    // Summary
    try stdout.print("\n{s}\n", .{"═" ** 50});
    try stdout.print("SUMMARY\n", .{});
    try stdout.print("{s}\n", .{"═" ** 50});
    try stdout.print("Total Functions:    {d}\n", .{function_count});
    try stdout.print("  ✓ Simple:         {d} ({d}%)\n", .{ simple_count, if (function_count > 0) simple_count * 100 / function_count else 0 });
    try stdout.print("  ○ Moderate:       {d} ({d}%)\n", .{ moderate_count, if (function_count > 0) moderate_count * 100 / function_count else 0 });
    try stdout.print("  ✗ Complex:        {d} ({d}%)\n", .{ complex_count, if (function_count > 0) complex_count * 100 / function_count else 0 });
    try stdout.print("Average Complexity: {d} nodes/function\n", .{if (function_count > 0) total_complexity / function_count else 0});

    try stdout.print("\nComplexity Thresholds:\n", .{});
    try stdout.print("  Simple:   < 20 nodes (good for inline)\n", .{});
    try stdout.print("  Moderate: 20-100 nodes\n", .{});
    try stdout.print("  Complex:  > 100 nodes (not recommended for inline)\n", .{});

    try stdout.flush();
}

/// Run full compilation pipeline with optional MLIR support
fn runFullCompilation(allocator: std.mem.Allocator, file_path: []const u8, mlir_options: MlirOptions, artifact_options: ArtifactOptions) !void {
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

    // Phase 3: MLIR Generation (if requested)
    if (mlir_options.emit_mlir) {
        try stdout.print("Phase 3: MLIR Generation\n", .{});
        try generateMlirOutput(allocator, ast_nodes, file_path, mlir_options, artifact_options, true);
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
fn runASTGeneration(allocator: std.mem.Allocator, file_path: []const u8, output_dir: ?[]const u8, artifact_options: ArtifactOptions) !void {
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
        try stdout.flush();
        std.process.exit(1);
    };
    defer allocator.free(tokens);

    var arena = lib.ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var parser = lib.Parser.init(tokens, &arena);
    parser.setFileId(1);
    const ast_nodes = parser.parse() catch |err| {
        try stdout.print("Parser error: {s}\n", .{@errorName(err)});
        try stdout.flush();
        std.process.exit(1);
    };
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

// ============================================================================
// SECTION 6: MLIR Integration & Code Generation
// ============================================================================

/// Advanced MLIR emission with full pass pipeline support
fn runMlirEmitAdvanced(allocator: std.mem.Allocator, file_path: []const u8, mlir_options: MlirOptions, artifact_options: ArtifactOptions) !void {
    try runMlirEmitAdvancedWithYul(allocator, file_path, mlir_options, artifact_options, true);
}

fn runMlirEmitAdvancedWithYul(allocator: std.mem.Allocator, file_path: []const u8, mlir_options: MlirOptions, artifact_options: ArtifactOptions, generate_yul: bool) !void {
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
        try stdout.flush();
        std.process.exit(1);
    };
    defer allocator.free(tokens);

    var arena = lib.ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var parser = lib.Parser.init(tokens, &arena);
    parser.setFileId(1);
    const ast_nodes = parser.parse() catch |err| {
        try stdout.print("Parser error: {s}\n", .{@errorName(err)});
        try stdout.flush();
        std.process.exit(1);
    };

    // Generate MLIR with advanced options
    try generateMlirOutput(allocator, ast_nodes, file_path, mlir_options, artifact_options, generate_yul);
    try stdout.flush();
}

/// Generate MLIR output with comprehensive options
fn generateMlirOutput(allocator: std.mem.Allocator, ast_nodes: []lib.AstNode, file_path: []const u8, mlir_options: MlirOptions, artifact_options: ArtifactOptions, generate_yul: bool) !void {
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    // Import MLIR modules
    const mlir = @import("mlir/mod.zig");
    const c = @import("mlir/c.zig").c;

    // Create arena allocator for MLIR lowering phase
    // This arena will be freed after MLIR generation completes
    var mlir_arena = std.heap.ArenaAllocator.init(allocator);
    defer mlir_arena.deinit();
    const mlir_allocator = mlir_arena.allocator();

    // Create MLIR context
    const h = mlir.createContext(mlir_allocator);
    defer mlir.destroyContext(h);

    // Choose lowering function based on options
    var lowering_result = if (mlir_options.passes != null or mlir_options.verify or mlir_options.timing or mlir_options.print_ir) blk: {
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
        defer custom_passes.deinit(mlir_allocator);

        // Parse custom passes if provided (command-line or build-time default)
        if (mlir_options.getDefaultPasses()) |passes_str| {
            var pass_iter = std.mem.splitSequence(u8, passes_str, ",");
            while (pass_iter.next()) |pass_name| {
                const trimmed = std.mem.trim(u8, pass_name, " \t");
                if (trimmed.len > 0) {
                    try custom_passes.append(mlir_allocator, trimmed);
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

        // Extract filename for location tracking
        const source_filename = std.fs.path.basename(file_path);

        if (mlir_options.getDefaultPasses()) |passes_str| {
            // Use pipeline string parsing
            break :blk try mlir.lower.lowerFunctionsToModuleWithPipelineString(h.ctx, ast_nodes, mlir_allocator, passes_str, source_filename);
        } else {
            // Use configuration-based approach
            break :blk try mlir.lower.lowerFunctionsToModuleWithPasses(h.ctx, ast_nodes, mlir_allocator, pass_config, source_filename);
        }
    } else blk: {
        // Extract filename for location tracking
        const source_filename = std.fs.path.basename(file_path);

        // Use basic lowering
        break :blk try mlir.lower.lowerFunctionsToModuleWithErrors(h.ctx, ast_nodes, mlir_allocator, source_filename);
    };
    defer lowering_result.deinit(mlir_allocator);

    // Check for errors first, before proceeding to YUL generation
    if (!lowering_result.success) {
        try stdout.print("MLIR lowering failed with {d} errors:\n", .{lowering_result.errors.len});
        for (lowering_result.errors) |err| {
            try stdout.print("  - {s}\n", .{err.message});
            if (err.suggestion) |suggestion| {
                try stdout.print("    Suggestion: {s}\n", .{suggestion});
            }
        }
        try stdout.flush();
        std.process.exit(1);
    }
    // Apply MLIR pipeline if requested
    var final_module = lowering_result.module;
    if (mlir_options.use_pipeline) {
        const opt_level = mlir_options.getOptimizationLevel();
        const pipeline_config = switch (opt_level) {
            .None => mlir_pipeline.no_opt_config,
            .Basic => mlir_pipeline.basic_config,
            .Aggressive => mlir_pipeline.aggressive_config,
        };

        var pipeline_result = mlir_pipeline.runMLIRPipeline(h.ctx, lowering_result.module, pipeline_config, mlir_allocator) catch |err| {
            try stdout.print("MLIR pipeline failed: {s}\n", .{@errorName(err)});
            return;
        };
        defer pipeline_result.deinit(mlir_allocator);

        if (!pipeline_result.success) {
            try stdout.print("MLIR pipeline failed: {s}\n", .{pipeline_result.error_message orelse "Unknown error"});
            try stdout.flush();
            std.process.exit(1);
        }

        // Pipeline optimization completed successfully
        // The module has been optimized in-place
        final_module = pipeline_result.optimized_module;

        // Report applied passes (only if generating YUL, otherwise keep output clean)
        if (generate_yul) {
            try stdout.print("MLIR pipeline applied {d} passes: ", .{pipeline_result.passes_applied.items.len});
            for (pipeline_result.passes_applied.items, 0..) |pass, i| {
                if (i > 0) try stdout.print(", ", .{});
                try stdout.print("{s}", .{pass});
            }
            try stdout.print("\n", .{});
        }
    }

    // Convert MLIR to Yul (only if requested)
    if (generate_yul) {
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
            try stdout.flush();
            std.process.exit(1);
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
            try stdout.flush();
            std.process.exit(1);
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

        // Create printing flags to enable location information
        const flags = c.mlirOpPrintingFlagsCreate();
        defer c.mlirOpPrintingFlagsDestroy(flags);
        c.mlirOpPrintingFlagsEnableDebugInfo(flags, true, false);

        c.mlirOperationPrintWithFlags(op, flags, callback.cb, @constCast(&stdout_file));
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
