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

/// MLIR-related command line options
const MlirOptions = struct {
    emit_mlir: bool,
    verify: bool,
    passes: ?[]const u8,
    opt_level: ?[]const u8,
    timing: bool,
    print_ir: bool,
    output_dir: ?[]const u8,

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
    };

    if (std.mem.eql(u8, cmd, "lex")) {
        try runLexer(allocator, file_path);
    } else if (std.mem.eql(u8, cmd, "parse")) {
        try runParser(allocator, file_path, !no_cst);
    } else if (std.mem.eql(u8, cmd, "ast")) {
        try runASTGeneration(allocator, file_path, output_dir, !no_cst);
    } else if (std.mem.eql(u8, cmd, "compile")) {
        try runFullCompilation(allocator, file_path, !no_cst, mlir_options);
    } else if (std.mem.eql(u8, cmd, "mlir")) {
        try runMlirEmitAdvanced(allocator, file_path, mlir_options);
    } else {
        try printUsage();
    }
}

fn printUsage() !void {
    const stdout = std.io.getStdOut().writer();
    try stdout.print("Ora Compiler v0.1\n", .{});
    try stdout.print("Usage: ora [options] <command> <file>\n", .{});
    try stdout.print("\nGeneral Options:\n", .{});
    try stdout.print("  -o, --output-dir <dir>  - Specify output directory for generated files\n", .{});
    try stdout.print("      --no-cst            - Disable CST building (enabled by default)\n", .{});
    try stdout.print("\nMLIR Options:\n", .{});
    try stdout.print("      --emit-mlir         - Generate MLIR output in addition to normal compilation\n", .{});
    try stdout.print("      --mlir-verify       - Run MLIR verification passes\n", .{});
    try stdout.print("      --mlir-passes <str> - Custom MLIR pass pipeline (e.g., 'canonicalize,cse')\n", .{});
    try stdout.print("      --mlir-opt <level>  - Optimization level: none, basic, aggressive\n", .{});
    try stdout.print("      --mlir-timing       - Enable pass timing statistics\n", .{});
    try stdout.print("      --mlir-print-ir     - Print IR before and after passes\n", .{});
    try stdout.print("\nCommands:\n", .{});
    try stdout.print("  lex <file>     - Tokenize a .ora file\n", .{});
    try stdout.print("  parse <file>   - Parse a .ora file to AST\n", .{});
    try stdout.print("  ast <file>     - Generate AST and save to JSON file\n", .{});
    try stdout.print("  compile <file> - Full frontend pipeline (lex -> parse -> [mlir])\n", .{});
    try stdout.print("  mlir <file>    - Run front-end and emit MLIR with advanced options\n", .{});
    try stdout.print("\nExamples:\n", .{});
    try stdout.print("  ora -o build ast example.ora\n", .{});
    try stdout.print("  ora --emit-mlir compile example.ora\n", .{});
    try stdout.print("  ora --mlir-opt aggressive --mlir-verify mlir example.ora\n", .{});
    try stdout.print("  ora --mlir-passes 'canonicalize,cse,sccp' --mlir-timing mlir example.ora\n", .{});
}

/// Run lexer on file and display tokens
fn runLexer(allocator: std.mem.Allocator, file_path: []const u8) !void {
    const stdout = std.io.getStdOut().writer();

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
}

/// Run parser on file and display AST
fn runParser(allocator: std.mem.Allocator, file_path: []const u8, enable_cst: bool) !void {
    const stdout = std.io.getStdOut().writer();

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

    if (enable_cst) {
        if (cst_builder_ptr) |builder| {
            const cst_root = try builder.buildRoot(tokens);
            _ = cst_root; // TODO: optional dump in future flag
            builder.deinit();
        }
    }
}

/// Run full compilation pipeline with optional MLIR support
fn runFullCompilation(allocator: std.mem.Allocator, file_path: []const u8, enable_cst: bool, mlir_options: MlirOptions) !void {
    const stdout = std.io.getStdOut().writer();

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
        try generateMlirOutput(allocator, ast_nodes, file_path, mlir_options);
    }

    try stdout.print("Frontend compilation completed successfully!\n", .{});
    try stdout.print("Pipeline: {d} tokens -> {d} AST nodes", .{ tokens.len, ast_nodes.len });
    if (mlir_options.emit_mlir) {
        try stdout.print(" -> MLIR module", .{});
    }
    try stdout.print("\n", .{});
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
fn runASTGeneration(allocator: std.mem.Allocator, file_path: []const u8, output_dir: ?[]const u8, enable_cst: bool) !void {
    const stdout = std.io.getStdOut().writer();

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

    const writer = file.writer();
    lib.ast.AstSerializer.serializeAST(ast_nodes, writer) catch |err| {
        try stdout.print("Error serializing AST: {s}\n", .{@errorName(err)});
        return;
    };

    try stdout.print("AST saved to {s}\n", .{output_file});
}

/// Advanced MLIR emission with full pass pipeline support
fn runMlirEmitAdvanced(allocator: std.mem.Allocator, file_path: []const u8, mlir_options: MlirOptions) !void {
    const stdout = std.io.getStdOut().writer();

    // Read source file
    const source = std.fs.cwd().readFileAlloc(allocator, file_path, 1024 * 1024) catch |err| {
        try stdout.print("Error reading file {s}: {s}\n", .{ file_path, @errorName(err) });
        return;
    };
    defer allocator.free(source);

    try stdout.print("Advanced MLIR compilation for {s}\n", .{file_path});
    try stdout.print("============================================================\n", .{});

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

    try stdout.print("Parsed {d} AST nodes\n", .{ast_nodes.len});

    // Generate MLIR with advanced options
    try generateMlirOutput(allocator, ast_nodes, file_path, mlir_options);
}

/// Generate MLIR output with comprehensive options
fn generateMlirOutput(allocator: std.mem.Allocator, ast_nodes: []lib.AstNode, file_path: []const u8, mlir_options: MlirOptions) !void {
    const stdout = std.io.getStdOut().writer();

    // Import MLIR modules
    const mlir = @import("mlir/mod.zig");
    const c = @import("mlir/c.zig").c;

    // Create MLIR context
    const h = mlir.ctx.createContext();
    defer mlir.ctx.destroyContext(h);

    try stdout.print("Lowering AST to MLIR...\n", .{});

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

        var custom_passes = std.ArrayList([]const u8).init(allocator);
        defer custom_passes.deinit();

        // Parse custom passes if provided (command-line or build-time default)
        if (mlir_options.getDefaultPasses()) |passes_str| {
            var pass_iter = std.mem.splitSequence(u8, passes_str, ",");
            while (pass_iter.next()) |pass_name| {
                const trimmed = std.mem.trim(u8, pass_name, " \t");
                if (trimmed.len > 0) {
                    try custom_passes.append(trimmed);
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
        fn cb(str: c.MlirStringRef, user: ?*anyopaque) callconv(.C) void {
            const W = std.fs.File.Writer;
            const w_const: *const W = @ptrCast(@alignCast(user.?));
            const w: *W = @constCast(w_const);
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

        const file_writer = file.writer();
        const op = c.mlirModuleGetOperation(lowering_result.module);
        c.mlirOperationPrint(op, callback.cb, @constCast(&file_writer));

        try stdout.print("MLIR saved to {s}\n", .{output_file});
    } else {
        // Print to stdout
        try stdout.print("=== MLIR Output ===\n", .{});
        const op = c.mlirModuleGetOperation(lowering_result.module);
        c.mlirOperationPrint(op, callback.cb, @constCast(&stdout));
        try stdout.print("\n", .{});
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
