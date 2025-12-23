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

/// MLIR-related command line options
const MlirOptions = struct {
    emit_mlir: bool,
    emit_mlir_sir: bool,
    opt_level: ?[]const u8,
    output_dir: ?[]const u8,
    canonicalize: bool = true,

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
    var input_file: ?[]const u8 = null;

    // Compilation stage control (--emit-X flags)
    var emit_tokens: bool = false;
    var emit_ast: bool = false;
    var emit_mlir: bool = false;
    var emit_mlir_sir: bool = false;
    var emit_cfg: bool = false;
    var canonicalize_mlir: bool = true;
    var analyze_state: bool = false;

    // MLIR options
    var mlir_opt_level: ?[]const u8 = null;

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
        } else if (std.mem.eql(u8, args[i], "--emit-mlir-sir")) {
            emit_mlir_sir = true;
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--emit-cfg")) {
            emit_cfg = true;
            i += 1;
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--analyze-state")) {
            analyze_state = true;
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
        } else if (std.mem.eql(u8, args[i], "--no-validate-mlir")) {
            // Validation is always enabled for MLIR (legacy flag, kept for compatibility)
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--no-canonicalize")) {
            canonicalize_mlir = false;
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

    // Handle state analysis (also a special analysis mode)
    if (analyze_state) {
        try runStateAnalysis(allocator, file_path);
        return;
    }

    // Determine compilation mode
    // If no --emit-X flag is set, default to MLIR generation
    if (!emit_tokens and !emit_ast and !emit_mlir and !emit_mlir_sir and !emit_cfg) {
        emit_mlir = true; // Default: emit MLIR
    }

    // Create MLIR options structure
    const mlir_options = MlirOptions{
        .emit_mlir = emit_mlir,
        .emit_mlir_sir = emit_mlir_sir,
        .opt_level = mlir_opt_level,
        .output_dir = output_dir,
        .canonicalize = canonicalize_mlir,
    };

    // Handle CFG generation (uses MLIR's built-in view-op-graph pass)
    if (emit_cfg) {
        try runCFGGeneration(allocator, file_path, mlir_options);
        return;
    }

    // Modern compiler-style behavior: process --emit-X flags
    // Stop at the earliest stage specified

    if (emit_tokens) {
        // Stop after lexer
        try runLexer(allocator, file_path);
    } else if (emit_ast) {
        // Stop after parser
        try runParser(allocator, file_path);
    } else if (emit_mlir) {
        // Run full MLIR pipeline (Ora MLIR)
        try runMlirEmitAdvanced(allocator, file_path, mlir_options);
    } else {
        // Default: emit MLIR
        try runMlirEmitAdvanced(allocator, file_path, mlir_options);
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
    try stdout.print("  (default)              - Emit MLIR\n", .{});
    try stdout.print("  --emit-tokens          - Stop after lexical analysis (emit tokens)\n", .{});
    try stdout.print("  --emit-ast             - Stop after parsing (emit AST)\n", .{});
    try stdout.print("  --emit-mlir            - Emit Ora MLIR (default)\n", .{});
    try stdout.print("  --emit-mlir-sir        - Emit SIR MLIR (after conversion)\n", .{});
    try stdout.print("  --emit-cfg             - Generate control flow graph (Graphviz DOT format)\n", .{});
    try stdout.print("\nOutput Options:\n", .{});
    try stdout.print("  -o <file>              - Write output to <file> (e.g., -o out.hex, -o out.mlir)\n", .{});
    try stdout.print("  -o <dir>/              - Write artifacts to <dir>/ (e.g., -o build/)\n", .{});
    try stdout.print("                           Default: ./<basename>_artifacts/ or current dir\n", .{});
    try stdout.print("\nOptimization Options:\n", .{});
    try stdout.print("  -O0, -Onone            - No optimization (default)\n", .{});
    try stdout.print("  -O1, -Obasic           - Basic optimizations\n", .{});
    try stdout.print("  -O2, -Oaggressive      - Aggressive optimizations\n", .{});
    try stdout.print("\nMLIR Options:\n", .{});
    try stdout.print("  --no-validate-mlir     - Disable automatic MLIR validation (not recommended)\n", .{});
    try stdout.print("  --no-canonicalize      - Skip Ora MLIR canonicalization pass\n", .{});
    try stdout.print("\nAnalysis Options:\n", .{});
    try stdout.print("  --analyze-state        - Analyze storage reads/writes per function\n", .{});
    try stdout.flush();
}

// ============================================================================
// SECTION 3: Command Handlers (lex, parse, ast, compile)
// ============================================================================

/// Run lexer on file and display tokens
fn runLexer(allocator: std.mem.Allocator, file_path: []const u8) !void {
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

    try stdout.flush();
}

// ============================================================================
// SECTION 5: Parser & Compilation Workflows
// ============================================================================

/// Run parser on file and display AST
fn runParser(allocator: std.mem.Allocator, file_path: []const u8) !void {
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

    // Run parser - use parseWithArena to keep arena alive for AST printing
    var parse_result = lib.parser.parseWithArena(allocator, tokens) catch |err| {
        try stdout.print("Parser error: {s}\n", .{@errorName(err)});
        try stdout.flush();
        std.process.exit(1);
    };
    defer parse_result.arena.deinit(); // Keep arena alive until after printing
    const ast_nodes = parse_result.nodes;

    try stdout.print("Generated {d} AST nodes\n\n", .{ast_nodes.len});

    // Display AST summary
    for (ast_nodes, 0..) |*node, i| {
        try stdout.print("[{d}] ", .{i});
        try printAstSummary(stdout, node, 0);
    }

    try stdout.flush();
}

/// Run state analysis on AST nodes (used during normal compilation)
fn runStateAnalysisForContracts(allocator: std.mem.Allocator, ast_nodes: []lib.AstNode) !void {
    const state_tracker = lib.state_tracker;

    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    // Analyze each contract
    for (ast_nodes) |*node| {
        switch (node.*) {
            .Contract => |*contract| {
                // Run state analysis on this contract
                var contract_analysis = state_tracker.analyzeContract(allocator, contract) catch |err| {
                    try stdout.print("State analysis error: {s}\n", .{@errorName(err)});
                    continue;
                };
                defer contract_analysis.deinit();

                // Print only warnings during compilation (not full analysis)
                try state_tracker.printWarnings(stdout, &contract_analysis);
            },
            else => {},
        }
    }

    try stdout.flush();
}

/// Run state analysis on all functions in the contract (standalone mode with --analyze-state)
fn runStateAnalysis(allocator: std.mem.Allocator, file_path: []const u8) !void {
    const state_tracker = lib.state_tracker;

    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    // Read source file
    const source = std.fs.cwd().readFileAlloc(allocator, file_path, 1024 * 1024) catch |err| {
        try stdout.print("Error reading file {s}: {s}\n", .{ file_path, @errorName(err) });
        return;
    };
    defer allocator.free(source);

    try stdout.print("Analyzing state changes for {s}\n", .{file_path});
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

    // Analyze each contract
    for (ast_nodes) |*node| {
        switch (node.*) {
            .Contract => |*contract| {
                // Run state analysis on this contract
                var contract_analysis = state_tracker.analyzeContract(allocator, contract) catch |err| {
                    try stdout.print("State analysis error: {s}\n", .{@errorName(err)});
                    continue;
                };
                defer contract_analysis.deinit();

                // Print results
                try state_tracker.printAnalysis(stdout, &contract_analysis);
            },
            else => {},
        }
    }

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
            // Print function body statements with indentation
            for (function.body.statements) |*stmt| {
                switch (stmt.*) {
                    .VariableDecl => |*var_decl| {
                        var stmt_indent: u32 = 0;
                        while (stmt_indent < indent + 1) : (stmt_indent += 1) {
                            try writer.print("  ", .{});
                        }
                        const mutability = switch (var_decl.kind) {
                            .Var => "var ",
                            .Let => "let ",
                            .Const => "const ",
                            .Immutable => "immutable ",
                        };
                        try writer.print("Variable {s}{s}'{s}'", .{ @tagName(var_decl.region), mutability, var_decl.name });
                        if (var_decl.type_info.ora_type) |ora_type| {
                            try writer.print(" : {s}", .{@tagName(ora_type)});
                        } else if (var_decl.type_info.category != .Unknown) {
                            try writer.print(" : {s}", .{@tagName(var_decl.type_info.category)});
                        } else {
                            try writer.print(" : <unresolved>", .{});
                        }
                        try writer.print("\n", .{});
                    },
                    else => {},
                }
            }
        },
        .VariableDecl => |*var_decl| {
            const mutability = switch (var_decl.kind) {
                .Var => "var ",
                .Let => "let ",
                .Const => "const ",
                .Immutable => "immutable ",
            };
            try writer.print("Variable {s}{s}'{s}'", .{ @tagName(var_decl.region), mutability, var_decl.name });
            if (var_decl.type_info.ora_type) |ora_type| {
                try writer.print(" : {s}", .{@tagName(ora_type)});
            } else if (var_decl.type_info.category != .Unknown) {
                try writer.print(" : {s}", .{@tagName(var_decl.type_info.category)});
            } else {
                try writer.print(" : <unresolved>", .{});
            }
            try writer.print("\n", .{});
        },
        .LogDecl => |*log_decl| {
            try writer.print("Log '{s}' ({d} fields)\n", .{ log_decl.name, log_decl.fields.len });
        },
        else => {
            try writer.print("AST Node\n", .{});
        },
    }
}

/// Generate Control Flow Graph using MLIR's built-in view-op-graph pass
fn runCFGGeneration(allocator: std.mem.Allocator, file_path: []const u8, mlir_options: MlirOptions) !void {
    const mlir = @import("mlir/mod.zig");
    const cfg_gen = @import("mlir/cfg.zig");
    const c = @import("mlir/c.zig").c;

    // First generate MLIR
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    // Read source file
    const source = std.fs.cwd().readFileAlloc(allocator, file_path, 1024 * 1024) catch |err| {
        try stdout.print("Error reading file {s}: {s}\n", .{ file_path, @errorName(err) });
        return;
    };
    defer allocator.free(source);

    // Parse to AST
    var lexer = lib.Lexer.init(allocator, source);
    defer lexer.deinit();
    const tokens = lexer.scanTokens() catch |err| {
        try stdout.print("Lexer error: {s}\n", .{@errorName(err)});
        return;
    };
    defer allocator.free(tokens);

    const ast_nodes = lib.parser.parse(allocator, tokens) catch |err| {
        try stdout.print("Parser error: {s}\n", .{@errorName(err)});
        return;
    };

    // Generate MLIR
    var mlir_arena = std.heap.ArenaAllocator.init(allocator);
    defer mlir_arena.deinit();
    const mlir_allocator = mlir_arena.allocator();

    const h = mlir.createContext(mlir_allocator);
    defer mlir.destroyContext(h);

    const source_filename = std.fs.path.basename(file_path);
    var lowering_result = try mlir.lower.lowerFunctionsToModuleWithErrors(h.ctx, ast_nodes, mlir_allocator, source_filename);
    defer lowering_result.deinit(mlir_allocator);

    if (!lowering_result.success) {
        try stdout.print("MLIR lowering failed\n", .{});
        return;
    }

    // Get MLIR as text by printing the module operation
    const module_op = c.mlirModuleGetOperation(lowering_result.module);
    var mlir_text_buffer = std.ArrayList(u8){};
    defer mlir_text_buffer.deinit(mlir_allocator);

    const PrintCallback = struct {
        buffer: *std.ArrayList(u8),
        allocator: std.mem.Allocator,
        fn callback(message: c.MlirStringRef, userData: ?*anyopaque) callconv(.c) void {
            const self = @as(*@This(), @ptrCast(@alignCast(userData)));
            const message_slice = message.data[0..message.length];
            self.buffer.appendSlice(self.allocator, message_slice) catch {};
        }
    };

    var callback = PrintCallback{ .buffer = &mlir_text_buffer, .allocator = mlir_allocator };
    c.mlirOperationPrint(module_op, PrintCallback.callback, @ptrCast(&callback));

    const mlir_text = try mlir_text_buffer.toOwnedSlice(mlir_allocator);
    defer mlir_allocator.free(mlir_text);

    // Convert Ora MLIR to SIR MLIR before generating CFG
    if (!c.oraConvertToSIR(h.ctx, lowering_result.module)) {
        try stdout.print("Error: Ora to SIR conversion failed\n", .{});
        try stdout.flush();
        std.process.exit(1);
    }

    // Use MLIR C++ API to generate CFG with dialect properly registered
    const dot_content = cfg_gen.generateCFG(h.ctx, lowering_result.module, mlir_allocator) catch |err| {
        try stdout.print("Failed to generate CFG: {s}\n", .{@errorName(err)});
        try stdout.print("Note: The view-op-graph pass may need the module to be in a specific format.\n", .{});
        return;
    };
    defer mlir_allocator.free(dot_content);

    // Output DOT content
    try stdout.print("{s}", .{dot_content});
    try stdout.flush();

    // Save to file if output directory specified
    if (mlir_options.output_dir) |output_dir| {
        // Create output directory if it doesn't exist
        std.fs.cwd().makeDir(output_dir) catch |err| switch (err) {
            error.PathAlreadyExists => {},
            else => return err,
        };

        const base_name = std.fs.path.stem(file_path);
        const dot_path = try std.fmt.allocPrint(allocator, "{s}/{s}.dot", .{ output_dir, base_name });
        defer allocator.free(dot_path);

        var dot_file = try std.fs.cwd().createFile(dot_path, .{});
        defer dot_file.close();
        try dot_file.writeAll(dot_content);
    }
}

// ============================================================================
// SECTION 6: MLIR Integration & Code Generation
// ============================================================================

/// Advanced MLIR emission with full pass pipeline support
fn runMlirEmitAdvanced(allocator: std.mem.Allocator, file_path: []const u8, mlir_options: MlirOptions) !void {
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

    const parse_result = lib.parser.parseWithArena(allocator, tokens) catch |err| {
        try stdout.print("Parser error: {s}\n", .{@errorName(err)});
        try stdout.flush();
        std.process.exit(1);
    };
    const ast_nodes = parse_result.nodes;
    var ast_arena = parse_result.arena;
    defer ast_arena.deinit();

    // Run state analysis automatically during compilation
    // Skip state analysis output when emitting MLIR to keep output clean
    if (!mlir_options.emit_mlir) {
        try runStateAnalysisForContracts(allocator, ast_nodes);
    }

    // Generate MLIR with advanced options
    try generateMlirOutput(allocator, ast_nodes, file_path, mlir_options);
    try stdout.flush();
}

/// Generate MLIR output with comprehensive options
fn generateMlirOutput(allocator: std.mem.Allocator, ast_nodes: []lib.AstNode, file_path: []const u8, mlir_options: MlirOptions) !void {
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    // Import MLIR modules directly (NOT through ora_lib to avoid circular dependencies)
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

    // Lower AST to MLIR (type resolution already done in parser.parse())
    const lower = @import("mlir/lower.zig");
    const source_filename = std.fs.path.basename(file_path);
    var lowering_result = try lower.lowerFunctionsToModuleWithErrors(h.ctx, ast_nodes, mlir_allocator, source_filename);
    defer lowering_result.deinit(mlir_allocator);

    // Check for errors first
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
    const final_module = lowering_result.module;

    // Run MLIR verification after generation (when emitting MLIR)
    if (mlir_options.emit_mlir) {
        const verification = @import("mlir/verification.zig");
        var verifier = verification.OraVerification.init(h.ctx, mlir_allocator);
        defer verifier.deinit();

        const validation_result = try verifier.verifyModule(final_module);

        if (!validation_result.success) {
            try stdout.print("❌ MLIR validation failed with {d} error(s):\n", .{validation_result.errors.len});
            for (validation_result.errors) |err| {
                try stdout.print("  - [{s}] {s}\n", .{ @tagName(err.type), err.message });
            }
            try stdout.flush();
            std.process.exit(1);
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

    // Run canonicalization on Ora MLIR before printing or conversion, unless
    // explicitly disabled via --no-canonicalize. This keeps the canonicalizer
    // "online" for normal usage but lets tests or debugging runs opt out if
    // they hit upstream MLIR issues.
    if (mlir_options.canonicalize and (mlir_options.emit_mlir or mlir_options.emit_mlir_sir)) {
        if (!c.oraCanonicalizeOraMLIR(h.ctx, lowering_result.module)) {
            try stdout.print("Warning: Ora MLIR canonicalization failed\n", .{});
            try stdout.flush();
        }
    }

    // Output Ora MLIR (after canonicalization, before conversion)
    if (mlir_options.emit_mlir) {
        try stdout.print("//===----------------------------------------------------------------------===//\n", .{});
        try stdout.print("// Ora MLIR (after canonicalization, before conversion)\n", .{});
        try stdout.print("//===----------------------------------------------------------------------===//\n\n", .{});
        try stdout.flush();

        const module_op_ora = c.mlirModuleGetOperation(lowering_result.module);
        const mlir_str_ora = c.oraPrintOperation(h.ctx, module_op_ora);
        defer if (mlir_str_ora.data != null) {
            const mlir_c = @import("mlir/c.zig");
            mlir_c.freeStringRef(mlir_str_ora);
        };

        if (mlir_str_ora.data != null and mlir_str_ora.length > 0) {
            const mlir_content_ora = mlir_str_ora.data[0..mlir_str_ora.length];
            try stdout.print("{s}\n", .{mlir_content_ora});
        }
        try stdout.flush();
    }

    // Convert Ora to SIR if emitting SIR MLIR
    if (mlir_options.emit_mlir_sir) {
        const conversion_success = c.oraConvertToSIR(h.ctx, lowering_result.module);
        if (!conversion_success) {
            try stdout.print("Error: Ora to SIR conversion failed\n", .{});
            try stdout.flush();
            std.process.exit(1);
        }
    }

    // Output SIR MLIR after conversion
    if (mlir_options.emit_mlir_sir) {
        try stdout.print("//===----------------------------------------------------------------------===//\n", .{});
        try stdout.print("// SIR MLIR (after conversion)\n", .{});
        try stdout.print("//===----------------------------------------------------------------------===//\n\n", .{});
        try stdout.flush();

        const module_op_sir = c.mlirModuleGetOperation(lowering_result.module);
        const mlir_str_sir = c.oraPrintOperation(h.ctx, module_op_sir);
        defer if (mlir_str_sir.data != null) {
            const mlir_c = @import("mlir/c.zig");
            mlir_c.freeStringRef(mlir_str_sir);
        };

        if (mlir_str_sir.data == null or mlir_str_sir.length == 0) {
            try stdout.print("Failed to print SIR MLIR\n", .{});
            return;
        }

        const mlir_content_sir = mlir_str_sir.data[0..mlir_str_sir.length];

        // Determine output destination
        if (mlir_options.output_dir) |output_dir| {
            // Save to file
            std.fs.cwd().makeDir(output_dir) catch |err| switch (err) {
                error.PathAlreadyExists => {},
                else => return err,
            };

            const basename = std.fs.path.stem(file_path);
            const extension = ".mlir";
            const filename = try std.mem.concat(allocator, u8, &[_][]const u8{ basename, extension });
            defer allocator.free(filename);
            const output_file = try std.fs.path.join(allocator, &[_][]const u8{ output_dir, filename });
            defer allocator.free(output_file);

            var mlir_file = try std.fs.cwd().createFile(output_file, .{});
            defer mlir_file.close();
            try mlir_file.writeAll(mlir_content_sir);
            try stdout.print("SIR MLIR saved to {s}\n", .{output_file});
        } else {
            // Print to stdout
            try stdout.print("{s}", .{mlir_content_sir});
        }
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
