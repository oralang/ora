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
const cli_args = @import("cli/args.zig");
const log = @import("log");
const ManagedArrayList = std.array_list.Managed;

/// MLIR-related command line options
const MlirOptions = struct {
    emit_mlir: bool,
    emit_mlir_sir: bool,
    emit_sir_text: bool,
    emit_bytecode: bool,
    opt_level: ?[]const u8,
    output_dir: ?[]const u8,
    debug_enabled: bool = false,
    canonicalize: bool = true,
    verify_z3: bool = true,
    cpp_lowering_stub: bool = false,

    fn getOptimizationLevel(self: MlirOptions) OptimizationLevel {
        if (self.opt_level) |level| {
            if (std.mem.eql(u8, level, "none")) return .None;
            if (std.mem.eql(u8, level, "basic")) return .Basic;
            if (std.mem.eql(u8, level, "aggressive")) return .Aggressive;
        }

        // use build-time default if no command-line option provided
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

    // Check for subcommands
    const is_fmt_command = args.len >= 2 and std.mem.eql(u8, args[1], "fmt");
    const is_compile_command = args.len >= 2 and std.mem.eql(u8, args[1], "compile");
    const args_to_parse = if (is_fmt_command or is_compile_command) args[2..] else args[1..];

    var parsed = cli_args.parseArgs(args_to_parse) catch {
        try printUsage();
        return;
    };

    if (is_fmt_command) {
        parsed.fmt = true;
    }
    if (is_compile_command) {
        parsed.emit_sir_text = true;
    }
    if (parsed.show_version) {
        try printVersion();
        return;
    }

    const output_dir: ?[]const u8 = parsed.output_dir;
    const input_file: ?[]const u8 = parsed.input_file;
    const emit_tokens: bool = parsed.emit_tokens;
    const emit_ast: bool = parsed.emit_ast;
    const emit_ast_format: ?[]const u8 = parsed.emit_ast_format;
    const emit_typed_ast: bool = parsed.emit_typed_ast;
    const emit_typed_ast_format: ?[]const u8 = parsed.emit_typed_ast_format;
    var emit_mlir: bool = parsed.emit_mlir;
    var emit_mlir_sir: bool = parsed.emit_mlir_sir;
    const emit_sir_text: bool = parsed.emit_sir_text;
    const emit_bytecode: bool = parsed.emit_bytecode;
    const emit_cfg: bool = parsed.emit_cfg;
    const emit_abi: bool = parsed.emit_abi;
    const emit_abi_solidity: bool = parsed.emit_abi_solidity;
    const canonicalize_mlir: bool = parsed.canonicalize_mlir;
    const analyze_state: bool = parsed.analyze_state;
    const verify_z3: bool = parsed.verify_z3;
    const cpp_lowering_stub: bool = parsed.cpp_lowering_stub;
    var debug_enabled: bool = parsed.debug;
    if (!debug_enabled) {
        if (std.posix.getenv("ORA_DEBUG")) |env_value| {
            if (env_value[0] != 0 and env_value[0] != '0') {
                debug_enabled = true;
            }
        }
    }
    const mlir_opt_level: ?[]const u8 = parsed.mlir_opt_level;
    const fmt: bool = parsed.fmt;
    const fmt_check: bool = parsed.fmt_check;
    const fmt_diff: bool = parsed.fmt_diff;
    const fmt_stdout: bool = parsed.fmt_stdout;
    const fmt_width: ?u32 = parsed.fmt_width;

    log.setDebugEnabled(debug_enabled);

    // handle fmt command
    if (fmt) {
        if (input_file == null) {
            std.debug.print("error: fmt requires input file(s)\n", .{});
            try printUsage();
            std.process.exit(2);
        }
        try runFmt(allocator, input_file.?, fmt_check, fmt_diff, fmt_stdout, fmt_width);
        return;
    }

    // require input file
    if (input_file == null) {
        try printUsage();
        return;
    }

    const file_path = input_file.?;

    // handle state analysis (also a special analysis mode)
    if (analyze_state) {
        try runStateAnalysis(allocator, file_path);
        return;
    }

    // determine compilation mode
    // if no --emit-X flag is set, default to MLIR generation
    if (!emit_tokens and !emit_ast and !emit_typed_ast and !emit_mlir and !emit_mlir_sir and !emit_sir_text and !emit_bytecode and !emit_cfg and !emit_abi and !emit_abi_solidity) {
        emit_mlir = true; // Default: emit MLIR
    }
    // Emit SIR MLIR only when explicitly requested or needed for SIR text/bytecode.
    if ((emit_sir_text or emit_bytecode) and !emit_mlir_sir) {
        emit_mlir_sir = true;
    }

    // create MLIR options structure
    const mlir_options = MlirOptions{
        .emit_mlir = emit_mlir,
        .emit_mlir_sir = emit_mlir_sir,
        .emit_sir_text = emit_sir_text,
        .emit_bytecode = emit_bytecode,
        .opt_level = mlir_opt_level,
        .output_dir = output_dir,
        .debug_enabled = debug_enabled,
        .canonicalize = canonicalize_mlir,
        .verify_z3 = verify_z3,
        .cpp_lowering_stub = cpp_lowering_stub,
    };

    // handle CFG generation (uses MLIR's built-in view-op-graph pass)
    if (emit_cfg) {
        try runCFGGeneration(allocator, file_path, mlir_options);
        return;
    }

    // modern compiler-style behavior: process --emit-X flags
    // stop at the earliest stage specified

    if (emit_abi or emit_abi_solidity) {
        try runAbiEmit(allocator, file_path, output_dir, emit_abi, emit_abi_solidity);
        const only_abi = !(emit_tokens or emit_ast or emit_typed_ast or emit_mlir or emit_mlir_sir or emit_sir_text or emit_bytecode);
        if (only_abi) return;
    }

    if (emit_tokens) {
        // stop after lexer
        try runLexer(allocator, file_path);
    } else if (emit_ast or emit_typed_ast) {
        // stop after parser
        const format = if (emit_typed_ast)
            (emit_typed_ast_format orelse "tree")
        else
            (emit_ast_format orelse "tree");
        try runParser(allocator, file_path, format, emit_typed_ast);
    } else if (emit_mlir) {
        // run full MLIR pipeline (Ora MLIR)
        try runMlirEmitAdvanced(allocator, file_path, mlir_options);
    } else {
        // default: emit MLIR
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
    try stdout.print("       ora compile [options] <file.ora>\n", .{});
    try stdout.print("       ora -v | --version\n", .{});
    try stdout.print("\nCompilation Control:\n", .{});
    try stdout.print("  (default)              - Emit MLIR\n", .{});
    try stdout.print("  compile                - Emit SIR text (Sensei format)\n", .{});
    try stdout.print("  --emit-tokens          - Stop after lexical analysis (emit tokens)\n", .{});
    try stdout.print("  --emit-ast             - Stop after parsing (emit AST)\n", .{});
    try stdout.print("  --emit-ast=json|tree   - Emit AST in JSON or tree format\n", .{});
    try stdout.print("  --emit-typed-ast       - Stop after parsing (emit typed AST)\n", .{});
    try stdout.print("  --emit-typed-ast=json|tree - Emit typed AST in JSON or tree format\n", .{});
    try stdout.print("  --emit-mlir            - Emit Ora MLIR only (default)\n", .{});
    try stdout.print("  --emit-mlir-sir        - Emit SIR MLIR only (after conversion)\n", .{});
    try stdout.print("  --emit-sir             - Alias for --emit-mlir-sir\n", .{});
    try stdout.print("  --emit-sir-text        - Emit Sensei SIR text (after conversion)\n", .{});
    try stdout.print("  --emit-bytecode        - Emit EVM bytecode from Sensei SIR text\n", .{});
    try stdout.print("  --emit-cfg             - Generate control flow graph (Graphviz DOT format)\n", .{});
    try stdout.print("  --emit-abi             - Emit Ora ABI manifest JSON\n", .{});
    try stdout.print("  --emit-abi-solidity    - Emit Solidity-compatible ABI JSON\n", .{});
    try stdout.print("  -v, --version          - Show version and logo\n", .{});
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
    try stdout.print("  --cpp-lowering-stub    - Use experimental C++ lowering stub (contract+func)\n", .{});
    try stdout.print("\nAnalysis Options:\n", .{});
    try stdout.print("  --analyze-state        - Analyze storage reads/writes per function\n", .{});
    try stdout.print("  --verify               - Run Z3 verification on MLIR annotations (default)\n", .{});
    try stdout.print("  --no-verify            - Disable Z3 verification\n", .{});
    try stdout.print("  --debug                - Enable compiler debug output\n", .{});
    try stdout.flush();
}

fn printVersion() !void {
    var stdout_buffer: [8192]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;
    const logo =
        \\=================
        \\                                       +====-----------====
        \\                                      +++++==---------==+=+=+
        \\                                    ++++++++==-------=+++++++++
        \\                                  ++++++++++===-----==+++++++++++
        \\                    +++          *++++++++++=====-====++++++++++*#         ++++
        \\                  +++++++        ##*++++++++==========++++++++*###        +++++++
        \\                 ++++++++++      ####*++++++===========+++++**####      ++++++++++
        \\                   ++++++++++    ######*+++======+======+++*######    ++++++++++
        \\                    =++++++++    ########+=====+++++=====+########    ++++++++
        \\                    ++=++++++      ########+=++******+=+########      +++++++++
        \\                   =++++++***        ########*********########        +**+++++++
        \\                   ++++++++**      ############*****############      **++++++++
        \\             ======++++++++++    #######*########*#########*######    +++++++++++=====
        \\           =====+++++++++++++    ######   ###############   ######    +++++++++++++=====
        \\         ======++++++++++++++    ######     ###########     ######    +++++++++++++++=====
        \\         ++==++++++   +++++++    #######    ###########    *######    +++++++   +++++++=++
        \\         ++++++++   ++++++++*      #############################      +++++++++   ++++++++
        \\         +++++++    =++++++          #########################          +++++++    +++++++
        \\         ++++++++     -++         ###############################         +++    +++++++++
        \\         ++++++++++               ##%#########################%%%               ++++++++++
        \\         +++++++++*##      +###########################################       ##**++++++++
        \\           +++++*######    =######%############################%%######     ######*+++++
        \\             +*##########  =######%%%#########################%%%######   *#########*+
        \\               *###################%%%#######################%%%####################
        \\                **##################%%%%###################%%%%%%*################
        \\           ##     +*############  ##%%%%%%################%%%%%%%  **############     ##
        \\         #*####     +*########    #############################%%    **#######      %#####
        \\        #*######                  #%%#########################%%%                  ########
        \\      ##########%%%%%%%%   =#%%%%%%%%%%##*##################%%%%%%%%%%%    #%%%%%%%%#########
        \\    ############%%%%%%%%   =#%%%%%%%%%%% +################ %%%%%%%%%%%%%  =#%%%%%%%############
        \\   ########**###%%%%%%%%   =#%%%%%%%%%   +################   %%%%%%%%%%%  =#%%%%%%%####*#########
        \\ ##########  *#%%%%%%%%%%%%#%%%%%%%%%   *#%%%%%##*##%%%%%%    ##%%%%%%%%####%%%%%%%%##  *#########
        \\ *#######     ##%%%%%%%##%%%%%%%%%%   *##%%%%%%#* =*%%%%%%%%    ###%%%%%%%%%%%%%%%%%      *#######
        \\ ######         ##############%##    ###%%%%%##     +#%%%%%###    **#############%%        *######
        \\ ######## ##%                      ##########%        *#########                      ### *#######
        \\  *##########%                   *#########%            ##########                  %############
        \\   *########%%%%               ##########%    *%   %@    ##########                %%%##########
        \\     #######%%%     %%%%%%%%# ##########    %%%%   %%%@    ########## #%%%%%%%%     ##########
        \\      ######%      %%%%%%%%%% #########   %%%%%%   %%%%%    *######## #%%%%%%%%%      ######
        \\        ###      %%%%%%%%%%%@ #######    %%%%%%%   %%%%%%     *###### #%%%%%%%%%%%     *###
        \\               %%%%%%%%%@%@@@ ##%%%%%%    #%%%%%   %%%%%%    %%%%%%%% #%%%%%%%%%%%%%
        \\                %%%%%%%%%%%   %%%%%%%%% %%%%%%%%   %%%%%%%  %%%%%%%%%   %%%%%%%%%%%
        \\                              *#%%%%%%%%%%%%%%%%   %%%%%%%%%%%%%%%%%%
        \\                                #%%%%%%%%%%%%%%%   %%%%%%%%%%%%%%%%
        \\                                  %%%%%@@@@%%%%     #%%%@@@@%%%%%
        \\                                   %%@@@@@@@@@       #%@@@@@@@%
        \\                                     %@@@@@@           %@@@@@@
        \\                                       %@@@             %@@@
    ;
    try stdout.print("{s}\n", .{logo});
    try stdout.print("Ora Compiler v0.1 - Asuka\n", .{});
    try stdout.flush();
}

// ============================================================================
// SECTION 3: Command Handlers (lex, parse, ast, compile)
// ============================================================================

/// Print unified diff between original and formatted code
fn printUnifiedDiff(_: std.mem.Allocator, original: []const u8, formatted: []const u8, file_path: []const u8) !void {
    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    try stdout.print("--- {s}\n", .{file_path});
    try stdout.print("+++ {s}\n", .{file_path});

    // Simple line-by-line diff
    var orig_lines = std.mem.splitScalar(u8, original, '\n');
    var fmt_lines = std.mem.splitScalar(u8, formatted, '\n');

    var line_num: u32 = 1;
    var orig_line = orig_lines.next();
    var fmt_line = fmt_lines.next();

    while (orig_line != null or fmt_line != null) {
        const orig = orig_line orelse "";
        const fmt = fmt_line orelse "";

        if (!std.mem.eql(u8, orig, fmt)) {
            if (orig_line != null) {
                try stdout.print("-{d}: {s}\n", .{ line_num, orig });
            }
            if (fmt_line != null) {
                try stdout.print("+{d}: {s}\n", .{ line_num, fmt });
            }
        } else {
            try stdout.print(" {d}: {s}\n", .{ line_num, orig });
        }

        if (orig_line != null) orig_line = orig_lines.next();
        if (fmt_line != null) fmt_line = fmt_lines.next();
        line_num += 1;
    }
}

/// Run formatter on file(s)
fn runFmt(allocator: std.mem.Allocator, file_path: []const u8, check: bool, diff: bool, stdout: bool, width: ?u32) !void {
    const fmt_mod = @import("fmt/mod.zig");
    const FormatOptions = fmt_mod.FormatOptions;

    const options = FormatOptions{
        .line_width = width orelse 100,
        .indent_size = 4,
    };

    // Read source file
    const source = std.fs.cwd().readFileAlloc(allocator, file_path, std.math.maxInt(usize)) catch |err| {
        std.debug.print("error: cannot read file '{s}': {s}\n", .{ file_path, @errorName(err) });
        std.process.exit(1);
    };
    defer allocator.free(source);

    // Format
    var formatter = fmt_mod.Formatter.init(allocator, source, options);
    defer formatter.deinit();

    const formatted = formatter.format() catch |err| {
        std.debug.print("error: failed to format {s}: {}\n", .{ file_path, err });
        std.process.exit(2);
    };
    defer allocator.free(formatted);

    // Check if already formatted
    const already_formatted = std.mem.eql(u8, source, formatted);

    if (check) {
        if (!already_formatted) {
            std.debug.print("{s} needs formatting\n", .{file_path});
            std.process.exit(1);
        }
        return;
    }

    if (diff) {
        if (!already_formatted) {
            // Generate unified diff
            try printUnifiedDiff(allocator, source, formatted, file_path);
            std.process.exit(1);
        }
        return;
    }

    if (stdout) {
        try std.fs.File.stdout().writeAll(formatted);
        return;
    }

    // Write formatted output
    if (!already_formatted) {
        try std.fs.cwd().writeFile(.{ .sub_path = file_path, .data = formatted });
    }
}

/// Run lexer on file and display tokens
fn runLexer(allocator: std.mem.Allocator, file_path: []const u8) !void {
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    // read source file
    const source = std.fs.cwd().readFileAlloc(allocator, file_path, 1024 * 1024) catch |err| {
        try stdout.print("error: cannot read file '{s}': {s}\n", .{ file_path, @errorName(err) });
        try stdout.flush();
        std.process.exit(1);
    };
    defer allocator.free(source);

    try stdout.print("Lexing {s}\n", .{file_path});
    try stdout.print("==================================================\n", .{});

    // run lexer
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

    // display all tokens without truncation
    for (tokens, 0..) |token, i| {
        try stdout.print("[{d:3}] {any}\n", .{ i, token });
    }

    try stdout.flush();
}

// ============================================================================
// SECTION 5: Parser & Compilation Workflows
// ============================================================================

/// Run parser on file and display AST
fn runParser(allocator: std.mem.Allocator, file_path: []const u8, format: []const u8, include_types: bool) !void {
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    if (!std.mem.eql(u8, format, "tree") and !std.mem.eql(u8, format, "json")) {
        try stdout.print("error: unsupported AST format '{s}' (use 'tree' or 'json')\n", .{format});
        try stdout.flush();
        std.process.exit(2);
    }

    // read source file
    const source = std.fs.cwd().readFileAlloc(allocator, file_path, 1024 * 1024) catch |err| {
        try stdout.print("error: cannot read file '{s}': {s}\n", .{ file_path, @errorName(err) });
        try stdout.flush();
        std.process.exit(1);
    };
    defer allocator.free(source);

    if (std.mem.eql(u8, format, "tree")) {
        try stdout.print("Parsing {s}\n", .{file_path});
        try stdout.print("==================================================\n", .{});
    }

    // run lexer
    var lexer = lib.Lexer.init(allocator, source);
    defer lexer.deinit();

    const tokens = lexer.scanTokens() catch |err| {
        try stdout.print("Lexer error: {s}\n", .{@errorName(err)});
        try stdout.flush();
        std.process.exit(1);
    };
    defer allocator.free(tokens);

    if (std.mem.eql(u8, format, "tree")) {
        try stdout.print("Lexed {d} tokens\n", .{tokens.len});
    }

    // run parser - use parseWithArena to keep arena alive for AST printing
    var parse_result = lib.parser.parseWithArena(allocator, tokens) catch |err| {
        try stdout.print("Parser error: {s}\n", .{@errorName(err)});
        try stdout.flush();
        std.process.exit(1);
    };
    defer parse_result.arena.deinit(); // Keep arena alive until after printing
    const ast_nodes = parse_result.nodes;

    if (std.mem.eql(u8, format, "json")) {
        var serializer = lib.AstSerializer.init(allocator, .{
            .include_types = include_types,
            .pretty_print = true,
        });
        defer serializer.deinit();
        serializer.serialize(ast_nodes, stdout) catch |err| {
            try stdout.print("error: failed to serialize AST: {s}\n", .{@errorName(err)});
            try stdout.flush();
            std.process.exit(1);
        };
    } else {
        try stdout.print("Generated {d} AST nodes\n\n", .{ast_nodes.len});
        for (ast_nodes, 0..) |*node, i| {
            try stdout.print("[{d}] ", .{i});
            try printAstSummary(stdout, node, 0);
        }
    }

    try stdout.flush();
}

/// Run state analysis on AST nodes (used during normal compilation)
fn runStateAnalysisForContracts(allocator: std.mem.Allocator, ast_nodes: []lib.AstNode) !void {
    const state_tracker = lib.state_tracker;

    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    // analyze each contract
    for (ast_nodes) |*node| {
        switch (node.*) {
            .Contract => |*contract| {
                // run state analysis on this contract
                var contract_analysis = state_tracker.analyzeContract(allocator, contract) catch |err| {
                    try stdout.print("State analysis error: {s}\n", .{@errorName(err)});
                    continue;
                };
                defer contract_analysis.deinit();

                // print only warnings during compilation (not full analysis)
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

    // read source file
    const source = std.fs.cwd().readFileAlloc(allocator, file_path, 1024 * 1024) catch |err| {
        try stdout.print("error: cannot read file '{s}': {s}\n", .{ file_path, @errorName(err) });
        try stdout.flush();
        std.process.exit(1);
    };
    defer allocator.free(source);

    try stdout.print("Analyzing state changes for {s}\n", .{file_path});
    try stdout.print("==================================================\n", .{});

    // run lexer
    var lexer = lib.Lexer.init(allocator, source);
    defer lexer.deinit();

    const tokens = lexer.scanTokens() catch |err| {
        try stdout.print("Lexer error: {s}\n", .{@errorName(err)});
        try stdout.flush();
        std.process.exit(1);
    };
    defer allocator.free(tokens);

    // run parser
    var arena = lib.ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var parser = lib.Parser.init(tokens, &arena);
    parser.setFileId(1);
    const ast_nodes = parser.parse() catch |err| {
        try stdout.print("Parser error: {s}\n", .{@errorName(err)});
        try stdout.flush();
        std.process.exit(1);
    };

    // analyze each contract
    for (ast_nodes) |*node| {
        switch (node.*) {
            .Contract => |*contract| {
                // run state analysis on this contract
                var contract_analysis = state_tracker.analyzeContract(allocator, contract) catch |err| {
                    try stdout.print("State analysis error: {s}\n", .{@errorName(err)});
                    continue;
                };
                defer contract_analysis.deinit();

                // print results
                try state_tracker.printAnalysis(stdout, &contract_analysis);
            },
            else => {},
        }
    }

    try stdout.flush();
}

/// Print a concise AST summary
fn printAstSummary(writer: anytype, node: *lib.AstNode, indent: u32) !void {
    // print indentation
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
            // print function body statements with indentation
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
    const c = @import("mlir_c_api").c;

    // first generate MLIR
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    // read source file
    const source = std.fs.cwd().readFileAlloc(allocator, file_path, 1024 * 1024) catch |err| {
        try stdout.print("error: cannot read file '{s}': {s}\n", .{ file_path, @errorName(err) });
        try stdout.flush();
        std.process.exit(1);
    };
    defer allocator.free(source);

    // parse to AST
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

    // generate MLIR
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

    // get MLIR as text by printing the module operation
    const module_op = c.oraModuleGetOperation(lowering_result.module);
    const mlir_text_ref = c.oraOperationPrintToString(module_op);
    defer if (mlir_text_ref.data != null) {
        const mlir_c = @import("mlir_c_api");
        mlir_c.freeStringRef(mlir_text_ref);
    };

    const mlir_text = if (mlir_text_ref.data != null and mlir_text_ref.length > 0)
        mlir_text_ref.data[0..mlir_text_ref.length]
    else
        "";
    _ = mlir_text;

    // convert Ora MLIR to SIR MLIR before generating CFG
    if (!c.oraConvertToSIR(h.ctx, lowering_result.module)) {
        try stdout.print("Error: Ora to SIR conversion failed\n", .{});
        try stdout.flush();
        std.process.exit(1);
    }

    // use MLIR C++ API to generate CFG with dialect properly registered
    const dot_content = cfg_gen.generateCFG(h.ctx, lowering_result.module, mlir_allocator) catch |err| {
        try stdout.print("Failed to generate CFG: {s}\n", .{@errorName(err)});
        try stdout.print("Note: The view-op-graph pass may need the module to be in a specific format.\n", .{});
        return;
    };
    defer mlir_allocator.free(dot_content);

    // output DOT content
    try stdout.print("{s}", .{dot_content});
    try stdout.flush();

    // save to file if output directory specified
    if (mlir_options.output_dir) |output_dir| {
        // create output directory if it doesn't exist
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

/// Generate Ora ABI outputs
fn runAbiEmit(
    allocator: std.mem.Allocator,
    file_path: []const u8,
    output_dir: ?[]const u8,
    emit_abi: bool,
    emit_abi_solidity: bool,
) !void {
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    // read source file
    const source = std.fs.cwd().readFileAlloc(allocator, file_path, 1024 * 1024) catch |err| {
        try stdout.print("error: cannot read file '{s}': {s}\n", .{ file_path, @errorName(err) });
        try stdout.flush();
        std.process.exit(1);
    };
    defer allocator.free(source);

    // parse to AST
    var lexer = lib.Lexer.init(allocator, source);
    defer lexer.deinit();
    const tokens = lexer.scanTokens() catch |err| {
        try stdout.print("Lexer error: {s}\n", .{@errorName(err)});
        return;
    };
    defer allocator.free(tokens);

    const parse_result = lib.parser.parseWithArena(allocator, tokens) catch |err| {
        try stdout.print("Parser error: {s}\n", .{@errorName(err)});
        return;
    };
    const ast_nodes = parse_result.nodes;
    var ast_arena = parse_result.arena;
    defer ast_arena.deinit();

    var generator = try lib.abi.AbiGenerator.init(allocator);
    defer generator.deinit();
    var contract_abi = try generator.generate(ast_nodes);
    defer contract_abi.deinit();

    const base_name = std.fs.path.stem(file_path);

    if (output_dir) |out_dir| {
        std.fs.cwd().makeDir(out_dir) catch |err| switch (err) {
            error.PathAlreadyExists => {},
            else => return err,
        };
        if (emit_abi) {
            const abi_json = try contract_abi.toJson(allocator);
            defer allocator.free(abi_json);
            const abi_path = try std.fmt.allocPrint(allocator, "{s}/{s}.abi.json", .{ out_dir, base_name });
            defer allocator.free(abi_path);
            var abi_file = try std.fs.cwd().createFile(abi_path, .{});
            defer abi_file.close();
            try abi_file.writeAll(abi_json);
        }
        if (emit_abi_solidity) {
            const abi_json = try contract_abi.toSolidityJson(allocator);
            defer allocator.free(abi_json);
            const abi_path = try std.fmt.allocPrint(allocator, "{s}/{s}.abi.sol.json", .{ out_dir, base_name });
            defer allocator.free(abi_path);
            var abi_file = try std.fs.cwd().createFile(abi_path, .{});
            defer abi_file.close();
            try abi_file.writeAll(abi_json);
        }
    } else {
        if (emit_abi) {
            const abi_json = try contract_abi.toJson(allocator);
            defer allocator.free(abi_json);
            try stdout.print("{s}\n", .{abi_json});
        }
        if (emit_abi_solidity) {
            const abi_json = try contract_abi.toSolidityJson(allocator);
            defer allocator.free(abi_json);
            try stdout.print("{s}\n", .{abi_json});
        }
        try stdout.flush();
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

    // read source file
    const source = std.fs.cwd().readFileAlloc(allocator, file_path, 1024 * 1024) catch |err| {
        try stdout.print("error: cannot read file '{s}': {s}\n", .{ file_path, @errorName(err) });
        try stdout.flush();
        std.process.exit(1);
    };
    defer allocator.free(source);

    // front half: lex + parse (ensures we have a valid AST before MLIR)
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

    // run state analysis automatically during compilation
    // skip state analysis output when emitting MLIR to keep output clean
    if (!mlir_options.emit_mlir and !mlir_options.emit_mlir_sir and !mlir_options.emit_sir_text and !mlir_options.emit_bytecode) {
        try runStateAnalysisForContracts(allocator, ast_nodes);
    }

    // generate MLIR with advanced options
    try generateMlirOutput(allocator, ast_nodes, file_path, mlir_options);
    try stdout.flush();
}

const OraTypeTag = enum(u32) {
    Void = 0,
    U256 = 1,
    I256 = 2,
    Bool = 3,
    Address = 4,
};

fn mapOraTypeTag(ora_type: lib.OraType) OraTypeTag {
    return switch (ora_type) {
        .u256 => .U256,
        .i256 => .I256,
        .bool => .Bool,
        .address => .Address,
        .void => .Void,
        .error_union => |succ| mapOraTypeTag(succ.*),
        else => .U256,
    };
}

fn mapParamTypeTag(type_info: lib.ast.Types.TypeInfo) OraTypeTag {
    if (type_info.ora_type) |ora_type| {
        return mapOraTypeTag(ora_type);
    }
    return .U256;
}

fn mapReturnTypeTag(type_info: ?lib.ast.Types.TypeInfo) OraTypeTag {
    if (type_info) |ti| {
        if (ti.ora_type) |ora_type| {
            return mapOraTypeTag(ora_type);
        }
    }
    return .Void;
}

fn resolveSenseiSirPath(allocator: std.mem.Allocator) ![]const u8 {
    if (std.posix.getenv("ORA_SENSEI_SIR")) |path| {
        return allocator.dupe(u8, path);
    }

    const default_path = "vendor/sensei/senseic/target/release/sir";
    if (std.fs.cwd().access(default_path, .{}) catch null) |_| {
        return allocator.dupe(u8, default_path);
    }

    return error.SenseiSirNotFound;
}

fn emitBytecodeFromSirText(
    allocator: std.mem.Allocator,
    sir_text: []const u8,
    file_path: []const u8,
    output_dir: ?[]const u8,
    stdout: anytype,
) !void {
    const sir_path = try resolveSenseiSirPath(allocator);
    defer allocator.free(sir_path);

    const basename = std.fs.path.stem(file_path);
    const sir_extension = ".sir";
    const sir_filename = try std.mem.concat(allocator, u8, &[_][]const u8{ basename, sir_extension });
    defer allocator.free(sir_filename);

    const temp_dir = "/tmp/ora_sir";
    std.fs.makeDirAbsolute(temp_dir) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return err,
    };
    const sir_file_path = try std.fs.path.join(allocator, &[_][]const u8{ temp_dir, sir_filename });
    defer allocator.free(sir_file_path);

    var sir_file = try std.fs.createFileAbsolute(sir_file_path, .{});
    defer sir_file.close();
    try sir_file.writeAll(sir_text);

    var argv = [_][]const u8{ sir_path, sir_file_path };
    var child = std.process.Child.init(&argv, allocator);
    child.stdin_behavior = .Ignore;
    child.stdout_behavior = .Pipe;
    child.stderr_behavior = .Pipe;

    try child.spawn();

    const max_output = 16 * 1024 * 1024;
    const stdout_bytes = try child.stdout.?.readToEndAlloc(allocator, max_output);
    defer allocator.free(stdout_bytes);
    const stderr_bytes = try child.stderr.?.readToEndAlloc(allocator, max_output);
    defer allocator.free(stderr_bytes);

    const term = try child.wait();
    switch (term) {
        .Exited => |code| {
            if (code != 0) {
                var stderr_buffer: [1024]u8 = undefined;
                var stderr_writer = std.fs.File.stderr().writer(&stderr_buffer);
                const stderr = &stderr_writer.interface;
                const trimmed_stderr = std.mem.trim(u8, stderr_bytes, " \n\r\t");
                if (trimmed_stderr.len > 0) {
                    try stderr.writeAll(trimmed_stderr);
                    try stderr.writeAll("\n");
                } else {
                    const trimmed_stdout = std.mem.trim(u8, stdout_bytes, " \n\r\t");
                    if (trimmed_stdout.len > 0) {
                        try stderr.writeAll(trimmed_stdout);
                        try stderr.writeAll("\n");
                    }
                }
                try stdout.print("Error: sensei sir failed ({d})\n", .{code});
                try stdout.print("SIR input saved to {s}\n", .{sir_file_path});
                return error.SenseiSirFailed;
            }
        },
        else => {
            try stdout.print("Error: sensei sir terminated unexpectedly\n", .{});
            try stdout.print("SIR input saved to {s}\n", .{sir_file_path});
            return error.SenseiSirFailed;
        },
    }

    const bytecode = std.mem.trim(u8, stdout_bytes, " \n\r\t");
    if (bytecode.len == 0) {
        try stdout.print("Error: sensei sir produced empty bytecode\n", .{});
        try stdout.print("SIR input saved to {s}\n", .{sir_file_path});
        return error.SenseiSirFailed;
    }

    if (output_dir) |out_dir| {
        const out_is_file = std.mem.endsWith(u8, out_dir, ".hex") or
            std.mem.endsWith(u8, out_dir, ".bytecode") or
            std.mem.endsWith(u8, out_dir, ".bin");

        if (out_is_file) {
            var out_file = if (std.fs.path.isAbsolute(out_dir))
                try std.fs.createFileAbsolute(out_dir, .{})
            else
                try std.fs.cwd().createFile(out_dir, .{});
            defer out_file.close();
            try out_file.writeAll(bytecode);
            try stdout.print("Bytecode saved to {s}\n", .{out_dir});
        } else {
            std.fs.cwd().makeDir(out_dir) catch |err| switch (err) {
                error.PathAlreadyExists => {},
                else => return err,
            };

            const extension = ".hex";
            const filename = try std.mem.concat(allocator, u8, &[_][]const u8{ basename, extension });
            defer allocator.free(filename);
            const output_file = try std.fs.path.join(allocator, &[_][]const u8{ out_dir, filename });
            defer allocator.free(output_file);

            var out_file = try std.fs.cwd().createFile(output_file, .{});
            defer out_file.close();
            try out_file.writeAll(bytecode);
            try stdout.print("Bytecode saved to {s}\n", .{output_file});
        }
    } else {
        try stdout.print("{s}\n", .{bytecode});
    }
}

const StubTarget = struct {
    contract: []const u8,
    function: []const u8,
    func_node: ?*const lib.FunctionNode,
};

fn getCppStubTarget(nodes: []lib.AstNode) StubTarget {
    var target = StubTarget{
        .contract = "StubContract",
        .function = "main",
        .func_node = null,
    };

    for (nodes) |node| {
        switch (node) {
            .Contract => |contract| {
                target.contract = contract.name;
                for (contract.body) |decl| {
                    if (decl == .Function) {
                        target.function = decl.Function.name;
                        target.func_node = &decl.Function;
                        return target;
                    }
                }
                return target;
            },
            .Module => |module_node| {
                for (module_node.declarations) |decl| {
                    if (decl == .Contract) {
                        const contract = decl.Contract;
                        target.contract = contract.name;
                        for (contract.body) |inner_decl| {
                            if (inner_decl == .Function) {
                                target.function = inner_decl.Function.name;
                                target.func_node = &inner_decl.Function;
                                return target;
                            }
                        }
                        return target;
                    }
                }
            },
            else => {},
        }
    }

    return target;
}

/// Generate MLIR output with comprehensive options
fn generateMlirOutput(allocator: std.mem.Allocator, ast_nodes: []lib.AstNode, file_path: []const u8, mlir_options: MlirOptions) !void {
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    // import MLIR modules directly (NOT through ora_lib to avoid circular dependencies)
    const mlir = @import("mlir/mod.zig");
    const c = @import("mlir_c_api").c;

    // create arena allocator for MLIR lowering phase
    // this arena will be freed after MLIR generation completes
    var mlir_arena = std.heap.ArenaAllocator.init(allocator);
    defer mlir_arena.deinit();
    const mlir_allocator = mlir_arena.allocator();

    // create MLIR context
    const h = mlir.createContext(mlir_allocator);
    defer mlir.destroyContext(h);

    const quiet_mlir_output = mlir_options.emit_mlir_sir and
        !mlir_options.emit_mlir and
        !mlir_options.emit_sir_text and
        !mlir_options.emit_bytecode and
        !mlir_options.debug_enabled;
    const final_module = if (mlir_options.cpp_lowering_stub) blk: {
        const target = getCppStubTarget(ast_nodes);
        const loc = c.oraLocationUnknownGet(h.ctx);
        const contract_ref = c.oraStringRefCreate(target.contract.ptr, target.contract.len);
        const func_ref = c.oraStringRefCreate(target.function.ptr, target.function.len);

        var param_tags: []u32 = &[_]u32{};
        var param_names: []c.MlirStringRef = &[_]c.MlirStringRef{};
        var return_tag: u32 = @intFromEnum(OraTypeTag.Void);
        if (target.func_node) |func_node| {
            if (func_node.parameters.len > 0) {
                param_tags = try mlir_allocator.alloc(u32, func_node.parameters.len);
                param_names = try mlir_allocator.alloc(c.MlirStringRef, func_node.parameters.len);
                for (func_node.parameters, 0..) |param, i| {
                    param_tags[i] = @intFromEnum(mapParamTypeTag(param.type_info));
                    param_names[i] = c.oraStringRefCreate(param.name.ptr, param.name.len);
                }
            }
            return_tag = @intFromEnum(mapReturnTypeTag(func_node.return_type_info));
        }

        const module = c.oraLowerContractStubWithSig(
            h.ctx,
            loc,
            contract_ref,
            func_ref,
            if (param_tags.len == 0) null else param_tags.ptr,
            param_tags.len,
            if (param_names.len == 0) null else param_names.ptr,
            param_names.len,
            return_tag,
        );
        if (c.oraModuleIsNull(module)) {
            try stdout.print("C++ lowering stub failed: module is null\n", .{});
            try stdout.flush();
            std.process.exit(1);
        }
        break :blk module;
    } else blk: {
        // lower AST to MLIR (type resolution already done in parser.parse())
        const lower = @import("mlir/lower.zig");
        const source_filename = std.fs.path.basename(file_path);
        var lowering_result = try lower.lowerFunctionsToModuleWithErrors(h.ctx, ast_nodes, mlir_allocator, source_filename);
        defer lowering_result.deinit(mlir_allocator);

        // check for errors first
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

        // print warnings if any
        if (!quiet_mlir_output and lowering_result.warnings.len > 0) {
            try stdout.print("MLIR lowering completed with {d} warnings:\n", .{lowering_result.warnings.len});
            for (lowering_result.warnings) |warn| {
                try stdout.print("  - {s}\n", .{warn.message});
            }
        }

        // print pass results if available
        if (!quiet_mlir_output) {
            if (lowering_result.pass_result) |pass_result| {
                if (pass_result.success) {
                    try stdout.print("Pass pipeline executed successfully\n", .{});
                } else {
                    try stdout.print("Pass pipeline failed: {s}\n", .{pass_result.error_message orelse "unknown error"});
                }
            }
        }

        break :blk lowering_result.module;
    };
    defer c.oraModuleDestroy(final_module);

    // run MLIR verification after generation (when emitting MLIR)
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

    var verification_result_opt: ?@import("z3/errors.zig").VerificationResult = null;
    defer {
        if (verification_result_opt) |*vr| {
            vr.deinit();
        }
    }
    // run Z3 verification pass (formal verification)
    if (mlir_options.verify_z3) {
        const z3_verification = @import("z3/verification.zig");
        var verifier = try z3_verification.VerificationPass.init(mlir_allocator);
        defer verifier.deinit();

        const verification_result = try verifier.runVerificationPass(final_module);

        if (!verification_result.success) {
            try stdout.print("❌ Z3 verification failed with {d} error(s):\n", .{verification_result.errors.items.len});
            for (verification_result.errors.items) |err| {
                try stdout.print("  - {s}\n", .{err.message});
                if (err.counterexample) |ce| {
                    if (ce.variables.get("__model")) |model| {
                        try stdout.print("    Model: {s}\n", .{model});
                    }
                }
            }
            try stdout.flush();
            std.process.exit(1);
        }

        verification_result_opt = verification_result;
    }

    // run canonicalization on Ora MLIR before printing or conversion, unless
    // explicitly disabled via --no-canonicalize. This keeps the canonicalizer
    // "online" for normal usage but lets tests or debugging runs opt out if
    // they hit upstream MLIR issues.
    if (mlir_options.canonicalize and (mlir_options.emit_mlir or mlir_options.emit_mlir_sir)) {
        if (!c.oraCanonicalizeOraMLIR(h.ctx, final_module)) {
            try stdout.print("Warning: Ora MLIR canonicalization failed\n", .{});
            try stdout.flush();
        }
    }

    if (mlir_options.emit_mlir_sir) {
        const refinement_guards = @import("mlir/refinement_guards.zig");
        if (verification_result_opt) |*vr| {
            refinement_guards.cleanupRefinementGuards(h.ctx, final_module, &vr.proven_guard_ids);
        } else {
            var empty_guards = std.StringHashMap(void).init(mlir_allocator);
            defer empty_guards.deinit();
            refinement_guards.cleanupRefinementGuards(h.ctx, final_module, &empty_guards);
        }
    }

    // output Ora MLIR (after canonicalization, before conversion)
    if (mlir_options.emit_mlir) {
        try stdout.print("//===----------------------------------------------------------------------===//\n", .{});
        try stdout.print("// Ora MLIR (before conversion)\n", .{});
        try stdout.print("//===----------------------------------------------------------------------===//\n\n", .{});
        try stdout.flush();

        const module_op_ora = c.oraModuleGetOperation(final_module);
        const mlir_str_ora = c.oraOperationPrintToString(module_op_ora);
        defer if (mlir_str_ora.data != null) {
            const mlir_c = @import("mlir_c_api");
            mlir_c.freeStringRef(mlir_str_ora);
        };

        if (mlir_str_ora.data != null and mlir_str_ora.length > 0) {
            const mlir_content_ora = mlir_str_ora.data[0..mlir_str_ora.length];
            try stdout.print("{s}\n", .{mlir_content_ora});
        }
        try stdout.flush();
    }

    // convert Ora to SIR if emitting SIR MLIR
    if (mlir_options.emit_mlir_sir) {
        try stdout.print("[main] Ora->SIR: begin (debug={any}, quiet_stderr={any})\n", .{
            mlir_options.debug_enabled, quiet_mlir_output,
        });
        if (quiet_mlir_output) {
            c.oraSetDebugEnabled(false);
        } else {
            c.oraSetDebugEnabled(mlir_options.debug_enabled);
        }
        var saved_stderr: ?std.posix.fd_t = null;
        var null_fd: ?std.posix.fd_t = null;
        if (quiet_mlir_output) {
            saved_stderr = try std.posix.dup(std.posix.STDERR_FILENO);
            null_fd = try std.posix.open("/dev/null", .{ .ACCMODE = .WRONLY }, 0);
            _ = try std.posix.dup2(null_fd.?, std.posix.STDERR_FILENO);
        }
        const conversion_success = c.oraConvertToSIR(h.ctx, final_module);
        if (quiet_mlir_output) {
            _ = std.posix.dup2(saved_stderr.?, std.posix.STDERR_FILENO) catch {};
            if (saved_stderr) |fd| std.posix.close(fd);
            if (null_fd) |fd| std.posix.close(fd);
        }
        if (!conversion_success) {
            try stdout.print("[main] Ora->SIR: failed\n", .{});
            try stdout.print("Error: Ora to SIR conversion failed\n", .{});
            try stdout.flush();
            std.process.exit(1);
        }
        try stdout.print("[main] Ora->SIR: success\n", .{});
    }

    const emit_sir_mlir_output = mlir_options.emit_mlir_sir and !mlir_options.emit_sir_text and !mlir_options.emit_bytecode;

    // output SIR MLIR after conversion
    if (emit_sir_mlir_output) {
        if (!quiet_mlir_output) {
            try stdout.print("//===----------------------------------------------------------------------===//\n", .{});
            try stdout.print("// SIR MLIR (after conversion)\n", .{});
            try stdout.print("//===----------------------------------------------------------------------===//\n\n", .{});
            try stdout.flush();
        }

        const module_op_sir = c.oraModuleGetOperation(final_module);
        const mlir_str_sir = c.oraOperationPrintToString(module_op_sir);
        defer if (mlir_str_sir.data != null) {
            const mlir_c = @import("mlir_c_api");
            mlir_c.freeStringRef(mlir_str_sir);
        };

        if (mlir_str_sir.data == null or mlir_str_sir.length == 0) {
            try stdout.print("Failed to print SIR MLIR\n", .{});
            return;
        }

        const mlir_content_sir = mlir_str_sir.data[0..mlir_str_sir.length];

        // determine output destination
        if (mlir_options.output_dir) |output_dir| {
            // save to file
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
            // print to stdout
            try stdout.print("{s}", .{mlir_content_sir});
        }
    }

    // emit SIR text after conversion
    if (mlir_options.emit_sir_text or mlir_options.emit_bytecode) {
        if (!c.oraBuildSIRDispatcher(h.ctx, final_module)) {
            try stdout.print("Error: SIR dispatcher build failed\n", .{});
            try stdout.flush();
            std.process.exit(1);
        }
        if (!c.oraLegalizeSIRText(h.ctx, final_module)) {
            try stdout.print("Error: SIR text legalizer failed\n", .{});
            try stdout.flush();
            std.process.exit(1);
        }

        const sir_text_ref = c.oraEmitSIRText(h.ctx, final_module);
        defer if (sir_text_ref.data != null) {
            const mlir_c = @import("mlir_c_api");
            mlir_c.freeStringRef(sir_text_ref);
        };

        if (sir_text_ref.data == null or sir_text_ref.length == 0) {
            try stdout.print("Failed to emit SIR text\n", .{});
            return;
        }

        const sir_text = sir_text_ref.data[0..sir_text_ref.length];
        if (mlir_options.emit_sir_text) {
            if (mlir_options.output_dir) |output_dir| {
                std.fs.cwd().makeDir(output_dir) catch |err| switch (err) {
                    error.PathAlreadyExists => {},
                    else => return err,
                };

                const basename = std.fs.path.stem(file_path);
                const extension = ".sir";
                const filename = try std.mem.concat(allocator, u8, &[_][]const u8{ basename, extension });
                defer allocator.free(filename);
                const output_file = try std.fs.path.join(allocator, &[_][]const u8{ output_dir, filename });
                defer allocator.free(output_file);

                var sir_file = try std.fs.cwd().createFile(output_file, .{});
                defer sir_file.close();
                try sir_file.writeAll(sir_text);
            } else {
                try stdout.print("{s}", .{sir_text});
            }
        }

        if (mlir_options.emit_bytecode) {
            if (mlir_options.emit_sir_text and mlir_options.output_dir == null) {
                try stdout.print("\n", .{});
            }
            try emitBytecodeFromSirText(allocator, sir_text, file_path, mlir_options.output_dir, stdout);
        }
    }

    try stdout.flush();
}

test "simple test" {
    var list = ManagedArrayList(i32).init(std.testing.allocator);
    defer list.deinit(); // Try commenting this out and see if zig detects the memory leak!
    try list.append(42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}

test "use lexer module" {
    var lexer = lib.Lexer.init(std.testing.allocator, "contract Test {}");
    defer lexer.deinit();

    const tokens = try lexer.scanTokens();
    defer std.testing.allocator.free(tokens);

    // should have at least: contract, Test, {, }, EOF = 5 tokens
    try std.testing.expect(tokens.len >= 5);
    try std.testing.expect(tokens[0].type == lib.TokenType.Contract);
    try std.testing.expect(tokens[1].type == lib.TokenType.Identifier);
    try std.testing.expect(tokens[tokens.len - 1].type == lib.TokenType.Eof);
}

test "fuzz example" {
    const Context = struct {
        fn testOne(context: @This(), input: []const u8) anyerror!void {
            _ = context;
            // try passing `--fuzz` to `zig build test` and see if it manages to fail this test case!
            try std.testing.expect(!std.mem.eql(u8, "canyoufindme", input));
        }
    };
    try std.testing.fuzz(Context{}, Context.testOne, .{});
}
