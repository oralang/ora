const std = @import("std");
pub const ast = @import("../ast.zig");
pub const typer = @import("../typer.zig");
const semantics_errors = @import("semantics_errors.zig");
const semantics_state = @import("semantics_state.zig");

// Forward declaration for SemanticAnalyzer
const SemanticAnalyzer = @import("semantics_core.zig").SemanticAnalyzer;

/// Analyze contract declaration
pub fn analyzeContract(analyzer: *SemanticAnalyzer, contract: *ast.ContractNode) semantics_errors.SemanticError!void {
    // Set current contract for immutable variable tracking
    analyzer.current_contract = contract;
    defer analyzer.current_contract = null;

    // Clear immutable variables from previous contract
    analyzer.immutable_variables.clearRetainingCapacity();

    // Initialize contract context
    var contract_ctx = semantics_state.ContractContext.init(analyzer.allocator, contract.name);
    // Ensure cleanup on error - this guarantees memory is freed even if analysis fails
    defer contract_ctx.deinit();

    // First pass: collect contract member names and immutable variables
    for (contract.body) |*member| {
        switch (member.*) {
            .VariableDecl => |*var_decl| {
                if (var_decl.region == .Storage or var_decl.region == .Immutable) {
                    try contract_ctx.storage_variables.append(var_decl.name);
                }

                // Track immutable variables (including storage const)
                if (var_decl.region == .Immutable or (var_decl.region == .Storage and !var_decl.mutable)) {
                    try analyzer.immutable_variables.put(var_decl.name, SemanticAnalyzer.ImmutableVarInfo{
                        .name = var_decl.name,
                        .declared_span = var_decl.span,
                        .initialized = var_decl.value != null,
                        .init_span = if (var_decl.value != null) var_decl.span else null,
                    });
                }
            },
            .LogDecl => |*log_decl| {
                try contract_ctx.events.append(log_decl.name);
            },
            else => {},
        }
    }

    // Second pass: analyze contract members
    for (contract.body) |*member| {
        switch (member.*) {
            .Function => |*function| {
                // Track init function
                if (std.mem.eql(u8, function.name, "init")) {
                    if (contract_ctx.has_init) {
                        try semantics_errors.addErrorStatic(analyzer, "Duplicate init function", function.span);
                        return semantics_errors.SemanticError.DuplicateInitFunction;
                    }
                    contract_ctx.has_init = true;
                    contract_ctx.init_is_public = function.visibility == .Public;

                    // Set constructor flag for immutable variable validation
                    analyzer.in_constructor = true;
                    defer analyzer.in_constructor = false;
                }

                try contract_ctx.functions.append(function.name);
                try analyzeFunction(analyzer, function);
            },
            .VariableDecl => |*var_decl| {
                try analyzeVariableDecl(analyzer, var_decl);
            },
            .LogDecl => |*log_decl| {
                try analyzeLogDecl(analyzer, log_decl);
            },
            else => {
                try analyzer.analyzeNode(member);
            },
        }
    }

    // Validate all immutable variables are initialized
    try validateImmutableInitialization(analyzer);

    // Validate contract after all members are analyzed
    try validateContract(analyzer, &contract_ctx);
}

/// Pre-initialize contract symbols in type checker before type checking runs
pub fn preInitializeContractSymbols(analyzer: *SemanticAnalyzer, contract: *ast.ContractNode) semantics_errors.SemanticError!void {
    // Import standard library explicitly (using null namespace to import directly)
    analyzer.type_checker.processImport("std", null, ast.SourceSpan{ .line = 0, .column = 0, .length = 0 }) catch |err| {
        switch (err) {
            typer.TyperError.OutOfMemory => return semantics_errors.SemanticError.OutOfMemory,
            else => {
                // Log the error but continue - standard library initialization is not critical
                try semantics_errors.addWarningStatic(analyzer, "Failed to import standard library", ast.SourceSpan{ .line = 0, .column = 0, .length = 0 });
            },
        }
    };

    // Add storage variables and events to type checker symbol table
    for (contract.body) |*member| {
        switch (member.*) {
            .VariableDecl => |*var_decl| {
                if (var_decl.region == .Storage or var_decl.region == .Immutable) {
                    // Add to type checker's global scope
                    const ora_type = analyzer.type_checker.convertTypeInfoToOraType(var_decl.type_info) catch |err| {
                        switch (err) {
                            typer.TyperError.TypeMismatch => return semantics_errors.SemanticError.TypeMismatch,
                            typer.TyperError.OutOfMemory => return semantics_errors.SemanticError.OutOfMemory,
                            else => return semantics_errors.SemanticError.TypeMismatch,
                        }
                    };

                    const symbol = typer.Symbol{
                        .name = var_decl.name,
                        .typ = ora_type,
                        .region = var_decl.region,
                        .mutable = var_decl.mutable,
                        .span = var_decl.span,
                        .namespace = null,
                    };
                    try analyzer.type_checker.current_scope.declare(symbol);
                }
            },
            .LogDecl => |*log_decl| {
                // Add event to symbol table
                const event_symbol = typer.Symbol{
                    .name = log_decl.name,
                    .typ = typer.OraType.Unknown, // Event type
                    .region = ast.MemoryRegion.Stack,
                    .mutable = false,
                    .span = log_decl.span,
                    .namespace = null,
                };
                try analyzer.type_checker.current_scope.declare(event_symbol);
            },
            else => {},
        }
    }
}

// Placeholder functions that will be implemented in other modules
fn analyzeFunction(analyzer: *SemanticAnalyzer, function: *ast.FunctionNode) semantics_errors.SemanticError!void {
    _ = analyzer;
    _ = function;
    // Will be implemented in function_analyzer module
}

fn analyzeVariableDecl(analyzer: *SemanticAnalyzer, var_decl: *ast.VariableDeclNode) semantics_errors.SemanticError!void {
    _ = analyzer;
    _ = var_decl;
    // Will be implemented in appropriate analyzer module
}

fn analyzeLogDecl(analyzer: *SemanticAnalyzer, log_decl: *ast.LogDeclNode) semantics_errors.SemanticError!void {
    _ = analyzer;
    _ = log_decl;
    // Will be implemented in appropriate analyzer module
}

fn validateImmutableInitialization(analyzer: *SemanticAnalyzer) semantics_errors.SemanticError!void {
    _ = analyzer;
    // Will be implemented in validation module
}

fn validateContract(analyzer: *SemanticAnalyzer, contract_ctx: *semantics_state.ContractContext) semantics_errors.SemanticError!void {
    _ = analyzer;
    _ = contract_ctx;
    // Will be implemented in validation module
}
