// Main MLIR lowering orchestrator - coordinates modular components
// This file contains only the main lowerFunctionsToModule function and orchestration logic.
// All specific lowering functionality has been moved to modular files:
// - Type mapping: types.zig
// - Expression lowering: expressions.zig
// - Statement lowering: statements.zig
// - Declaration lowering: declarations.zig
// - Memory management: memory.zig
// - Symbol table: symbols.zig
// - Location tracking: locations.zig

const std = @import("std");
const lib = @import("ora_lib");
const c = @import("c.zig").c;

// Import modular components
const TypeMapper = @import("types.zig").TypeMapper;
const ExpressionLowerer = @import("expressions.zig").ExpressionLowerer;
const StatementLowerer = @import("statements.zig").StatementLowerer;
const DeclarationLowerer = @import("declarations.zig").DeclarationLowerer;
const MemoryManager = @import("memory.zig").MemoryManager;
const StorageMap = @import("memory.zig").StorageMap;
const SymbolTable = @import("symbols.zig").SymbolTable;
const ParamMap = @import("symbols.zig").ParamMap;
const LocalVarMap = @import("symbols.zig").LocalVarMap;
const LocationTracker = @import("locations.zig").LocationTracker;
const ErrorHandler = @import("error_handling.zig").ErrorHandler;
const ErrorContext = @import("error_handling.zig").ErrorContext;
const PassManager = @import("pass_manager.zig").PassManager;
const PassPipelineConfig = @import("pass_manager.zig").PassPipelineConfig;

/// Enhanced lowering result with error information and pass results
pub const LoweringResult = struct {
    module: c.MlirModule,
    errors: []const @import("error_handling.zig").LoweringError,
    warnings: []const @import("error_handling.zig").LoweringWarning,
    success: bool,
    pass_result: ?@import("pass_manager.zig").PassResult,
};

/// Main entry point for lowering Ora AST nodes to MLIR module with comprehensive error handling
/// This function orchestrates the modular lowering components and provides robust error reporting
pub fn lowerFunctionsToModuleWithErrors(ctx: c.MlirContext, nodes: []lib.AstNode, allocator: std.mem.Allocator) !LoweringResult {
    const loc = c.mlirLocationUnknownGet(ctx);
    const module = c.mlirModuleCreateEmpty(loc);
    const body = c.mlirModuleGetBody(module);

    // Initialize error handler
    var error_handler = ErrorHandler.init(allocator);
    defer error_handler.deinit();

    // Initialize modular components with error handling
    var type_mapper = TypeMapper.init(ctx, allocator);
    defer type_mapper.deinit();

    const locations = LocationTracker.init(ctx);
    const decl_lowerer = DeclarationLowerer.init(ctx, &type_mapper, locations);

    // Create global symbol table and storage map for the module
    var symbol_table = SymbolTable.init(allocator);
    defer symbol_table.deinit();

    var global_storage_map = StorageMap.init(allocator);
    defer global_storage_map.deinit();

    // Process all AST nodes using modular lowering components with error handling
    for (nodes) |node| {
        switch (node) {
            .Function => |func| {
                // Set error context for function lowering
                try error_handler.pushContext(ErrorContext.function(func.name));
                defer error_handler.popContext();

                // Validate function AST node
                const is_valid = error_handler.validateAstNode(func, func.span) catch {
                    try error_handler.reportError(.MalformedAst, func.span, "function validation failed", "check function structure");
                    continue; // Skip malformed function
                };
                if (!is_valid) {
                    continue; // Skip malformed function
                }

                // Lower function declaration using the modular declaration lowerer
                var local_var_map = LocalVarMap.init(allocator);
                defer local_var_map.deinit();

                const func_op = decl_lowerer.lowerFunction(&func, &global_storage_map, &local_var_map);

                // Validate the created MLIR operation
                if (error_handler.validateMlirOperation(func_op, func.span) catch false) {
                    c.mlirBlockAppendOwnedOperation(body, func_op);
                }
            },
            .Contract => |contract| {
                // Set error context for contract lowering
                try error_handler.pushContext(ErrorContext.contract(contract.name));
                defer error_handler.popContext();

                // Validate contract AST node
                const contract_valid = error_handler.validateAstNode(contract, contract.span) catch {
                    try error_handler.reportError(.MalformedAst, contract.span, "contract validation failed", "check contract structure");
                    continue; // Skip malformed contract
                };
                if (!contract_valid) {
                    continue; // Skip malformed contract
                }

                // Lower contract declaration using the modular declaration lowerer
                const contract_op = decl_lowerer.lowerContract(&contract);

                // Validate the created MLIR operation
                if (error_handler.validateMlirOperation(contract_op, contract.span) catch false) {
                    c.mlirBlockAppendOwnedOperation(body, contract_op);
                }
            },
            .VariableDecl => |var_decl| {
                // Validate variable declaration
                const var_valid = error_handler.validateAstNode(var_decl, var_decl.span) catch {
                    try error_handler.reportError(.MalformedAst, var_decl.span, "variable declaration validation failed", "check variable structure");
                    continue; // Skip malformed variable declaration
                };
                if (!var_valid) {
                    continue; // Skip malformed variable declaration
                }

                // Validate memory region
                const region_name = switch (var_decl.region) {
                    .Storage => "storage",
                    .Memory => "memory",
                    .TStore => "tstore",
                    .Stack => "stack",
                };

                const is_valid = error_handler.validateMemoryRegion(region_name, "variable declaration", var_decl.span) catch false;
                if (!is_valid) {
                    continue; // Skip invalid memory region
                }

                // Lower global variable declarations
                switch (var_decl.region) {
                    .Storage => {
                        if (var_decl.kind == .Immutable) {
                            // Handle immutable storage variables
                            const immutable_op = decl_lowerer.lowerImmutableDecl(&var_decl);
                            if (error_handler.validateMlirOperation(immutable_op, var_decl.span) catch false) {
                                c.mlirBlockAppendOwnedOperation(body, immutable_op);
                            }
                        } else {
                            const global_op = decl_lowerer.createGlobalDeclaration(&var_decl);
                            if (error_handler.validateMlirOperation(global_op, var_decl.span) catch false) {
                                c.mlirBlockAppendOwnedOperation(body, global_op);
                            }
                        }
                        _ = global_storage_map.getOrCreateAddress(var_decl.name) catch {};
                    },
                    .Memory => {
                        if (var_decl.kind == .Immutable) {
                            // Handle immutable memory variables
                            const immutable_op = decl_lowerer.lowerImmutableDecl(&var_decl);
                            if (error_handler.validateMlirOperation(immutable_op, var_decl.span) catch false) {
                                c.mlirBlockAppendOwnedOperation(body, immutable_op);
                            }
                        } else {
                            const memory_global_op = decl_lowerer.createMemoryGlobalDeclaration(&var_decl);
                            if (error_handler.validateMlirOperation(memory_global_op, var_decl.span) catch false) {
                                c.mlirBlockAppendOwnedOperation(body, memory_global_op);
                            }
                        }
                    },
                    .TStore => {
                        if (var_decl.kind == .Immutable) {
                            // Handle immutable transient storage variables
                            const immutable_op = decl_lowerer.lowerImmutableDecl(&var_decl);
                            if (error_handler.validateMlirOperation(immutable_op, var_decl.span) catch false) {
                                c.mlirBlockAppendOwnedOperation(body, immutable_op);
                            }
                        } else {
                            const tstore_global_op = decl_lowerer.createTStoreGlobalDeclaration(&var_decl);
                            if (error_handler.validateMlirOperation(tstore_global_op, var_decl.span) catch false) {
                                c.mlirBlockAppendOwnedOperation(body, tstore_global_op);
                            }
                        }
                    },
                    .Stack => {
                        // Stack variables at module level are not allowed
                        try error_handler.reportError(.InvalidMemoryRegion, var_decl.span, "stack variables are not allowed at module level", "use 'storage', 'memory', or 'tstore' instead");
                    },
                }
            },
            .StructDecl => |struct_decl| {
                const struct_valid = error_handler.validateAstNode(struct_decl, struct_decl.span) catch {
                    try error_handler.reportError(.MalformedAst, struct_decl.span, "struct declaration validation failed", "check struct structure");
                    continue;
                };
                if (struct_valid) {
                    const struct_op = decl_lowerer.lowerStruct(&struct_decl);
                    if (error_handler.validateMlirOperation(struct_op, struct_decl.span) catch false) {
                        c.mlirBlockAppendOwnedOperation(body, struct_op);
                    }
                }
            },
            .EnumDecl => |enum_decl| {
                const enum_valid = error_handler.validateAstNode(enum_decl, enum_decl.span) catch {
                    try error_handler.reportError(.MalformedAst, enum_decl.span, "enum declaration validation failed", "check enum structure");
                    continue;
                };
                if (enum_valid) {
                    const enum_op = decl_lowerer.lowerEnum(&enum_decl);
                    if (error_handler.validateMlirOperation(enum_op, enum_decl.span) catch false) {
                        c.mlirBlockAppendOwnedOperation(body, enum_op);
                    }
                }
            },
            .Import => |import_decl| {
                const import_valid = error_handler.validateAstNode(import_decl, import_decl.span) catch {
                    try error_handler.reportError(.MalformedAst, import_decl.span, "import declaration validation failed", "check import structure");
                    continue;
                };
                if (import_valid) {
                    const import_op = decl_lowerer.lowerImport(&import_decl);
                    if (error_handler.validateMlirOperation(import_op, import_decl.span) catch false) {
                        c.mlirBlockAppendOwnedOperation(body, import_op);
                    }
                }
            },
            .Constant => |const_decl| {
                const const_valid = error_handler.validateAstNode(const_decl, const_decl.span) catch {
                    try error_handler.reportError(.MalformedAst, const_decl.span, "constant declaration validation failed", "check constant structure");
                    continue;
                };
                if (const_valid) {
                    const const_op = decl_lowerer.lowerConstDecl(&const_decl);
                    if (error_handler.validateMlirOperation(const_op, const_decl.span) catch false) {
                        c.mlirBlockAppendOwnedOperation(body, const_op);
                    }
                }
            },
            .LogDecl => |log_decl| {
                const log_valid = error_handler.validateAstNode(log_decl, log_decl.span) catch {
                    try error_handler.reportError(.MalformedAst, log_decl.span, "log declaration validation failed", "check log structure");
                    continue;
                };
                if (log_valid) {
                    const log_op = decl_lowerer.lowerLogDecl(&log_decl);
                    if (error_handler.validateMlirOperation(log_op, log_decl.span) catch false) {
                        c.mlirBlockAppendOwnedOperation(body, log_op);
                    }
                }
            },
            .ErrorDecl => |error_decl| {
                const error_valid = error_handler.validateAstNode(error_decl, error_decl.span) catch {
                    try error_handler.reportError(.MalformedAst, error_decl.span, "error declaration validation failed", "check error structure");
                    continue;
                };
                if (error_valid) {
                    const error_op = decl_lowerer.lowerErrorDecl(&error_decl);
                    if (error_handler.validateMlirOperation(error_op, error_decl.span) catch false) {
                        c.mlirBlockAppendOwnedOperation(body, error_op);
                    }
                }
            },
            .Module => |module_node| {
                // Handle module-level declarations by processing their contents
                for (module_node.declarations) |decl| {
                    // Recursively process module declarations
                    // This could be implemented as a recursive call to lowerFunctionsToModuleWithErrors
                    try error_handler.reportWarning(.DeprecatedFeature, null, "nested modules are not fully supported yet");
                    _ = decl;
                }
            },
            .Block => |block| {
                // Blocks at top level are unusual - report as warning
                try error_handler.reportWarning(.DeprecatedFeature, null, "top-level blocks are not recommended");
                _ = block;
            },
            .Expression => |expr| {
                // Top-level expressions are unusual - report as warning
                try error_handler.reportWarning(.DeprecatedFeature, null, "top-level expressions are not recommended");
                _ = expr;
            },
            .Statement => |stmt| {
                // Top-level statements are unusual - report as warning
                try error_handler.reportWarning(.DeprecatedFeature, null, "top-level statements are not recommended");
                _ = stmt;
            },
            .TryBlock => |try_block| {
                // Try blocks at top level are unusual - report as warning
                try error_handler.reportWarning(.DeprecatedFeature, null, "top-level try blocks are not recommended");
                _ = try_block;
            },
        }
    }

    // Create and return the lowering result
    const result = LoweringResult{
        .module = module,
        .errors = try allocator.dupe(@import("error_handling.zig").LoweringError, error_handler.getErrors()),
        .warnings = try allocator.dupe(@import("error_handling.zig").LoweringWarning, error_handler.getWarnings()),
        .success = !error_handler.hasErrors(),
        .pass_result = null,
    };

    return result;
}

/// Main entry point with pass management support
pub fn lowerFunctionsToModuleWithPasses(ctx: c.MlirContext, nodes: []lib.AstNode, allocator: std.mem.Allocator, pass_config: ?PassPipelineConfig) !LoweringResult {
    // First, perform the basic lowering
    var lowering_result = try lowerFunctionsToModuleWithErrors(ctx, nodes, allocator);

    // If lowering failed, return early
    if (!lowering_result.success) {
        return lowering_result;
    }

    // Apply passes if configuration is provided
    if (pass_config) |config| {
        var pass_manager = PassManager.init(ctx, allocator);
        defer pass_manager.deinit();

        // Configure the pass pipeline
        pass_manager.configurePipeline(config);

        // Enable timing if requested
        if (config.enable_timing) {
            pass_manager.enableTiming();
        }

        // Enable IR printing if requested
        pass_manager.enableIRPrinting(config.ir_printing);

        // Run the passes
        const pass_result = try pass_manager.runPasses(lowering_result.module);

        // Verify the module after passes
        if (pass_result.success) {
            const verification_success = pass_manager.verifyModule(lowering_result.module);
            if (!verification_success) {
                // Create a new error for verification failure
                var error_handler = ErrorHandler.init(allocator);
                defer error_handler.deinit();

                try error_handler.reportError(.MlirOperationFailed, null, "module verification failed after pass execution", "check pass configuration and module structure");

                // Update the result with verification error
                const verification_errors = try allocator.dupe(@import("error_handling.zig").LoweringError, error_handler.getErrors());
                const combined_errors = try allocator.alloc(@import("error_handling.zig").LoweringError, lowering_result.errors.len + verification_errors.len);
                std.mem.copyForwards(@import("error_handling.zig").LoweringError, combined_errors[0..lowering_result.errors.len], lowering_result.errors);
                std.mem.copyForwards(@import("error_handling.zig").LoweringError, combined_errors[lowering_result.errors.len..], verification_errors);

                lowering_result.errors = combined_errors;
                lowering_result.success = false;
            }
        }

        // Update the result with pass information
        lowering_result.pass_result = pass_result;
        lowering_result.module = pass_result.modified_module;

        if (!pass_result.success) {
            lowering_result.success = false;
        }
    }

    return lowering_result;
}

/// Backward compatibility function - maintains the original interface
pub fn lowerFunctionsToModule(ctx: c.MlirContext, nodes: []lib.AstNode) c.MlirModule {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    const result = lowerFunctionsToModuleWithErrors(ctx, nodes, arena.allocator()) catch |err| {
        std.debug.print("Error during MLIR lowering: {s}\n", .{@errorName(err)});
        // Return empty module on error
        const loc = c.mlirLocationUnknownGet(ctx);
        return c.mlirModuleCreateEmpty(loc);
    };

    // Print diagnostics if there are any errors or warnings
    if (result.errors.len > 0 or result.warnings.len > 0) {
        var error_handler = ErrorHandler.init(arena.allocator());
        defer error_handler.deinit();

        // Add errors and warnings back to handler for printing
        for (result.errors) |err| {
            error_handler.errors.append(err) catch {};
        }
        for (result.warnings) |warn| {
            error_handler.warnings.append(warn) catch {};
        }

        error_handler.printDiagnostics(std.io.getStdErr().writer()) catch {};
    }

    return result.module;
}

/// Convenience function for debug builds with verification passes
pub fn lowerFunctionsToModuleDebug(ctx: c.MlirContext, nodes: []lib.AstNode, allocator: std.mem.Allocator) !LoweringResult {
    const debug_config = PassPipelineConfig.debug();
    return lowerFunctionsToModuleWithPasses(ctx, nodes, allocator, debug_config);
}

/// Convenience function for release builds with aggressive optimization
pub fn lowerFunctionsToModuleRelease(ctx: c.MlirContext, nodes: []lib.AstNode, allocator: std.mem.Allocator) !LoweringResult {
    const release_config = PassPipelineConfig.release();
    return lowerFunctionsToModuleWithPasses(ctx, nodes, allocator, release_config);
}

/// Convenience function with custom pass pipeline string
pub fn lowerFunctionsToModuleWithPipelineString(ctx: c.MlirContext, nodes: []lib.AstNode, allocator: std.mem.Allocator, pipeline_str: []const u8) !LoweringResult {
    // First, perform the basic lowering
    var lowering_result = try lowerFunctionsToModuleWithErrors(ctx, nodes, allocator);

    // If lowering failed, return early
    if (!lowering_result.success) {
        return lowering_result;
    }

    // Create pass manager and parse pipeline string
    var pass_manager = PassManager.init(ctx, allocator);
    defer pass_manager.deinit();

    try @import("pass_manager.zig").OraPassUtils.parsePipelineString(&pass_manager, pipeline_str);

    // Run the passes
    const pass_result = try pass_manager.runPasses(lowering_result.module);

    // Update the result
    lowering_result.pass_result = pass_result;
    lowering_result.module = pass_result.modified_module;

    if (!pass_result.success) {
        lowering_result.success = false;
    }

    return lowering_result;
}
