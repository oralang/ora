// ============================================================================
// MLIR Lowering Orchestrator
// ============================================================================
//
// Main entry point for converting Ora AST to MLIR IR.
//
// ARCHITECTURE:
//   Coordinates modular components for type mapping, expression lowering,
//   statement lowering, declarations, memory management, and symbol tables.
//
// COMPONENTS:
//   • types.zig - Type mapping (Ora → MLIR types)
//   • expressions.zig - Expression lowering
//   • statements.zig - Statement lowering
//   • declarations.zig - Declaration lowering
//   • memory.zig - Memory management
//   • symbols.zig - Symbol tables
//   • locations.zig - Location tracking
//
// ============================================================================

const std = @import("std");
const lib = @import("ora_lib");
const c = @import("c.zig").c;
pub const LocationTracker = @import("locations.zig").LocationTracker;

// MLIR constants used throughout the lowering system
pub const DEFAULT_INTEGER_BITS: u32 = 256;
pub const DEFAULT_INTEGER_TYPE_NAME: []const u8 = "i256";

// MLIR Context Management
pub const MlirContextHandle = struct {
    ctx: c.MlirContext,
    ora_dialect: @import("dialect.zig").OraDialect,
};

pub fn createContext(allocator: std.mem.Allocator) MlirContextHandle {
    const ctx = c.mlirContextCreate();

    // Register all standard MLIR dialects
    const registry = c.mlirDialectRegistryCreate();
    c.mlirRegisterAllDialects(registry);
    c.mlirContextAppendDialectRegistry(ctx, registry);
    c.mlirDialectRegistryDestroy(registry);
    c.mlirContextLoadAllAvailableDialects(ctx);

    // Initialize the Ora dialect
    var ora_dialect = @import("dialect.zig").OraDialect.init(ctx, allocator);
    ora_dialect.register() catch |err| {
        std.log.warn("Failed to register Ora dialect: {}", .{err});
    };

    return .{
        .ctx = ctx,
        .ora_dialect = ora_dialect,
    };
}

pub fn destroyContext(handle: MlirContextHandle) void {
    c.mlirContextDestroy(handle.ctx);
}

// Re-export ParamMap from symbols.zig to avoid duplication
pub const ParamMap = @import("symbols.zig").ParamMap;

// Re-export LocalVarMap from symbols.zig to avoid duplication
pub const LocalVarMap = @import("symbols.zig").LocalVarMap;

/// Symbol information structure
pub const SymbolInfo = struct {
    name: []const u8,
    type: c.MlirType,
    region: []const u8, // "storage", "memory", "tstore", "stack"
    value: ?c.MlirValue, // For variables that have been assigned values
    span: ?[]const u8, // Source span information
    symbol_kind: SymbolKind, // What kind of symbol this is
    variable_kind: ?lib.ast.Statements.VariableKind, // var, let, const, immutable (only for variables)
};

/// Different kinds of symbols that can be stored in the symbol table
pub const SymbolKind = enum {
    Variable,
    Function,
    Type,
    Parameter,
    Constant,
    Error,
};

/// Function symbol information
pub const FunctionSymbol = struct {
    name: []const u8,
    operation: c.MlirOperation, // The MLIR function operation
    param_types: []c.MlirType,
    return_type: c.MlirType,
    visibility: []const u8, // "pub", "private"
    attributes: std.StringHashMap(c.MlirAttribute), // Function attributes like inline, requires, ensures

    pub fn init(allocator: std.mem.Allocator, name: []const u8, operation: c.MlirOperation, param_types: []c.MlirType, return_type: c.MlirType) FunctionSymbol {
        return .{
            .name = name,
            .operation = operation,
            .param_types = param_types,
            .return_type = return_type,
            .visibility = "private",
            .attributes = std.StringHashMap(c.MlirAttribute).init(allocator),
        };
    }

    pub fn deinit(self: *FunctionSymbol) void {
        self.attributes.deinit();
    }
};

/// Type symbol information for structs and enums
pub const TypeSymbol = struct {
    name: []const u8,
    type_kind: TypeKind,
    mlir_type: c.MlirType,
    fields: ?[]FieldInfo, // For struct types
    variants: ?[]VariantInfo, // For enum types
    allocator: std.mem.Allocator, // Store allocator for cleanup

    pub const TypeKind = enum {
        Struct,
        Enum,
        Contract,
        Alias,
    };

    pub const FieldInfo = struct {
        name: []const u8,
        field_type: c.MlirType,
        offset: ?usize,
    };

    pub const VariantInfo = struct {
        name: []const u8,
        value: ?i64,
    };

    pub fn deinit(self: *TypeSymbol) void {
        if (self.fields) |fields| {
            self.allocator.free(fields);
        }
        if (self.variants) |variants| {
            self.allocator.free(variants);
        }
    }

    /// Get the index of an enum variant by name
    pub fn getVariantIndex(self: *const TypeSymbol, variant_name: []const u8) ?usize {
        if (self.variants) |variants| {
            for (variants, 0..) |variant, i| {
                if (std.mem.eql(u8, variant.name, variant_name)) {
                    return i;
                }
            }
        }
        return null;
    }
};

/// Symbol table with scope management
///
/// Memory ownership:
/// - Owns: scopes ArrayList and all StringHashMaps within it
/// - Owns: functions HashMap and all FunctionSymbol values
/// - Owns: types HashMap and all TypeSymbol array slices
/// - Owns: TypeSymbol.variants and TypeSymbol.fields slices
/// - Borrows: String keys (point to AST, caller owns)
/// - Must call: deinit() to avoid leaks
pub const SymbolTable = struct {
    allocator: std.mem.Allocator,
    scopes: std.ArrayList(std.StringHashMap(SymbolInfo)),
    current_scope: usize,

    // Separate tables for different symbol kinds
    functions: std.StringHashMap(FunctionSymbol),
    types: std.StringHashMap([]TypeSymbol),
    // Store constant declarations for lazy value creation
    constants: std.StringHashMap(*const lib.ast.ConstantNode),

    pub fn init(allocator: std.mem.Allocator) SymbolTable {
        var scopes = std.ArrayList(std.StringHashMap(SymbolInfo)){};
        const global_scope = std.StringHashMap(SymbolInfo).init(allocator);
        scopes.append(allocator, global_scope) catch unreachable;

        return .{
            .allocator = allocator,
            .scopes = scopes,
            .current_scope = 0,
            .functions = std.StringHashMap(FunctionSymbol).init(allocator),
            .types = std.StringHashMap([]TypeSymbol).init(allocator),
            .constants = std.StringHashMap(*const lib.ast.ConstantNode).init(allocator),
        };
    }

    pub fn deinit(self: *SymbolTable) void {
        for (self.scopes.items) |*scope| {
            scope.*.deinit();
        }
        self.scopes.deinit(self.allocator);

        // Clean up function symbols
        var func_iter = self.functions.iterator();
        while (func_iter.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.functions.deinit();

        // Clean up type symbols
        var type_iter = self.types.iterator();

        // Constants map doesn't own the AST nodes, just references them
        self.constants.deinit();
        while (type_iter.next()) |entry| {
            const type_array = entry.value_ptr.*;
            // First deinit the TypeSymbol's internal allocations
            type_array[0].deinit();
            // Then free the array itself
            self.allocator.free(type_array);
        }
        self.types.deinit();
    }

    /// Push a new scope
    pub fn pushScope(self: *SymbolTable) !void {
        const new_scope = std.StringHashMap(SymbolInfo).init(self.allocator);
        try self.scopes.append(self.allocator, new_scope);
        self.current_scope += 1;
    }

    /// Pop the current scope
    pub fn popScope(self: *SymbolTable) void {
        if (self.current_scope > 0) {
            var scope = self.scopes.orderedRemove(self.current_scope);
            scope.deinit();
            self.current_scope -= 1;
        }
    }

    /// Add a variable symbol to the current scope
    pub fn addSymbol(self: *SymbolTable, name: []const u8, type_info: c.MlirType, region: lib.ast.Statements.MemoryRegion, span: ?[]const u8, variable_kind: ?lib.ast.Statements.VariableKind) !void {
        const region_str = switch (region) {
            .Storage => "storage",
            .Memory => "memory",
            .TStore => "tstore",
            .Stack => "stack",
        };
        std.debug.print("[addSymbol] Adding symbol: {s}, variable_kind: {any}, scope: {}\n", .{ name, variable_kind, self.current_scope });
        const symbol_info = SymbolInfo{
            .name = name,
            .type = type_info,
            .region = region_str,
            .value = null,
            .span = span,
            .symbol_kind = .Variable,
            .variable_kind = variable_kind,
        };

        try self.scopes.items[self.current_scope].put(name, symbol_info);
        std.debug.print("[addSymbol] Symbol added successfully: {s}, stored variable_kind: {any}\n", .{ name, symbol_info.variable_kind });
    }

    /// Add a parameter symbol to the current scope
    pub fn addParameter(self: *SymbolTable, name: []const u8, type_info: c.MlirType, value: c.MlirValue, span: ?[]const u8) !void {
        const symbol_info = SymbolInfo{
            .name = name,
            .type = type_info,
            .region = "stack", // Parameters are stack-based
            .value = value,
            .span = span,
            .symbol_kind = .Parameter,
            .variable_kind = null, // Parameters don't have variable_kind
        };

        try self.scopes.items[self.current_scope].put(name, symbol_info);
    }

    /// Add a constant symbol to the current scope
    pub fn addConstant(self: *SymbolTable, name: []const u8, type_info: c.MlirType, value: ?c.MlirValue, span: ?[]const u8) !void {
        const symbol_info = SymbolInfo{
            .name = name,
            .type = type_info,
            .region = "stack", // Constants are stack-based
            .value = value,
            .span = span,
            .symbol_kind = .Constant,
            .variable_kind = null, // Constants don't have variable_kind (they use symbol_kind)
        };

        try self.scopes.items[self.current_scope].put(name, symbol_info);
    }

    /// Add an error symbol to the current scope
    pub fn addError(self: *SymbolTable, name: []const u8, type_info: c.MlirType, span: ?[]const u8) !void {
        const symbol_info = SymbolInfo{
            .name = name,
            .type = type_info,
            .region = "stack", // Errors are stack-based (for error values)
            .value = null,
            .span = span,
            .symbol_kind = .Error,
            .variable_kind = null, // Errors don't have variable_kind
        };

        try self.scopes.items[self.current_scope].put(name, symbol_info);
    }

    /// Register a constant declaration for lazy value creation
    pub fn registerConstantDecl(self: *SymbolTable, name: []const u8, const_decl: *const lib.ast.ConstantNode) !void {
        try self.constants.put(name, const_decl);
    }

    /// Look up a constant declaration
    pub fn lookupConstantDecl(self: *const SymbolTable, name: []const u8) ?*const lib.ast.ConstantNode {
        return self.constants.get(name);
    }

    /// Add a function symbol to the global function table
    pub fn addFunction(self: *SymbolTable, name: []const u8, operation: c.MlirOperation, param_types: []c.MlirType, return_type: c.MlirType) !void {
        const func_symbol = FunctionSymbol.init(self.allocator, name, operation, param_types, return_type);
        try self.functions.put(name, func_symbol);
    }

    /// Add a type symbol (struct, enum) to the global type table
    pub fn addType(self: *SymbolTable, name: []const u8, type_symbol: TypeSymbol) !void {
        const type_symbol_array = try self.allocator.alloc(TypeSymbol, 1);
        type_symbol_array[0] = type_symbol;
        try self.types.put(name, type_symbol_array);
    }

    /// Look up a symbol starting from the current scope and going outward
    pub fn lookupSymbol(self: *const SymbolTable, name: []const u8) ?SymbolInfo {
        var scope_idx: usize = self.current_scope;
        while (true) {
            if (self.scopes.items[scope_idx].get(name)) |symbol| {
                std.debug.print("[lookupSymbol] Found symbol: {s} in scope {}, symbol_kind: {s}, variable_kind: {any}\n", .{ name, scope_idx, @tagName(symbol.symbol_kind), symbol.variable_kind });
                return symbol;
            }
            if (scope_idx == 0) break;
            scope_idx -= 1;
        }
        std.debug.print("[lookupSymbol] Symbol not found: {s}\n", .{name});
        return null;
    }

    /// Update a symbol's value
    pub fn updateSymbolValue(self: *SymbolTable, name: []const u8, value: c.MlirValue) !void {
        var scope_idx: usize = self.current_scope;
        while (true) {
            if (self.scopes.items[scope_idx].get(name)) |symbol| {
                std.debug.print("[updateSymbolValue] Found symbol: {s} in scope {}, variable_kind before: {any}\n", .{ name, scope_idx, symbol.variable_kind });
                var updated_symbol = symbol;
                updated_symbol.value = value;
                std.debug.print("[updateSymbolValue] variable_kind after: {any}\n", .{updated_symbol.variable_kind});
                try self.scopes.items[scope_idx].put(name, updated_symbol);
                return;
            }
            if (scope_idx == 0) break;
            scope_idx -= 1;
        }
        // If symbol not found, add it to current scope
        std.debug.print("[updateSymbolValue] WARNING: Symbol not found: {s}, adding new symbol with variable_kind=null\n", .{name});
        try self.addSymbol(name, c.mlirValueGetType(value), lib.ast.Statements.MemoryRegion.Stack, null, null);
        if (self.scopes.items[self.current_scope].get(name)) |symbol| {
            var updated_symbol = symbol;
            updated_symbol.value = value;
            try self.scopes.items[self.current_scope].put(name, updated_symbol);
        }
    }

    /// Update a symbol's type
    pub fn updateSymbolType(self: *SymbolTable, name: []const u8, type_info: c.MlirType) !void {
        var scope_idx: usize = self.current_scope;
        while (true) {
            if (self.scopes.items[scope_idx].get(name)) |symbol| {
                var updated_symbol = symbol;
                updated_symbol.type = type_info;
                try self.scopes.items[scope_idx].put(name, updated_symbol);
                return;
            }
            if (scope_idx == 0) break;
            scope_idx -= 1;
        }
        // If symbol not found, this is an error - we can't update a non-existent symbol
        return error.SymbolNotFound;
    }

    /// Check if a symbol exists in any scope
    pub fn hasSymbol(self: *const SymbolTable, name: []const u8) bool {
        return self.lookupSymbol(name) != null;
    }

    /// Look up a function symbol
    pub fn lookupFunction(self: *const SymbolTable, name: []const u8) ?FunctionSymbol {
        return self.functions.get(name);
    }

    /// Look up a type symbol
    pub fn lookupType(self: *const SymbolTable, name: []const u8) ?*TypeSymbol {
        if (self.types.get(name)) |type_array| {
            return &type_array[0];
        }
        return null;
    }

    /// Check if a function exists
    pub fn hasFunction(self: *const SymbolTable, name: []const u8) bool {
        return self.functions.contains(name);
    }

    /// Check if a type exists
    pub fn hasType(self: *const SymbolTable, name: []const u8) bool {
        return self.types.contains(name);
    }
};

// Import modular components
const TypeMapper = @import("types.zig").TypeMapper;
const ExpressionLowerer = @import("expressions.zig").ExpressionLowerer;
const StatementLowerer = @import("statements.zig").StatementLowerer;
const DeclarationLowerer = @import("declarations.zig").DeclarationLowerer;
const pass_manager = @import("pass_manager.zig");
const verification = @import("verification.zig");
const MemoryManager = @import("memory.zig").MemoryManager;
const StorageMap = @import("memory.zig").StorageMap;
const ErrorHandler = @import("error_handling.zig").ErrorHandler;
const ErrorContext = @import("error_handling.zig").ErrorContext;
const LoweringError = @import("error_handling.zig").LoweringError;
const LoweringWarning = @import("error_handling.zig").LoweringWarning;
const error_handling = @import("error_handling.zig");
const PassManager = @import("pass_manager.zig").PassManager;
const PassPipelineConfig = @import("pass_manager.zig").PassPipelineConfig;

/// Enhanced lowering result with error information and pass results
pub const LoweringResult = struct {
    module: c.MlirModule,
    errors: []LoweringError,
    warnings: []LoweringWarning,
    success: bool,
    pass_result: ?@import("pass_manager.zig").PassResult,

    /// Free the allocated errors and warnings
    pub fn deinit(self: *LoweringResult, allocator: std.mem.Allocator) void {
        // Free deep-copied error messages and suggestions
        for (self.errors) |err| {
            allocator.free(err.message);
            if (err.suggestion) |s| {
                allocator.free(s);
            }
        }
        // Free deep-copied warning messages
        for (self.warnings) |warn| {
            allocator.free(warn.message);
        }
        allocator.free(self.errors);
        allocator.free(self.warnings);
        if (self.pass_result) |*pr| {
            pr.deinit();
        }
    }
};

/// Convert semantic analysis symbol table to MLIR symbol table
/// Note: Type registration (enums, structs) is now handled in the MLIR lowering phase
/// This function only handles variable and function symbols from semantic analysis
pub fn convertSemanticSymbolTable(semantic_table: *const lib.semantics.state.SymbolTable, mlir_table: *SymbolTable, ctx: c.MlirContext, allocator: std.mem.Allocator) !void {
    // Note: Type registration (enums, structs) is now handled directly in the MLIR lowering phase
    // where we have access to the MLIR context and can create proper MLIR types.
    // This function is kept for future use if we need to convert other semantic symbols.
    _ = semantic_table;
    _ = mlir_table;
    _ = ctx;
    _ = allocator;
}

/// Helper to get span from AstNode
fn getNodeSpan(node: *const lib.AstNode) ?lib.ast.SourceSpan {
    return switch (node.*) {
        .Contract => |contract| contract.span,
        .Function => |func| func.span,
        .VariableDecl => |var_decl| var_decl.span,
        .StructDecl => |struct_decl| struct_decl.span,
        .EnumDecl => |enum_decl| enum_decl.span,
        .LogDecl => |log_decl| log_decl.span,
        .Import => |import| import.span,
        .ErrorDecl => |error_decl| error_decl.span,
        .ContractInvariant => |invariant| invariant.span,
        else => null,
    };
}

/// Main entry point for lowering Ora AST nodes to MLIR module with semantic analysis symbol table
/// This function uses the semantic analysis symbol table for type resolution
pub fn lowerFunctionsToModuleWithSemanticTable(ctx: c.MlirContext, nodes: []lib.AstNode, allocator: std.mem.Allocator, semantic_table: *const lib.semantics.state.SymbolTable, source_filename: ?[]const u8) !LoweringResult {
    // Create location tracker to get proper module location
    const location_tracker = if (source_filename) |fname|
        LocationTracker.initWithFilename(ctx, fname)
    else
        LocationTracker.init(ctx);

    // Use first node's location if available, otherwise unknown (module wrapper has no single location)
    const loc = if (nodes.len > 0)
        location_tracker.createLocation(getNodeSpan(&nodes[0]))
    else
        location_tracker.getUnknownLocation();

    const module = c.mlirModuleCreateEmpty(loc);
    _ = c.mlirModuleGetBody(module);

    // Initialize error handler
    var error_handler = ErrorHandler.init(allocator);
    defer error_handler.deinit();

    // Initialize Ora dialect
    var ora_dialect = @import("dialect.zig").OraDialect.init(ctx, allocator);
    ora_dialect.register() catch |err| {
        std.log.warn("Failed to register Ora dialect: {}", .{err});
    };

    const locations = if (source_filename) |fname|
        LocationTracker.initWithFilename(ctx, fname)
    else
        LocationTracker.init(ctx);

    // Create global symbol table and storage map for the module
    var symbol_table = SymbolTable.init(allocator);
    defer symbol_table.deinit();

    // Convert semantic analysis symbol table to MLIR symbol table
    try convertSemanticSymbolTable(semantic_table, &symbol_table, ctx, allocator);

    // Create builtin registry for standard library functions
    var builtin_registry = lib.semantics.builtins.BuiltinRegistry.init(allocator) catch {
        std.debug.print("FATAL: Failed to initialize builtin registry\n", .{});
        @panic("Builtin registry initialization failed");
    };
    defer builtin_registry.deinit();

    // Initialize modular components with error handling and symbol table
    var type_mapper = TypeMapper.initWithSymbolTable(ctx, allocator, &symbol_table);
    defer type_mapper.deinit();

    const decl_lowerer = DeclarationLowerer.withErrorHandlerAndDialectAndSymbolTable(ctx, &type_mapper, locations, &error_handler, &ora_dialect, &symbol_table, &builtin_registry);

    var global_storage_map = StorageMap.init(allocator);
    defer global_storage_map.deinit();

    // Process all declarations (enums, structs, functions, contracts)
    // Note: Type declarations are already registered from semantic analysis
    for (nodes) |node| {
        switch (node) {
            .Function => |func| {
                try decl_lowerer.lowerFunction(func, &global_storage_map);
            },
            .Contract => |contract| {
                try decl_lowerer.lowerContract(contract, &global_storage_map);
            },
            .VariableDecl => |var_decl| {
                try decl_lowerer.lowerVariableDecl(var_decl, &global_storage_map);
            },
            .Import => |import_decl| {
                try decl_lowerer.lowerImport(import_decl);
            },
            .Constant => |const_decl| {
                try decl_lowerer.lowerConstant(const_decl);
            },
            .LogDecl => |log_decl| {
                try decl_lowerer.lowerLogDecl(log_decl);
            },
            .ErrorDecl => |error_decl| {
                try decl_lowerer.lowerErrorDecl(error_decl);
            },
            .Block => |block| {
                try decl_lowerer.lowerBlock(block);
            },
            .Expression => |expr| {
                try decl_lowerer.lowerExpression(expr);
            },
            .Statement => |stmt| {
                try decl_lowerer.lowerStatement(stmt);
            },
            .TryBlock => |try_block| {
                try decl_lowerer.lowerTryBlock(try_block);
            },
            .EnumDecl, .StructDecl => {
                // Skip enum and struct declarations - already processed in semantic analysis
            },
            .ContractInvariant => {
                // Skip contract invariants - specification-only, don't generate code
            },
        }
    }

    // Collect errors and warnings
    const errors = try error_handler.getErrors();
    const warnings = try error_handler.getWarnings();

    const result = LoweringResult{
        .module = module,
        .errors = errors,
        .warnings = warnings,
        .success = errors.len == 0,
        .pass_result = null,
    };

    return result;
}

/// Main entry point for lowering Ora AST nodes to MLIR module with comprehensive error handling
/// This function orchestrates the modular lowering components and provides robust error reporting
pub fn lowerFunctionsToModuleWithErrors(ctx: c.MlirContext, nodes: []lib.AstNode, allocator: std.mem.Allocator, source_filename: ?[]const u8) !LoweringResult {
    const location_tracker = if (source_filename) |fname|
        LocationTracker.initWithFilename(ctx, fname)
    else
        LocationTracker.init(ctx);
    const loc = location_tracker.getUnknownLocation();
    const module = c.mlirModuleCreateEmpty(loc);
    const body = c.mlirModuleGetBody(module);

    // Initialize error handler
    var error_handler = ErrorHandler.init(allocator);
    defer error_handler.deinit();

    // Initialize Ora dialect
    var ora_dialect = @import("dialect.zig").OraDialect.init(ctx, allocator);
    ora_dialect.register() catch |err| {
        std.log.warn("Failed to register Ora dialect: {}", .{err});
    };

    const locations = if (source_filename) |fname|
        LocationTracker.initWithFilename(ctx, fname)
    else
        LocationTracker.init(ctx);

    // Create global symbol table and storage map for the module
    var symbol_table = SymbolTable.init(allocator);
    defer symbol_table.deinit();

    // Create builtin registry for standard library functions
    var builtin_registry = lib.semantics.builtins.BuiltinRegistry.init(allocator) catch {
        std.debug.print("FATAL: Failed to initialize builtin registry\n", .{});
        @panic("Builtin registry initialization failed");
    };
    defer builtin_registry.deinit();

    // Initialize modular components with error handling and symbol table
    var type_mapper = TypeMapper.initWithSymbolTable(ctx, allocator, &symbol_table);
    defer type_mapper.deinit();

    const decl_lowerer = DeclarationLowerer.withErrorHandlerAndDialectAndSymbolTable(ctx, &type_mapper, locations, &error_handler, &ora_dialect, &symbol_table, &builtin_registry);

    var global_storage_map = StorageMap.init(allocator);
    defer global_storage_map.deinit();

    // First pass: Process all type declarations (enums, structs) to register them in symbol table
    for (nodes) |node| {
        switch (node) {
            .EnumDecl => |enum_decl| {
                const enum_valid = error_handler.validateAstNode(enum_decl, enum_decl.span) catch {
                    try error_handler.reportError(.MalformedAst, enum_decl.span, "enum declaration validation failed", "check enum structure");
                    continue;
                };
                if (enum_valid) {
                    const enum_op = decl_lowerer.lowerEnum(&enum_decl);
                    if (error_handler.validateMlirOperation(enum_op, enum_decl.span) catch false) {
                        c.mlirBlockAppendOwnedOperation(body, enum_op);

                        // Register enum type in symbol table
                        const enum_type = decl_lowerer.createEnumType(&enum_decl);

                        // Allocate variant info array directly
                        const variants_slice = allocator.alloc(@import("lower.zig").TypeSymbol.VariantInfo, enum_decl.variants.len) catch {
                            std.debug.print("ERROR: Failed to allocate variants slice for enum: {s}\n", .{enum_decl.name});
                            continue;
                        };

                        for (enum_decl.variants, 0..) |variant, i| {
                            // Use index as value for implicit enum variants
                            // Explicit value evaluation handled by constant expression evaluator
                            variants_slice[i] = .{
                                .name = variant.name,
                                .value = null, // Will be set to index during symbol table creation
                            };
                        }

                        // Check if type already exists before allocating
                        if (symbol_table.lookupType(enum_decl.name)) |_| {
                            std.debug.print("WARNING: Duplicate enum type: {s}, skipping\n", .{enum_decl.name});
                            allocator.free(variants_slice);
                            continue;
                        }

                        const type_symbol = @import("lower.zig").TypeSymbol{
                            .name = enum_decl.name,
                            .type_kind = .Enum,
                            .mlir_type = enum_type,
                            .fields = null,
                            .variants = variants_slice,
                            .allocator = allocator,
                        };

                        symbol_table.addType(enum_decl.name, type_symbol) catch {
                            // Free the variants_slice if addType fails
                            allocator.free(variants_slice);
                            std.debug.print("ERROR: Failed to register enum type: {s}\n", .{enum_decl.name});
                        };
                    }
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

                        // Register struct type in symbol table
                        const struct_type = decl_lowerer.createStructType(&struct_decl);

                        // Allocate field info array directly
                        const fields_slice = allocator.alloc(@import("lower.zig").TypeSymbol.FieldInfo, struct_decl.fields.len) catch {
                            std.debug.print("ERROR: Failed to allocate fields slice for struct: {s}\n", .{struct_decl.name});
                            continue;
                        };

                        // Calculate field offsets based on cumulative sizes
                        var current_offset: usize = 0;
                        for (struct_decl.fields, 0..) |field, i| {
                            const field_type = decl_lowerer.type_mapper.toMlirType(field.type_info);

                            // Calculate offset: for EVM, each field occupies a storage slot
                            // In memory/stack, we use byte offsets based on type width
                            // For simplicity, use slot-based offsets (32 bytes per field in EVM)
                            fields_slice[i] = .{
                                .name = field.name,
                                .field_type = field_type,
                                .offset = current_offset,
                            };

                            // Increment offset: in EVM storage, each slot is 32 bytes
                            // For proper layout, this should consider actual type sizes
                            current_offset += 32; // EVM storage slot size in bytes
                        }

                        // Check if type already exists before allocating
                        if (symbol_table.lookupType(struct_decl.name)) |_| {
                            std.debug.print("WARNING: Duplicate struct type: {s}, skipping\n", .{struct_decl.name});
                            allocator.free(fields_slice);
                            continue;
                        }

                        const type_symbol = @import("lower.zig").TypeSymbol{
                            .name = struct_decl.name,
                            .type_kind = .Struct,
                            .mlir_type = struct_type,
                            .fields = fields_slice,
                            .variants = null,
                            .allocator = allocator,
                        };

                        symbol_table.addType(struct_decl.name, type_symbol) catch {
                            // Free the fields_slice if addType fails
                            allocator.free(fields_slice);
                            std.debug.print("ERROR: Failed to register struct type: {s}\n", .{struct_decl.name});
                        };
                    }
                }
            },
            else => {
                // Skip non-type declarations in first pass
            },
        }
    }

    // Second pass: Process all other declarations (functions, contracts, etc.)
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
                const is_valid = error_handler.validateMemoryRegion(var_decl.region, "variable declaration", var_decl.span) catch false;
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

                        // Register constant declaration for lazy value creation
                        symbol_table.registerConstantDecl(const_decl.name, &const_decl) catch {
                            std.debug.print("ERROR: Failed to register constant declaration: {s}\n", .{const_decl.name});
                        };
                        // Add constant to symbol table with null value - will be created lazily when referenced
                        const const_type = type_mapper.toMlirType(const_decl.typ);
                        symbol_table.addConstant(const_decl.name, const_type, null, null) catch {
                            std.debug.print("ERROR: Failed to add constant to symbol table: {s}\n", .{const_decl.name});
                        };
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
            .ContractInvariant => {
                // Skip contract invariants - they are specification-only and don't generate code
                continue;
            },
            .Module => |module_node| {
                // Set error context for module lowering
                try error_handler.pushContext(ErrorContext.module(module_node.name orelse "unnamed"));
                defer error_handler.popContext();

                // Validate module AST node
                const module_valid = error_handler.validateAstNode(module_node, module_node.span) catch {
                    try error_handler.reportError(.MalformedAst, module_node.span, "module validation failed", "check module structure");
                    continue;
                };
                if (!module_valid) {
                    continue;
                }

                // Process module imports first
                for (module_node.imports) |import| {
                    const import_valid = error_handler.validateAstNode(import, import.span) catch {
                        try error_handler.reportError(.MalformedAst, import.span, "import validation failed", "check import structure");
                        continue;
                    };
                    if (import_valid) {
                        const import_op = decl_lowerer.lowerImport(&import);
                        if (error_handler.validateMlirOperation(import_op, import.span) catch false) {
                            c.mlirBlockAppendOwnedOperation(body, import_op);
                        }
                    }
                }

                // Process module declarations recursively
                for (module_node.declarations) |decl| {
                    // Recursively process module declarations
                    // This creates a proper module structure in MLIR
                    // Note: We can't call lowerModule on individual declarations
                    // Instead, we need to handle them based on their type
                    switch (decl) {
                        .Function => |func| {
                            // Create a local variable map for this function
                            var local_var_map = LocalVarMap.init(allocator);
                            defer local_var_map.deinit();

                            const func_op = decl_lowerer.lowerFunction(&func, &global_storage_map, &local_var_map);
                            if (error_handler.validateMlirOperation(func_op, func.span) catch false) {
                                c.mlirBlockAppendOwnedOperation(body, func_op);
                            }
                        },
                        .Contract => |contract| {
                            const contract_op = decl_lowerer.lowerContract(&contract);
                            if (error_handler.validateMlirOperation(contract_op, contract.span) catch false) {
                                c.mlirBlockAppendOwnedOperation(body, contract_op);
                            }
                        },
                        .VariableDecl => |var_decl| {
                            // Handle variable declarations within module with graceful degradation
                            try error_handler.reportGracefulDegradation("variable declarations within modules", "global variable declarations", var_decl.span);
                            // Create a placeholder operation to allow compilation to continue
                            const placeholder_op = decl_lowerer.createVariablePlaceholder(&var_decl);
                            if (error_handler.validateMlirOperation(placeholder_op, var_decl.span) catch false) {
                                c.mlirBlockAppendOwnedOperation(body, placeholder_op);
                            }
                        },
                        .StructDecl => |struct_decl| {
                            const struct_op = decl_lowerer.lowerStruct(&struct_decl);
                            if (error_handler.validateMlirOperation(struct_op, struct_decl.span) catch false) {
                                c.mlirBlockAppendOwnedOperation(body, struct_op);
                            }
                        },
                        .EnumDecl => |enum_decl| {
                            const enum_op = decl_lowerer.lowerEnum(&enum_decl);
                            if (error_handler.validateMlirOperation(enum_op, enum_decl.span) catch false) {
                                c.mlirBlockAppendOwnedOperation(body, enum_op);
                            }
                        },
                        .Import => |import_decl| {
                            const import_op = decl_lowerer.lowerImport(&import_decl);
                            if (error_handler.validateMlirOperation(import_op, import_decl.span) catch false) {
                                c.mlirBlockAppendOwnedOperation(body, import_op);
                            }
                        },
                        .Constant => |const_decl| {
                            const const_op = decl_lowerer.lowerConstDecl(&const_decl);
                            if (error_handler.validateMlirOperation(const_op, const_decl.span) catch false) {
                                c.mlirBlockAppendOwnedOperation(body, const_op);
                            }
                        },
                        .LogDecl => |log_decl| {
                            const log_op = decl_lowerer.lowerLogDecl(&log_decl);
                            if (error_handler.validateMlirOperation(log_op, log_decl.span) catch false) {
                                c.mlirBlockAppendOwnedOperation(body, log_op);
                            }
                        },
                        .ErrorDecl => |error_decl| {
                            const error_op = decl_lowerer.lowerErrorDecl(&error_decl);
                            if (error_handler.validateMlirOperation(error_op, error_decl.span) catch false) {
                                c.mlirBlockAppendOwnedOperation(body, error_op);
                            }
                        },
                        .ContractInvariant => {
                            // Skip contract invariants - they are specification-only
                            continue;
                        },
                        .Module => |nested_module| {
                            // Recursively handle nested modules with graceful degradation
                            try error_handler.reportGracefulDegradation("nested modules", "flat module structure", nested_module.span);
                            // Create a placeholder operation to allow compilation to continue
                            const placeholder_op = decl_lowerer.createModulePlaceholder(&nested_module);
                            if (error_handler.validateMlirOperation(placeholder_op, nested_module.span) catch false) {
                                c.mlirBlockAppendOwnedOperation(body, placeholder_op);
                            }
                        },
                        .Block => |block| {
                            const block_op = decl_lowerer.lowerBlock(&block);
                            if (error_handler.validateMlirOperation(block_op, block.span) catch false) {
                                c.mlirBlockAppendOwnedOperation(body, block_op);
                            }
                        },
                        .Expression => |expr| {
                            // Handle expressions within module with graceful degradation
                            try error_handler.reportGracefulDegradation("expressions within modules", "expression capture operations", error_handling.getSpanFromExpression(expr));
                            // Create a placeholder operation to allow compilation to continue
                            const expr_lowerer = ExpressionLowerer.init(ctx, body, &type_mapper, null, null, null, &symbol_table, &builtin_registry, locations, &ora_dialect);
                            const expr_value = expr_lowerer.lowerExpression(expr);
                            const expr_op = expr_lowerer.createExpressionCapture(expr_value, error_handling.getSpanFromExpression(expr));
                            if (error_handler.validateMlirOperation(expr_op, error_handling.getSpanFromExpression(expr)) catch false) {
                                c.mlirBlockAppendOwnedOperation(body, expr_op);
                            }
                        },
                        .Statement => |stmt| {
                            // Handle statements within modules with graceful degradation
                            try error_handler.reportGracefulDegradation("statements within modules", "statement lowering operations", error_handling.getSpanFromStatement(stmt));
                            // Create a placeholder operation to allow compilation to continue
                            const expr_lowerer = ExpressionLowerer.init(ctx, body, &type_mapper, null, null, null, &symbol_table, &builtin_registry, locations, &ora_dialect);
                            const stmt_lowerer = StatementLowerer.init(ctx, body, &type_mapper, &expr_lowerer, null, null, null, locations, &symbol_table, &builtin_registry, std.heap.page_allocator, null, null, &ora_dialect, &[_]*lib.ast.Expressions.ExprNode{});
                            stmt_lowerer.lowerStatement(stmt) catch {
                                try error_handler.reportError(.MlirOperationFailed, error_handling.getSpanFromStatement(stmt), "failed to lower top-level statement", "check statement structure and dependencies");
                                continue;
                            };
                        },
                        .TryBlock => |try_block| {
                            const try_block_op = decl_lowerer.lowerTryBlock(&try_block);
                            if (error_handler.validateMlirOperation(try_block_op, try_block.span) catch false) {
                                c.mlirBlockAppendOwnedOperation(body, try_block_op);
                            }
                        },
                    }
                }
            },
            .Block => |block| {
                // Set error context for block lowering
                try error_handler.pushContext(ErrorContext.block("top-level"));
                defer error_handler.popContext();

                // Validate block AST node
                const block_valid = error_handler.validateAstNode(block, block.span) catch {
                    try error_handler.reportError(.MalformedAst, block.span, "block validation failed", "check block structure");
                    continue;
                };
                if (!block_valid) {
                    continue;
                }

                // Lower top-level block using the declaration lowerer
                const block_op = decl_lowerer.lowerBlock(&block);
                if (error_handler.validateMlirOperation(block_op, block.span) catch false) {
                    c.mlirBlockAppendOwnedOperation(body, block_op);
                }
            },
            .Expression => |expr| {
                // Set error context for expression lowering
                try error_handler.pushContext(ErrorContext.expression());
                defer error_handler.popContext();

                // Validate expression AST node
                const expr_valid = error_handler.validateAstNode(expr, error_handling.getSpanFromExpression(expr)) catch {
                    try error_handler.reportError(.MalformedAst, error_handling.getSpanFromExpression(expr), "expression validation failed", "check expression structure");
                    continue;
                };
                if (!expr_valid) {
                    continue;
                }

                // Create a temporary expression lowerer for top-level expressions
                const expr_lowerer = ExpressionLowerer.init(ctx, body, &type_mapper, null, null, null, &symbol_table, null, locations, &ora_dialect);
                const expr_value = expr_lowerer.lowerExpression(expr);

                // For top-level expressions, we need to create a proper operation
                // This could be a constant or a call to a function that evaluates the expression
                // For now, we'll create a simple operation that captures the expression value
                const expr_op = expr_lowerer.createExpressionCapture(expr_value, error_handling.getSpanFromExpression(expr));
                if (error_handler.validateMlirOperation(expr_op, error_handling.getSpanFromExpression(expr)) catch false) {
                    c.mlirBlockAppendOwnedOperation(body, expr_op);
                }
            },
            .Statement => |stmt| {
                // Set error context for statement lowering
                try error_handler.pushContext(ErrorContext.statement());
                defer error_handler.popContext();

                // Validate statement AST node
                const stmt_valid = error_handler.validateAstNode(stmt, error_handling.getSpanFromStatement(stmt)) catch {
                    try error_handler.reportError(.MalformedAst, error_handling.getSpanFromStatement(stmt), "statement validation failed", "check statement structure");
                    continue;
                };
                if (!stmt_valid) {
                    continue;
                }

                // Create a temporary statement lowerer for top-level statements
                const expr_lowerer = ExpressionLowerer.init(ctx, body, &type_mapper, null, null, null, &symbol_table, null, locations, &ora_dialect);
                const stmt_lowerer = StatementLowerer.init(ctx, body, &type_mapper, &expr_lowerer, null, null, null, locations, null, &builtin_registry, std.heap.page_allocator, null, null, &ora_dialect, &[_]*lib.ast.Expressions.ExprNode{});
                stmt_lowerer.lowerStatement(stmt) catch {
                    try error_handler.reportError(.MlirOperationFailed, error_handling.getSpanFromStatement(stmt), "failed to lower top-level statement", "check statement structure and dependencies");
                    continue;
                };
            },
            .TryBlock => |try_block| {
                // Set error context for try block lowering
                try error_handler.pushContext(ErrorContext.try_block("top-level"));
                defer error_handler.popContext();

                // Validate try block AST node
                const try_block_valid = error_handler.validateAstNode(try_block, try_block.span) catch {
                    try error_handler.reportError(.MalformedAst, try_block.span, "try block validation failed", "check try block structure");
                    continue;
                };
                if (!try_block_valid) {
                    continue;
                }

                // Lower top-level try block using the declaration lowerer
                const try_block_op = decl_lowerer.lowerTryBlock(&try_block);
                if (error_handler.validateMlirOperation(try_block_op, try_block.span) catch false) {
                    c.mlirBlockAppendOwnedOperation(body, try_block_op);
                }
            },
            .EnumDecl => |enum_decl| {
                // Already handled in the first pass, but ensure MLIR type is created
                const enum_type = decl_lowerer.createEnumType(&enum_decl);
                if (symbol_table.lookupType(enum_decl.name)) |type_symbol| {
                    // Note: We can't modify the type_symbol here as it's const
                    // The MLIR type should have been set during the first pass
                    _ = enum_type;
                    _ = type_symbol;
                }
            },
            .StructDecl => |struct_decl| {
                // Already handled in the first pass, but ensure MLIR type is created
                const struct_type = decl_lowerer.createStructType(&struct_decl);
                if (symbol_table.lookupType(struct_decl.name)) |type_symbol| {
                    // Note: We can't modify the type_symbol here as it's const
                    // The MLIR type should have been set during the first pass
                    _ = struct_type;
                    _ = type_symbol;
                }
            },
        }
    }

    // Deep-copy errors and warnings out of the error handler before it is deinitialized.
    const handler_errors = error_handler.getErrors();
    const handler_warnings = error_handler.getWarnings();

    var errors = try allocator.alloc(LoweringError, handler_errors.len);
    for (handler_errors, 0..) |e, i| {
        const msg_copy = try allocator.dupe(u8, e.message);
        const sugg_copy: ?[]const u8 = if (e.suggestion) |s|
            try allocator.dupe(u8, s)
        else
            null;

        errors[i] = LoweringError{
            .error_type = e.error_type,
            .span = e.span,
            .message = msg_copy,
            .suggestion = sugg_copy,
            .context = e.context,
        };
    }

    var warnings = try allocator.alloc(LoweringWarning, handler_warnings.len);
    for (handler_warnings, 0..) |w, i| {
        const msg_copy = try allocator.dupe(u8, w.message);
        warnings[i] = LoweringWarning{
            .warning_type = w.warning_type,
            .span = w.span,
            .message = msg_copy,
        };
    }

    // Create and return the lowering result
    const result = LoweringResult{
        .module = module,
        .errors = errors,
        .warnings = warnings,
        .success = !error_handler.hasErrors(),
        .pass_result = null,
    };

    return result;
}

/// Main entry point with pass management support
pub fn lowerFunctionsToModuleWithPasses(ctx: c.MlirContext, nodes: []lib.AstNode, allocator: std.mem.Allocator, pass_config: ?PassPipelineConfig, source_filename: ?[]const u8) !LoweringResult {
    // First, perform the basic lowering
    var lowering_result = try lowerFunctionsToModuleWithErrors(ctx, nodes, allocator, source_filename);

    // If lowering failed, return early
    if (!lowering_result.success) {
        return lowering_result;
    }

    // Apply passes if configuration is provided
    if (pass_config) |config| {
        // Create pass manager with configuration
        var pm = try pass_manager.OraPassUtils.createOraPassManager(ctx, allocator, config);
        defer pm.deinit();

        // Note: IR printing configuration is handled in createOraPassManager

        // Run the passes
        const pass_result = try pm.run(lowering_result.module);

        // Verify the module after passes
        if (pass_result) {
            var verifier = verification.OraVerification.init(ctx, allocator);
            defer verifier.deinit();
            const verification_result = try verifier.verifyModule(lowering_result.module);
            if (!verification_result.success) {
                // Create a new error for verification failure
                var error_handler = ErrorHandler.init(allocator);
                defer error_handler.deinit();

                try error_handler.reportError(.MlirOperationFailed, null, "module verification failed after pass execution", "check pass configuration and module structure");

                // Update the result with verification error
                const verification_errors = try allocator.dupe(LoweringError, error_handler.getErrors());
                const combined_errors = try allocator.alloc(LoweringError, lowering_result.errors.len + verification_errors.len);
                std.mem.copyForwards(LoweringError, combined_errors[0..lowering_result.errors.len], lowering_result.errors);
                std.mem.copyForwards(LoweringError, combined_errors[lowering_result.errors.len..], verification_errors);

                lowering_result.errors = combined_errors;
                lowering_result.success = false;
            } else {
                // Run Ora-specific verification (temporarily disabled due to C API issues)
                // const ora_verification_result = pass_manager.runOraVerification(lowering_result.module) catch |err| {
                //     // Create a new error for Ora verification failure
                //     var error_handler = ErrorHandler.init(allocator);
                //     defer error_handler.deinit();

                //     try error_handler.reportError(.MlirOperationFailed, null, "Ora verification failed", @errorName(err));

                //     // Update the result with verification error
                //     const verification_errors = try allocator.dupe(LoweringError, error_handler.getErrors());
                //     const combined_errors = try allocator.alloc(LoweringError, lowering_result.errors.len + verification_errors.len);
                //     std.mem.copyForwards(LoweringError, combined_errors[0..lowering_result.errors.len], lowering_result.errors);
                //     std.mem.copyForwards(LoweringError, combined_errors[lowering_result.errors.len..], verification_errors);

                //     lowering_result.errors = combined_errors;
                //     lowering_result.success = false;
                //     return lowering_result;
                // };
                // defer ora_verification_result.deinit(allocator);

                // if (!ora_verification_result.success) {
                //     // Create errors for each Ora verification failure
                //     var error_handler = ErrorHandler.init(allocator);
                //     defer error_handler.deinit();

                //     for (ora_verification_result.errors) |ora_error| {
                //         try error_handler.reportError(.MlirOperationFailed, null, ora_error.message, "Ora verification failed");
                //     }

                //     // Update the result with verification errors
                //     const verification_errors = try allocator.dupe(LoweringError, error_handler.getErrors());
                //     const combined_errors = try allocator.alloc(LoweringError, lowering_result.errors.len + verification_errors.len);
                //     std.mem.copyForwards(LoweringError, combined_errors[0..lowering_result.errors.len], lowering_result.errors);
                //     std.mem.copyForwards(LoweringError, combined_errors[lowering_result.errors.len..], verification_errors);

                //     lowering_result.errors = combined_errors;
                //     lowering_result.success = false;
                // }
            }
        }

        // Update the result with pass information
        lowering_result.pass_result = .{
            .success = pass_result,
            .error_message = null,
            .passes_run = std.ArrayList([]const u8){},
            .timing_info = null,
            .allocator = allocator,
        };

        if (!pass_result) {
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
pub fn lowerFunctionsToModuleWithPipelineString(ctx: c.MlirContext, nodes: []lib.AstNode, allocator: std.mem.Allocator, pipeline_str: []const u8, source_filename: ?[]const u8) !LoweringResult {
    // First, perform the basic lowering
    var lowering_result = try lowerFunctionsToModuleWithErrors(ctx, nodes, allocator, source_filename);

    // If lowering failed, return early
    if (!lowering_result.success) {
        return lowering_result;
    }

    // Create pass manager and parse pipeline string
    var pm = PassManager.init(ctx, allocator);
    defer pm.deinit();

    try @import("pass_manager.zig").OraPassUtils.parsePipelineString(&pm, pipeline_str);

    // Run the passes
    const pass_success = try pm.run(lowering_result.module);

    // Update the result
    lowering_result.pass_result = .{
        .success = pass_success,
        .error_message = null,
        .passes_run = std.ArrayList([]const u8){},
        .timing_info = null,
        .allocator = allocator,
    };

    if (!pass_success) {
        lowering_result.success = false;
    }

    return lowering_result;
}
