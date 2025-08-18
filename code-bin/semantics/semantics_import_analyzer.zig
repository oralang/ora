const std = @import("std");
pub const ast = @import("../ast.zig");
const semantics_errors = @import("semantics_errors.zig");
const semantics_memory_safety = @import("semantics_memory_safety.zig");

// Forward declaration for SemanticAnalyzer
const SemanticAnalyzer = @import("semantics_core.zig").SemanticAnalyzer;

/// Module registry for tracking available modules and their symbols
pub const ModuleRegistry = struct {
    modules: std.HashMap([]const u8, ModuleInfo, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),
    standard_library: StandardLibraryDefinitions,
    allocator: std.mem.Allocator,

    pub const ModuleInfo = struct {
        name: []const u8,
        symbols: std.ArrayList(SymbolInfo),
        is_standard_library: bool,

        pub const SymbolInfo = struct {
            name: []const u8,
            symbol_type: SymbolType,
            span: ast.SourceSpan,

            pub const SymbolType = enum {
                Function,
                Variable,
                Type,
                Constant,
            };
        };
    };

    pub fn init(allocator: std.mem.Allocator) ModuleRegistry {
        return ModuleRegistry{
            .modules = std.HashMap([]const u8, ModuleInfo, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
            .standard_library = StandardLibraryDefinitions.init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ModuleRegistry) void {
        var iterator = self.modules.iterator();
        while (iterator.next()) |entry| {
            entry.value_ptr.symbols.deinit();
        }
        self.modules.deinit();
        self.standard_library.deinit();
    }

    pub fn registerModule(self: *ModuleRegistry, name: []const u8, is_standard: bool) !*ModuleInfo {
        const owned_name = try self.allocator.dupe(u8, name);
        const module_info = ModuleInfo{
            .name = owned_name,
            .symbols = std.ArrayList(ModuleInfo.SymbolInfo).init(self.allocator),
            .is_standard_library = is_standard,
        };

        try self.modules.put(owned_name, module_info);
        return self.modules.getPtr(owned_name).?;
    }

    pub fn getModule(self: *ModuleRegistry, name: []const u8) ?*ModuleInfo {
        return self.modules.getPtr(name);
    }
};

/// Standard library definitions for import validation
pub const StandardLibraryDefinitions = struct {
    modules: std.HashMap([]const u8, []const []const u8, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) StandardLibraryDefinitions {
        var std_lib = StandardLibraryDefinitions{
            .modules = std.HashMap([]const u8, []const []const u8, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
            .allocator = allocator,
        };

        // Initialize with common standard library modules
        std_lib.initializeStandardModules() catch {};
        return std_lib;
    }

    pub fn deinit(self: *StandardLibraryDefinitions) void {
        var iterator = self.modules.iterator();
        while (iterator.next()) |entry| {
            self.allocator.free(entry.value_ptr.*);
        }
        self.modules.deinit();
    }

    fn initializeStandardModules(self: *StandardLibraryDefinitions) !void {
        // Math module
        const math_symbols = [_][]const u8{ "abs", "max", "min", "sqrt", "pow" };
        const owned_math_symbols = try self.allocator.dupe([]const u8, &math_symbols);
        try self.modules.put("math", owned_math_symbols);

        // String module
        const string_symbols = [_][]const u8{ "length", "concat", "substring", "indexOf" };
        const owned_string_symbols = try self.allocator.dupe([]const u8, &string_symbols);
        try self.modules.put("string", owned_string_symbols);

        // Array module
        const array_symbols = [_][]const u8{ "length", "push", "pop", "slice" };
        const owned_array_symbols = try self.allocator.dupe([]const u8, &array_symbols);
        try self.modules.put("array", owned_array_symbols);
    }

    pub fn hasModule(self: *StandardLibraryDefinitions, name: []const u8) bool {
        return self.modules.contains(name);
    }

    pub fn getModuleSymbols(self: *StandardLibraryDefinitions, name: []const u8) ?[]const []const u8 {
        return self.modules.get(name);
    }
};

/// Import dependency graph for circular dependency detection
pub const ImportDependencyGraph = struct {
    dependencies: std.HashMap([]const u8, std.ArrayList([]const u8), std.hash_map.StringContext, std.hash_map.default_max_load_percentage),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) ImportDependencyGraph {
        return ImportDependencyGraph{
            .dependencies = std.HashMap([]const u8, std.ArrayList([]const u8), std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ImportDependencyGraph) void {
        var iterator = self.dependencies.iterator();
        while (iterator.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.dependencies.deinit();
    }

    pub fn addDependency(self: *ImportDependencyGraph, from: []const u8, to: []const u8) !void {
        const owned_from = try self.allocator.dupe(u8, from);
        const owned_to = try self.allocator.dupe(u8, to);

        const result = try self.dependencies.getOrPut(owned_from);
        if (!result.found_existing) {
            result.value_ptr.* = std.ArrayList([]const u8).init(self.allocator);
        }

        try result.value_ptr.append(owned_to);
    }

    pub fn detectCircularDependencies(self: *ImportDependencyGraph) ![][]const u8 {
        var cycles = std.ArrayList([]const u8).init(self.allocator);
        var visited = std.HashMap([]const u8, bool, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(self.allocator);
        defer visited.deinit();

        var iterator = self.dependencies.iterator();
        while (iterator.next()) |entry| {
            if (!visited.contains(entry.key_ptr.*)) {
                try self.detectCycleFromNode(entry.key_ptr.*, &visited, &cycles);
            }
        }

        return cycles.toOwnedSlice();
    }

    fn detectCycleFromNode(self: *ImportDependencyGraph, node: []const u8, visited: *std.HashMap([]const u8, bool, std.hash_map.StringContext, std.hash_map.default_max_load_percentage), cycles: *std.ArrayList([]const u8)) !void {
        try visited.put(node, true);

        if (self.dependencies.get(node)) |deps| {
            for (deps.items) |dep| {
                if (visited.contains(dep)) {
                    try cycles.append(try self.allocator.dupe(u8, dep));
                } else {
                    try self.detectCycleFromNode(dep, visited, cycles);
                }
            }
        }
    }
};

/// Circular import detector
pub const CircularImportDetector = struct {
    import_graph: ImportDependencyGraph,

    pub fn init(allocator: std.mem.Allocator) CircularImportDetector {
        return CircularImportDetector{
            .import_graph = ImportDependencyGraph.init(allocator),
        };
    }

    pub fn deinit(self: *CircularImportDetector) void {
        self.import_graph.deinit();
    }

    pub fn addImport(self: *CircularImportDetector, from: []const u8, to: []const u8) !void {
        try self.import_graph.addDependency(from, to);
    }

    pub fn detectCircularImports(self: *CircularImportDetector) ![][]const u8 {
        return self.import_graph.detectCircularDependencies();
    }
};

/// Import analysis context
pub const ImportContext = struct {
    current_module: ?[]const u8,
    import_path: []const u8,
    alias: ?[]const u8,
    span: ast.SourceSpan,
};

/// Symbol resolution result
pub const SymbolResolutionResult = struct {
    found: bool,
    symbol_type: ModuleRegistry.ModuleInfo.SymbolInfo.SymbolType,
    module_name: []const u8,
    error_message: ?[]const u8,
};

/// Import analysis result
pub const ImportAnalysisResult = struct {
    success: bool,
    module_found: bool,
    symbols_validated: bool,
    error_message: ?[]const u8,
};

/// Module access result
pub const ModuleAccessResult = struct {
    valid: bool,
    field_exists: bool,
    field_type: ?ModuleRegistry.ModuleInfo.SymbolInfo.SymbolType,
    error_message: ?[]const u8,
};

/// Circular import report
pub const CircularImportReport = struct {
    has_cycles: bool,
    cycles: [][]const u8,
    affected_modules: [][]const u8,
};

/// Import analyzer for handling import system analysis
pub const ImportAnalyzer = struct {
    module_registry: ModuleRegistry,
    circular_detector: CircularImportDetector,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) ImportAnalyzer {
        return ImportAnalyzer{
            .module_registry = ModuleRegistry.init(allocator),
            .circular_detector = CircularImportDetector.init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ImportAnalyzer) void {
        self.module_registry.deinit();
        self.circular_detector.deinit();
    }

    /// Analyze import statement
    pub fn analyzeImport(self: *ImportAnalyzer, analyzer: *SemanticAnalyzer, import: *ast.ImportNode) semantics_errors.SemanticError!ImportAnalysisResult {
        // Validate import path string
        if (!semantics_memory_safety.isValidString(analyzer, import.path)) {
            try semantics_errors.addErrorStatic(analyzer, "Invalid import path", import.span);
            return ImportAnalysisResult{
                .success = false,
                .module_found = false,
                .symbols_validated = false,
                .error_message = "Invalid import path",
            };
        }

        // Check if module exists in standard library
        const is_standard_module = self.module_registry.standard_library.hasModule(import.path);

        if (!is_standard_module) {
            // For now, only standard library modules are supported
            try semantics_errors.addErrorStatic(analyzer, "Only standard library imports are currently supported", import.span);
            return ImportAnalysisResult{
                .success = false,
                .module_found = false,
                .symbols_validated = false,
                .error_message = "Module not found",
            };
        }

        // Register module if not already registered
        if (self.module_registry.getModule(import.path) == null) {
            const module_info = try self.module_registry.registerModule(import.path, true);

            // Add standard library symbols to module
            if (self.module_registry.standard_library.getModuleSymbols(import.path)) |symbols| {
                for (symbols) |symbol_name| {
                    const symbol_info = ModuleRegistry.ModuleInfo.SymbolInfo{
                        .name = symbol_name,
                        .symbol_type = .Function, // Default to function for standard library
                        .span = import.span,
                    };
                    try module_info.symbols.append(symbol_info);
                }
            }
        }

        // Add to circular dependency detection
        const current_module = if (analyzer.current_contract) |contract| contract.name else "main";
        try self.circular_detector.addImport(current_module, import.path);

        return ImportAnalysisResult{
            .success = true,
            .module_found = true,
            .symbols_validated = true,
            .error_message = null,
        };
    }

    /// Validate module field access
    pub fn validateModuleAccess(self: *ImportAnalyzer, analyzer: *SemanticAnalyzer, field_access: *ast.FieldAccessNode) semantics_errors.SemanticError!ModuleAccessResult {
        // Extract module name from field access
        const module_name = switch (field_access.object.*) {
            .Identifier => |*ident| ident.name,
            else => {
                try semantics_errors.addErrorStatic(analyzer, "Invalid module access", field_access.span);
                return ModuleAccessResult{
                    .valid = false,
                    .field_exists = false,
                    .field_type = null,
                    .error_message = "Invalid module access",
                };
            },
        };

        // Validate field name string
        if (!semantics_memory_safety.isValidString(analyzer, field_access.field)) {
            try semantics_errors.addErrorStatic(analyzer, "Invalid field name", field_access.span);
            return ModuleAccessResult{
                .valid = false,
                .field_exists = false,
                .field_type = null,
                .error_message = "Invalid field name",
            };
        }

        // Check if module exists
        const module_info = self.module_registry.getModule(module_name) orelse {
            try semantics_errors.addErrorStatic(analyzer, "Module not found", field_access.span);
            return ModuleAccessResult{
                .valid = false,
                .field_exists = false,
                .field_type = null,
                .error_message = "Module not found",
            };
        };

        // Check if field exists in module
        for (module_info.symbols.items) |symbol| {
            if (std.mem.eql(u8, symbol.name, field_access.field)) {
                return ModuleAccessResult{
                    .valid = true,
                    .field_exists = true,
                    .field_type = symbol.symbol_type,
                    .error_message = null,
                };
            }
        }

        // Field not found
        const error_msg = try std.fmt.allocPrint(analyzer.allocator, "Field '{s}' not found in module '{s}'", .{ field_access.field, module_name });
        try semantics_errors.addError(analyzer, error_msg, field_access.span);

        return ModuleAccessResult{
            .valid = false,
            .field_exists = false,
            .field_type = null,
            .error_message = "Field not found in module",
        };
    }

    /// Detect circular imports
    pub fn detectCircularImports(self: *ImportAnalyzer) !CircularImportReport {
        const cycles = try self.circular_detector.detectCircularImports();

        return CircularImportReport{
            .has_cycles = cycles.len > 0,
            .cycles = cycles,
            .affected_modules = cycles, // For simplicity, same as cycles
        };
    }

    /// Resolve symbol in import context
    pub fn resolveSymbol(self: *ImportAnalyzer, symbol: []const u8, context: ImportContext) SymbolResolutionResult {
        // Check if symbol exists in the imported module
        if (self.module_registry.getModule(context.import_path)) |module_info| {
            for (module_info.symbols.items) |symbol_info| {
                if (std.mem.eql(u8, symbol_info.name, symbol)) {
                    return SymbolResolutionResult{
                        .found = true,
                        .symbol_type = symbol_info.symbol_type,
                        .module_name = module_info.name,
                        .error_message = null,
                    };
                }
            }
        }

        return SymbolResolutionResult{
            .found = false,
            .symbol_type = .Function, // Default
            .module_name = context.import_path,
            .error_message = "Symbol not found in module",
        };
    }
};

/// Analyze import node
pub fn analyzeImport(analyzer: *SemanticAnalyzer, import: *ast.ImportNode) semantics_errors.SemanticError!void {
    // This will be integrated with the main analyzer's import analyzer instance
    // For now, just validate the import path
    if (!semantics_memory_safety.isValidString(analyzer, import.path)) {
        try semantics_errors.addErrorStatic(analyzer, "Invalid import path", import.span);
        return;
    }

    // Basic validation - more comprehensive analysis will be done by ImportAnalyzer
    if (import.path.len == 0) {
        try semantics_errors.addErrorStatic(analyzer, "Empty import path", import.span);
        return;
    }
}
