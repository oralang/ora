// ============================================================================
// Symbol Tables & Variable Mapping
// ============================================================================
//
// Manages symbol tables for parameters, local variables, and declarations.
//
// KEY COMPONENTS:
//   • ParamMap: Function parameter name → MLIR value mapping
//   • LocalVarMap: Local variable name → MLIR value mapping
//   • SymbolTable: Global symbol tracking for functions, types, contracts
//
// ============================================================================

const std = @import("std");
const c = @import("c.zig").c;
const lib = @import("ora_lib");

// Parameter mapping structure for function parameters
pub const ParamMap = struct {
    names: std.StringHashMap(usize), // parameter name -> block argument index
    block_args: std.StringHashMap(c.MlirValue), // parameter name -> block argument value

    pub fn init(allocator: std.mem.Allocator) ParamMap {
        return .{
            .names = std.StringHashMap(usize).init(allocator),
            .block_args = std.StringHashMap(c.MlirValue).init(allocator),
        };
    }

    pub fn deinit(self: *ParamMap) void {
        self.names.deinit();
        self.block_args.deinit();
    }

    pub fn addParam(self: *ParamMap, name: []const u8, index: usize) !void {
        try self.names.put(name, index);
    }

    pub fn getParamIndex(self: *const ParamMap, name: []const u8) ?usize {
        return self.names.get(name);
    }

    pub fn setBlockArgument(self: *ParamMap, name: []const u8, block_arg: c.MlirValue) !void {
        try self.block_args.put(name, block_arg);
    }

    pub fn getBlockArgument(self: *const ParamMap, name: []const u8) ?c.MlirValue {
        return self.block_args.get(name);
    }
};

// Local variable mapping for function-local variables
pub const LocalVarMap = struct {
    variables: std.StringHashMap(c.MlirValue),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) LocalVarMap {
        return .{
            .variables = std.StringHashMap(c.MlirValue).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *LocalVarMap) void {
        self.variables.deinit();
    }

    pub fn addLocalVar(self: *LocalVarMap, name: []const u8, value: c.MlirValue) !void {
        try self.variables.put(name, value);
    }

    pub fn getLocalVar(self: *const LocalVarMap, name: []const u8) ?c.MlirValue {
        return self.variables.get(name);
    }

    pub fn hasLocalVar(self: *const LocalVarMap, name: []const u8) bool {
        return self.variables.contains(name);
    }
};

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
};

/// Symbol table with scope management
pub const SymbolTable = struct {
    allocator: std.mem.Allocator,
    scopes: std.ArrayList(std.StringHashMap(SymbolInfo)),
    current_scope: usize,

    // Separate tables for different symbol kinds
    functions: std.StringHashMap(FunctionSymbol),
    types: std.StringHashMap(TypeSymbol),

    pub fn init(allocator: std.mem.Allocator) SymbolTable {
        var scopes = std.ArrayList(std.StringHashMap(SymbolInfo)){};
        const global_scope = std.StringHashMap(SymbolInfo).init(allocator);
        scopes.append(allocator, global_scope) catch unreachable;

        return .{
            .allocator = allocator,
            .scopes = scopes,
            .current_scope = 0,
            .functions = std.StringHashMap(FunctionSymbol).init(allocator),
            .types = std.StringHashMap(TypeSymbol).init(allocator),
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

    /// Add a function symbol to the global function table
    pub fn addFunction(self: *SymbolTable, name: []const u8, operation: c.MlirOperation, param_types: []c.MlirType, return_type: c.MlirType) !void {
        const func_symbol = FunctionSymbol.init(self.allocator, name, operation, param_types, return_type);
        try self.functions.put(name, func_symbol);
    }

    /// Add a type symbol (struct, enum) to the global type table
    pub fn addType(self: *SymbolTable, name: []const u8, type_symbol: TypeSymbol) !void {
        try self.types.put(name, type_symbol);
    }

    /// Add a struct type symbol
    pub fn addStructType(self: *SymbolTable, name: []const u8, mlir_type: c.MlirType, fields: []TypeSymbol.FieldInfo) !void {
        const type_symbol = TypeSymbol{
            .name = name,
            .type_kind = .Struct,
            .mlir_type = mlir_type,
            .fields = fields,
            .variants = null,
        };
        try self.addType(name, type_symbol);
    }

    /// Add an enum type symbol
    pub fn addEnumType(self: *SymbolTable, name: []const u8, mlir_type: c.MlirType, variants: []TypeSymbol.VariantInfo) !void {
        const type_symbol = TypeSymbol{
            .name = name,
            .type_kind = .Enum,
            .mlir_type = mlir_type,
            .fields = null,
            .variants = variants,
        };
        try self.addType(name, type_symbol);
    }

    /// Look up a symbol starting from the current scope and going outward
    pub fn lookupSymbol(self: *const SymbolTable, name: []const u8) ?SymbolInfo {
        var scope_idx: usize = self.current_scope;
        while (true) {
            if (self.scopes.items[scope_idx].get(name)) |symbol| {
                return symbol;
            }
            if (scope_idx == 0) break;
            scope_idx -= 1;
        }
        return null;
    }

    /// Update a symbol's value
    pub fn updateSymbolValue(self: *SymbolTable, name: []const u8, value: c.MlirValue) !void {
        var scope_idx: usize = self.current_scope;
        while (true) {
            if (self.scopes.items[scope_idx].get(name)) |symbol| {
                var updated_symbol = symbol;
                updated_symbol.value = value;
                try self.scopes.items[scope_idx].put(name, updated_symbol);
                return;
            }
            if (scope_idx == 0) break;
            scope_idx -= 1;
        }
        // If symbol not found, add it to current scope
        try self.addSymbol(name, c.mlirValueGetType(value), lib.ast.Statements.MemoryRegion.Stack, null, null);
        if (self.scopes.items[self.current_scope].get(name)) |symbol| {
            var updated_symbol = symbol;
            updated_symbol.value = value;
            try self.scopes.items[self.current_scope].put(name, updated_symbol);
        }
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
    pub fn lookupType(self: *const SymbolTable, name: []const u8) ?TypeSymbol {
        return self.types.get(name);
    }

    /// Check if a function exists
    pub fn hasFunction(self: *const SymbolTable, name: []const u8) bool {
        return self.functions.contains(name);
    }

    /// Check if a type exists
    pub fn hasType(self: *const SymbolTable, name: []const u8) bool {
        return self.types.contains(name);
    }

    /// Get current scope level
    pub fn getCurrentScopeLevel(self: *const SymbolTable) usize {
        return self.current_scope;
    }

    /// Get symbol count in current scope
    pub fn getSymbolCount(self: *const SymbolTable) usize {
        return self.scopes.items[self.current_scope].count();
    }

    /// Get function count
    pub fn getFunctionCount(self: *const SymbolTable) usize {
        return self.functions.count();
    }

    /// Get type count
    pub fn getTypeCount(self: *const SymbolTable) usize {
        return self.types.count();
    }

    /// Update function attributes (for inline, requires, ensures clauses)
    pub fn updateFunctionAttribute(self: *SymbolTable, func_name: []const u8, attr_name: []const u8, attr_value: c.MlirAttribute) !void {
        if (self.functions.getPtr(func_name)) |func_symbol| {
            try func_symbol.attributes.put(attr_name, attr_value);
        }
    }

    /// Get function attribute
    pub fn getFunctionAttribute(self: *const SymbolTable, func_name: []const u8, attr_name: []const u8) ?c.MlirAttribute {
        if (self.functions.get(func_name)) |func_symbol| {
            return func_symbol.attributes.get(attr_name);
        }
        return null;
    }
};
