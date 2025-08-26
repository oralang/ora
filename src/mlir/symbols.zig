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
};

/// Symbol table with scope management
pub const SymbolTable = struct {
    allocator: std.mem.Allocator,
    scopes: std.ArrayList(std.StringHashMap(SymbolInfo)),
    current_scope: usize,

    pub fn init(allocator: std.mem.Allocator) SymbolTable {
        var scopes = std.ArrayList(std.StringHashMap(SymbolInfo)).init(allocator);
        const global_scope = std.StringHashMap(SymbolInfo).init(allocator);
        scopes.append(global_scope) catch unreachable;

        return .{
            .allocator = allocator,
            .scopes = scopes,
            .current_scope = 0,
        };
    }

    pub fn deinit(self: *SymbolTable) void {
        for (self.scopes.items) |*scope| {
            scope.deinit();
        }
        self.scopes.deinit();
    }

    /// Push a new scope
    pub fn pushScope(self: *SymbolTable) !void {
        const new_scope = std.StringHashMap(SymbolInfo).init(self.allocator);
        try self.scopes.append(new_scope);
        self.current_scope += 1;
    }

    /// Pop the current scope
    pub fn popScope(self: *SymbolTable) void {
        if (self.current_scope > 0) {
            const scope = self.scopes.orderedRemove(self.current_scope);
            scope.deinit();
            self.current_scope -= 1;
        }
    }

    /// Add a symbol to the current scope
    pub fn addSymbol(self: *SymbolTable, name: []const u8, type_info: c.MlirType, region: lib.ast.Statements.MemoryRegion, span: ?[]const u8) !void {
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
        };

        try self.scopes.items[self.current_scope].put(name, symbol_info);
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
            if (self.scopes.items[scope_idx].get(name)) |*symbol| {
                symbol.value = value;
                try self.scopes.items[scope_idx].put(name, symbol.*);
                return;
            }
            if (scope_idx == 0) break;
            scope_idx -= 1;
        }
        // If symbol not found, add it to current scope
        try self.addSymbol(name, c.mlirValueGetType(value), "stack", null);
        if (self.scopes.items[self.current_scope].get(name)) |*symbol| {
            symbol.value = value;
            try self.scopes.items[self.current_scope].put(name, symbol.*);
        }
    }

    /// Check if a symbol exists in any scope
    pub fn hasSymbol(self: *const SymbolTable, name: []const u8) bool {
        return self.lookupSymbol(name) != null;
    }
};
