// ============================================================================
// Semantic State - Symbol Table & Scope Management
// ============================================================================
//
// Core data structures for semantic analysis.
//
// DATA STRUCTURES:
//   • **Symbol**: name, kind, type, span, mutability, memory region
//   • **Scope**: symbols list, parent pointer, hierarchical lookup
//   • **SymbolTable**: root scope, child scopes, specialized maps
//     (contracts, functions, blocks, logs, enums)
//
// LOOKUP METHODS:
//   findInCurrent, findUp, safeFindUp, isScopeKnown
//
// ============================================================================

const std = @import("std");
const ast = @import("../ast.zig");
const builtins = @import("../semantics.zig").builtins;

pub const SymbolKind = enum {
    Contract,
    Function,
    Struct,
    Enum,
    Log,
    Error,
    Var,
    Param,
    Module,
};

pub const Symbol = struct {
    name: []const u8,
    kind: SymbolKind,
    typ: ?ast.Types.TypeInfo = null,
    span: ast.SourceSpan,
    mutable: bool = false,
    region: ?ast.statements.MemoryRegion = null,
    typ_owned: bool = false,
};

pub const Scope = struct {
    name: ?[]const u8 = null,
    symbols: std.ArrayList(Symbol),
    parent: ?*Scope = null,

    pub fn init(_: std.mem.Allocator, parent: ?*Scope, name: ?[]const u8) Scope {
        return .{ .name = name, .symbols = std.ArrayList(Symbol){}, .parent = parent };
    }

    pub fn deinit(self: *Scope) void {
        self.symbols.deinit();
    }

    pub fn findInCurrent(self: *const Scope, name: []const u8) ?usize {
        for (self.symbols.items, 0..) |s, i| if (std.mem.eql(u8, s.name, name)) return i;
        return null;
    }
};

pub const SymbolTable = struct {
    allocator: std.mem.Allocator,
    root: Scope,
    scopes: std.ArrayList(*Scope),
    contract_scopes: std.StringHashMap(*Scope),
    function_scopes: std.StringHashMap(*Scope),
    log_signatures: std.StringHashMap([]const ast.LogField),
    function_allowed_errors: std.StringHashMap([][]const u8),
    function_success_types: std.StringHashMap(ast.Types.OraType),
    block_scopes: std.AutoHashMap(usize, *Scope),
    enum_variants: std.StringHashMap([][]const u8),
    struct_fields: std.StringHashMap([]const ast.StructField), // struct name → fields
    builtin_registry: builtins.BuiltinRegistry, // Built-in stdlib functions/constants

    pub fn init(allocator: std.mem.Allocator) SymbolTable {
        const root_scope = Scope.init(allocator, null, null);
        const builtin_reg = builtins.BuiltinRegistry.init(allocator) catch {
            std.debug.print("FATAL: Failed to initialize builtin registry\n", .{});
            @panic("Builtin registry initialization failed");
        };
        return .{
            .allocator = allocator,
            .root = root_scope,
            .scopes = std.ArrayList(*Scope){},
            .contract_scopes = std.StringHashMap(*Scope).init(allocator),
            .function_scopes = std.StringHashMap(*Scope).init(allocator),
            .log_signatures = std.StringHashMap([]const ast.LogField).init(allocator),
            .function_allowed_errors = std.StringHashMap([][]const u8).init(allocator),
            .function_success_types = std.StringHashMap(ast.Types.OraType).init(allocator),
            .block_scopes = std.AutoHashMap(usize, *Scope).init(allocator),
            .enum_variants = std.StringHashMap([][]const u8).init(allocator),
            .struct_fields = std.StringHashMap([]const ast.StructField).init(allocator),
            .builtin_registry = builtin_reg,
        };
    }

    pub fn deinit(self: *SymbolTable) void {
        const type_info = @import("../ast/type_info.zig");
        // Deinit types in root symbols
        for (self.root.symbols.items) |*s| {
            if (s.typ_owned) {
                if (s.typ) |*ti| type_info.deinitTypeInfo(self.allocator, ti);
            }
        }
        // Deinit types in child scopes then deinit the scope containers
        for (self.scopes.items) |sc| {
            for (sc.symbols.items) |*sym| {
                if (sym.typ_owned) {
                    if (sym.typ) |*ti| type_info.deinitTypeInfo(self.allocator, ti);
                }
            }
            sc.deinit();
            self.allocator.destroy(sc);
        }
        self.scopes.deinit();
        self.contract_scopes.deinit();
        self.function_scopes.deinit();
        self.log_signatures.deinit();
        var it = self.function_allowed_errors.valueIterator();
        while (it.next()) |slice_ptr| {
            self.allocator.free(slice_ptr.*);
        }
        self.function_allowed_errors.deinit();
        self.function_success_types.deinit();
        var bs_it = self.block_scopes.valueIterator();
        while (bs_it.next()) |_| {
            // Scopes themselves are owned by self.scopes; nothing to free per entry
        }
        self.block_scopes.deinit();
        // Free enum variant name slices
        var ev_it = self.enum_variants.valueIterator();
        while (ev_it.next()) |slice_ptr| {
            self.allocator.free(slice_ptr.*);
        }
        self.enum_variants.deinit();
        // Note: struct_fields values are direct references to AST nodes, not owned copies
        self.struct_fields.deinit();
        self.builtin_registry.deinit();
        self.root.deinit();
    }

    pub fn declare(self: *SymbolTable, scope: *Scope, sym: Symbol) !?Symbol {
        if (scope.findInCurrent(sym.name)) |_| {
            return sym; // signal duplicate by returning the conflicting symbol content
        }
        try scope.symbols.append(self.allocator, sym);
        return null;
    }

    pub fn findUp(scope: ?*const Scope, name: []const u8) ?Symbol {
        var cur = scope;
        while (cur) |s| : (cur = s.parent) {
            if (s.findInCurrent(name)) |idx| return s.symbols.items[idx];
        }
        return null;
    }

    /// Check if a scope is registered in the symbol table
    pub fn isScopeKnown(self: *const SymbolTable, scope: *const Scope) bool {
        if (scope == &self.root) return true;
        for (self.scopes.items) |sc| {
            if (sc == scope) return true;
        }
        return false;
    }

    /// Safe version of findUp that checks if scopes are known before traversing
    pub fn safeFindUp(self: *const SymbolTable, scope: *const Scope, name: []const u8) ?Symbol {
        var cur: ?*const Scope = scope;
        while (cur) |s| : (cur = s.parent) {
            if (!self.isScopeKnown(s)) return null;
            if (s.findInCurrent(name)) |idx| return s.symbols.items[idx];
        }
        return null;
    }
};
