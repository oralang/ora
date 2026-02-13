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
const TypeInfo = @import("../ast/type_info.zig").TypeInfo;
const builtins = @import("../semantics.zig").builtins;
const log = @import("log");

pub const SymbolKind = enum {
    Contract,
    Function,
    Struct,
    Bitfield,
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
    /// True if this symbol's type was derived from control flow analysis
    /// (e.g., inside `if (x == 0)`, x is flow-refined to have value 0).
    /// Flow refinements should be used for reads but NOT for assignment targets.
    is_flow_refinement: bool = false,
};

pub const FunctionEffect = union(enum) {
    Pure,
    Reads: std.ArrayList([]const u8),
    Writes: std.ArrayList([]const u8),
    ReadsWrites: struct {
        reads: std.ArrayList([]const u8),
        writes: std.ArrayList([]const u8),
    },

    pub fn pure() FunctionEffect {
        return .Pure;
    }

    pub fn reads(list: std.ArrayList([]const u8)) FunctionEffect {
        return .{ .Reads = list };
    }

    pub fn writes(list: std.ArrayList([]const u8)) FunctionEffect {
        return .{ .Writes = list };
    }

    pub fn readsWrites(read_slots: std.ArrayList([]const u8), write_slots: std.ArrayList([]const u8)) FunctionEffect {
        return .{ .ReadsWrites = .{ .reads = read_slots, .writes = write_slots } };
    }

    pub fn deinit(self: *FunctionEffect, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .Pure => {},
            .Reads => |*slots| slots.deinit(allocator),
            .Writes => |*slots| slots.deinit(allocator),
            .ReadsWrites => |*rw| {
                rw.reads.deinit(allocator);
                rw.writes.deinit(allocator);
            },
        }
    }
};

pub const Scope = struct {
    name: ?[]const u8 = null,
    symbols: std.ArrayListUnmanaged(Symbol) = .{},
    parent: ?*Scope = null,

    pub fn init(_: std.mem.Allocator, parent: ?*Scope, name: ?[]const u8) Scope {
        return .{ .name = name, .symbols = .{}, .parent = parent };
    }

    pub fn deinit(self: *Scope, allocator: std.mem.Allocator) void {
        self.symbols.deinit(allocator);
    }

    pub fn findInCurrent(self: *const Scope, name: []const u8) ?usize {
        for (self.symbols.items, 0..) |s, i| if (std.mem.eql(u8, s.name, name)) return i;
        return null;
    }
};

pub const SymbolTable = struct {
    allocator: std.mem.Allocator,
    root: *Scope, // Heap-allocated to avoid dangling pointers when table is copied
    scopes: std.ArrayList(*Scope),
    contract_scopes: std.StringHashMap(*Scope),
    function_scopes: std.StringHashMap(*Scope),
    // top-level (non-contract) log signatures
    log_signatures: std.StringHashMap([]const ast.LogField),
    // contract-scoped log signatures: contract name -> (event name -> fields)
    contract_log_signatures: std.StringHashMap(std.StringHashMap([]const ast.LogField)),
    error_signatures: std.StringHashMap(?[]const ast.ParameterNode), // error name → parameters (null if no params)
    function_allowed_errors: std.StringHashMap([][]const u8),
    function_success_types: std.StringHashMap(ast.Types.OraType),
    function_effects: std.StringHashMap(FunctionEffect),
    block_scopes: std.AutoHashMap(usize, *Scope),
    enum_variants: std.StringHashMap([][]const u8),
    struct_fields: std.StringHashMap([]const ast.StructField), // struct name → fields
    bitfield_fields: std.StringHashMap([]const ast.BitfieldField), // bitfield name → fields
    builtin_registry: builtins.BuiltinRegistry, // Built-in stdlib functions/constants

    pub fn init(allocator: std.mem.Allocator) SymbolTable {
        // Allocate root scope on heap to avoid dangling pointers when table is copied/returned
        const root_scope = allocator.create(Scope) catch @panic("Failed to allocate root scope");
        root_scope.* = Scope.init(allocator, null, null);
        const builtin_reg = builtins.BuiltinRegistry.init(allocator) catch {
            log.err("Failed to initialize builtin registry\n", .{});
            @panic("Builtin registry initialization failed");
        };
        return .{
            .allocator = allocator,
            .root = root_scope,
            .scopes = std.ArrayList(*Scope){},
            .contract_scopes = std.StringHashMap(*Scope).init(allocator),
            .function_scopes = std.StringHashMap(*Scope).init(allocator),
            .log_signatures = std.StringHashMap([]const ast.LogField).init(allocator),
            .contract_log_signatures = std.StringHashMap(std.StringHashMap([]const ast.LogField)).init(allocator),
            .error_signatures = std.StringHashMap(?[]const ast.ParameterNode).init(allocator),
            .function_allowed_errors = std.StringHashMap([][]const u8).init(allocator),
            .function_success_types = std.StringHashMap(ast.Types.OraType).init(allocator),
            .function_effects = std.StringHashMap(FunctionEffect).init(allocator),
            .block_scopes = std.AutoHashMap(usize, *Scope).init(allocator),
            .enum_variants = std.StringHashMap([][]const u8).init(allocator),
            .struct_fields = std.StringHashMap([]const ast.StructField).init(allocator),
            .bitfield_fields = std.StringHashMap([]const ast.BitfieldField).init(allocator),
            .builtin_registry = builtin_reg,
        };
    }

    pub fn deinit(self: *SymbolTable) void {
        const type_info_mod = @import("../ast/type_info.zig");
        // deinit types in root symbols
        for (self.root.symbols.items) |*s| {
            if (s.typ_owned) {
                if (s.typ) |*ti| type_info_mod.deinitTypeInfo(self.allocator, ti);
            }
        }
        // deinit root scope (heap-allocated)
        self.root.deinit(self.allocator);
        self.allocator.destroy(self.root);
        // deinit types in child scopes then deinit the scope containers
        for (self.scopes.items) |sc| {
            for (sc.symbols.items) |*sym| {
                if (sym.typ_owned) {
                    if (sym.typ) |*ti| type_info_mod.deinitTypeInfo(self.allocator, ti);
                }
            }
            sc.deinit(self.allocator);
            self.allocator.destroy(sc);
        }
        self.scopes.deinit(self.allocator);
        self.contract_scopes.deinit();
        self.function_scopes.deinit();
        self.log_signatures.deinit();
        var log_it = self.contract_log_signatures.valueIterator();
        while (log_it.next()) |map_ptr| {
            map_ptr.*.deinit();
        }
        self.contract_log_signatures.deinit();
        self.error_signatures.deinit();
        var it = self.function_allowed_errors.valueIterator();
        while (it.next()) |slice_ptr| {
            self.allocator.free(slice_ptr.*);
        }
        self.function_allowed_errors.deinit();
        // deallocate types stored in function_success_types (may contain pointers for refinement types)
        var fst_it = self.function_success_types.iterator();
        while (fst_it.next()) |entry| {
            type_info_mod.deinitOraType(self.allocator, @constCast(&entry.value_ptr.*));
        }
        self.function_success_types.deinit();
        var fe_it = self.function_effects.valueIterator();
        while (fe_it.next()) |eff| {
            eff.deinit(self.allocator);
        }
        self.function_effects.deinit();
        var bs_it = self.block_scopes.valueIterator();
        while (bs_it.next()) |_| {
            // scopes themselves are owned by self.scopes; nothing to free per entry
        }
        self.block_scopes.deinit();
        // free enum variant name slices
        var ev_it = self.enum_variants.valueIterator();
        while (ev_it.next()) |slice_ptr| {
            self.allocator.free(slice_ptr.*);
        }
        self.enum_variants.deinit();
        // note: struct_fields/bitfield_fields values are direct references to AST nodes, not owned copies
        self.struct_fields.deinit();
        self.bitfield_fields.deinit();
        self.builtin_registry.deinit();
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

    /// Find the original declared symbol, skipping flow-refined shadowing symbols.
    /// This is used for assignment targets where we want the declared type, not flow refinements.
    pub fn findDeclaredUp(scope: ?*const Scope, name: []const u8) ?Symbol {
        var cur = scope;
        while (cur) |s| : (cur = s.parent) {
            if (s.findInCurrent(name)) |idx| {
                const sym = s.symbols.items[idx];
                // Skip flow-refined symbols - they shadow the declared type
                if (!sym.is_flow_refinement) {
                    return sym;
                }
                // Continue up the scope chain to find the original
            }
        }
        return null;
    }

    pub fn findEnclosingContractName(self: *const SymbolTable, scope: ?*const Scope) ?[]const u8 {
        var cur = scope;
        while (cur) |s| : (cur = s.parent) {
            if (s.name) |scope_name| {
                if (self.contract_scopes.get(scope_name) != null or
                    self.contract_log_signatures.getPtr(scope_name) != null)
                {
                    return scope_name;
                }
            }
        }
        return null;
    }

    pub fn getContractLogSignatures(self: *SymbolTable, scope: ?*const Scope) ?*std.StringHashMap([]const ast.LogField) {
        const contract_name = self.findEnclosingContractName(scope) orelse return null;
        return self.contract_log_signatures.getPtr(contract_name);
    }

    /// Find the scope containing a symbol by name, searching from the given scope up to root
    /// Returns the scope and the symbol index if found
    /// NOTE: This only searches UP the parent chain, not down into nested scopes
    pub fn findScopeContaining(scope: ?*Scope, name: []const u8) ?struct { scope: *Scope, idx: usize } {
        var cur = scope;
        while (cur) |s| : (cur = s.parent) {
            if (s.findInCurrent(name)) |idx| {
                return .{ .scope = s, .idx = idx };
            }
        }
        return null;
    }

    /// Safe variant that bails if the scope pointer is unknown to the symbol table.
    pub fn findScopeContainingSafe(self: *const SymbolTable, scope: ?*Scope, name: []const u8) ?struct { scope: *Scope, idx: usize } {
        var cur = scope;
        while (cur) |s| : (cur = s.parent) {
            if (!self.isScopeKnown(s)) return null;
            if (s.findInCurrent(name)) |idx| {
                return .{ .scope = s, .idx = idx };
            }
        }
        return null;
    }
    /// Find the scope containing a symbol by name, searching from the given scope down into nested scopes first, then up
    /// This is needed when current_scope is set to a block scope but the symbol might be in a nested block scope
    fn findScopeContainingRecursive(scope: ?*Scope, name: []const u8) ?struct { scope: *Scope, idx: usize } {
        // first check current scope
        if (scope) |s| {
            if (s.findInCurrent(name)) |idx| {
                return .{ .scope = s, .idx = idx };
            }
            // todo: Search down into nested scopes if we track them
            // for now, just search up
        }
        // fall back to searching up
        return findScopeContaining(scope, name);
    }

    /// Update a symbol's type in the symbol table
    /// Finds the symbol by name starting from the given scope (checks current scope first, then searches up)
    /// and updates its type. Properly deallocates the old type if it was owned.
    /// NOTE: Only sets typ_owned=true if the symbol is in a function scope, not a block scope
    pub fn updateSymbolType(self: *SymbolTable, scope: ?*Scope, name: []const u8, new_type: TypeInfo, new_typ_owned: bool) !void {
        const type_info = @import("../ast/type_info.zig");

        // first check the current scope (in case symbol is in a nested block scope)
        if (scope) |s| {
            if (s.findInCurrent(name)) |idx| {
                // found in current scope - update it
                const old_symbol = &s.symbols.items[idx];
                // only deallocate if the old type was owned AND has a valid ora_type
                // skip deallocation if old type is null or points to arena memory (ora_type null)
                if (old_symbol.typ_owned) {
                    if (old_symbol.typ) |*old_typ| {
                        // only deallocate if old type has a valid ora_type (not pointing to arena)
                        if (old_typ.ora_type != null) {
                            type_info.deinitTypeInfo(self.allocator, old_typ);
                        }
                    }
                }
                old_symbol.typ = new_type;
                // only set typ_owned=true if this scope is a function scope, not a block scope
                old_symbol.typ_owned = new_typ_owned and isFunctionScope(self, s);
                return;
            }
        }

        // find the scope containing the symbol (searches up from given scope)
        if (self.findScopeContainingSafe(scope, name)) |found| {
            // found the symbol in a parent scope - deallocate old type if it was owned
            const old_symbol = &found.scope.symbols.items[found.idx];
            // only deallocate if the old type was owned AND has a valid ora_type
            if (old_symbol.typ_owned) {
                if (old_symbol.typ) |*old_typ| {
                    // only deallocate if old type has a valid ora_type (not pointing to arena)
                    if (old_typ.ora_type != null) {
                        type_info.deinitTypeInfo(self.allocator, old_typ);
                    }
                }
            }
            // update the symbol with new type and ownership flag
            // only set typ_owned=true if this scope is a function scope, not a block scope
            old_symbol.typ = new_type;
            old_symbol.typ_owned = new_typ_owned and isFunctionScope(self, found.scope);
            return;
        }

        // symbol not found - this can happen if:
        // 1. The symbol was stored in a nested scope below the given scope (locals_binder case)
        // 2. The symbol was never stored (skipped in collectSymbols due to null ora_type)
        return error.SymbolNotFound;
    }

    /// Check if a scope is registered in the symbol table
    pub fn isScopeKnown(self: *const SymbolTable, scope: *const Scope) bool {
        if (scope == self.root) return true;
        for (self.scopes.items) |sc| {
            if (sc == scope) return true;
        }
        return false;
    }

    /// Check if a scope is a function scope (not a block scope)
    pub fn isFunctionScope(self: *const SymbolTable, scope: ?*Scope) bool {
        if (scope) |s| {
            // check if this scope is in function_scopes
            var it = self.function_scopes.iterator();
            while (it.next()) |entry| {
                if (entry.value_ptr.* == s) return true;
            }
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

    /// Safe findUp that accepts optional scopes.
    pub fn safeFindUpOpt(self: *const SymbolTable, scope: ?*const Scope, name: []const u8) ?Symbol {
        if (scope) |s| {
            return self.safeFindUp(s, name);
        }
        return null;
    }
};
