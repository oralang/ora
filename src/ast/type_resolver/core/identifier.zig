// ============================================================================
// Identifier Resolution
// ============================================================================
// Phase 4: Extract identifier lookup logic
// ============================================================================

const std = @import("std");
const ast = @import("../../../ast.zig");
const TypeInfo = @import("../../type_info.zig").TypeInfo;
const OraType = @import("../../type_info.zig").OraType;
const state = @import("../../../semantics/state.zig");
const SymbolTable = state.SymbolTable;
const Scope = state.Scope;
const TypeResolutionError = @import("../mod.zig").TypeResolutionError;

const CoreResolver = @import("mod.zig").CoreResolver;
const FunctionNode = ast.FunctionNode;

/// Look up identifier in symbol table
pub fn lookupIdentifier(
    self: *CoreResolver,
    name: []const u8,
) ?TypeInfo {
    const symbol = SymbolTable.findUp(self.current_scope, name) orelse return null;
    return symbol.typ;
}

/// Resolve identifier type (used by expression synthesis)
pub fn resolveIdentifierType(
    self: *CoreResolver,
    id: *ast.Expressions.IdentifierExpr,
) TypeResolutionError!void {
    // First try to find in symbol table (for variables, functions, etc.)
    const symbol = SymbolTable.findUp(self.current_scope, id.name);

    // If not found in symbol table, check function registry as fallback
    // This handles cases where functions aren't in symbol table yet (e.g., ghost functions)
    if (symbol) |found_symbol| {
        if (found_symbol.typ) |typ| {
            // Ensure category matches ora_type if present (fixes refinement types)
            // Fix category BEFORE checking isResolved() to handle refinement types correctly
            var resolved_typ = typ;
            if (typ.ora_type) |ot| {
                // Fix custom type names: if parser assumed struct_type but it's actually enum_type
                if (ot == .struct_type) {
                    const type_name = ot.struct_type;
                    // Look up the TYPE symbol (not the variable symbol) to check if it's an enum
                    // Search from root scope to find type declarations
                    // Access symbol_table through CoreResolver
                    const root_scope: ?*const Scope = if (@hasField(@TypeOf(self.*), "symbol_table"))
                        @as(?*const Scope, @ptrCast(&self.symbol_table.root))
                    else
                        null;
                    const type_symbol = SymbolTable.findUp(root_scope, type_name);
                    if (type_symbol) |tsym| {
                        if (tsym.kind == .Enum) {
                            // Fix: change struct_type to enum_type
                            resolved_typ.ora_type = OraType{ .enum_type = type_name };
                            resolved_typ.category = .Enum;
                        } else {
                            // Use the category from the ora_type
                            const derived_category = ot.getCategory();
                            resolved_typ.category = derived_category;
                        }
                    } else {
                        // Type not found, use category from ora_type
                        const derived_category = ot.getCategory();
                        resolved_typ.category = derived_category;
                    }
                } else {
                    // Use the category from the ora_type
                    const derived_category = ot.getCategory();
                    resolved_typ.category = derived_category;
                }
            } else {
                // If ora_type is null, this symbol was stored during collection phase with unresolved type
                // This should have been fixed by updateSymbolType in resolveVariableDecl
                // If we're seeing this, it means the symbol wasn't updated, which is a bug
                // For now, return UnresolvedType to allow compilation to continue and report the error
                return TypeResolutionError.UnresolvedType;
            }
            // Now check if resolved after fixing category
            if (!resolved_typ.isResolved()) {
                return TypeResolutionError.UnresolvedType;
            }
            id.type_info = resolved_typ;
        } else {
            // Symbol found but no type - handle special cases
            if (found_symbol.kind == .Error) {
                // Error symbols don't have types, create Error type
                id.type_info = TypeInfo{
                    .category = .Error,
                    .ora_type = null,
                    .source = .inferred,
                    .span = id.span,
                };
                return;
            }
            // For variables with null type (e.g., catch block error variables), create a placeholder type
            // TODO: Properly type error variables based on the error union type of the try expression
            if (found_symbol.kind == .Var) {
                // Create a placeholder Error type for catch block error variables
                // This allows them to be used in switch statements and other error handling code
                id.type_info = TypeInfo{
                    .category = .Error,
                    .ora_type = null,
                    .source = .inferred,
                    .span = id.span,
                };
                return;
            }
            return TypeResolutionError.UnresolvedType;
        }
    } else {
        // Not in symbol table - check function registry
        if (self.function_registry) |registry| {
            // Type-erased registry - need to cast back to get function
            const registry_map = @as(*std.StringHashMap(*ast.FunctionNode), @ptrCast(@alignCast(registry)));
            if (registry_map.get(id.name)) |function| {
                // Found in function registry - create a function type from the function
                // Extract return type if available
                if (function.return_type_info) |ret_info| {
                    id.type_info = ret_info;
                    return;
                } else {
                    // No return type - mark as unknown and let call resolver handle it
                    id.type_info = TypeInfo.unknown();
                    return;
                }
            }
        }
        // Check builtin registry for constants (e.g., std.constants.ZERO_ADDRESS)
        // Note: This handles the "std" part of field accesses like std.constants.ZERO_ADDRESS
        // The full path resolution happens in synthFieldAccess
        if (self.builtin_registry) |_| {
            // Check if "std" is a known namespace in the builtin registry
            // We can't resolve the full path here, but we can check if it's a known namespace
            // The actual resolution will happen in synthFieldAccess when we have the full path
            if (std.mem.eql(u8, id.name, "std")) {
                // "std" is a builtin namespace - mark as unknown for now, will be resolved in field access
                id.type_info = TypeInfo.unknown();
                return;
            }
        }
        return TypeResolutionError.UndefinedIdentifier;
    }
}
