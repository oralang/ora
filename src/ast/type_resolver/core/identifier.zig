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
    const scope = if (self.current_scope) |s| s else self.symbol_table.root;
    var symbol = self.symbol_table.safeFindUpOpt(scope, name);
    if (symbol == null and scope != self.symbol_table.root) {
        symbol = self.symbol_table.safeFindUpOpt(self.symbol_table.root, name);
    }
    if (symbol == null) return null;
    return symbol.typ;
}

/// Resolve identifier type (used by expression synthesis)
pub fn resolveIdentifierType(
    self: *CoreResolver,
    id: *ast.Expressions.IdentifierExpr,
) TypeResolutionError!void {
    // first try to find in symbol table (for variables, functions, etc.)
    const scope = if (self.current_scope) |s| s else self.symbol_table.root;
    var symbol = self.symbol_table.safeFindUpOpt(scope, id.name);
    if (symbol == null and scope != self.symbol_table.root) {
        symbol = self.symbol_table.safeFindUpOpt(self.symbol_table.root, id.name);
    }

    // if not found in symbol table, check function registry as fallback
    // this handles cases where functions aren't in symbol table yet (e.g., ghost functions)
    if (symbol) |found_symbol| {
        if (found_symbol.typ) |typ| {
            // ensure category matches ora_type if present (fixes refinement types)
            // fix category BEFORE checking isResolved() to handle refinement types correctly
            var resolved_typ = typ;
            if (typ.ora_type) |ot| {
                // fix custom type names: if parser assumed struct_type but it's actually enum_type
                if (ot == .struct_type) {
                    const type_name = ot.struct_type;
                    // look up the TYPE symbol (not the variable symbol) to check if it's an enum
                    // search from root scope to find type declarations
                    // access symbol_table through CoreResolver
                    const root_scope: ?*const Scope = if (@hasField(@TypeOf(self.*), "symbol_table"))
                        @as(?*const Scope, @ptrCast(self.symbol_table.root))
                    else
                        null;
                    const type_symbol = SymbolTable.findUp(root_scope, type_name);
                    if (type_symbol) |tsym| {
                        if (tsym.kind == .Enum) {
                            // fix: change struct_type to enum_type
                            resolved_typ.ora_type = OraType{ .enum_type = type_name };
                            resolved_typ.category = .Enum;
                        } else {
                            // use the category from the ora_type
                            var derived_category = ot.getCategory();
                            if (ot == ._union and ot._union.len > 0 and ot._union[0] == .error_union) {
                                derived_category = .ErrorUnion;
                            }
                            resolved_typ.category = derived_category;
                        }
                    } else {
                        // type not found, use category from ora_type
                        var derived_category = ot.getCategory();
                        if (ot == ._union and ot._union.len > 0 and ot._union[0] == .error_union) {
                            derived_category = .ErrorUnion;
                        }
                        resolved_typ.category = derived_category;
                    }
                } else {
                    // use the category from the ora_type
                    var derived_category = ot.getCategory();
                    if (ot == ._union and ot._union.len > 0 and ot._union[0] == .error_union) {
                        derived_category = .ErrorUnion;
                    }
                    resolved_typ.category = derived_category;
                }
            } else {
                // if ora_type is null, this symbol was stored during collection phase with unresolved type
                // this should have been fixed by updateSymbolType in resolveVariableDecl
                // if we're seeing this, it means the symbol wasn't updated, which is a bug
                // for now, return UnresolvedType to allow compilation to continue and report the error
                return TypeResolutionError.UnresolvedType;
            }
            // now check if resolved after fixing category
            if (!resolved_typ.isResolved()) {
                return TypeResolutionError.UnresolvedType;
            }
            id.type_info = resolved_typ;
            id.type_info.region = found_symbol.region;
        } else {
            // symbol found but no type - handle special cases
            if (found_symbol.kind == .Error) {
                // error symbols don't have types, create Error type
                id.type_info = TypeInfo{
                    .category = .Error,
                    .ora_type = null,
                    .source = .inferred,
                    .span = id.span,
                };
                return;
            }
            // for variables with null type (e.g., catch block error variables), create a placeholder type
            // todo: Properly type error variables based on the error union type of the try expression
            if (found_symbol.kind == .Var) {
                // create a placeholder Error type for catch block error variables
                // this allows them to be used in switch statements and other error handling code
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
        // not in symbol table - check function registry
        if (self.function_registry) |registry| {
            // type-erased registry - need to cast back to get function
            const registry_map = @as(*std.StringHashMap(*ast.FunctionNode), @ptrCast(@alignCast(registry)));
            if (registry_map.get(id.name)) |function| {
                // found in function registry - create a function type from the function
                // extract return type if available
                if (function.return_type_info) |ret_info| {
                    id.type_info = ret_info;
                    return;
                } else {
                    // no return type - mark as unknown and let call resolver handle it
                    id.type_info = TypeInfo.unknown();
                    return;
                }
            }
        }
        // check builtin registry for constants (e.g., std.constants.ZERO_ADDRESS)
        // note: This handles the "std" part of field accesses like std.constants.ZERO_ADDRESS
        // the full path resolution happens in synthFieldAccess
        if (self.builtin_registry) |_| {
            // check if "std" is a known namespace in the builtin registry
            // we can't resolve the full path here, but we can check if it's a known namespace
            // the actual resolution will happen in synthFieldAccess when we have the full path
            if (std.mem.eql(u8, id.name, "std")) {
                // "std" is a builtin namespace - mark as unknown for now, will be resolved in field access
                id.type_info = TypeInfo.unknown();
                return;
            }
        }
        return TypeResolutionError.UndefinedIdentifier;
    }
}
