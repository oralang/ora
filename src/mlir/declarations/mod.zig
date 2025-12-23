// ============================================================================
// Declaration Lowering - Core Module
// ============================================================================
//
// Core DeclarationLowerer struct and initialization.
// Other modules handle specific declaration types.
//
// ============================================================================

const std = @import("std");
const c = @import("../c.zig").c;
const lib = @import("ora_lib");
const TypeMapper = @import("../types.zig").TypeMapper;
const LocationTracker = @import("../locations.zig").LocationTracker;

/// Declaration lowering system for converting Ora top-level declarations to MLIR
pub const DeclarationLowerer = struct {
    ctx: c.MlirContext,
    type_mapper: *const TypeMapper,
    locations: LocationTracker,
    error_handler: ?*const @import("../error_handling.zig").ErrorHandler,
    ora_dialect: *@import("../dialect.zig").OraDialect,
    symbol_table: ?*@import("../lower.zig").SymbolTable = null,
    builtin_registry: ?*const lib.semantics.builtins.BuiltinRegistry = null,

    pub fn init(ctx: c.MlirContext, type_mapper: *const TypeMapper, locations: LocationTracker, ora_dialect: *@import("../dialect.zig").OraDialect) DeclarationLowerer {
        return .{
            .ctx = ctx,
            .type_mapper = type_mapper,
            .locations = locations,
            .error_handler = null,
            .ora_dialect = ora_dialect,
        };
    }

    pub fn withErrorHandler(ctx: c.MlirContext, type_mapper: *const TypeMapper, locations: LocationTracker, error_handler: *const @import("../error_handling.zig").ErrorHandler, ora_dialect: *@import("../dialect.zig").OraDialect) DeclarationLowerer {
        return .{
            .ctx = ctx,
            .type_mapper = type_mapper,
            .locations = locations,
            .error_handler = error_handler,
            .ora_dialect = ora_dialect,
        };
    }

    pub fn withErrorHandlerAndDialect(ctx: c.MlirContext, type_mapper: *const TypeMapper, locations: LocationTracker, error_handler: *const @import("../error_handling.zig").ErrorHandler, ora_dialect: *@import("../dialect.zig").OraDialect) DeclarationLowerer {
        return .{
            .ctx = ctx,
            .type_mapper = type_mapper,
            .locations = locations,
            .error_handler = error_handler,
            .ora_dialect = ora_dialect,
        };
    }

    pub fn withErrorHandlerAndDialectAndSymbolTable(ctx: c.MlirContext, type_mapper: *const TypeMapper, locations: LocationTracker, error_handler: *const @import("../error_handling.zig").ErrorHandler, ora_dialect: *@import("../dialect.zig").OraDialect, symbol_table: *@import("../lower.zig").SymbolTable, builtin_registry: ?*const lib.semantics.builtins.BuiltinRegistry) DeclarationLowerer {
        return .{
            .ctx = ctx,
            .type_mapper = type_mapper,
            .locations = locations,
            .error_handler = error_handler,
            .ora_dialect = ora_dialect,
            .symbol_table = symbol_table,
            .builtin_registry = builtin_registry,
        };
    }

    // Re-export all lowering functions from submodules
    pub const lowerFunction = @import("function.zig").lowerFunction;
    pub const lowerContract = @import("contract.zig").lowerContract;
    pub const lowerStruct = @import("types.zig").lowerStruct;
    pub const lowerEnum = @import("types.zig").lowerEnum;
    pub const createStructType = @import("types.zig").createStructType;
    pub const createEnumType = @import("types.zig").createEnumType;
    pub const createErrorType = @import("types.zig").createErrorType;
    pub const lowerConstDecl = @import("globals.zig").lowerConstDecl;
    pub const lowerImmutableDecl = @import("globals.zig").lowerImmutableDecl;
    pub const createGlobalDeclaration = @import("globals.zig").createGlobalDeclaration;
    pub const createMemoryGlobalDeclaration = @import("globals.zig").createMemoryGlobalDeclaration;
    pub const createTStoreGlobalDeclaration = @import("globals.zig").createTStoreGlobalDeclaration;
    pub const lowerModule = @import("module.zig").lowerModule;
    pub const lowerBlock = @import("module.zig").lowerBlock;
    pub const lowerTryBlock = @import("module.zig").lowerTryBlock;
    pub const lowerImport = @import("module.zig").lowerImport;
    pub const lowerLogDecl = @import("logs.zig").lowerLogDecl;
    pub const lowerErrorDecl = @import("logs.zig").lowerErrorDecl;
    pub const insertRefinementGuard = @import("refinements.zig").insertRefinementGuard;
    pub const lowerQuantifiedExpression = @import("verification.zig").lowerQuantifiedExpression;
    pub const addVerificationAttributes = @import("verification.zig").addVerificationAttributes;
    pub const lowerFormalVerificationConstructs = @import("verification.zig").lowerFormalVerificationConstructs;
    pub const createFileLocation = @import("helpers.zig").createFileLocation;
    pub const getExpressionSpan = @import("helpers.zig").getExpressionSpan;
    pub const createConstant = @import("helpers.zig").createConstant;
    pub const createVariablePlaceholder = @import("helpers.zig").createVariablePlaceholder;
    pub const createModulePlaceholder = @import("helpers.zig").createModulePlaceholder;
    pub const oraTypeToString = @import("helpers.zig").oraTypeToString;
    pub const createFunctionType = @import("helpers.zig").createFunctionType;
};
