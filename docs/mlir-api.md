# MLIR Lowering API Reference

## Overview

This document provides comprehensive API documentation for the MLIR lowering system's public interfaces. The API is designed to be modular, extensible, and easy to integrate with the existing Ora compiler pipeline.

**Important:** This API documentation covers the Ora-specific MLIR lowering components that integrate with the LLVM MLIR framework (https://mlir.llvm.org). We use the existing MLIR C API and infrastructure, extending it with Ora-specific dialects, operations, and lowering logic.

## Core Types and Structures

### LoweringResult

The main result type returned by the MLIR lowering system.

```zig
pub const LoweringResult = struct {
    module: c.MlirModule,           // Generated MLIR module
    errors: []const LoweringError, // Compilation errors
    warnings: []const LoweringWarning, // Compilation warnings
    success: bool,                  // Overall success status
    pass_result: ?PassResult,       // Optional pass execution results
    
    pub fn deinit(self: *LoweringResult, allocator: std.mem.Allocator) void;
    pub fn hasErrors(self: *const LoweringResult) bool;
    pub fn hasWarnings(self: *const LoweringResult) bool;
};
```

### LoweringError

Represents compilation errors during MLIR lowering.

```zig
pub const LoweringError = struct {
    kind: ErrorKind,
    message: []const u8,
    span: ?lib.ast.SourceSpan,
    suggestion: ?[]const u8,
    
    pub const ErrorKind = enum {
        UnsupportedAstNode,
        TypeMismatch,
        UndefinedSymbol,
        InvalidMemoryRegion,
        MalformedExpression,
        MlirOperationFailed,
        SymbolTableError,
        LocationTrackingError,
    };
};
```

### LoweringWarning

Represents compilation warnings during MLIR lowering.

```zig
pub const LoweringWarning = struct {
    kind: WarningKind,
    message: []const u8,
    span: ?lib.ast.SourceSpan,
    suggestion: ?[]const u8,
    
    pub const WarningKind = enum {
        UnusedVariable,
        DeadCode,
        PerformanceHint,
        MemoryRegionMismatch,
        DeprecatedFeature,
    };
};
```

## Main Entry Points

### lowerFunctionsToModuleWithErrors

The primary entry point for MLIR lowering with comprehensive error handling.

```zig
pub fn lowerFunctionsToModuleWithErrors(
    ctx: c.MlirContext, 
    nodes: []lib.AstNode, 
    allocator: std.mem.Allocator
) !LoweringResult
```

**Parameters:**
- `ctx`: MLIR context for operation creation
- `nodes`: Array of AST nodes to lower
- `allocator`: Memory allocator for temporary allocations

**Returns:** `LoweringResult` containing the MLIR module and any errors/warnings

**Example:**
```zig
const result = try lowerFunctionsToModuleWithErrors(ctx, ast_nodes, allocator);
defer result.deinit(allocator);

if (result.hasErrors()) {
    for (result.errors) |err| {
        std.debug.print("Error: {s}\n", .{err.message});
    }
    return;
}

// Use result.module for further processing
```

### lowerFunctionsToModule

Legacy entry point for backward compatibility.

```zig
pub fn lowerFunctionsToModule(
    ctx: c.MlirContext, 
    nodes: []lib.AstNode, 
    allocator: std.mem.Allocator
) c.MlirModule
```

**Note:** This function panics on errors. Use `lowerFunctionsToModuleWithErrors` for production code.

## Type Mapping API

### TypeMapper

The type mapping system converts Ora types to MLIR types.

```zig
pub const TypeMapper = struct {
    ctx: c.MlirContext,
    inference_ctx: ?*TypeInference.InferenceContext,
    
    pub fn init(ctx: c.MlirContext) TypeMapper;
    pub fn initWithInference(ctx: c.MlirContext, inference_ctx: *TypeInference.InferenceContext) TypeMapper;
    
    // Core type mapping functions
    pub fn toMlirType(self: *const TypeMapper, ora_type: lib.ast.type_info.TypeInfo) !c.MlirType;
    pub fn createOraDialectType(self: *const TypeMapper, type_name: []const u8, params: []c.MlirType) !c.MlirType;
    
    // Primitive type mapping
    pub fn mapPrimitiveType(self: *const TypeMapper, primitive: lib.ast.type_info.PrimitiveType) c.MlirType;
    pub fn mapIntegerType(self: *const TypeMapper, width: u32, is_signed: bool) c.MlirType;
    pub fn mapBooleanType(self: *const TypeMapper) c.MlirType;
    pub fn mapAddressType(self: *const TypeMapper) c.MlirType;
    pub fn mapStringType(self: *const TypeMapper) c.MlirType;
    pub fn mapBytesType(self: *const TypeMapper) c.MlirType;
    pub fn mapVoidType(self: *const TypeMapper) c.MlirType;
    
    // Complex type mapping
    pub fn mapArrayType(self: *const TypeMapper, elem_type: *const lib.ast.type_info.OraType, length: u64) !c.MlirType;
    pub fn mapSliceType(self: *const TypeMapper, elem_type: *const lib.ast.type_info.OraType) !c.MlirType;
    pub fn mapMapType(self: *const TypeMapper, key_type: *const lib.ast.type_info.OraType, value_type: *const lib.ast.type_info.OraType) !c.MlirType;
    pub fn mapDoubleMapType(self: *const TypeMapper, key1_type: *const lib.ast.type_info.OraType, key2_type: *const lib.ast.type_info.OraType, value_type: *const lib.ast.type_info.OraType) !c.MlirType;
    pub fn mapStructType(self: *const TypeMapper, struct_type: lib.ast.type_info.StructType) !c.MlirType;
    pub fn mapEnumType(self: *const TypeMapper, enum_type: lib.ast.type_info.EnumType) !c.MlirType;
    pub fn mapErrorType(self: *const TypeMapper, error_type: *const lib.ast.type_info.OraType) !c.MlirType;
    pub fn mapErrorUnionType(self: *const TypeMapper, union_types: []const *const lib.ast.type_info.OraType) !c.MlirType;
    
    // Type validation and utilities
    pub fn validateTypeMapping(self: *const TypeMapper, ora_type: lib.ast.type_info.TypeInfo, mlir_type: c.MlirType) bool;
    pub fn getTypeSize(self: *const TypeMapper, mlir_type: c.MlirType) ?u64;
    pub fn isCompatibleType(self: *const TypeMapper, type1: c.MlirType, type2: c.MlirType) bool;
};
```

### Type Inference

Advanced type system features for generic types and type aliases.

```zig
pub const TypeInference = struct {
    pub const TypeVariable = struct {
        name: []const u8,
        constraints: []const lib.ast.type_info.OraType,
        resolved_type: ?lib.ast.type_info.OraType,
    };
    
    pub const TypeAlias = struct {
        name: []const u8,
        target_type: lib.ast.type_info.OraType,
        generic_params: []const TypeVariable,
    };
    
    pub const InferenceContext = struct {
        pub fn init(allocator: std.mem.Allocator) InferenceContext;
        pub fn deinit(self: *InferenceContext) void;
        
        pub fn addTypeVariable(self: *InferenceContext, name: []const u8, constraints: []const lib.ast.type_info.OraType) !void;
        pub fn addTypeAlias(self: *InferenceContext, name: []const u8, target_type: lib.ast.type_info.OraType, generic_params: []const TypeVariable) !void;
        pub fn resolveTypeVariable(self: *InferenceContext, name: []const u8, resolved_type: lib.ast.type_info.OraType) !void;
        pub fn lookupTypeAlias(self: *const InferenceContext, name: []const u8) ?TypeAlias;
        pub fn inferType(self: *InferenceContext, expr: *const lib.ast.Expressions.ExprNode) !lib.ast.type_info.OraType;
    };
};
```

## Expression Lowering API

### ExpressionLowerer

Handles lowering of all expression types to MLIR operations.

```zig
pub const ExpressionLowerer = struct {
    ctx: c.MlirContext,
    block: c.MlirBlock,
    type_mapper: *const TypeMapper,
    param_map: ?*const ParamMap,
    storage_map: ?*const StorageMap,
    local_var_map: ?*const LocalVarMap,
    locations: LocationTracker,
    
    pub fn init(
        ctx: c.MlirContext, 
        block: c.MlirBlock, 
        type_mapper: *const TypeMapper, 
        param_map: ?*const ParamMap, 
        storage_map: ?*const StorageMap, 
        local_var_map: ?*const LocalVarMap, 
        locations: LocationTracker
    ) ExpressionLowerer;
    
    // Main expression lowering
    pub fn lowerExpression(self: *const ExpressionLowerer, expr: *const lib.ast.Expressions.ExprNode) c.MlirValue;
    
    // Literal lowering
    pub fn lowerLiteral(self: *const ExpressionLowerer, literal: *const lib.ast.Expressions.LiteralNode) c.MlirValue;
    pub fn lowerIntegerLiteral(self: *const ExpressionLowerer, value: u256, width: u32) c.MlirValue;
    pub fn lowerBooleanLiteral(self: *const ExpressionLowerer, value: bool) c.MlirValue;
    pub fn lowerStringLiteral(self: *const ExpressionLowerer, value: []const u8) c.MlirValue;
    pub fn lowerAddressLiteral(self: *const ExpressionLowerer, value: []const u8) c.MlirValue;
    
    // Binary operation lowering
    pub fn lowerBinary(self: *const ExpressionLowerer, binary: *const lib.ast.Expressions.BinaryNode) c.MlirValue;
    pub fn lowerArithmeticOp(self: *const ExpressionLowerer, op: lib.ast.Expressions.BinaryOp, lhs: c.MlirValue, rhs: c.MlirValue, span: lib.ast.SourceSpan) c.MlirValue;
    pub fn lowerComparisonOp(self: *const ExpressionLowerer, op: lib.ast.Expressions.BinaryOp, lhs: c.MlirValue, rhs: c.MlirValue, span: lib.ast.SourceSpan) c.MlirValue;
    pub fn lowerLogicalOp(self: *const ExpressionLowerer, op: lib.ast.Expressions.BinaryOp, lhs: c.MlirValue, rhs: c.MlirValue, span: lib.ast.SourceSpan) c.MlirValue;
    pub fn lowerBitwiseOp(self: *const ExpressionLowerer, op: lib.ast.Expressions.BinaryOp, lhs: c.MlirValue, rhs: c.MlirValue, span: lib.ast.SourceSpan) c.MlirValue;
    
    // Unary operation lowering
    pub fn lowerUnary(self: *const ExpressionLowerer, unary: *const lib.ast.Expressions.UnaryNode) c.MlirValue;
    pub fn lowerLogicalNot(self: *const ExpressionLowerer, operand: c.MlirValue, span: lib.ast.SourceSpan) c.MlirValue;
    pub fn lowerArithmeticNegation(self: *const ExpressionLowerer, operand: c.MlirValue, span: lib.ast.SourceSpan) c.MlirValue;
    pub fn lowerUnaryPlus(self: *const ExpressionLowerer, operand: c.MlirValue, span: lib.ast.SourceSpan) c.MlirValue;
    
    // Complex expression lowering
    pub fn lowerIdentifier(self: *const ExpressionLowerer, identifier: *const lib.ast.Expressions.IdentifierNode) c.MlirValue;
    pub fn lowerCall(self: *const ExpressionLowerer, call: *const lib.ast.Expressions.CallNode) c.MlirValue;
    pub fn lowerFieldAccess(self: *const ExpressionLowerer, field_access: *const lib.ast.Expressions.FieldAccessNode) c.MlirValue;
    pub fn lowerIndex(self: *const ExpressionLowerer, index: *const lib.ast.Expressions.IndexNode) c.MlirValue;
    pub fn lowerCast(self: *const ExpressionLowerer, cast: *const lib.ast.Expressions.CastNode) c.MlirValue;
    pub fn lowerSwitchExpression(self: *const ExpressionLowerer, switch_expr: *const lib.ast.Expressions.SwitchExpressionNode) c.MlirValue;
    
    // Assignment lowering
    pub fn lowerAssignment(self: *const ExpressionLowerer, assignment: *const lib.ast.Expressions.AssignmentNode) c.MlirValue;
    pub fn lowerCompoundAssignment(self: *const ExpressionLowerer, comp_assign: *const lib.ast.Expressions.CompoundAssignmentNode) c.MlirValue;
    
    // Advanced expression lowering
    pub fn lowerComptime(self: *const ExpressionLowerer, comptime_expr: *const lib.ast.Expressions.ComptimeNode) c.MlirValue;
    pub fn lowerOld(self: *const ExpressionLowerer, old: *const lib.ast.Expressions.OldNode) c.MlirValue;
    pub fn lowerQuantified(self: *const ExpressionLowerer, quantified: *const lib.ast.Expressions.QuantifiedNode) c.MlirValue;
    pub fn lowerTuple(self: *const ExpressionLowerer, tuple: *const lib.ast.Expressions.TupleNode) c.MlirValue;
    pub fn lowerArrayLiteral(self: *const ExpressionLowerer, array: *const lib.ast.Expressions.ArrayLiteralNode) c.MlirValue;
    pub fn lowerStructInstantiation(self: *const ExpressionLowerer, struct_inst: *const lib.ast.Expressions.StructInstantiationNode) c.MlirValue;
    pub fn lowerAnonymousStruct(self: *const ExpressionLowerer, anon_struct: *const lib.ast.Expressions.AnonymousStructNode) c.MlirValue;
};
```

## Statement Lowering API

### StatementLowerer

Handles lowering of all statement types to MLIR operations and control flow.

```zig
pub const StatementLowerer = struct {
    ctx: c.MlirContext,
    block: c.MlirBlock,
    type_mapper: *const TypeMapper,
    expr_lowerer: *const ExpressionLowerer,
    param_map: ?*const ParamMap,
    storage_map: ?*const StorageMap,
    local_var_map: ?*LocalVarMap,
    locations: LocationTracker,
    
    pub fn init(
        ctx: c.MlirContext, 
        block: c.MlirBlock, 
        type_mapper: *const TypeMapper, 
        expr_lowerer: *const ExpressionLowerer, 
        param_map: ?*const ParamMap, 
        storage_map: ?*const StorageMap, 
        local_var_map: ?*LocalVarMap, 
        locations: LocationTracker
    ) StatementLowerer;
    
    // Main statement lowering
    pub fn lowerStatement(self: *StatementLowerer, stmt: *const lib.ast.Statements.StmtNode) void;
    
    // Variable declarations and assignments
    pub fn lowerVariableDeclaration(self: *StatementLowerer, var_decl: *const lib.ast.Statements.VariableDeclarationNode) void;
    pub fn lowerAssignmentStatement(self: *StatementLowerer, assignment: *const lib.ast.Statements.AssignmentNode) void;
    pub fn lowerDestructuringAssignment(self: *StatementLowerer, destructuring: *const lib.ast.Statements.DestructuringAssignmentNode) void;
    
    // Control flow statements
    pub fn lowerIf(self: *StatementLowerer, if_stmt: *const lib.ast.Statements.IfNode) void;
    pub fn lowerWhile(self: *StatementLowerer, while_stmt: *const lib.ast.Statements.WhileNode) void;
    pub fn lowerForLoop(self: *StatementLowerer, for_loop: *const lib.ast.Statements.ForLoopNode) void;
    pub fn lowerSwitch(self: *StatementLowerer, switch_stmt: *const lib.ast.Statements.SwitchNode) void;
    
    // Jump statements
    pub fn lowerReturn(self: *StatementLowerer, return_stmt: *const lib.ast.Statements.ReturnNode) void;
    pub fn lowerBreak(self: *StatementLowerer, break_stmt: *const lib.ast.Statements.BreakNode) void;
    pub fn lowerContinue(self: *StatementLowerer, continue_stmt: *const lib.ast.Statements.ContinueNode) void;
    
    // Ora-specific statements
    pub fn lowerMove(self: *StatementLowerer, move_stmt: *const lib.ast.Statements.MoveNode) void;
    pub fn lowerLog(self: *StatementLowerer, log_stmt: *const lib.ast.Statements.LogNode) void;
    pub fn lowerLock(self: *StatementLowerer, lock_stmt: *const lib.ast.Statements.LockNode) void;
    pub fn lowerUnlock(self: *StatementLowerer, unlock_stmt: *const lib.ast.Statements.UnlockNode) void;
    pub fn lowerTryBlock(self: *StatementLowerer, try_block: *const lib.ast.Statements.TryBlockNode) void;
    
    // Contract statements
    pub fn lowerRequires(self: *StatementLowerer, requires_stmt: *const lib.ast.Statements.RequiresNode) void;
    pub fn lowerEnsures(self: *StatementLowerer, ensures_stmt: *const lib.ast.Statements.EnsuresNode) void;
    pub fn lowerInvariant(self: *StatementLowerer, invariant_stmt: *const lib.ast.Statements.InvariantNode) void;
    
    // Block and labeled statements
    pub fn lowerBlock(self: *StatementLowerer, block: *const lib.ast.Statements.BlockNode) void;
    pub fn lowerLabeledBlock(self: *StatementLowerer, labeled_block: *const lib.ast.Statements.LabeledBlockNode) void;
};
```

## Declaration Lowering API

### DeclarationLowerer

Handles lowering of top-level declarations to MLIR module-level constructs.

```zig
pub const DeclarationLowerer = struct {
    ctx: c.MlirContext,
    module: c.MlirModule,
    type_mapper: *const TypeMapper,
    locations: LocationTracker,
    
    pub fn init(ctx: c.MlirContext, module: c.MlirModule, type_mapper: *const TypeMapper, locations: LocationTracker) DeclarationLowerer;
    
    // Main declaration lowering
    pub fn lowerDeclaration(self: *const DeclarationLowerer, decl: lib.AstNode) !void;
    
    // Function declarations
    pub fn lowerFunction(self: *const DeclarationLowerer, func: *const lib.ast.FunctionNode) !void;
    pub fn createFunctionType(self: *const DeclarationLowerer, func: *const lib.ast.FunctionNode) !c.MlirType;
    pub fn lowerFunctionBody(self: *const DeclarationLowerer, func: *const lib.ast.FunctionNode, mlir_func: c.MlirOperation) !void;
    
    // Contract declarations
    pub fn lowerContract(self: *const DeclarationLowerer, contract: *const lib.ast.ContractNode) !void;
    pub fn lowerContractMember(self: *const DeclarationLowerer, member: lib.AstNode, contract_name: []const u8) !void;
    
    // Type declarations
    pub fn lowerStructDecl(self: *const DeclarationLowerer, struct_decl: *const lib.ast.StructDeclNode) !void;
    pub fn lowerEnumDecl(self: *const DeclarationLowerer, enum_decl: *const lib.ast.EnumDeclNode) !void;
    pub fn lowerErrorDecl(self: *const DeclarationLowerer, error_decl: *const lib.ast.ErrorDeclNode) !void;
    
    // Event declarations
    pub fn lowerLogDecl(self: *const DeclarationLowerer, log_decl: *const lib.ast.LogDeclNode) !void;
    
    // Global declarations
    pub fn lowerConstant(self: *const DeclarationLowerer, constant: *const lib.ast.ConstantNode) !void;
    pub fn lowerImmutable(self: *const DeclarationLowerer, immutable: *const lib.ast.ImmutableNode) !void;
    pub fn lowerImport(self: *const DeclarationLowerer, import: *const lib.ast.ImportNode) !void;
    
    // Module declarations
    pub fn lowerModule(self: *const DeclarationLowerer, module_node: *const lib.ast.ModuleNode) !void;
    
    // Utility functions
    pub fn attachVisibilityAttribute(self: *const DeclarationLowerer, operation: c.MlirOperation, visibility: lib.ast.Visibility) void;
    pub fn attachInlineAttribute(self: *const DeclarationLowerer, operation: c.MlirOperation) void;
    pub fn attachContractAttribute(self: *const DeclarationLowerer, operation: c.MlirOperation, contract_name: []const u8) void;
};
```

## Memory Management API

### MemoryManager

Manages memory region semantics and ensures correct memory space usage.

```zig
pub const MemoryManager = struct {
    pub const MemoryRegion = enum {
        Storage,
        Memory,
        TStore,
        
        pub fn toSpace(self: MemoryRegion) u32 {
            return switch (self) {
                .Storage => 1,
                .Memory => 0,
                .TStore => 2,
            };
        }
        
        pub fn toString(self: MemoryRegion) []const u8 {
            return switch (self) {
                .Storage => "storage",
                .Memory => "memory",
                .TStore => "tstore",
            };
        }
    };
    
    ctx: c.MlirContext,
    
    pub fn init(ctx: c.MlirContext) MemoryManager;
    
    // Memory space management
    pub fn getMemorySpace(region: MemoryRegion) u32;
    pub fn createRegionAttribute(self: *const MemoryManager, region: MemoryRegion) c.MlirAttribute;
    pub fn validateMemoryAccess(region: MemoryRegion, access_type: AccessType) bool;
    
    // Memory operation creation
    pub fn createAllocaOp(self: *const MemoryManager, block: c.MlirBlock, mlir_type: c.MlirType, region: MemoryRegion, span: lib.ast.SourceSpan) c.MlirValue;
    pub fn createStoreOp(self: *const MemoryManager, block: c.MlirBlock, value: c.MlirValue, address: c.MlirValue, region: MemoryRegion, span: lib.ast.SourceSpan) void;
    pub fn createLoadOp(self: *const MemoryManager, block: c.MlirBlock, address: c.MlirValue, region: MemoryRegion, span: lib.ast.SourceSpan) c.MlirValue;
    
    // Region validation
    pub fn validateRegionConstraints(self: *const MemoryManager, operations: []c.MlirOperation) []const MemoryViolation;
    pub fn checkRegionCompatibility(self: *const MemoryManager, source_region: MemoryRegion, target_region: MemoryRegion) bool;
};

pub const StorageMap = std.HashMap([]const u8, StorageEntry, std.hash_map.StringContext, std.hash_map.default_max_load_percentage);

pub const StorageEntry = struct {
    value: c.MlirValue,
    type: c.MlirType,
    region: MemoryManager.MemoryRegion,
    is_mutable: bool,
};
```

## Symbol Table API

### SymbolTable

Tracks variable bindings, scopes, and symbol resolution during lowering.

```zig
pub const SymbolTable = struct {
    scopes: std.ArrayList(Scope),
    allocator: std.mem.Allocator,
    
    pub const SymbolEntry = struct {
        name: []const u8,
        value: c.MlirValue,
        type: c.MlirType,
        region: ?MemoryManager.MemoryRegion,
        is_mutable: bool,
        scope_level: u32,
    };
    
    pub const Scope = std.HashMap([]const u8, SymbolEntry, std.hash_map.StringContext, std.hash_map.default_max_load_percentage);
    
    pub fn init(allocator: std.mem.Allocator) SymbolTable;
    pub fn deinit(self: *SymbolTable) void;
    
    // Scope management
    pub fn pushScope(self: *SymbolTable) !void;
    pub fn popScope(self: *SymbolTable) void;
    pub fn getCurrentScopeLevel(self: *const SymbolTable) u32;
    
    // Symbol management
    pub fn addSymbol(self: *SymbolTable, name: []const u8, value: c.MlirValue, mlir_type: c.MlirType, region: ?MemoryManager.MemoryRegion, is_mutable: bool) !void;
    pub fn lookupSymbol(self: *const SymbolTable, name: []const u8) ?SymbolEntry;
    pub fn updateSymbol(self: *SymbolTable, name: []const u8, new_value: c.MlirValue) !void;
    pub fn removeSymbol(self: *SymbolTable, name: []const u8) bool;
    
    // Function and type symbols
    pub fn addFunction(self: *SymbolTable, name: []const u8, func_op: c.MlirOperation) !void;
    pub fn lookupFunction(self: *const SymbolTable, name: []const u8) ?c.MlirOperation;
    pub fn addType(self: *SymbolTable, name: []const u8, mlir_type: c.MlirType) !void;
    pub fn lookupType(self: *const SymbolTable, name: []const u8) ?c.MlirType;
    
    // Utility functions
    pub fn symbolExists(self: *const SymbolTable, name: []const u8) bool;
    pub fn getSymbolsInCurrentScope(self: *const SymbolTable) []const SymbolEntry;
    pub fn getAllSymbols(self: *const SymbolTable) []const SymbolEntry;
};

pub const ParamMap = std.HashMap([]const u8, c.MlirValue, std.hash_map.StringContext, std.hash_map.default_max_load_percentage);
pub const LocalVarMap = std.HashMap([]const u8, c.MlirValue, std.hash_map.StringContext, std.hash_map.default_max_load_percentage);
```

## Location Tracking API

### LocationTracker

Preserves source location information throughout the lowering process.

```zig
pub const LocationTracker = struct {
    ctx: c.MlirContext,
    file_name: []const u8,
    
    pub fn init(ctx: c.MlirContext, file_name: []const u8) LocationTracker;
    
    // Location creation
    pub fn createLocation(self: *const LocationTracker, span: lib.ast.SourceSpan) c.MlirLocation;
    pub fn createFileLocation(self: *const LocationTracker, line: u32, column: u32) c.MlirLocation;
    pub fn createUnknownLocation(self: *const LocationTracker) c.MlirLocation;
    
    // Location attachment
    pub fn attachLocationToOp(self: *const LocationTracker, operation: c.MlirOperation, span: lib.ast.SourceSpan) void;
    pub fn attachLocationToValue(self: *const LocationTracker, value: c.MlirValue, span: lib.ast.SourceSpan) void;
    
    // Span utilities
    pub fn preserveSpanInfo(self: *const LocationTracker, span: lib.ast.SourceSpan) c.MlirAttribute;
    pub fn extractSpanFromLocation(self: *const LocationTracker, location: c.MlirLocation) ?lib.ast.SourceSpan;
    
    // Debug information
    pub fn createDebugInfo(self: *const LocationTracker, span: lib.ast.SourceSpan, additional_info: []const u8) c.MlirAttribute;
    pub fn attachDebugInfo(self: *const LocationTracker, operation: c.MlirOperation, debug_info: c.MlirAttribute) void;
};
```

## Error Handling API

### ErrorHandler

Comprehensive error reporting and recovery system.

```zig
pub const ErrorHandler = struct {
    errors: std.ArrayList(LoweringError),
    warnings: std.ArrayList(LoweringWarning),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) ErrorHandler;
    pub fn deinit(self: *ErrorHandler) void;
    
    // Error reporting
    pub fn reportError(self: *ErrorHandler, kind: LoweringError.ErrorKind, message: []const u8, span: ?lib.ast.SourceSpan, suggestion: ?[]const u8) void;
    pub fn reportWarning(self: *ErrorHandler, kind: LoweringWarning.WarningKind, message: []const u8, span: ?lib.ast.SourceSpan, suggestion: ?[]const u8) void;
    
    // Error queries
    pub fn hasErrors(self: *const ErrorHandler) bool;
    pub fn hasWarnings(self: *const ErrorHandler) bool;
    pub fn getErrorCount(self: *const ErrorHandler) usize;
    pub fn getWarningCount(self: *const ErrorHandler) usize;
    
    // Error retrieval
    pub fn getErrors(self: *const ErrorHandler) []const LoweringError;
    pub fn getWarnings(self: *const ErrorHandler) []const LoweringWarning;
    pub fn getErrorsOfKind(self: *const ErrorHandler, kind: LoweringError.ErrorKind) []const LoweringError;
    pub fn getWarningsOfKind(self: *const ErrorHandler, kind: LoweringWarning.WarningKind) []const LoweringWarning;
    
    // Error formatting
    pub fn formatError(self: *const ErrorHandler, error_info: LoweringError, writer: anytype) !void;
    pub fn formatWarning(self: *const ErrorHandler, warning: LoweringWarning, writer: anytype) !void;
    pub fn formatAllErrors(self: *const ErrorHandler, writer: anytype) !void;
    
    // Error recovery
    pub fn canRecover(self: *const ErrorHandler, error_info: LoweringError) bool;
    pub fn suggestRecovery(self: *const ErrorHandler, error_info: LoweringError) ?[]const u8;
    pub fn clearErrors(self: *ErrorHandler) void;
    pub fn clearWarnings(self: *ErrorHandler) void;
};

pub const ErrorContext = struct {
    span: lib.ast.SourceSpan,
    message: []const u8,
    suggestion: ?[]const u8,
    recovery_action: ?RecoveryAction,
    
    pub const RecoveryAction = enum {
        SkipNode,
        UseDefault,
        InsertPlaceholder,
        ContinueWithWarning,
    };
};
```

## Pass Management API

### PassManager

Integrates with MLIR's optimization infrastructure.

```zig
pub const PassManager = struct {
    mlir_pm: c.MlirPassManager,
    ctx: c.MlirContext,
    
    pub fn init(ctx: c.MlirContext) !PassManager;
    pub fn deinit(self: *PassManager) void;
    
    // Pass pipeline management
    pub fn addPass(self: *PassManager, pass_name: []const u8) !void;
    pub fn addPassPipeline(self: *PassManager, pipeline: []const []const u8) !void;
    pub fn runPasses(self: *PassManager, module: c.MlirModule) !PassResult;
    
    // Standard passes
    pub fn addCanonicalizationPass(self: *PassManager) !void;
    pub fn addCSEPass(self: *PassManager) !void;
    pub fn addInlinerPass(self: *PassManager) !void;
    pub fn addSCCPPass(self: *PassManager) !void;
    
    // Ora-specific passes
    pub fn addOraVerificationPass(self: *PassManager) !void;
    pub fn addMemoryRegionValidationPass(self: *PassManager) !void;
    pub fn addContractAnalysisPass(self: *PassManager) !void;
    pub fn addGasOptimizationPass(self: *PassManager) !void;
    
    // Pass configuration
    pub fn configurePassPipeline(self: *PassManager, config: PassPipelineConfig) !void;
    pub fn enableTiming(self: *PassManager) void;
    pub fn enableStatistics(self: *PassManager) void;
};

pub const PassResult = struct {
    success: bool,
    timing_info: ?PassTimingInfo,
    statistics: ?PassStatistics,
    diagnostics: []const PassDiagnostic,
    
    pub const PassTimingInfo = struct {
        total_time: f64,
        pass_times: std.HashMap([]const u8, f64, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),
    };
    
    pub const PassStatistics = struct {
        operations_processed: u64,
        transformations_applied: u64,
        optimizations_performed: u64,
    };
    
    pub const PassDiagnostic = struct {
        level: DiagnosticLevel,
        message: []const u8,
        location: ?c.MlirLocation,
        
        pub const DiagnosticLevel = enum {
            Note,
            Warning,
            Error,
        };
    };
};

pub const PassPipelineConfig = struct {
    optimization_level: OptimizationLevel,
    enable_verification: bool,
    enable_timing: bool,
    enable_statistics: bool,
    custom_passes: []const []const u8,
    
    pub const OptimizationLevel = enum {
        None,
        Basic,
        Aggressive,
        Size,
    };
};
```

## Usage Examples

### Basic Lowering

```zig
const std = @import("std");
const mlir = @import("mlir");

pub fn lowerContract(ast_nodes: []lib.AstNode, allocator: std.mem.Allocator) !void {
    // Create MLIR context
    const ctx = c.mlirContextCreate();
    defer c.mlirContextDestroy(ctx);
    
    // Lower AST to MLIR
    const result = try mlir.lowerFunctionsToModuleWithErrors(ctx, ast_nodes, allocator);
    defer result.deinit(allocator);
    
    // Check for errors
    if (result.hasErrors()) {
        for (result.errors) |err| {
            std.debug.print("Error: {s}\n", .{err.message});
            if (err.suggestion) |suggestion| {
                std.debug.print("Suggestion: {s}\n", .{suggestion});
            }
        }
        return;
    }
    
    // Use the MLIR module
    const module = result.module;
    // ... further processing
}
```

### Custom Type Mapping

```zig
pub fn customTypeMapping(allocator: std.mem.Allocator) !void {
    const ctx = c.mlirContextCreate();
    defer c.mlirContextDestroy(ctx);
    
    // Create type mapper with inference context
    var inference_ctx = TypeInference.InferenceContext.init(allocator);
    defer inference_ctx.deinit();
    
    const type_mapper = TypeMapper.initWithInference(ctx, &inference_ctx);
    
    // Add custom type alias
    try inference_ctx.addTypeAlias("Balance", 
        lib.ast.type_info.OraType{ .primitive = .U256 }, 
        &[_]TypeInference.TypeVariable{});
    
    // Map Ora type to MLIR
    const ora_type = lib.ast.type_info.TypeInfo{ 
        .category = .Primitive, 
        .ora_type = lib.ast.type_info.OraType{ .primitive = .U256 } 
    };
    const mlir_type = try type_mapper.toMlirType(ora_type);
    
    // Use the mapped type
    // ...
}
```

### Expression Lowering with Error Handling

```zig
pub fn lowerExpressionWithErrorHandling(expr: *const lib.ast.Expressions.ExprNode, allocator: std.mem.Allocator) !c.MlirValue {
    const ctx = c.mlirContextCreate();
    defer c.mlirContextDestroy(ctx);
    
    // Create error handler
    var error_handler = ErrorHandler.init(allocator);
    defer error_handler.deinit();
    
    // Set up lowering components
    const type_mapper = TypeMapper.init(ctx);
    const locations = LocationTracker.init(ctx, "example.ora");
    
    // Create MLIR module and block
    const loc = locations.createUnknownLocation();
    const module = c.mlirModuleCreateEmpty(loc);
    const body = c.mlirModuleGetBody(module);
    const block = c.mlirBlockCreate(0, null, null);
    c.mlirRegionAppendOwnedBlock(body, block);
    
    // Create expression lowerer
    const expr_lowerer = ExpressionLowerer.init(ctx, block, &type_mapper, null, null, null, locations);
    
    // Lower expression with error handling
    const result = expr_lowerer.lowerExpression(expr);
    
    // Check for errors
    if (error_handler.hasErrors()) {
        for (error_handler.getErrors()) |err| {
            std.debug.print("Expression lowering error: {s}\n", .{err.message});
        }
        return error.ExpressionLoweringFailed;
    }
    
    return result;
}
```

### Pass Pipeline Configuration

```zig
pub fn runOptimizationPasses(module: c.MlirModule, allocator: std.mem.Allocator) !PassResult {
    const ctx = c.mlirModuleGetContext(module);
    
    // Create pass manager
    var pass_manager = try PassManager.init(ctx);
    defer pass_manager.deinit();
    
    // Configure optimization pipeline
    const config = PassPipelineConfig{
        .optimization_level = .Aggressive,
        .enable_verification = true,
        .enable_timing = true,
        .enable_statistics = true,
        .custom_passes = &[_][]const u8{ "ora-contract-analysis", "ora-gas-optimization" },
    };
    
    try pass_manager.configurePassPipeline(config);
    
    // Add standard passes
    try pass_manager.addCanonicalizationPass();
    try pass_manager.addCSEPass();
    try pass_manager.addInlinerPass();
    
    // Add Ora-specific passes
    try pass_manager.addOraVerificationPass();
    try pass_manager.addMemoryRegionValidationPass();
    try pass_manager.addGasOptimizationPass();
    
    // Run passes
    const result = try pass_manager.runPasses(module);
    
    if (!result.success) {
        for (result.diagnostics) |diag| {
            std.debug.print("Pass diagnostic: {s}\n", .{diag.message});
        }
        return error.PassExecutionFailed;
    }
    
    return result;
}
```

## Error Handling Patterns

### Graceful Error Recovery

```zig
pub fn lowerWithRecovery(nodes: []lib.AstNode, allocator: std.mem.Allocator) !LoweringResult {
    var error_handler = ErrorHandler.init(allocator);
    defer error_handler.deinit();
    
    // Attempt lowering with error recovery
    for (nodes) |node| {
        switch (node) {
            .Function => |func| {
                lowerFunctionWithRecovery(func, &error_handler) catch |err| {
                    // Log error and continue with next node
                    error_handler.reportError(.MalformedExpression, 
                        "Failed to lower function", 
                        func.span, 
                        "Check function syntax and try again");
                    continue;
                };
            },
            // Handle other node types...
            else => {
                error_handler.reportWarning(.UnsupportedFeature, 
                    "Unsupported AST node type", 
                    null, 
                    "This feature is not yet implemented");
            },
        }
    }
    
    // Return result with accumulated errors
    return LoweringResult{
        .module = module,
        .errors = error_handler.getErrors(),
        .warnings = error_handler.getWarnings(),
        .success = !error_handler.hasErrors(),
        .pass_result = null,
    };
}
```

This API reference provides comprehensive documentation for all public interfaces in the MLIR lowering system, enabling developers to effectively use and extend the system.