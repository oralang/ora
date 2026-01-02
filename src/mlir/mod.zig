// ============================================================================
// MLIR Module - Public API
// ============================================================================
//
// MLIR lowering system for converting Ora AST to MLIR IR.
//
// ARCHITECTURE:
//   Modular design with specialized components for types, expressions,
//   statements, declarations, memory management, and code generation.
//
// ============================================================================

// Core MLIR functionality - consolidated into lower.zig
pub const c = @import("mlir_c_api").c;
pub const lower = @import("lower.zig");
pub const dialect = @import("dialect.zig");
pub const pass_manager = @import("pass_manager.zig");
pub const verification = @import("verification.zig");

// Helper utilities for reducing MLIR C API boilerplate
pub const helpers = @import("helpers.zig");

// Modular components
pub const types = @import("types.zig");
pub const expressions = @import("expressions.zig");
pub const statements = @import("statements.zig");
pub const declarations = @import("declarations.zig");
pub const memory = @import("memory.zig");
pub const refinement_guards = @import("refinement_guards.zig");

// Re-export commonly used types for convenience
pub const TypeMapper = types.TypeMapper;
pub const ExpressionLowerer = expressions.ExpressionLowerer;
pub const StatementLowerer = statements.StatementLowerer;
pub const DeclarationLowerer = declarations.DeclarationLowerer;
pub const MemoryManager = memory.MemoryManager;
pub const StorageMap = memory.StorageMap;

// Re-export consolidated types from lower.zig
pub const MlirContextHandle = lower.MlirContextHandle;
pub const createContext = lower.createContext;
pub const destroyContext = lower.destroyContext;
pub const LocationTracker = lower.LocationTracker;
pub const SymbolTable = lower.SymbolTable;
pub const ParamMap = lower.ParamMap;
pub const LocalVarMap = lower.LocalVarMap;
pub const SymbolInfo = lower.SymbolInfo;
pub const FunctionSymbol = lower.FunctionSymbol;
pub const TypeSymbol = lower.TypeSymbol;
