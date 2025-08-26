/// MLIR lowering system for the Ora compiler
/// This module provides a comprehensive system for converting Ora AST to MLIR IR

// Core MLIR functionality
pub const ctx = @import("context.zig");
pub const emit = @import("emit.zig");
pub const lower = @import("lower.zig");
pub const dialect = @import("dialect.zig");

// New modular components
pub const types = @import("types.zig");
pub const expressions = @import("expressions.zig");
pub const statements = @import("statements.zig");
pub const declarations = @import("declarations.zig");
pub const memory = @import("memory.zig");
pub const symbols = @import("symbols.zig");
pub const locations = @import("locations.zig");

// Re-export commonly used types for convenience
pub const TypeMapper = types.TypeMapper;
pub const ExpressionLowerer = expressions.ExpressionLowerer;
pub const StatementLowerer = statements.StatementLowerer;
pub const DeclarationLowerer = declarations.DeclarationLowerer;
pub const MemoryManager = memory.MemoryManager;
pub const StorageMap = memory.StorageMap;
pub const SymbolTable = symbols.SymbolTable;
pub const ParamMap = symbols.ParamMap;
pub const LocalVarMap = symbols.LocalVarMap;
pub const LocationTracker = locations.LocationTracker;
