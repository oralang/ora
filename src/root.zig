//! Ora DSL Compiler Library - Core Components
//!
//! This is the main library module for the Ora language compiler frontend, providing
//! lexical analysis, parsing, AST construction, and semantic analysis.
//! Backend compilation to Yul is supported but not exposed as user commands.
//!
//! The Ora language is a domain-specific language for smart contract development
//! with formal verification capabilities and memory safety guarantees.

const std = @import("std");
const testing = std.testing;

// Export core compiler modules
pub const lexer = @import("lexer.zig");
pub const parser = @import("parser.zig");
pub const ast = @import("ast.zig");
pub const ast_visitor = @import("ast/ast_visitor.zig");
pub const ast_arena = @import("ast/ast_arena.zig");
pub const ast_builder = @import("ast/ast_builder.zig");
pub const ast_serializer = @import("ast/ast_serializer.zig");
pub const semantics = @import("semantics.zig");
pub const state_tracker = @import("analysis/state_tracker.zig");
pub const abi = @import("abi.zig");

// Note: MLIR and Z3 are NOT exported from ora_lib because they import ora_lib themselves,
// which would create circular dependencies. They should be imported directly by main.zig.

// Re-export key types for convenience
/// Lexical analyzer for Ora source code
pub const Lexer = lexer.Lexer;
/// Parser for generating AST from tokens
pub const Parser = parser.Parser;
/// Token representation
pub const Token = lexer.Token;
/// Token type enumeration
pub const TokenType = lexer.TokenType;

// AST types and functions
/// Abstract Syntax Tree node
pub const AstNode = ast.AstNode;
/// Contract declaration node
pub const ContractNode = ast.ContractNode;
/// Function declaration node
pub const FunctionNode = ast.FunctionNode;
/// Variable declaration node
pub const VariableDeclNode = ast.Statements.VariableDeclNode;
/// Expression node
pub const ExprNode = ast.Expressions.ExprNode;
/// Type reference
/// Memory region specification
pub const MemoryRegion = ast.Memory.Region;
/// Cleanup function for AST nodes
pub const deinitAstNodes = ast.deinitAstNodes;

// AST Arena memory management
/// Arena-based allocator for AST nodes
pub const AstArena = ast_arena.AstArena;
/// Memory statistics for AST arena
pub const MemoryStats = ast_arena.MemoryStats;
/// Error type for AST arena operations
pub const AstArenaError = ast_arena.AstArenaError;

// AST Serialization
/// Enhanced AST serializer with comprehensive customization options
pub const AstSerializer = ast_serializer.AstSerializer;
/// Serialization options for customizing output
pub const SerializationOptions = ast_serializer.SerializationOptions;

// Core analysis types
/// Type representation (unified AST type info)
pub const OraType = ast.type_info.OraType;
pub const TypeInfo = ast.type_info.TypeInfo;
