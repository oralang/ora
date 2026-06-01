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
pub const lexer = @import("ora_lexer");
pub const ast_arena = @import("ora_types").ast_arena;
pub const abi = @import("abi.zig");
pub const abi_layout = @import("abi/layout.zig");
pub const abi_layout_context = @import("abi/layout_context.zig");
pub const abi_type_names = @import("abi/type_names.zig");
pub const abi_comptime_decoder = @import("abi/comptime_decoder.zig");
pub const abi_comptime_encoder = @import("abi/comptime_encoder.zig");
pub const abi_comptime_decoder_test_support = @import("abi/comptime_decoder_test_support.zig");
pub const lsp = @import("lsp/mod.zig");
pub const compiler = @import("compiler.zig");

// Note: MLIR and Z3 are NOT exported from ora_lib because they import ora_lib themselves,
// which would create circular dependencies. They should be imported directly by main.zig.

// Re-export key types for convenience
/// Lexical analyzer for Ora source code
pub const Lexer = lexer.Lexer;
/// Token representation
pub const Token = lexer.Token;
/// Token type enumeration
pub const TokenType = lexer.TokenType;

// AST Arena memory management
/// Arena-based allocator for AST nodes
pub const AstArena = ast_arena.AstArena;
pub const comptime_eval = @import("comptime/mod.zig");
