//! Ora shared type definitions.
//!
//! Foundation types used across all compiler pipeline stages.
//! No dependency on AST nodes, semantics, or backends.

pub const source_span = @import("source_span.zig");
pub const region = @import("region.zig");
pub const builtin = @import("builtin.zig");
pub const integer_constants = @import("integer_constants.zig");
pub const semantic = @import("semantic.zig");
pub const value = @import("value.zig");
pub const refinement_semantics = @import("refinement_semantics.zig");
pub const ast_arena = @import("ast_arena.zig");
pub const refinements = @import("ora_refinements");

pub const SourceSpan = source_span.SourceSpan;
pub const MemoryRegion = region.MemoryRegion;
pub const BuiltinTypeId = builtin.BuiltinTypeId;
pub const BuiltinTypeSpec = builtin.BuiltinTypeSpec;
pub const TypeCategory = builtin.TypeCategory;
pub const TypeKind = semantic.TypeKind;
pub const IntegerType = semantic.IntegerType;
pub const ComptimeIntegerType = semantic.ComptimeIntegerType;
pub const FixedBytesType = semantic.FixedBytesType;
pub const RefinementArg = semantic.RefinementArg;
pub const RefinementIntegerArg = semantic.RefinementIntegerArg;
pub const RefinementType = semantic.RefinementType;
pub const AnonymousStructField = semantic.AnonymousStructField;
pub const SemanticType = semantic.Type;
pub const LocatedType = semantic.LocatedType;
pub const ConstValue = value.ConstValue;
pub const AstArena = ast_arena.AstArena;
pub const AstArenaError = ast_arena.AstArenaError;
pub const MemoryStats = ast_arena.MemoryStats;
