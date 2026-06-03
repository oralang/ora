//! Ora shared type definitions.
//!
//! Foundation types used across all compiler pipeline stages.
//! No dependency on AST nodes, semantics, or backends.

pub const source_span = @import("source_span.zig");
pub const region = @import("region.zig");
pub const builtin = @import("builtin.zig");
pub const semantic = @import("semantic.zig");
pub const value = @import("value.zig");
pub const refinement_semantics = @import("refinement_semantics.zig");
pub const type_info = @import("type_info.zig");
pub const ast_arena = @import("ast_arena.zig");
pub const refinements = @import("ora_refinements");

pub const SourceSpan = source_span.SourceSpan;
pub const MemoryRegion = region.MemoryRegion;
pub const BuiltinTypeId = builtin.BuiltinTypeId;
pub const BuiltinTypeSpec = builtin.BuiltinTypeSpec;
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
pub const TypeInfo = type_info.TypeInfo;
pub const TypeCategory = type_info.TypeCategory;
pub const OraType = type_info.OraType;
pub const TypeSource = type_info.TypeSource;
pub const MapType = type_info.MapType;
pub const FunctionType = type_info.FunctionType;
pub const ResultType = type_info.ResultType;
pub const AnonymousStructFieldType = type_info.AnonymousStructFieldType;
pub const AstArena = ast_arena.AstArena;
pub const AstArenaError = ast_arena.AstArenaError;
pub const MemoryStats = ast_arena.MemoryStats;
