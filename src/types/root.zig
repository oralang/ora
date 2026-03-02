//! Ora shared type definitions.
//!
//! Foundation types used across all compiler pipeline stages.
//! No dependency on AST nodes, semantics, or backends.

pub const source_span = @import("source_span.zig");
pub const region = @import("region.zig");
pub const type_info = @import("type_info.zig");
pub const ast_arena = @import("ast_arena.zig");

pub const SourceSpan = source_span.SourceSpan;
pub const MemoryRegion = region.MemoryRegion;
pub const TypeInfo = type_info.TypeInfo;
pub const TypeCategory = type_info.TypeCategory;
pub const OraType = type_info.OraType;
pub const TypeSource = type_info.TypeSource;
pub const MapType = type_info.MapType;
pub const FunctionType = type_info.FunctionType;
pub const ResultType = type_info.ResultType;
pub const AnonymousStructFieldType = type_info.AnonymousStructFieldType;
pub const CommonTypes = type_info.CommonTypes;
pub const deinitTypeInfo = type_info.deinitTypeInfo;
pub const deinitOraType = type_info.deinitOraType;
pub const AstArena = ast_arena.AstArena;
pub const AstArenaError = ast_arena.AstArenaError;
pub const MemoryStats = ast_arena.MemoryStats;
