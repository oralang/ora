const std = @import("std");

/// Source span to track position info in the source code
pub const SourceSpan = struct {
    // File identity for multi-file projects (0 = unknown/default)
    file_id: u32 = 0,
    // 1-based caret position
    line: u32,
    column: u32,
    // Byte length of the span
    length: u32,
    // Start byte offset within file (for precise mapping)
    byte_offset: u32 = 0,
    // Optional original slice
    lexeme: ?[]const u8 = null,
};

/// Type system matching the actual Ora specification from typer.zig
pub const TypeRef = union(enum) {
    // Primitive types
    Bool: void,
    Address: void,
    U8: void,
    U16: void,
    U32: void,
    U64: void,
    U128: void,
    U256: void,
    I8: void,
    I16: void,
    I32: void,
    I64: void,
    I128: void,
    I256: void,
    String: void,
    Bytes: void,

    // Complex types
    Slice: *const TypeRef,
    Mapping: *const MappingType,
    DoubleMap: *const DoubleMapType,
    Tuple: *const TupleType,

    // Custom types
    Struct: []const u8, // Reference to struct name
    Enum: []const u8, // Reference to enum name
    Identifier: []const u8, // User-defined type identifier
    Contract: []const u8, // Reference to contract name

    // Function type
    Function: *const FunctionType,

    // Error union type for error handling
    ErrorUnion: *const ErrorUnionType,

    // Special types
    Void: void,
    Unknown: void,
    Error: void,
    Module: ?[]const u8, // Module type with optional module name
};

/// Mapping type for key-value storage
pub const MappingType = struct {
    key: *const TypeRef,
    value: *const TypeRef,
};

/// Double mapping type for two-key storage
pub const DoubleMapType = struct {
    key1: *const TypeRef,
    key2: *const TypeRef,
    value: *const TypeRef,
};

/// Tuple type for multiple values
pub const TupleType = struct {
    types: []const TypeRef,
};

// Result type removed; prefer error unions '!T | E'

/// Error union type for error handling (!T)
pub const ErrorUnionType = struct {
    success_type: *const TypeRef,
};

/// Function type for function pointers/references
pub const FunctionType = struct {
    params: []TypeRef,
    return_type: ?*const TypeRef, // None for void functions
};

/// Free a type reference and its associated data
pub fn deinitTypeRef(allocator: std.mem.Allocator, type_ref: *TypeRef) void {
    switch (type_ref.*) {
        .Slice => |elem_type| {
            deinitTypeRef(allocator, @constCast(elem_type));
            allocator.destroy(elem_type);
        },
        .Mapping => |mapping| {
            deinitTypeRef(allocator, @constCast(mapping.key));
            deinitTypeRef(allocator, @constCast(mapping.value));
            allocator.destroy(mapping.key);
            allocator.destroy(mapping.value);
            allocator.destroy(mapping);
        },
        .DoubleMap => |doublemap| {
            deinitTypeRef(allocator, @constCast(doublemap.key1));
            deinitTypeRef(allocator, @constCast(doublemap.key2));
            deinitTypeRef(allocator, @constCast(doublemap.value));
            allocator.destroy(doublemap.key1);
            allocator.destroy(doublemap.key2);
            allocator.destroy(doublemap.value);
            allocator.destroy(doublemap);
        },
        .Tuple => |tuple| {
            for (tuple.types) |*element| {
                deinitTypeRef(allocator, @constCast(element));
            }
            allocator.free(tuple.types);
            allocator.destroy(tuple);
        },
        .Function => |func| {
            for (func.params) |*param| {
                deinitTypeRef(allocator, @constCast(param));
            }
            allocator.free(func.params);
            if (func.return_type) |ret_type| {
                deinitTypeRef(allocator, @constCast(ret_type));
                allocator.destroy(ret_type);
            }
            allocator.destroy(func);
        },
        .Result => |result| {
            deinitTypeRef(allocator, @constCast(result.ok_type));
            deinitTypeRef(allocator, @constCast(result.error_type));
            allocator.destroy(result.ok_type);
            allocator.destroy(result.error_type);
            allocator.destroy(result);
        },
        .ErrorUnion => |error_union| {
            deinitTypeRef(allocator, @constCast(error_union.success_type));
            allocator.destroy(error_union.success_type);
            allocator.destroy(error_union);
        },
        .Contract, .Struct, .Enum, .Identifier, .Bool, .Address, .U8, .U16, .U32, .U64, .U128, .U256, .I8, .I16, .I32, .I64, .I128, .I256, .String, .Bytes, .Void, .Unknown, .Error, .Module => {
            // Primitive types and named references don't need cleanup
        },
    }
}
