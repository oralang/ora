const std = @import("std");
const ast = @import("ast.zig");
const typer = @import("typer.zig");

/// Enhanced type utilities for Ora's explicit typing system
pub const TypeUtilities = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) TypeUtilities {
        return TypeUtilities{
            .allocator = allocator,
        };
    }

    /// Check if two types are exactly the same (no implicit conversions in Ora)
    pub fn isExactMatch(self: *TypeUtilities, type1: typer.OraType, type2: typer.OraType) bool {
        _ = self;
        return std.meta.eql(type1, type2);
    }

    /// Check if two TypeRef are exactly the same
    pub fn isTypeRefExactMatch(self: *TypeUtilities, type1: *const ast.TypeRef, type2: *const ast.TypeRef) bool {
        return switch (type1.*) {
            .Bool => type2.* == .Bool,
            .Address => type2.* == .Address,
            .U8 => type2.* == .U8,
            .U16 => type2.* == .U16,
            .U32 => type2.* == .U32,
            .U64 => type2.* == .U64,
            .U128 => type2.* == .U128,
            .U256 => type2.* == .U256,
            .I8 => type2.* == .I8,
            .I16 => type2.* == .I16,
            .I32 => type2.* == .I32,
            .I64 => type2.* == .I64,
            .I128 => type2.* == .I128,
            .I256 => type2.* == .I256,
            .String => type2.* == .String,
            .Bytes => type2.* == .Bytes,
            .Identifier => |name1| switch (type2.*) {
                .Identifier => |name2| std.mem.eql(u8, name1, name2),
                else => false,
            },
            .Slice => |elem1| switch (type2.*) {
                .Slice => |elem2| self.isTypeRefExactMatch(elem1, elem2),
                else => false,
            },
            .Mapping => |map1| switch (type2.*) {
                .Mapping => |map2| self.isTypeRefExactMatch(map1.key, map2.key) and self.isTypeRefExactMatch(map1.value, map2.value),
                else => false,
            },
            .DoubleMap => |dmap1| switch (type2.*) {
                .DoubleMap => |dmap2| self.isTypeRefExactMatch(dmap1.key1, dmap2.key1) and
                    self.isTypeRefExactMatch(dmap1.key2, dmap2.key2) and
                    self.isTypeRefExactMatch(dmap1.value, dmap2.value),
                else => false,
            },
            .Tuple => |tuple1| switch (type2.*) {
                .Tuple => |tuple2| blk: {
                    if (tuple1.types.len != tuple2.types.len) break :blk false;
                    for (tuple1.types, tuple2.types) |*t1, *t2| {
                        if (!self.isTypeRefExactMatch(t1, t2)) break :blk false;
                    }
                    break :blk true;
                },
                else => false,
            },
            .ErrorUnion => |eu1| switch (type2.*) {
                .ErrorUnion => |eu2| self.isTypeRefExactMatch(eu1.success_type, eu2.success_type),
                else => false,
            },
            .Result => |res1| switch (type2.*) {
                .Result => |res2| self.isTypeRefExactMatch(res1.ok_type, res2.ok_type) and
                    self.isTypeRefExactMatch(res1.error_type, res2.error_type),
                else => false,
            },
            .Unknown => type2.* == .Unknown,
            .Inferred => |inf1| switch (type2.*) {
                .Inferred => |inf2| self.isTypeRefExactMatch(&inf1.base_type, &inf2.base_type),
                else => false,
            },
        };
    }

    /// Get the size of a type in bytes
    pub fn getSize(self: *TypeUtilities, type_ref: *const ast.TypeRef) u32 {
        return switch (type_ref.*) {
            .Bool => 1,
            .Address => 20,
            .U8, .I8 => 1,
            .U16, .I16 => 2,
            .U32, .I32 => 4,
            .U64, .I64 => 8,
            .U128, .I128 => 16,
            .U256, .I256 => 32,
            .String, .Bytes => 32, // Dynamic size, stored as pointer
            .Slice => 32, // Dynamic size, stored as pointer
            .Mapping, .DoubleMap => 32, // Storage slot reference
            .Tuple => |tuple| blk: {
                var size: u32 = 0;
                for (tuple.types) |*elem_type| {
                    size += self.getSize(elem_type);
                }
                break :blk size;
            },
            .ErrorUnion => |error_union| self.getSize(error_union.success_type),
            .Result => |result| self.getSize(result.ok_type) + self.getSize(result.error_type),
            .Identifier => 32, // Default size for custom types
            .Unknown, .Inferred => 32, // Default size
        };
    }

    /// Check if a type is nullable (can hold null values)
    pub fn isNullable(self: *TypeUtilities, type_ref: *const ast.TypeRef) bool {
        _ = self;
        return switch (type_ref.*) {
            // Primitive types are not nullable in Ora
            .Bool, .Address, .U8, .U16, .U32, .U64, .U128, .U256, .I8, .I16, .I32, .I64, .I128, .I256, .String, .Bytes => false,

            // Complex types might be nullable depending on implementation
            .Slice, .Mapping, .DoubleMap => false, // Not nullable in Ora
            .Tuple => false, // Tuples are not nullable
            .ErrorUnion, .Result => false, // Error types handle absence differently
            .Identifier => false, // Custom types are not nullable by default
            .Unknown, .Inferred => false, // Unknown types assumed non-nullable
        };
    }

    /// Get the inner type for container types
    pub fn getInnerType(self: *TypeUtilities, type_ref: *const ast.TypeRef) ?*const ast.TypeRef {
        _ = self;
        return switch (type_ref.*) {
            .Slice => |element_type| element_type,
            .ErrorUnion => |error_union| error_union.success_type,
            .Result => |result| result.ok_type,
            .Inferred => |inferred| &inferred.base_type,
            else => null, // No inner type
        };
    }

    /// Check if a type is a primitive type
    pub fn isPrimitive(self: *TypeUtilities, type_ref: *const ast.TypeRef) bool {
        _ = self;
        return switch (type_ref.*) {
            .Bool, .Address, .U8, .U16, .U32, .U64, .U128, .U256, .I8, .I16, .I32, .I64, .I128, .I256, .String, .Bytes => true,
            else => false,
        };
    }

    /// Check if a type is numeric
    pub fn isNumeric(self: *TypeUtilities, type_ref: *const ast.TypeRef) bool {
        _ = self;
        return switch (type_ref.*) {
            .U8, .U16, .U32, .U64, .U128, .U256, .I8, .I16, .I32, .I64, .I128, .I256 => true,
            else => false,
        };
    }

    /// Check if a type is integer
    pub fn isInteger(self: *TypeUtilities, type_ref: *const ast.TypeRef) bool {
        return self.isNumeric(type_ref);
    }

    /// Check if a type is unsigned integer
    pub fn isUnsignedInteger(self: *TypeUtilities, type_ref: *const ast.TypeRef) bool {
        _ = self;
        return switch (type_ref.*) {
            .U8, .U16, .U32, .U64, .U128, .U256 => true,
            else => false,
        };
    }

    /// Check if a type is signed integer
    pub fn isSignedInteger(self: *TypeUtilities, type_ref: *const ast.TypeRef) bool {
        _ = self;
        return switch (type_ref.*) {
            .I8, .I16, .I32, .I64, .I128, .I256 => true,
            else => false,
        };
    }

    /// Check if a type is a container type (holds other types)
    pub fn isContainer(self: *TypeUtilities, type_ref: *const ast.TypeRef) bool {
        _ = self;
        return switch (type_ref.*) {
            .Slice, .Mapping, .DoubleMap, .Tuple, .ErrorUnion, .Result => true,
            else => false,
        };
    }

    /// Check if a type requires explicit annotation (cannot be inferred)
    pub fn requiresExplicitAnnotation(self: *TypeUtilities, type_ref: *const ast.TypeRef) bool {
        _ = self;
        return switch (type_ref.*) {
            // Integer literals require explicit type annotation in Ora
            .Unknown => true,
            .Inferred => true,
            // All other types are explicit when declared
            else => false,
        };
    }

    /// Validate that a type annotation is explicit and complete
    pub fn validateExplicitAnnotation(self: *TypeUtilities, type_ref: *const ast.TypeRef) !bool {
        return switch (type_ref.*) {
            .Unknown => false, // Must have explicit type
            .Inferred => false, // Inference not allowed in Ora
            .Slice => |element_type| self.validateExplicitAnnotation(element_type),
            .Mapping => |mapping| self.validateExplicitAnnotation(mapping.key) and self.validateExplicitAnnotation(mapping.value),
            .DoubleMap => |double_map| self.validateExplicitAnnotation(double_map.key1) and
                self.validateExplicitAnnotation(double_map.key2) and
                self.validateExplicitAnnotation(double_map.value),
            .Tuple => |tuple| blk: {
                for (tuple.types) |*elem_type| {
                    if (!try self.validateExplicitAnnotation(elem_type)) break :blk false;
                }
                break :blk true;
            },
            .ErrorUnion => |error_union| self.validateExplicitAnnotation(error_union.success_type),
            .Result => |result| self.validateExplicitAnnotation(result.ok_type) and self.validateExplicitAnnotation(result.error_type),
            else => true, // All other types are explicit
        };
    }

    /// Get the alignment requirement for a type
    pub fn getAlignment(self: *TypeUtilities, type_ref: *const ast.TypeRef) u32 {
        return switch (type_ref.*) {
            .Bool, .U8, .I8 => 1,
            .U16, .I16 => 2,
            .U32, .I32 => 4,
            .U64, .I64 => 8,
            .U128, .I128 => 16,
            .U256, .I256, .Address => 32,
            .String, .Bytes, .Slice, .Mapping, .DoubleMap => 32,
            .Tuple => |tuple| blk: {
                var max_align: u32 = 1;
                for (tuple.types) |*elem_type| {
                    const elem_align = self.getAlignment(elem_type);
                    if (elem_align > max_align) max_align = elem_align;
                }
                break :blk max_align;
            },
            .ErrorUnion => |error_union| self.getAlignment(error_union.success_type),
            .Result => |result| @max(self.getAlignment(result.ok_type), self.getAlignment(result.error_type)),
            .Identifier, .Unknown, .Inferred => 32, // Default alignment
        };
    }

    /// Check if a type can be used as a mapping key
    pub fn isValidMappingKey(self: *TypeUtilities, type_ref: *const ast.TypeRef) bool {
        _ = self;
        return switch (type_ref.*) {
            // Primitive types can be mapping keys
            .Bool, .Address, .U8, .U16, .U32, .U64, .U128, .U256, .I8, .I16, .I32, .I64, .I128, .I256, .String, .Bytes => true,

            // Complex types generally cannot be mapping keys
            .Slice, .Mapping, .DoubleMap, .Tuple, .ErrorUnion, .Result => false,

            // Custom types might be valid depending on implementation
            .Identifier => true, // Assume custom types can be keys if they implement required traits

            .Unknown, .Inferred => false, // Cannot determine validity
        };
    }

    /// Check if a type supports arithmetic operations
    pub fn supportsArithmetic(self: *TypeUtilities, type_ref: *const ast.TypeRef) bool {
        return self.isNumeric(type_ref);
    }

    /// Check if a type supports bitwise operations
    pub fn supportsBitwise(self: *TypeUtilities, type_ref: *const ast.TypeRef) bool {
        return self.isInteger(type_ref);
    }

    /// Check if a type supports comparison operations
    pub fn supportsComparison(self: *TypeUtilities, type_ref: *const ast.TypeRef) bool {
        _ = self;
        return switch (type_ref.*) {
            // Most primitive types support comparison
            .Bool, .Address, .U8, .U16, .U32, .U64, .U128, .U256, .I8, .I16, .I32, .I64, .I128, .I256, .String, .Bytes => true,

            // Complex types might support comparison depending on implementation
            .Slice, .Tuple => false, // Generally don't support comparison
            .Mapping, .DoubleMap => false, // Mappings don't support comparison
            .ErrorUnion, .Result => false, // Error types have special comparison semantics

            .Identifier => true, // Assume custom types can implement comparison
            .Unknown, .Inferred => false, // Cannot determine
        };
    }

    /// Get a human-readable string representation of a type
    pub fn toString(self: *TypeUtilities, type_ref: *const ast.TypeRef) ![]const u8 {
        return switch (type_ref.*) {
            .Bool => "bool",
            .Address => "address",
            .U8 => "u8",
            .U16 => "u16",
            .U32 => "u32",
            .U64 => "u64",
            .U128 => "u128",
            .U256 => "u256",
            .I8 => "i8",
            .I16 => "i16",
            .I32 => "i32",
            .I64 => "i64",
            .I128 => "i128",
            .I256 => "i256",
            .String => "string",
            .Bytes => "bytes",
            .Identifier => |name| name,
            .Slice => |element_type| try std.fmt.allocPrint(self.allocator, "[]const {s}", .{try self.toString(element_type)}),
            .Mapping => |mapping| try std.fmt.allocPrint(self.allocator, "mapping({s} => {s})", .{ try self.toString(mapping.key), try self.toString(mapping.value) }),
            .DoubleMap => |double_map| try std.fmt.allocPrint(self.allocator, "double_map({s}, {s} => {s})", .{ try self.toString(double_map.key1), try self.toString(double_map.key2), try self.toString(double_map.value) }),
            .Tuple => |tuple| blk: {
                var result = std.ArrayList(u8).init(self.allocator);
                try result.appendSlice("(");
                for (tuple.types, 0..) |*elem_type, i| {
                    if (i > 0) try result.appendSlice(", ");
                    try result.appendSlice(try self.toString(elem_type));
                }
                try result.appendSlice(")");
                break :blk result.toOwnedSlice();
            },
            .ErrorUnion => |error_union| try std.fmt.allocPrint(self.allocator, "!{s}", .{try self.toString(error_union.success_type)}),
            .Result => |result| try std.fmt.allocPrint(self.allocator, "Result({s}, {s})", .{ try self.toString(result.ok_type), try self.toString(result.error_type) }),
            .Unknown => "unknown",
            .Inferred => |inferred| try std.fmt.allocPrint(self.allocator, "inferred({s})", .{try self.toString(&inferred.base_type)}),
        };
    }

    /// Check if a type conversion is valid without explicit casting
    /// In Ora, this should always return false except for identical types
    pub fn isImplicitlyConvertible(self: *TypeUtilities, from: *const ast.TypeRef, to: *const ast.TypeRef) bool {
        // In Ora, no implicit conversions are allowed
        return self.isTypeRefExactMatch(from, to);
    }

    /// Check if an explicit cast is valid
    pub fn isExplicitlyConvertible(self: *TypeUtilities, from: *const ast.TypeRef, to: *const ast.TypeRef) bool {
        _ = self;
        // Most explicit casts are allowed in Ora, but some combinations don't make sense
        return switch (from.*) {
            .Unknown => false, // Cannot cast from unknown type
            else => switch (to.*) {
                .Unknown => false, // Cannot cast to unknown type
                else => true, // Most explicit casts are allowed
            },
        };
    }

    /// Get the default value representation for a type
    pub fn getDefaultValue(self: *TypeUtilities, type_ref: *const ast.TypeRef) ![]const u8 {
        _ = self;
        return switch (type_ref.*) {
            .Bool => "false",
            .Address => "0x0000000000000000000000000000000000000000",
            .U8, .U16, .U32, .U64, .U128, .U256, .I8, .I16, .I32, .I64, .I128, .I256 => "0",
            .String => "\"\"",
            .Bytes => "\"\"",
            .Slice => "[]",
            .Mapping => "{}", // Empty mapping
            .DoubleMap => "{}", // Empty double map
            .Tuple => "()", // Empty tuple
            .Identifier => "default", // Custom types have custom defaults
            .ErrorUnion, .Result, .Unknown, .Inferred => "undefined",
        };
    }
};
