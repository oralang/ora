const std = @import("std");
const c = @import("c.zig").c;
const lib = @import("ora_lib");
const constants = @import("constants.zig");

/// Type alias for array struct to match AST definition
const ArrayStruct = struct { elem: *const lib.ast.type_info.OraType, len: u64 };

/// Comprehensive type mapping system for converting Ora types to MLIR types
pub const TypeMapper = struct {
    ctx: c.MlirContext,

    pub fn init(ctx: c.MlirContext) TypeMapper {
        return .{ .ctx = ctx };
    }

    /// Convert any Ora type to its corresponding MLIR type
    pub fn toMlirType(self: *const TypeMapper, ora_type: anytype) c.MlirType {
        if (ora_type.ora_type) |ora_ty| {
            return switch (ora_ty) {
                // Unsigned integer types - map to appropriate bit widths
                .u8 => c.mlirIntegerTypeGet(self.ctx, 8),
                .u16 => c.mlirIntegerTypeGet(self.ctx, 16),
                .u32 => c.mlirIntegerTypeGet(self.ctx, 32),
                .u64 => c.mlirIntegerTypeGet(self.ctx, 64),
                .u128 => c.mlirIntegerTypeGet(self.ctx, 128),
                .u256 => c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS),

                // Signed integer types - map to appropriate bit widths
                .i8 => c.mlirIntegerTypeGet(self.ctx, 8),
                .i16 => c.mlirIntegerTypeGet(self.ctx, 16),
                .i32 => c.mlirIntegerTypeGet(self.ctx, 32),
                .i64 => c.mlirIntegerTypeGet(self.ctx, 64),
                .i128 => c.mlirIntegerTypeGet(self.ctx, 128),
                .i256 => c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS),

                // Other primitive types
                .bool => c.mlirIntegerTypeGet(self.ctx, 1),
                .address => c.mlirIntegerTypeGet(self.ctx, 160), // Ethereum address is 20 bytes (160 bits)
                .void => c.mlirNoneTypeGet(self.ctx),

                // Complex types - implement comprehensive mapping
                .string => self.mapStringType(ora_ty.string),
                .bytes => self.mapBytesType(ora_ty.bytes),
                .struct_type => self.mapStructType(ora_ty.struct_type),
                .enum_type => self.mapEnumType(ora_ty.enum_type),
                .contract_type => self.mapContractType(ora_ty.contract_type),
                .array => self.mapArrayType(ora_ty.array),
                .slice => self.mapSliceType(ora_ty.slice),
                .mapping => self.mapMappingType(ora_ty.mapping),
                .double_map => self.mapDoubleMapType(ora_ty.double_map),
                .tuple => self.mapTupleType(ora_ty.tuple),
                .function => self.mapFunctionType(ora_ty.function),
                .error_union => self.mapErrorUnionType(ora_ty.error_union),
                ._union => self.mapUnionType(ora_ty._union),
                .anonymous_struct => self.mapAnonymousStructType(ora_ty.anonymous_struct),
                .module => self.mapModuleType(ora_ty.module),
            };
        } else {
            // Default to i256 for unknown types
            return c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
        }
    }

    /// Convert primitive integer types with proper bit width
    pub fn mapIntegerType(self: *const TypeMapper, bit_width: u32, is_signed: bool) c.MlirType {
        _ = is_signed; // For now, we use the same bit width for signed/unsigned
        return c.mlirIntegerTypeGet(self.ctx, @intCast(bit_width));
    }

    /// Convert boolean type
    pub fn mapBoolType(self: *const TypeMapper) c.MlirType {
        return c.mlirIntegerTypeGet(self.ctx, 1);
    }

    /// Convert address type (Ethereum address)
    pub fn mapAddressType(self: *const TypeMapper) c.MlirType {
        return c.mlirIntegerTypeGet(self.ctx, 160);
    }

    /// Convert string type
    pub fn mapStringType(self: *const TypeMapper, string_info: anytype) c.MlirType {
        _ = string_info; // String length info
        // For now, use i256 as placeholder for string type
        // In the future, this could be a proper MLIR string type or pointer type
        return c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
    }

    /// Convert bytes type
    pub fn mapBytesType(self: *const TypeMapper, bytes_info: anytype) c.MlirType {
        _ = bytes_info; // Bytes length info
        // For now, use i256 as placeholder for bytes type
        // In the future, this could be a proper MLIR vector type or pointer type
        return c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
    }

    /// Convert void type
    pub fn mapVoidType(self: *const TypeMapper) c.MlirType {
        return c.mlirNoneTypeGet(self.ctx);
    }

    /// Convert struct type
    pub fn mapStructType(self: *const TypeMapper, struct_info: anytype) c.MlirType {
        _ = struct_info; // Struct field information
        // For now, use i256 as placeholder for struct type
        // In the future, this could be a proper MLIR struct type
        return c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
    }

    /// Convert enum type
    pub fn mapEnumType(self: *const TypeMapper, enum_info: anytype) c.MlirType {
        _ = enum_info; // Enum variant information
        // For now, use i256 as placeholder for enum type
        // In the future, this could be a proper MLIR integer type with appropriate width
        return c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
    }

    /// Convert contract type
    pub fn mapContractType(self: *const TypeMapper, contract_info: anytype) c.MlirType {
        _ = contract_info; // Contract information
        // For now, use i256 as placeholder for contract type
        // In the future, this could be a proper MLIR pointer type or custom type
        return c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
    }

    /// Convert array type
    pub fn mapArrayType(self: *const TypeMapper, array_info: anytype) c.MlirType {
        _ = array_info; // For now, use placeholder
        // For now, use i256 as placeholder for array type
        // In the future, this could be a proper MLIR array type or vector type
        return c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
    }

    /// Convert slice type
    pub fn mapSliceType(self: *const TypeMapper, slice_info: anytype) c.MlirType {
        _ = slice_info; // Slice element type information
        // For now, use i256 as placeholder for slice type
        // In the future, this could be a proper MLIR vector type or pointer type
        return c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
    }

    /// Convert mapping type
    pub fn mapMappingType(self: *const TypeMapper, mapping_info: lib.ast.type_info.MappingType) c.MlirType {
        _ = mapping_info; // Key and value type information
        // For now, use i256 as placeholder for mapping type
        // In the future, this could be a proper MLIR struct type or custom type
        return c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
    }

    /// Convert double mapping type
    pub fn mapDoubleMapType(self: *const TypeMapper, double_map_info: lib.ast.type_info.DoubleMapType) c.MlirType {
        _ = double_map_info; // Two keys and value type information
        // For now, use i256 as placeholder for double mapping type
        // In the future, this could be a proper MLIR struct type or custom type
        return c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
    }

    /// Convert tuple type
    pub fn mapTupleType(self: *const TypeMapper, tuple_info: anytype) c.MlirType {
        _ = tuple_info; // Tuple element types information
        // For now, use i256 as placeholder for tuple type
        // In the future, this could be a proper MLIR tuple type or struct type
        return c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
    }

    /// Convert function type
    pub fn mapFunctionType(self: *const TypeMapper, function_info: lib.ast.type_info.FunctionType) c.MlirType {
        _ = function_info; // Parameter and return type information
        // For now, use i256 as placeholder for function type
        // In the future, this could be a proper MLIR function type
        return c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
    }

    /// Convert error union type
    pub fn mapErrorUnionType(self: *const TypeMapper, error_union_info: anytype) c.MlirType {
        _ = error_union_info; // Error and success type information
        // For now, use i256 as placeholder for error union type
        // In the future, this could be a proper MLIR union type or custom type
        return c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
    }

    /// Convert union type
    pub fn mapUnionType(self: *const TypeMapper, union_info: anytype) c.MlirType {
        _ = union_info; // Union variant types information
        // For now, use i256 as placeholder for union type
        // In the future, this could be a proper MLIR union type or custom type
        return c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
    }

    /// Convert anonymous struct type
    pub fn mapAnonymousStructType(self: *const TypeMapper, fields: []const lib.ast.type_info.AnonymousStructFieldType) c.MlirType {
        _ = fields; // Anonymous struct field information
        // For now, use i256 as placeholder for anonymous struct type
        // In the future, this could be a proper MLIR struct type
        return c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
    }

    /// Convert module type
    pub fn mapModuleType(self: *const TypeMapper, module_info: anytype) c.MlirType {
        _ = module_info; // Module information
        // For now, use i256 as placeholder for module type
        // In the future, this could be a proper MLIR module type or custom type
        return c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
    }

    /// Get the bit width for an integer type
    pub fn getIntegerBitWidth(self: *const TypeMapper, ora_type: lib.ast.type_info.OraType) ?u32 {
        _ = self;
        return switch (ora_type) {
            .u8, .i8 => 8,
            .u16, .i16 => 16,
            .u32, .i32 => 32,
            .u64, .i64 => 64,
            .u128, .i128 => 128,
            .u256, .i256 => constants.DEFAULT_INTEGER_BITS,
            else => null,
        };
    }

    /// Check if a type is signed
    pub fn isSigned(self: *const TypeMapper, ora_type: lib.ast.type_info.OraType) bool {
        _ = self;
        return switch (ora_type) {
            .i8, .i16, .i32, .i64, .i128, .i256 => true,
            else => false,
        };
    }

    /// Check if a type is unsigned
    pub fn isUnsigned(self: *const TypeMapper, ora_type: lib.ast.type_info.OraType) bool {
        _ = self;
        return switch (ora_type) {
            .u8, .u16, .u32, .u64, .u128, .u256 => true,
            else => false,
        };
    }

    /// Check if a type is an integer
    pub fn isInteger(self: *const TypeMapper, ora_type: lib.ast.type_info.OraType) bool {
        _ = self;
        return switch (ora_type) {
            .u8, .u16, .u32, .u64, .u128, .u256, .i8, .i16, .i32, .i64, .i128, .i256 => true,
            else => false,
        };
    }

    /// Check if a type is a boolean
    pub fn isBoolean(self: *const TypeMapper, ora_type: lib.ast.type_info.OraType) bool {
        _ = self;
        return ora_type == .bool;
    }

    /// Check if a type is void
    pub fn isVoid(self: *const TypeMapper, ora_type: lib.ast.type_info.OraType) bool {
        _ = self;
        return ora_type == .void;
    }

    /// Check if a type is an address
    pub fn isAddress(self: *const TypeMapper, ora_type: lib.ast.type_info.OraType) bool {
        _ = self;
        return ora_type == .address;
    }

    /// Check if a type is a string
    pub fn isString(self: *const TypeMapper, ora_type: lib.ast.type_info.OraType) bool {
        _ = self;
        return ora_type == .string;
    }

    /// Check if a type is bytes
    pub fn isBytes(self: *const TypeMapper, ora_type: lib.ast.type_info.OraType) bool {
        _ = self;
        return ora_type == .bytes;
    }

    /// Check if a type is a complex type (struct, enum, contract, array, etc.)
    pub fn isComplex(self: *const TypeMapper, ora_type: lib.ast.type_info.OraType) bool {
        _ = self;
        return switch (ora_type) {
            .struct_type, .enum_type, .contract_type, .array, .slice, .mapping, .double_map, .tuple, .function, .error_union, ._union, .anonymous_struct, .module => true,
            else => false,
        };
    }
};
