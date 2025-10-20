const std = @import("std");
const c = @import("c.zig").c;
const lib = @import("ora_lib");
const constants = @import("lower.zig");

/// Type alias for array struct to match AST definition
const ArrayStruct = struct { elem: *const lib.ast.type_info.OraType, len: u64 };

/// Advanced type system features for MLIR lowering
pub const TypeInference = struct {
    /// Type variable for generic type parameters
    pub const TypeVariable = struct {
        name: []const u8,
        constraints: []const lib.ast.type_info.OraType,
        resolved_type: ?lib.ast.type_info.OraType,
    };

    /// Type alias definition
    pub const TypeAlias = struct {
        name: []const u8,
        target_type: lib.ast.type_info.OraType,
        generic_params: []const TypeVariable,
    };

    /// Type inference context
    pub const InferenceContext = struct {
        type_variables: std.HashMap([]const u8, TypeVariable, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),
        type_aliases: std.HashMap([]const u8, TypeAlias, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator) InferenceContext {
            return .{
                .type_variables = std.HashMap([]const u8, TypeVariable, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
                .type_aliases = std.HashMap([]const u8, TypeAlias, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *InferenceContext) void {
            self.type_variables.deinit();
            self.type_aliases.deinit();
        }

        /// Add a type variable for generic type parameters
        pub fn addTypeVariable(self: *InferenceContext, name: []const u8, constraints: []const lib.ast.type_info.OraType) !void {
            const type_var = TypeVariable{
                .name = name,
                .constraints = constraints,
                .resolved_type = null,
            };
            try self.type_variables.put(name, type_var);
        }

        /// Resolve a type variable to a concrete type
        pub fn resolveTypeVariable(self: *InferenceContext, name: []const u8, concrete_type: lib.ast.type_info.OraType) !void {
            if (self.type_variables.getPtr(name)) |type_var| {
                // Check constraints
                for (type_var.constraints) |constraint| {
                    if (!self.isTypeCompatible(concrete_type, constraint)) {
                        return error.TypeConstraintViolation;
                    }
                }
                type_var.resolved_type = concrete_type;
            }
        }

        /// Add a type alias
        pub fn addTypeAlias(self: *InferenceContext, name: []const u8, target_type: lib.ast.type_info.OraType, generic_params: []const TypeVariable) !void {
            const alias = TypeAlias{
                .name = name,
                .target_type = target_type,
                .generic_params = generic_params,
            };
            try self.type_aliases.put(name, alias);
        }

        /// Resolve a type alias to its target type
        pub fn resolveTypeAlias(self: *InferenceContext, name: []const u8) ?lib.ast.type_info.OraType {
            if (self.type_aliases.get(name)) |alias| {
                return alias.target_type;
            }
            return null;
        }

        /// Check if two types are compatible for inference
        pub fn isTypeCompatible(self: *InferenceContext, type1: lib.ast.type_info.OraType, type2: lib.ast.type_info.OraType) bool {
            _ = self;
            // Basic compatibility check - can be extended for more complex rules
            return lib.ast.type_info.OraType.equals(type1, type2) or
                (type1.isInteger() and type2.isInteger()) or
                (type1.isUnsignedInteger() and type2.isUnsignedInteger()) or
                (type1.isSignedInteger() and type2.isSignedInteger());
        }

        /// Infer the type of an expression based on context
        pub fn inferExpressionType(self: *InferenceContext, expr_type: lib.ast.type_info.OraType, context_type: ?lib.ast.type_info.OraType) lib.ast.type_info.OraType {
            if (context_type) |ctx_type| {
                if (self.isTypeCompatible(expr_type, ctx_type)) {
                    return ctx_type; // Use context type if compatible
                }
            }
            return expr_type; // Fall back to expression's own type
        }
    };
};

/// Comprehensive type mapping system for converting Ora types to MLIR types
pub const TypeMapper = struct {
    ctx: c.MlirContext,
    inference_ctx: TypeInference.InferenceContext,
    symbol_table: ?*@import("lower.zig").SymbolTable,

    pub fn init(ctx: c.MlirContext, allocator: std.mem.Allocator) TypeMapper {
        return .{
            .ctx = ctx,
            .inference_ctx = TypeInference.InferenceContext.init(allocator),
            .symbol_table = null,
        };
    }

    pub fn initWithSymbolTable(ctx: c.MlirContext, allocator: std.mem.Allocator, symbol_table: *@import("lower.zig").SymbolTable) TypeMapper {
        return .{
            .ctx = ctx,
            .inference_ctx = TypeInference.InferenceContext.init(allocator),
            .symbol_table = symbol_table,
        };
    }

    pub fn deinit(self: *TypeMapper) void {
        self.inference_ctx.deinit();
    }

    /// Convert any Ora type to its corresponding MLIR type
    /// Supports all primitive types (u8-u256, i8-i256, bool, address, string, bytes, void)
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
                .string => self.mapStringType(),
                .bytes => self.mapBytesType(),
                .void => c.mlirNoneTypeGet(self.ctx),

                // Complex types - comprehensive mapping
                .struct_type => self.mapStructType(ora_ty.struct_type),
                .enum_type => self.mapEnumType(ora_ty.enum_type),
                .contract_type => self.mapContractType(ora_ty.contract_type),
                .array => self.mapArrayType(ora_ty.array),
                .slice => self.mapSliceType(ora_ty.slice),
                .map => self.mapMapType(ora_ty.map),
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

    /// Convert string type - maps to i256 for now (could be pointer type in future)
    pub fn mapStringType(self: *const TypeMapper) c.MlirType {
        // String types are represented as i256 for compatibility with EVM
        // In the future, this could be a proper MLIR string type or pointer type
        return c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
    }

    /// Convert bytes type - maps to i256 for now (could be vector type in future)
    pub fn mapBytesType(self: *const TypeMapper) c.MlirType {
        // Bytes types are represented as i256 for compatibility with EVM
        // In the future, this could be a proper MLIR vector type or pointer type
        return c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
    }

    /// Convert void type
    pub fn mapVoidType(self: *const TypeMapper) c.MlirType {
        return c.mlirNoneTypeGet(self.ctx);
    }

    /// Convert struct type to `!llvm.struct<...>`
    pub fn mapStructType(self: *const TypeMapper, struct_info: anytype) c.MlirType {
        // TODO: Implement proper struct type mapping to !llvm.struct<...>
        // For now, use i256 as placeholder until we can create LLVM struct types
        // In a full implementation, this would:
        // 1. Iterate through struct fields from struct_info
        // 2. Convert each field type recursively
        // 3. Create !llvm.struct<field1_type, field2_type, ...> type
        // 4. Eventually migrate to !ora.struct<fields> for better Ora semantics
        _ = struct_info;
        return c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
    }

    /// Convert enum type to `!ora.enum<name, repr>`
    pub fn mapEnumType(self: *const TypeMapper, enum_info: anytype) c.MlirType {
        // TODO: Implement proper enum type mapping to !ora.enum<name, repr> dialect type
        // For now, use the underlying integer representation
        // In a full implementation, this would:
        // 1. Get enum name from enum_info
        // 2. Determine underlying integer representation (i8, i16, i32, etc.)
        // 3. Create !ora.enum<name, repr = iN> dialect type
        // 4. For now, just return the underlying integer type
        _ = enum_info;
        // Default to i32 for enum representation
        return c.mlirIntegerTypeGet(self.ctx, 32);
    }

    /// Convert contract type
    pub fn mapContractType(self: *const TypeMapper, contract_info: anytype) c.MlirType {
        _ = contract_info; // Contract information
        // For now, use i256 as placeholder for contract type
        // In the future, this could be a proper MLIR pointer type or custom type
        return c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
    }

    /// Convert array type `[T; N]` to `memref<NxT, space>`
    pub fn mapArrayType(self: *const TypeMapper, array_info: anytype) c.MlirType {
        // TODO: Implement proper array type mapping to memref<NxT, space>
        // For now, use i256 as placeholder until we can access element type and length
        // In a full implementation, this would:
        // 1. Get element type from array_info.elem and convert it recursively
        // 2. Get array length from array_info.len
        // 3. Create memref type with appropriate memory space attribute
        _ = array_info;
        return c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
    }

    /// Convert slice type `slice[T]` to `!ora.slice<T>` or `memref<?xT, space>`
    pub fn mapSliceType(self: *const TypeMapper, slice_info: anytype) c.MlirType {
        // TODO: Implement proper slice type mapping to !ora.slice<T> dialect type
        // For now, use i256 as placeholder until we can create custom dialect types
        // In a full implementation, this would:
        // 1. Get element type from slice_info and convert it recursively
        // 2. Create !ora.slice<T> dialect type or memref<?xT, space> with dynamic shape
        // 3. Add appropriate ora.slice attributes
        _ = slice_info;
        return c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
    }

    /// Convert mapping type `map[K, V]` to `!ora.map<K, V>`
    pub fn mapMapType(self: *const TypeMapper, mapping_info: lib.ast.type_info.MapType) c.MlirType {
        // TODO: Implement proper mapping type to !ora.map<K, V> dialect type
        // For now, use i256 as placeholder until we can create custom dialect types
        // In a full implementation, this would:
        // 1. Get key type from mapping_info.key and convert it recursively
        // 2. Get value type from mapping_info.value and convert it recursively
        // 3. Create !ora.map<K, V> dialect type
        _ = mapping_info;
        return c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
    }

    /// Convert double mapping type `doublemap[K1, K2, V]` to `!ora.doublemap<K1, K2, V>`
    pub fn mapDoubleMapType(self: *const TypeMapper, double_map_info: lib.ast.type_info.DoubleMapType) c.MlirType {
        // TODO: Implement proper double mapping type to !ora.doublemap<K1, K2, V> dialect type
        // For now, use i256 as placeholder until we can create custom dialect types
        // In a full implementation, this would:
        // 1. Get first key type from double_map_info.key1 and convert it recursively
        // 2. Get second key type from double_map_info.key2 and convert it recursively
        // 3. Get value type from double_map_info.value and convert it recursively
        // 4. Create !ora.doublemap<K1, K2, V> dialect type
        _ = double_map_info;
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

    /// Convert error union type `!T1 | T2` to `!ora.error_union<T1, T2, ...>`
    pub fn mapErrorUnionType(self: *const TypeMapper, error_union_info: anytype) c.MlirType {
        // TODO: Implement proper error union type mapping to !ora.error_union<T1, T2, ...>
        // For now, use i256 as placeholder until we can create custom dialect types
        // In a full implementation, this would:
        // 1. Get all error types from error_union_info
        // 2. Convert each error type recursively
        // 3. Create !ora.error_union<T1, T2, ...> logical sum type
        _ = error_union_info;
        return c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
    }

    /// Convert error type `!T` to `!ora.error<T>`
    pub fn mapErrorType(self: *const TypeMapper, error_info: anytype) c.MlirType {
        // TODO: Implement proper error type mapping to !ora.error<T>
        // For now, use i256 as placeholder until we can create custom dialect types
        // In a full implementation, this would:
        // 1. Get the success type T from error_info
        // 2. Convert the success type recursively
        // 3. Create !ora.error<T> logical error capability type
        _ = error_info;
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
            .struct_type, .enum_type, .contract_type, .array, .slice, .Map, .double_map, .tuple, .function, .error_union, ._union, .anonymous_struct, .module => true,
            else => false,
        };
    }

    /// Create memref type with memory space for arrays `[T; N]` -> `memref<NxT, space>`
    pub fn createMemRefType(self: *const TypeMapper, element_type: c.MlirType, size: i64, memory_space: u32) c.MlirType {
        _ = self;
        // TODO: Implement proper memref type creation with memory space
        // For now, return the element type as placeholder
        // In a full implementation, this would:
        // 1. Create shaped type with dimensions [size]
        // 2. Set element type to element_type
        // 3. Add memory space attribute (0=memory, 1=storage, 2=tstore)
        _ = size;
        _ = memory_space;
        return element_type;
    }

    /// Create Ora dialect type (placeholder for future dialect implementation)
    pub fn createOraDialectType(self: *const TypeMapper, type_name: []const u8, param_types: []const c.MlirType) c.MlirType {
        // TODO: Implement Ora dialect type creation
        // For now, return i256 as placeholder
        // In a full implementation, this would create custom dialect types like:
        // - !ora.slice<T>
        // - !ora.map<K, V>
        // - !ora.doublemap<K1, K2, V>
        // - !ora.enum<name, repr>
        // - !ora.error<T>
        // - !ora.error_union<T1, T2, ...>
        _ = type_name;
        _ = param_types;
        return c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
    }

    /// Advanced type conversion with inference support
    pub fn convertTypeWithInference(self: *TypeMapper, ora_type: lib.ast.type_info.OraType, context_type: ?lib.ast.type_info.OraType) c.MlirType {
        const inferred_type = self.inference_ctx.inferExpressionType(ora_type, context_type);
        return self.toMlirType(.{ .ora_type = inferred_type });
    }

    /// Handle generic type instantiation
    pub fn instantiateGenericType(self: *TypeMapper, generic_type: lib.ast.type_info.OraType, type_args: []const lib.ast.type_info.OraType) !c.MlirType {
        // TODO: Implement generic type instantiation
        // For now, just convert the base type
        _ = type_args;
        return self.toMlirType(.{ .ora_type = generic_type });
    }

    /// Check if a type conversion is valid
    pub fn isValidConversion(self: *const TypeMapper, from_type: lib.ast.type_info.OraType, to_type: lib.ast.type_info.OraType) bool {
        return self.inference_ctx.isTypeCompatible(from_type, to_type);
    }

    /// Get the most specific common type between two types
    pub fn getCommonType(self: *const TypeMapper, type1: lib.ast.type_info.OraType, type2: lib.ast.type_info.OraType) ?lib.ast.type_info.OraType {
        // If types are equal, return either one
        if (lib.ast.type_info.OraType.equals(type1, type2)) {
            return type1;
        }

        // Handle integer type promotion
        if (type1.isInteger() and type2.isInteger()) {
            // Both signed or both unsigned
            if ((type1.isSignedInteger() and type2.isSignedInteger()) or
                (type1.isUnsignedInteger() and type2.isUnsignedInteger()))
            {

                // Get bit widths and return the larger type
                const width1 = self.getIntegerBitWidth(type1) orelse return null;
                const width2 = self.getIntegerBitWidth(type2) orelse return null;

                if (width1 >= width2) return type1;
                return type2;
            }

            // Mixed signed/unsigned - promote to signed with larger width
            const width1 = self.getIntegerBitWidth(type1) orelse return null;
            const width2 = self.getIntegerBitWidth(type2) orelse return null;
            const max_width = @max(width1, width2);

            return switch (max_width) {
                8 => lib.ast.type_info.OraType{ .i8 = {} },
                16 => lib.ast.type_info.OraType{ .i16 = {} },
                32 => lib.ast.type_info.OraType{ .i32 = {} },
                64 => lib.ast.type_info.OraType{ .i64 = {} },
                128 => lib.ast.type_info.OraType{ .i128 = {} },
                256 => lib.ast.type_info.OraType{ .i256 = {} },
                else => null,
            };
        }

        // No common type found
        return null;
    }

    /// Create a type conversion operation if needed
    pub fn createConversionOp(self: *const TypeMapper, block: c.MlirBlock, value: c.MlirValue, target_type: c.MlirType, span: ?lib.ast.SourceSpan) c.MlirValue {
        const value_type = c.mlirValueGetType(value);

        // If types are already the same, no conversion needed
        if (c.mlirTypeEqual(value_type, target_type)) {
            return value;
        }

        // Create location for the conversion operation
        const location = if (span) |s|
            c.mlirLocationFileLineColGet(self.ctx, c.mlirStringRefCreateFromCString(""), @intCast(s.start), @intCast(s.start))
        else
            c.mlirLocationUnknownGet(self.ctx);

        // For integer types, use arith.extui, arith.extsi, or arith.trunci
        if (c.mlirTypeIsAInteger(value_type) and c.mlirTypeIsAInteger(target_type)) {
            const value_width = c.mlirIntegerTypeGetWidth(value_type);
            const target_width = c.mlirIntegerTypeGetWidth(target_type);

            if (value_width < target_width) {
                // Extension - use unsigned extension for now
                const op_name = c.mlirStringRefCreateFromCString("arith.extui");
                const op_state = c.mlirOperationStateGet(op_name, location);
                c.mlirOperationStateAddOperands(&op_state, 1, &value);
                c.mlirOperationStateAddResults(&op_state, 1, &target_type);
                const op = c.mlirOperationCreate(&op_state);
                c.mlirBlockAppendOwnedOperation(block, op);
                return c.mlirOperationGetResult(op, 0);
            } else if (value_width > target_width) {
                // Truncation
                const op_name = c.mlirStringRefCreateFromCString("arith.trunci");
                const op_state = c.mlirOperationStateGet(op_name, location);
                c.mlirOperationStateAddOperands(&op_state, 1, &value);
                c.mlirOperationStateAddResults(&op_state, 1, &target_type);
                const op = c.mlirOperationCreate(&op_state);
                c.mlirBlockAppendOwnedOperation(block, op);
                return c.mlirOperationGetResult(op, 0);
            }
        }

        // For other types, return the original value for now
        // TODO: Implement more sophisticated type conversions
        return value;
    }

    /// Handle type alias resolution
    pub fn resolveTypeAlias(self: *TypeMapper, type_name: []const u8) ?lib.ast.type_info.OraType {
        return self.inference_ctx.resolveTypeAlias(type_name);
    }

    /// Resolve type name using symbol table
    pub fn resolveTypeName(self: *TypeMapper, type_name: []const u8) ?c.MlirType {
        if (self.symbol_table) |st| {
            if (st.lookupType(type_name)) |type_symbol| {
                return type_symbol.mlir_type;
            }
        }
        return null;
    }

    /// Add a type alias to the inference context
    pub fn addTypeAlias(self: *TypeMapper, name: []const u8, target_type: lib.ast.type_info.OraType) !void {
        try self.inference_ctx.addTypeAlias(name, target_type, &[_]TypeInference.TypeVariable{});
    }

    /// Handle complex type relationships and conversions
    pub fn handleComplexTypeRelationship(self: *const TypeMapper, type1: lib.ast.type_info.OraType, type2: lib.ast.type_info.OraType) TypeRelationship {
        if (lib.ast.type_info.OraType.equals(type1, type2)) {
            return .Identical;
        }

        if (self.isValidConversion(type1, type2)) {
            return .Convertible;
        }

        if (self.getCommonType(type1, type2) != null) {
            return .Compatible;
        }

        return .Incompatible;
    }

    /// Type relationship classification
    pub const TypeRelationship = enum {
        Identical, // Types are exactly the same
        Convertible, // One type can be converted to the other
        Compatible, // Types have a common supertype
        Incompatible, // Types cannot be used together
    };

    /// Get memory space attribute for different storage regions
    pub fn getMemorySpaceAttribute(self: *const TypeMapper, region: []const u8) c.MlirAttribute {
        const space_value: i64 = if (std.mem.eql(u8, region, "storage"))
            1
        else if (std.mem.eql(u8, region, "memory"))
            0
        else if (std.mem.eql(u8, region, "tstore"))
            2
        else
            0; // default to memory space

        return c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), space_value);
    }

    /// Create region attribute for attaching `ora.region` attributes
    pub fn createRegionAttribute(self: *const TypeMapper, region: []const u8) c.MlirAttribute {
        const region_ref = c.mlirStringRefCreate(region.ptr, region.len);
        return c.mlirStringAttrGet(self.ctx, region_ref);
    }
};
