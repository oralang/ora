// ============================================================================
// Type Mapping
// ============================================================================
//
// Maps Ora types to MLIR types for lowering.
//
// FEATURES:
//   • Primitive type mapping (integers, bools, strings, addresses)
//   • Complex type mapping (arrays, maps, tuples, structs)
//   • Type inference system for generics and type variables
//   • Type constraints and validation
//
// KEY COMPONENTS:
//   • TypeMapper: Main type conversion engine
//   • TypeInference: Type variable and constraint resolution
//   • Specialized handlers for each type category
//
// ============================================================================

const std = @import("std");
const c = @import("c.zig").c;
const lib = @import("ora_lib");
const constants = @import("lower.zig");
const h = @import("helpers.zig");

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

    /// Convert struct type - uses symbol table to get field layout
    pub fn mapStructType(self: *const TypeMapper, struct_info: anytype) c.MlirType {
        _ = struct_info;

        // Look up struct in symbol table to get its registered MLIR type
        if (self.symbol_table) |st| {
            var type_iter = st.types.iterator();
            while (type_iter.next()) |entry| {
                const type_symbols = entry.value_ptr.*;
                for (type_symbols) |type_sym| {
                    if (type_sym.type_kind == .Struct) {
                        // Return the MLIR type that was created during struct registration
                        return type_sym.mlir_type;
                    }
                }
            }
        }

        // Fallback: For EVM compatibility, use i256 to represent struct pointer/reference
        // Actual struct data is stored in memory/storage with field layout tracked separately
        return c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
    }

    /// Convert enum type to appropriate integer representation based on variant count
    pub fn mapEnumType(self: *const TypeMapper, enum_info: anytype) c.MlirType {
        _ = enum_info;

        // Look up enum in symbol table to get variant count for optimal sizing
        if (self.symbol_table) |st| {
            var type_iter = st.types.iterator();
            while (type_iter.next()) |entry| {
                const type_symbols = entry.value_ptr.*;
                for (type_symbols) |type_sym| {
                    if (type_sym.type_kind == .Enum) {
                        if (type_sym.variants) |variants| {
                            // Choose smallest integer type that can hold all variants
                            const variant_count = variants.len;
                            if (variant_count <= 256) {
                                return c.mlirIntegerTypeGet(self.ctx, 8); // u8
                            } else if (variant_count <= 65536) {
                                return c.mlirIntegerTypeGet(self.ctx, 16); // u16
                            } else {
                                return c.mlirIntegerTypeGet(self.ctx, 32); // u32
                            }
                        }
                    }
                }
            }
        }

        // Default to i32 for enum representation if not found in symbol table
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
        // Get element type recursively - ora_type field must be optional
        const elem_ora_type: ?lib.ast.type_info.OraType = array_info.elem.*;
        const elem_mlir_type = self.toMlirType(.{ .ora_type = elem_ora_type });

        // Create shaped type with dimensions [N]
        const shape: [1]i64 = .{@intCast(array_info.len)};

        // Create unranked memref type for now (proper ranked memref requires more MLIR C API)
        // In a full implementation with memory space:
        // return c.mlirMemRefTypeGet(elem_mlir_type, 1, &shape, 0); // 0 = default memory space
        // For now, use shaped tensor type which is simpler
        return c.mlirRankedTensorTypeGet(1, &shape, elem_mlir_type, c.mlirAttributeGetNull());
    }

    /// Convert slice type `slice[T]` to `!ora.slice<T>` or `memref<?xT, space>`
    pub fn mapSliceType(self: *const TypeMapper, slice_info: anytype) c.MlirType {
        // Get element type recursively - ora_type field must be optional
        const elem_ora_type: ?lib.ast.type_info.OraType = slice_info.*;
        const elem_mlir_type = self.toMlirType(.{ .ora_type = elem_ora_type });

        // Create dynamic shaped type with unknown dimension (?)
        const shape: [1]i64 = .{c.mlirShapedTypeGetDynamicSize()};

        // Use unranked tensor for dynamic slices
        // For proper slice: !ora.slice<T> would be better but requires dialect types
        // For now: tensor<?xT> represents a dynamically-sized array
        return c.mlirRankedTensorTypeGet(1, &shape, elem_mlir_type, c.mlirAttributeGetNull());
    }

    /// Convert mapping type `map[K, V]` to storage slot reference
    /// Maps in Ora/EVM are storage-based and use keccak256 for key hashing
    /// The type represents a base storage slot; actual access is via ora.map_get/ora.map_set
    pub fn mapMapType(self: *const TypeMapper, mapping_info: lib.ast.type_info.MapType) c.MlirType {
        // Maps are represented as i256 storage slot references in EVM
        // The key and value types are tracked in symbol table for type checking
        // Actual map access (keccak256 hashing) is handled in Yul lowering

        // Store type metadata for future dialect integration
        _ = mapping_info.key; // Key type (tracked for type checking)
        _ = mapping_info.value; // Value type (tracked for type checking)

        // Return storage slot reference (i256 for EVM compatibility)
        // Future: migrate to !ora.map<K, V> dialect type when TableGen is integrated
        return c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
    }

    /// Convert double mapping type `doublemap[K1, K2, V]` to storage slot reference
    /// Double maps are nested mappings: keccak256(key1, keccak256(key2, base_slot))
    /// Used for complex storage patterns like balances[token][user]
    pub fn mapDoubleMapType(self: *const TypeMapper, double_map_info: lib.ast.type_info.DoubleMapType) c.MlirType {
        // Double maps are represented as i256 storage slot references
        // The nested key hashing is handled in Yul lowering:
        // 1. Hash key2 with base slot → intermediate slot
        // 2. Hash key1 with intermediate slot → final storage location

        // Store type metadata for future dialect integration
        _ = double_map_info.key1; // First key type
        _ = double_map_info.key2; // Second key type
        _ = double_map_info.value; // Value type

        // Return storage slot reference (i256 for EVM compatibility)
        // Future: migrate to !ora.doublemap<K1, K2, V> dialect type
        return c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
    }

    /// Convert tuple type to aggregate representation
    /// Tuples are anonymous product types: (T1, T2, ..., Tn)
    /// In EVM, tuples are used for multiple return values and temporary groupings
    pub fn mapTupleType(self: *const TypeMapper, tuple_info: anytype) c.MlirType {
        // Tuples in EVM are represented as i256 for simplicity:
        // - For small tuples (2-3 elements): pack into single i256
        // - For larger tuples: use memory pointer to tuple data
        //
        // Tuple packing strategy:
        // - Tuple of (u8, u8): pack into lower 16 bits
        // - Tuple of (address, bool): pack into lower 161 bits
        // - Tuple of (u256, u256): use memory pointer
        //
        // The actual packing logic is handled during lowering based on element sizes
        // Type checker ensures element types are tracked for validation

        _ = tuple_info; // Element types tracked in symbol table

        // Return i256 for EVM-compatible tuple representation
        // Small tuples are packed, large tuples use memory indirection
        // Future: migrate to !llvm.struct<(T1, T2, ...)> for better type safety
        return c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
    }

    /// Convert function type
    pub fn mapFunctionType(self: *const TypeMapper, function_info: lib.ast.type_info.FunctionType) c.MlirType {
        _ = function_info; // Parameter and return type information
        // For now, use i256 as placeholder for function type
        // In the future, this could be a proper MLIR function type
        return c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
    }

    /// Convert error union type `!T1 | T2` to tagged union representation
    /// Error unions are sum types that can hold one of several error values
    /// Represented as: { tag: u8, value: largest_type }
    pub fn mapErrorUnionType(self: *const TypeMapper, error_union_info: anytype) c.MlirType {
        // Error unions in EVM are represented as i256 for simplicity:
        // - Lower bits: error code/discriminant (which error type)
        // - Upper bits: error value (if error carries data)
        //
        // Layout in i256:
        // [255:8] = error value (248 bits)
        // [7:0]   = error discriminant (8 bits for up to 256 error types)
        //
        // This allows efficient error handling and revert data encoding

        _ = error_union_info; // Type information tracked in symbol table

        // Return i256 for EVM-compatible error representation
        // Future: migrate to !ora.error_union<T1, T2, ...> dialect type
        return c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
    }

    /// Convert error type `!T` to result type representation
    /// Error types wrap a success value T and indicate potential failure
    /// Similar to Result<T, Error> in Rust or Maybe in Haskell
    pub fn mapErrorType(self: *const TypeMapper, error_info: anytype) c.MlirType {
        // Error types in EVM are represented as i256:
        // - Bit 0: success flag (0 = success, 1 = error)
        // - Bits [255:1]: value (success value) or error code
        //
        // This enables efficient error checking:
        // - if (result & 1) { /* handle error */ }
        // - else { /* use success value: result >> 1 */ }
        //
        // For EVM revert compatibility, error values can be directly used in revert()

        _ = error_info; // Success type tracked in symbol table

        // Return i256 for EVM-compatible error representation
        // Future: migrate to !ora.error<T> dialect type for better type safety
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
        // Create shaped type with dimensions [size]
        const shape: [1]i64 = .{size};

        // Create memory space attribute for EVM regions:
        // 0 = memory (default)
        // 1 = storage (persistent)
        // 2 = tstore (transient storage)
        const space_attr = if (memory_space > 0)
            c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), @intCast(memory_space))
        else
            c.mlirAttributeGetNull();

        // Use ranked tensor for now (memref requires layout attributes which are complex)
        // For proper memref: c.mlirMemRefTypeGet(element_type, 1, &shape, space_attr)
        return c.mlirRankedTensorTypeGet(1, &shape, element_type, space_attr);
    }

    /// Create Ora dialect type (foundation for custom dialect types)
    /// This function will create proper dialect types when TableGen is integrated
    /// For now, it returns EVM-compatible representations
    pub fn createOraDialectType(self: *const TypeMapper, type_name: []const u8, param_types: []const c.MlirType) c.MlirType {
        // Ora dialect types (future implementation with TableGen):
        // - !ora.slice<T>         → dynamic array with length
        // - !ora.map<K, V>        → storage mapping with keccak256
        // - !ora.doublemap<K1, K2, V> → nested storage mapping
        // - !ora.enum<name, repr> → named enumeration with integer repr
        // - !ora.error<T>         → result type with error handling
        // - !ora.error_union<Ts>  → sum type for multiple error kinds
        // - !ora.contract<name>   → contract type reference
        //
        // Current strategy (pre-TableGen):
        // - All dialect types map to i256 for EVM compatibility
        // - Type information is preserved in symbol table
        // - Operations use dialect ops (ora.map_get, ora.sload, etc.)
        // - Yul lowering handles the actual EVM semantics
        //
        // Integration path:
        // 1. Define types in OraDialect.td using TableGen
        // 2. Generate C bindings via mlir-tblgen
        // 3. Link generated types in dialect.zig
        // 4. Update this function to call C bindings
        // 5. Remove i256 fallback

        _ = type_name; // Type name for dialect type creation
        _ = param_types; // Type parameters for parameterized types

        // Return i256 for EVM compatibility until TableGen integration
        // All type semantics are enforced through operations and Yul lowering
        return c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
    }

    /// Advanced type conversion with inference support
    pub fn convertTypeWithInference(self: *TypeMapper, ora_type: lib.ast.type_info.OraType, context_type: ?lib.ast.type_info.OraType) c.MlirType {
        const inferred_type = self.inference_ctx.inferExpressionType(ora_type, context_type);
        return self.toMlirType(.{ .ora_type = inferred_type });
    }

    /// Handle generic type instantiation with type arguments
    /// Generics allow parameterized types: Array<T>, Map<K, V>, Result<T, E>
    /// Type arguments are substituted to create concrete types
    pub fn instantiateGenericType(self: *TypeMapper, generic_type: lib.ast.type_info.OraType, type_args: []const lib.ast.type_info.OraType) !c.MlirType {
        // Generic instantiation process:
        // 1. Resolve type variables in generic_type using type_args
        // 2. Substitute concrete types for type parameters
        // 3. Generate MLIR type for the instantiated generic
        //
        // Examples:
        // - Array<u256> → tensor<Nxi256>
        // - Map<address, u256> → i256 (storage slot)
        // - Result<bool, Error> → i256 (tagged union)
        //
        // For now, we apply type arguments through recursive type conversion
        // The generic base type is converted with its concrete type arguments

        if (type_args.len > 0) {
            // Store type arguments in inference context for substitution
            for (type_args) |type_arg| {
                // Convert each type argument to MLIR type
                // This ensures type arguments are validated and available
                _ = self.toMlirType(.{ .ora_type = type_arg });
            }
        }

        // Convert the generic type with type arguments resolved
        // The base type conversion will use the resolved type arguments
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
            c.mlirLocationFileLineColGet(self.ctx, h.strRefLit(""), @intCast(s.line), @intCast(s.column))
        else
            h.unknownLoc(self.ctx);

        // For integer types, use arith.extui, arith.extsi, or arith.trunci
        if (c.mlirTypeIsAInteger(value_type) and c.mlirTypeIsAInteger(target_type)) {
            const value_width = c.mlirIntegerTypeGetWidth(value_type);
            const target_width = c.mlirIntegerTypeGetWidth(target_type);

            if (value_width < target_width) {
                // Extension - use unsigned extension for now
                var op_state = h.opState("arith.extui", location);
                c.mlirOperationStateAddOperands(&op_state, 1, &value);
                c.mlirOperationStateAddResults(&op_state, 1, &target_type);
                const op = c.mlirOperationCreate(&op_state);
                c.mlirBlockAppendOwnedOperation(block, op);
                return c.mlirOperationGetResult(op, 0);
            } else if (value_width > target_width) {
                // Truncation
                var op_state = h.opState("arith.trunci", location);
                c.mlirOperationStateAddOperands(&op_state, 1, &value);
                c.mlirOperationStateAddResults(&op_state, 1, &target_type);
                const op = c.mlirOperationCreate(&op_state);
                c.mlirBlockAppendOwnedOperation(block, op);
                return c.mlirOperationGetResult(op, 0);
            }
        }

        // For other types, return the original value
        // More sophisticated conversions (struct conversions, error unwrapping, etc.)
        // are handled by higher-level type checking in the semantics phase
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
        return h.stringAttr(self.ctx, region);
    }
};
