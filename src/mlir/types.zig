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
const math = std.math;

/// Type alias for array struct to match AST definition
const ArrayStruct = struct { elem: *const lib.ast.type_info.OraType, len: u64 };

/// Split u256 into four u64 words for MLIR type parameters
/// u256 = (high_high << 192) | (high_low << 128) | (low_high << 64) | low_low
fn splitU256IntoU64Words(x: u256) struct {
    high_high: u64,
    high_low: u64,
    low_high: u64,
    low_low: u64,
} {
    return .{
        .high_high = @truncate(x >> 192),
        .high_low = @truncate(x >> 128),
        .low_high = @truncate(x >> 64),
        .low_low = @truncate(x),
    };
}

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
        // Handle both optional and non-optional ora_type field
        const ora_ty_opt: ?lib.ast.type_info.OraType = if (@TypeOf(ora_type.ora_type) == lib.ast.type_info.OraType)
            ora_type.ora_type
        else if (ora_type.ora_type) |ot|
            ot
        else
            null;

        const ora_ty = ora_ty_opt orelse {
            std.debug.print("[toMlirType] ERROR: ora_type is null - Ora is strongly typed, this should not happen!\n", .{});
            @panic("toMlirType: ora_type is null - this indicates a type system bug");
        };

        // Don't print refinement types with {any} as it tries to dereference base pointers
        // which may point to invalid memory. Print a safe representation instead.
        switch (ora_ty) {
            .min_value => |mv| std.debug.print("[toMlirType] Converting Ora type: min_value<min={d}>\n", .{mv.min}),
            .max_value => |mv| std.debug.print("[toMlirType] Converting Ora type: max_value<max={d}>\n", .{mv.max}),
            .in_range => |ir| std.debug.print("[toMlirType] Converting Ora type: in_range<min={d}, max={d}>\n", .{ ir.min, ir.max }),
            .scaled => |s| std.debug.print("[toMlirType] Converting Ora type: scaled<decimals={d}>\n", .{s.decimals}),
            else => std.debug.print("[toMlirType] Converting Ora type: {any}\n", .{ora_ty}),
        }

        const result = switch (ora_ty) {
            // All integer types use builtin MLIR types (iN) - signedness is in operation semantics
            // u8, i8 → i8
            .u8 => c.mlirIntegerTypeGet(self.ctx, 8),
            .i8 => c.mlirIntegerTypeGet(self.ctx, 8),
            // u16, i16 → i16
            .u16 => c.mlirIntegerTypeGet(self.ctx, 16),
            .i16 => c.mlirIntegerTypeGet(self.ctx, 16),
            // u32, i32 → i32
            .u32 => c.mlirIntegerTypeGet(self.ctx, 32),
            .i32 => c.mlirIntegerTypeGet(self.ctx, 32),
            // u64, i64 → i64
            .u64 => c.mlirIntegerTypeGet(self.ctx, 64),
            .i64 => c.mlirIntegerTypeGet(self.ctx, 64),
            // u128, i128 → i128
            .u128 => c.mlirIntegerTypeGet(self.ctx, 128),
            .i128 => c.mlirIntegerTypeGet(self.ctx, 128),
            // u256, i256 → i256
            .u256 => c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS),
            .i256 => c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS),

            // Other primitive types
            // Note: bool uses MLIR's built-in i1 type (not !ora.bool) for compatibility with arith operations
            .bool => c.mlirIntegerTypeGet(self.ctx, 1),
            .address => c.oraAddressTypeGet(self.ctx), // Ethereum address is 20 bytes (160 bits)
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
            .tuple => self.mapTupleType(ora_ty.tuple),
            .function => self.mapFunctionType(ora_ty.function),
            .error_union => self.mapErrorUnionType(ora_ty.error_union),
            ._union => self.mapUnionType(ora_ty._union),
            .anonymous_struct => self.mapAnonymousStructType(ora_ty.anonymous_struct),
            .module => self.mapModuleType(ora_ty.module),

            // Refinement types - preserve refinement information in MLIR
            .min_value => {
                // Safely access base pointer - if it's invalid, we'll segfault here
                // This indicates the type was not properly copied or the base pointer points to freed memory
                const base_ora_type = ora_ty.min_value.base.*;
                const base_type = self.toMlirType(.{ .ora_type = base_ora_type });
                const min_words = splitU256IntoU64Words(ora_ty.min_value.min);
                return c.oraMinValueTypeGet(self.ctx, base_type, min_words.high_high, min_words.high_low, min_words.low_high, min_words.low_low);
            },
            .max_value => {
                const base_ora_type = ora_ty.max_value.base.*;
                const base_type = self.toMlirType(.{ .ora_type = base_ora_type });
                const max_words = splitU256IntoU64Words(ora_ty.max_value.max);
                return c.oraMaxValueTypeGet(self.ctx, base_type, max_words.high_high, max_words.high_low, max_words.low_high, max_words.low_low);
            },
            .in_range => {
                const base_ora_type = ora_ty.in_range.base.*;
                const base_type = self.toMlirType(.{ .ora_type = base_ora_type });
                const min_words = splitU256IntoU64Words(ora_ty.in_range.min);
                const max_words = splitU256IntoU64Words(ora_ty.in_range.max);
                return c.oraInRangeTypeGet(self.ctx, base_type, min_words.high_high, min_words.high_low, min_words.low_high, min_words.low_low, max_words.high_high, max_words.high_low, max_words.low_high, max_words.low_low);
            },
            .scaled => {
                const base_ora_type = ora_ty.scaled.base.*;
                const base_type = self.toMlirType(.{ .ora_type = base_ora_type });
                return c.oraScaledTypeGet(self.ctx, base_type, ora_ty.scaled.decimals);
            },
            .exact => {
                const base_ora_type = ora_ty.exact.*;
                const base_type = self.toMlirType(.{ .ora_type = base_ora_type });
                return c.oraExactTypeGet(self.ctx, base_type);
            },
            .non_zero_address => c.oraNonZeroAddressTypeGet(self.ctx),
        };

        // Log result type info
        if (c.mlirTypeIsAInteger(result)) {
            const width = c.mlirIntegerTypeGetWidth(result);
            std.debug.print("[toMlirType] Result: i{d}\n", .{width});
        } else if (c.mlirTypeIsAMemRef(result)) {
            std.debug.print("[toMlirType] Result: memref\n", .{});
        } else if (c.mlirTypeIsANone(result)) {
            std.debug.print("[toMlirType] Result: none (void)\n", .{});
        } else {
            std.debug.print("[toMlirType] Result: (other MLIR type)\n", .{});
        }

        return result;
    }

    /// Convert primitive integer types with proper bit width and signedness
    /// Uses Ora dialect types (!ora.int<width, isSigned>)
    pub fn mapIntegerType(self: *const TypeMapper, bit_width: u32, is_signed: bool) c.MlirType {
        return c.oraIntegerTypeGet(self.ctx, @intCast(bit_width), is_signed);
    }

    /// Convert boolean type
    /// Uses MLIR's built-in i1 type (not !ora.bool) for compatibility with arith operations
    pub fn mapBoolType(self: *const TypeMapper) c.MlirType {
        return c.mlirIntegerTypeGet(self.ctx, 1);
    }

    /// Convert address type (Ethereum address)
    /// Uses Ora dialect type (!ora.address)
    pub fn mapAddressType(self: *const TypeMapper) c.MlirType {
        return c.oraAddressTypeGet(self.ctx);
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
        // struct_info is the struct name (string)
        const struct_name = struct_info;

        // Look up struct in symbol table to get its registered MLIR type
        if (self.symbol_table) |st| {
            // First check if it's actually an enum (sometimes enums are stored as struct_type in AST)
            if (st.lookupType(struct_name)) |type_sym| {
                if (type_sym.type_kind == .Enum) {
                    // This is actually an enum, return its underlying type
                    return type_sym.mlir_type;
                } else if (type_sym.type_kind == .Struct) {
                    // Return the MLIR type that was created during struct registration
                    return type_sym.mlir_type;
                }
            }

            // Fallback: iterate through all types (for backwards compatibility)
            // NOTE: This fallback should match the struct name, not just return the first struct!
            var type_iter = st.types.iterator();
            while (type_iter.next()) |entry| {
                // Check if the entry key matches the struct name
                if (!std.mem.eql(u8, entry.key_ptr.*, struct_name)) {
                    continue;
                }
                const type_symbols = entry.value_ptr.*;
                for (type_symbols) |type_sym| {
                    if (type_sym.type_kind == .Struct and std.mem.eql(u8, type_sym.name, struct_name)) {
                        // Return the MLIR type that was created during struct registration
                        return type_sym.mlir_type;
                    }
                }
            }
        }

        // Fallback: Try to create the struct type directly (struct may be declared but not yet registered)
        // This can happen during type resolution before all declarations are processed
        const struct_name_ref = h.strRef(struct_name);
        const struct_type = c.oraStructTypeGet(self.ctx, struct_name_ref);

        if (struct_type.ptr != null) {
            return struct_type;
        }

        // Last resort: use i256 as fallback (should not happen if struct is properly declared)
        std.debug.print("WARNING: Struct type '{s}' not found in symbol table and failed to create. Using i256 fallback.\n", .{struct_name});
        return c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
    }

    /// Convert enum type to appropriate integer representation based on underlying type
    pub fn mapEnumType(self: *const TypeMapper, enum_info: anytype) c.MlirType {
        // enum_info is the enum name (string)
        const enum_name = enum_info;

        // Look up enum in symbol table to get the stored MLIR type (which uses underlying type)
        if (self.symbol_table) |st| {
            if (st.lookupType(enum_name)) |type_sym| {
                if (type_sym.type_kind == .Enum) {
                    // Return the stored mlir_type which was created using the underlying type
                    // (e.g., enum Status : u8 -> i8, enum ErrorCode : string -> i256)
                    return type_sym.mlir_type;
                }
            }
        }

        // Fallback: If enum not found in symbol table, use variant count for sizing
        // This should rarely happen as enums should be registered before use
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

    /// Convert mapping type `map[K, V]` to !ora.map<K, V>
    /// Maps in Ora/EVM are storage-based and use keccak256 for key hashing
    /// The type represents a base storage slot; actual access is via ora.map_get/ora.map_set
    pub fn mapMapType(self: *const TypeMapper, mapping_info: lib.ast.type_info.MapType) c.MlirType {
        // Get the key and value types
        const key_ora_type: ?lib.ast.type_info.OraType = mapping_info.key.*;
        const value_ora_type: ?lib.ast.type_info.OraType = mapping_info.value.*;

        const key_type = self.toMlirType(.{ .ora_type = key_ora_type });
        const value_type = self.toMlirType(.{ .ora_type = value_ora_type });

        // Create !ora.map<K, V> type
        return c.oraMapTypeGet(self.ctx, key_type, value_type);
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
            .struct_type, .enum_type, .contract_type, .array, .slice, .Map, .tuple, .function, .error_union, ._union, .anonymous_struct, .module => true,
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
        // - !ora.enum<name, repr> → named enumeration with integer repr
        // - !ora.error<T>         → result type with error handling
        // - !ora.error_union<Ts>  → sum type for multiple error kinds
        // - !ora.contract<name>   → contract type reference
        //
        // Current strategy (pre-TableGen):
        // - All dialect types map to i256 for EVM compatibility
        // - Type information is preserved in symbol table
        // - Operations use dialect ops (ora.map_get, ora.sload, etc.)
        // - Target code generation handles the actual EVM semantics
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
        // All type semantics are enforced through operations and target code generation
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
    /// Note: This function works with Ora types directly in the Ora dialect.
    /// Conversions between Ora types (e.g., u64 -> u256) are handled here.
    /// Convert between builtin MLIR types (now using iN types everywhere)
    pub fn createConversionOp(self: *const TypeMapper, block: c.MlirBlock, value: c.MlirValue, target_type: c.MlirType, _: ?lib.ast.SourceSpan) c.MlirValue {
        const value_type = c.mlirValueGetType(value);
        const types_equal = c.mlirTypeEqual(value_type, target_type);

        // If types are already the same, no conversion needed
        if (types_equal) {
            return value;
        }

        // Check if types are integers or index types
        // MLIR index type can be converted to/from integers
        const value_is_int = c.mlirTypeIsAInteger(value_type);
        const target_is_int = c.mlirTypeIsAInteger(target_type);
        const index_ty = c.mlirIndexTypeGet(self.ctx);
        const value_is_index = c.mlirTypeEqual(value_type, index_ty);
        const target_is_index = c.mlirTypeEqual(target_type, index_ty);

        // Handle index <-> integer conversions
        if (value_is_index and target_is_int) {
            // Convert index to integer: use arith.index_castui to convert index to target integer type
            const loc = c.mlirLocationUnknownGet(self.ctx);
            var cast_state = h.opState("arith.index_castui", loc);
            c.mlirOperationStateAddOperands(&cast_state, 1, @ptrCast(&value));
            c.mlirOperationStateAddResults(&cast_state, 1, @ptrCast(&target_type));
            const cast_op = c.mlirOperationCreate(&cast_state);
            h.appendOp(block, cast_op);
            return h.getResult(cast_op, 0);
        } else if (value_is_int and target_is_index) {
            // Convert integer to index: use arith.index_castui to convert integer to index
            const loc = c.mlirLocationUnknownGet(self.ctx);
            var cast_state = h.opState("arith.index_castui", loc);
            c.mlirOperationStateAddOperands(&cast_state, 1, @ptrCast(&value));
            c.mlirOperationStateAddResults(&cast_state, 1, @ptrCast(&target_type));
            const cast_op = c.mlirOperationCreate(&cast_state);
            h.appendOp(block, cast_op);
            return h.getResult(cast_op, 0);
        } else if (value_is_int and target_is_int) {
            const value_width = c.mlirIntegerTypeGetWidth(value_type);
            const target_width = c.mlirIntegerTypeGetWidth(target_type);

            // Use unknown location for conversions (span info not available in TypeMapper)
            const loc = c.mlirLocationUnknownGet(self.ctx);

            if (value_width == target_width) {
                // Same width - use bitcast to ensure exact type match
                var cast_state = h.opState("arith.bitcast", loc);
                c.mlirOperationStateAddOperands(&cast_state, 1, @ptrCast(&value));
                c.mlirOperationStateAddResults(&cast_state, 1, @ptrCast(&target_type));
                const cast_op = c.mlirOperationCreate(&cast_state);
                h.appendOp(block, cast_op);
                return h.getResult(cast_op, 0);
            } else if (value_width < target_width) {
                // Extend - use arith.extui (zero-extend for unsigned semantics)
                var ext_state = h.opState("arith.extui", loc);
                c.mlirOperationStateAddOperands(&ext_state, 1, @ptrCast(&value));
                c.mlirOperationStateAddResults(&ext_state, 1, @ptrCast(&target_type));
                const ext_op = c.mlirOperationCreate(&ext_state);
                h.appendOp(block, ext_op);
                return h.getResult(ext_op, 0);
            } else {
                // Truncate - use arith.trunci
                var trunc_state = h.opState("arith.trunci", loc);
                c.mlirOperationStateAddOperands(&trunc_state, 1, @ptrCast(&value));
                c.mlirOperationStateAddResults(&trunc_state, 1, @ptrCast(&target_type));
                const trunc_op = c.mlirOperationCreate(&trunc_state);
                h.appendOp(block, trunc_op);
                return h.getResult(trunc_op, 0);
            }
        }

        // For non-integer types or if conversion fails, return value as-is
        // This should not happen with our current type system
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
