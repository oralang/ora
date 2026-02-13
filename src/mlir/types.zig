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
const c = @import("mlir_c_api").c;
const lib = @import("ora_lib");
const constants = @import("lower.zig");
const h = @import("helpers.zig");
const ErrorHandler = @import("error_handling.zig").ErrorHandler;
const math = std.math;
const log = @import("log");

fn hashOraType(hasher: *std.hash.Wyhash, ora_type: lib.ast.type_info.OraType) void {
    const tag_val: u32 = @intFromEnum(std.meta.activeTag(ora_type));
    hasher.update(std.mem.asBytes(&tag_val));
    switch (ora_type) {
        .u8, .u16, .u32, .u64, .u128, .u256, .i8, .i16, .i32, .i64, .i128, .i256, .bool, .string, .address, .bytes, .void, .non_zero_address => {},
        .struct_type => |name| hasher.update(name),
        .bitfield_type => |name| hasher.update(name),
        .enum_type => |name| hasher.update(name),
        .contract_type => |name| hasher.update(name),
        .array => |arr| {
            hashOraType(hasher, arr.elem.*);
            hasher.update(std.mem.asBytes(&arr.len));
        },
        .slice => |elem| hashOraType(hasher, elem.*),
        .map => |m| {
            hashOraType(hasher, m.key.*);
            hashOraType(hasher, m.value.*);
        },
        .tuple => |types| {
            hasher.update(std.mem.asBytes(&types.len));
            for (types) |t| hashOraType(hasher, t);
        },
        .function => |fn_ty| {
            hasher.update(std.mem.asBytes(&fn_ty.params.len));
            for (fn_ty.params) |p| hashOraType(hasher, p);
            if (fn_ty.return_type) |ret| {
                hashOraType(hasher, ret.*);
            } else {
                hashOraType(hasher, .{ .void = {} });
            }
        },
        .error_union => |t| hashOraType(hasher, t.*),
        ._union => |types| {
            hasher.update(std.mem.asBytes(&types.len));
            for (types) |t| hashOraType(hasher, t);
        },
        .anonymous_struct => |fields| {
            hasher.update(std.mem.asBytes(&fields.len));
            for (fields) |field| {
                hasher.update(field.name);
                hashOraType(hasher, field.typ.*);
            }
        },
        .module => |name_opt| if (name_opt) |name| hasher.update(name),
        .min_value => |mv| {
            hashOraType(hasher, mv.base.*);
            hasher.update(std.mem.asBytes(&mv.min));
        },
        .max_value => |mv| {
            hashOraType(hasher, mv.base.*);
            hasher.update(std.mem.asBytes(&mv.max));
        },
        .in_range => |ir| {
            hashOraType(hasher, ir.base.*);
            hasher.update(std.mem.asBytes(&ir.min));
            hasher.update(std.mem.asBytes(&ir.max));
        },
        .scaled => |s| {
            hashOraType(hasher, s.base.*);
            hasher.update(std.mem.asBytes(&s.decimals));
        },
        .exact => |base| hashOraType(hasher, base.*),
    }
}

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
                // check constraints
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
            // basic compatibility check - can be extended for more complex rules
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
    error_handler: ?*ErrorHandler,
    anon_structs: std.StringHashMap([]const lib.ast.type_info.AnonymousStructFieldType),

    pub fn init(ctx: c.MlirContext, allocator: std.mem.Allocator) TypeMapper {
        return .{
            .ctx = ctx,
            .inference_ctx = TypeInference.InferenceContext.init(allocator),
            .symbol_table = null,
            .error_handler = null,
            .anon_structs = std.StringHashMap([]const lib.ast.type_info.AnonymousStructFieldType).init(allocator),
        };
    }

    pub fn initWithSymbolTable(ctx: c.MlirContext, allocator: std.mem.Allocator, symbol_table: *@import("lower.zig").SymbolTable) TypeMapper {
        return .{
            .ctx = ctx,
            .inference_ctx = TypeInference.InferenceContext.init(allocator),
            .symbol_table = symbol_table,
            .error_handler = null,
            .anon_structs = std.StringHashMap([]const lib.ast.type_info.AnonymousStructFieldType).init(allocator),
        };
    }

    pub fn deinit(self: *TypeMapper) void {
        self.inference_ctx.deinit();
        self.anon_structs.deinit();
    }

    pub fn setErrorHandler(self: *TypeMapper, error_handler: ?*ErrorHandler) void {
        self.error_handler = error_handler;
    }

    /// Convert any Ora type to its corresponding MLIR type
    /// Supports all primitive types (u8-u256, i8-i256, bool, address, string, bytes, void)
    pub fn toMlirType(self: *const TypeMapper, ora_type: anytype) c.MlirType {
        const span_opt: ?lib.ast.SourceSpan = if (@hasField(@TypeOf(ora_type), "span"))
            ora_type.span
        else
            null;

        // handle both optional and non-optional ora_type field
        const ora_ty_opt: ?lib.ast.type_info.OraType = if (@TypeOf(ora_type.ora_type) == lib.ast.type_info.OraType)
            ora_type.ora_type
        else if (ora_type.ora_type) |ot|
            ot
        else
            null;

        const ora_ty = ora_ty_opt orelse {
            if (self.error_handler) |handler| {
                handler.reportError(
                    .InternalError,
                    span_opt,
                    "Missing Ora type during MLIR lowering",
                    "Ensure type resolution runs before MLIR lowering.",
                ) catch {};
            } else {
                log.debug("[toMlirType] ERROR: ora_type is null - Ora is strongly typed, this should not happen!\n", .{});
            }
            return c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);
        };

        // don't print refinement types with {any} as it tries to dereference base pointers
        // which may point to invalid memory. Print a safe representation instead.
        switch (ora_ty) {
            .min_value => |mv| log.debug("[toMlirType] Converting Ora type: min_value<min={d}>\n", .{mv.min}),
            .max_value => |mv| log.debug("[toMlirType] Converting Ora type: max_value<max={d}>\n", .{mv.max}),
            .in_range => |ir| log.debug("[toMlirType] Converting Ora type: in_range<min={d}, max={d}>\n", .{ ir.min, ir.max }),
            .scaled => |s| log.debug("[toMlirType] Converting Ora type: scaled<decimals={d}>\n", .{s.decimals}),
            else => log.debug("[toMlirType] Converting Ora type: {any}\n", .{ora_ty}),
        }

        const result = switch (ora_ty) {
            // all integer types use builtin MLIR types (iN) - signedness is in operation semantics
            // u8, i8 → i8
            .u8 => c.oraIntegerTypeCreate(self.ctx, 8),
            .i8 => c.oraIntegerTypeCreate(self.ctx, 8),
            // u16, i16 → i16
            .u16 => c.oraIntegerTypeCreate(self.ctx, 16),
            .i16 => c.oraIntegerTypeCreate(self.ctx, 16),
            // u32, i32 → i32
            .u32 => c.oraIntegerTypeCreate(self.ctx, 32),
            .i32 => c.oraIntegerTypeCreate(self.ctx, 32),
            // u64, i64 → i64
            .u64 => c.oraIntegerTypeCreate(self.ctx, 64),
            .i64 => c.oraIntegerTypeCreate(self.ctx, 64),
            // u128, i128 → i128
            .u128 => c.oraIntegerTypeCreate(self.ctx, 128),
            .i128 => c.oraIntegerTypeCreate(self.ctx, 128),
            // u256, i256 → i256
            .u256 => c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS),
            .i256 => c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS),

            // other primitive types
            // note: bool uses MLIR's built-in i1 type (not !ora.bool) for compatibility with arith operations
            .bool => c.oraIntegerTypeCreate(self.ctx, 1),
            .address => c.oraAddressTypeGet(self.ctx), // Ethereum address is 20 bytes (160 bits)
            .string => self.mapStringType(),
            .bytes => self.mapBytesType(),
            .void => c.oraNoneTypeCreate(self.ctx),

            // complex types - comprehensive mapping
            .struct_type => self.mapStructType(ora_ty.struct_type),
            // bitfield maps to its base integer type (u256 by default)
            .bitfield_type => c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS),
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

            // refinement types - preserve refinement information in MLIR
            .min_value => {
                // safely access base pointer - if it's invalid, we'll segfault here
                // this indicates the type was not properly copied or the base pointer points to freed memory
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

        // log result type info
        if (c.oraTypeIsAInteger(result)) {
            const width = c.oraIntegerTypeGetWidth(result);
            log.debug("[toMlirType] Result: i{d}\n", .{width});
        } else if (c.oraTypeIsAMemRef(result)) {
            log.debug("[toMlirType] Result: memref\n", .{});
        } else if (c.oraTypeIsANone(result)) {
            log.debug("[toMlirType] Result: none (void)\n", .{});
        } else {
            log.debug("[toMlirType] Result: (other MLIR type)\n", .{});
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
        return c.oraIntegerTypeCreate(self.ctx, 1);
    }

    /// Convert address type (Ethereum address)
    /// Uses Ora dialect type (!ora.address)
    pub fn mapAddressType(self: *const TypeMapper) c.MlirType {
        return c.oraAddressTypeGet(self.ctx);
    }

    /// Convert string type
    pub fn mapStringType(self: *const TypeMapper) c.MlirType {
        return c.oraStringTypeGet(self.ctx);
    }

    /// Convert bytes type
    pub fn mapBytesType(self: *const TypeMapper) c.MlirType {
        return c.oraBytesTypeGet(self.ctx);
    }

    /// Convert void type
    pub fn mapVoidType(self: *const TypeMapper) c.MlirType {
        return c.oraNoneTypeCreate(self.ctx);
    }

    /// Convert struct type - uses symbol table to get field layout
    pub fn mapStructType(self: *const TypeMapper, struct_info: anytype) c.MlirType {
        // struct_info is the struct name (string)
        const struct_name = struct_info;

        // look up struct in symbol table to get its registered MLIR type
        if (self.symbol_table) |st| {
            // first check if it's actually an enum (sometimes enums are stored as struct_type in AST)
            if (st.lookupType(struct_name)) |type_sym| {
                if (type_sym.type_kind == .Enum) {
                    // this is actually an enum, return its underlying type
                    return type_sym.mlir_type;
                } else if (type_sym.type_kind == .Struct) {
                    // return the MLIR type that was created during struct registration
                    return type_sym.mlir_type;
                }
            }

            // fallback: iterate through all types (for backwards compatibility)
            // note: This fallback should match the struct name, not just return the first struct!
            var type_iter = st.types.iterator();
            while (type_iter.next()) |entry| {
                // check if the entry key matches the struct name
                if (!std.mem.eql(u8, entry.key_ptr.*, struct_name)) {
                    continue;
                }
                const type_symbols = entry.value_ptr.*;
                for (type_symbols) |type_sym| {
                    if (type_sym.type_kind == .Struct and std.mem.eql(u8, type_sym.name, struct_name)) {
                        // return the MLIR type that was created during struct registration
                        return type_sym.mlir_type;
                    }
                }
            }
        }

        // fallback: Try to create the struct type directly (struct may be declared but not yet registered)
        // this can happen during type resolution before all declarations are processed
        const struct_name_ref = h.strRef(struct_name);
        const struct_type = c.oraStructTypeGet(self.ctx, struct_name_ref);

        if (struct_type.ptr != null) {
            return struct_type;
        }

        // last resort: use i256 as fallback (should not happen if struct is properly declared)
        log.debug("WARNING: Struct type '{s}' not found in symbol table and failed to create. Using i256 fallback.\n", .{struct_name});
        return c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);
    }

    /// Convert enum type to appropriate integer representation based on underlying type
    pub fn mapEnumType(self: *const TypeMapper, enum_info: anytype) c.MlirType {
        // enum_info is the enum name (string)
        const enum_name = enum_info;

        // look up enum in symbol table to get the stored MLIR type (which uses underlying type)
        if (self.symbol_table) |st| {
            if (st.lookupType(enum_name)) |type_sym| {
                if (type_sym.type_kind == .Enum) {
                    // return the stored mlir_type which was created using the underlying type
                    // (e.g., enum Status : u8 -> i8, enum ErrorCode : string -> i256)
                    return type_sym.mlir_type;
                }
            }
        }

        // fallback: If enum not found in symbol table, use variant count for sizing
        // this should rarely happen as enums should be registered before use
        if (self.symbol_table) |st| {
            var type_iter = st.types.iterator();
            while (type_iter.next()) |entry| {
                const type_symbols = entry.value_ptr.*;
                for (type_symbols) |type_sym| {
                    if (type_sym.type_kind == .Enum) {
                        if (type_sym.variants) |variants| {
                            // choose smallest integer type that can hold all variants
                            const variant_count = variants.len;
                            if (variant_count <= 256) {
                                return c.oraIntegerTypeCreate(self.ctx, 8); // u8
                            } else if (variant_count <= 65536) {
                                return c.oraIntegerTypeCreate(self.ctx, 16); // u16
                            } else {
                                return c.oraIntegerTypeCreate(self.ctx, 32); // u32
                            }
                        }
                    }
                }
            }
        }

        // default to i32 for enum representation if not found in symbol table
        return c.oraIntegerTypeCreate(self.ctx, 32);
    }

    /// Convert contract type
    pub fn mapContractType(self: *const TypeMapper, contract_info: anytype) c.MlirType {
        _ = contract_info; // Contract information
        // for now, use i256 as placeholder for contract type
        // in the future, this could be a proper MLIR pointer type or custom type
        return c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);
    }

    /// Convert array type `[T; N]` to `memref<NxT, space>`
    pub fn mapArrayType(self: *const TypeMapper, array_info: anytype) c.MlirType {
        // get element type recursively - ora_type field must be optional
        const elem_ora_type: ?lib.ast.type_info.OraType = array_info.elem.*;
        const elem_mlir_type = self.toMlirType(.{ .ora_type = elem_ora_type });

        // create shaped type with dimensions [N]
        const shape: [1]i64 = .{@intCast(array_info.len)};

        // create unranked memref type for now (proper ranked memref requires more MLIR C API)
        // in a full implementation with memory space:
        // for now, use shaped tensor type which is simpler
        return h.rankedTensorType(self.ctx, 1, &shape[0], elem_mlir_type, h.nullAttr());
    }

    /// Convert slice type `slice[T]` to `!ora.slice<T>` or `memref<?xT, space>`
    pub fn mapSliceType(self: *const TypeMapper, slice_info: anytype) c.MlirType {
        // get element type recursively - ora_type field must be optional
        const elem_ora_type: ?lib.ast.type_info.OraType = slice_info.*;
        const elem_mlir_type = self.toMlirType(.{ .ora_type = elem_ora_type });

        // create dynamic shaped type with unknown dimension (?)
        const shape: [1]i64 = .{c.oraShapedTypeDynamicSize()};

        // use unranked tensor for dynamic slices
        // for proper slice: !ora.slice<T> would be better but requires dialect types
        // for now: tensor<?xT> represents a dynamically-sized array
        return h.rankedTensorType(self.ctx, 1, &shape[0], elem_mlir_type, h.nullAttr());
    }

    /// Convert mapping type `map<K, V>` to !ora.map<K, V>
    /// Maps in Ora/EVM are storage-based and use keccak256 for key hashing
    /// The type represents a base storage slot; actual access is via ora.map_get/ora.map_set
    pub fn mapMapType(self: *const TypeMapper, mapping_info: lib.ast.type_info.MapType) c.MlirType {
        // get the key and value types
        const key_ora_type: ?lib.ast.type_info.OraType = mapping_info.key.*;
        const value_ora_type: ?lib.ast.type_info.OraType = mapping_info.value.*;

        const key_type = self.toMlirType(.{ .ora_type = key_ora_type });
        const value_type = self.toMlirType(.{ .ora_type = value_ora_type });

        // create !ora.map<K, V> type
        return c.oraMapTypeGet(self.ctx, key_type, value_type);
    }

    /// Convert tuple type to aggregate representation
    /// Tuples are anonymous product types: (T1, T2, ..., Tn)
    /// In EVM, tuples are used for multiple return values and temporary groupings
    pub fn mapTupleType(self: *const TypeMapper, tuple_info: anytype) c.MlirType {
        if (tuple_info.len == 0) {
            return c.oraNoneTypeCreate(self.ctx);
        }

        const fields = self.inference_ctx.allocator.alloc(lib.ast.type_info.AnonymousStructFieldType, tuple_info.len) catch {
            log.warn("Failed to allocate tuple field types; using i256 fallback\n", .{});
            return c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);
        };

        var i: usize = 0;
        while (i < tuple_info.len) : (i += 1) {
            // Tuple fields use numeric names ("0", "1", ...) to align with t.0 syntax.
            const field_name = std.fmt.allocPrint(self.inference_ctx.allocator, "{d}", .{i}) catch {
                log.warn("Failed to allocate tuple field name; using i256 fallback\n", .{});
                return c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);
            };
            const elem_ptr = self.inference_ctx.allocator.create(lib.ast.type_info.OraType) catch {
                log.warn("Failed to allocate tuple field type; using i256 fallback\n", .{});
                return c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);
            };
            elem_ptr.* = tuple_info[i];
            fields[i] = .{ .name = field_name, .typ = elem_ptr };
        }

        return self.mapAnonymousStructType(fields);
    }

    /// Convert function type
    pub fn mapFunctionType(self: *const TypeMapper, function_info: lib.ast.type_info.FunctionType) c.MlirType {
        _ = function_info; // Parameter and return type information
        // for now, use i256 as placeholder for function type
        // in the future, this could be a proper MLIR function type
        return c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);
    }

    /// Convert error union type `!T1 | T2` to tagged union representation
    /// Error unions are sum types that can hold one of several error values
    /// Represented as: { tag: u8, value: largest_type }
    pub fn mapErrorUnionType(self: *const TypeMapper, error_union_info: anytype) c.MlirType {
        const success_type = self.toMlirType(.{ .ora_type = error_union_info.* });
        return c.oraErrorUnionTypeGet(self.ctx, success_type);
    }

    /// Convert error type `!T` to result type representation
    /// Error types wrap a success value T and indicate potential failure
    /// Similar to Result<T, Error> in Rust or Maybe in Haskell
    pub fn mapErrorType(self: *const TypeMapper, error_info: anytype) c.MlirType {
        // error types in EVM are represented as i256:
        // - Bit 0: success flag (0 = success, 1 = error)
        // - Bits [255:1]: value (success value) or error code
        //
        // this enables efficient error checking:
        // - if (result & 1) { /* handle error */ }
        // - else { /* use success value: result >> 1 */ }
        //
        // for EVM revert compatibility, error values can be directly used in revert()

        _ = error_info; // Success type tracked in symbol table

        // return i256 for EVM-compatible error representation
        // future: migrate to !ora.error<T> dialect type for better type safety
        return c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);
    }

    /// Convert union type
    pub fn mapUnionType(self: *const TypeMapper, union_info: anytype) c.MlirType {
        // Union variant types information.
        // If this is an error union (first member is error_union), map to !ora.error_union<T>.
        if (union_info.len > 0 and union_info[0] == .error_union) {
            return self.mapErrorUnionType(union_info[0].error_union);
        }

        // for now, use i256 as placeholder for union type
        // in the future, this could be a proper MLIR union type or custom type
        return c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);
    }

    /// Convert anonymous struct type
    pub fn mapAnonymousStructType(self: *const TypeMapper, fields: []const lib.ast.type_info.AnonymousStructFieldType) c.MlirType {
        var hasher = std.hash.Wyhash.init(0);
        hashOraType(&hasher, .{ .anonymous_struct = fields });
        const hash = hasher.final();
        const name = std.fmt.allocPrint(self.inference_ctx.allocator, "__anon_struct_{x}", .{hash}) catch {
            log.warn("Failed to allocate anonymous struct name; using i256 fallback\n", .{});
            return c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);
        };
        if (!self.anon_structs.contains(name)) {
            const owned_fields = self.copyAnonymousStructFields(fields) catch {
                log.warn("Failed to copy anonymous struct fields; using i256 fallback\n", .{});
                return c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);
            };
            _ = @constCast(self).anon_structs.put(name, owned_fields) catch {};
        }
        const struct_name_ref = h.strRef(name);
        const struct_type = c.oraStructTypeGet(self.ctx, struct_name_ref);
        if (struct_type.ptr != null) {
            return struct_type;
        }
        log.debug("WARNING: Anonymous struct type '{s}' could not be created. Using i256 fallback.\n", .{name});
        return c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);
    }

    fn copyAnonymousStructFields(
        self: *const TypeMapper,
        fields: []const lib.ast.type_info.AnonymousStructFieldType,
    ) ![]const lib.ast.type_info.AnonymousStructFieldType {
        const owned = try self.inference_ctx.allocator.alloc(lib.ast.type_info.AnonymousStructFieldType, fields.len);
        for (fields, 0..) |field, i| {
            const name_copy = try self.inference_ctx.allocator.dupe(u8, field.name);
            const typ_ptr = try self.inference_ctx.allocator.create(lib.ast.type_info.OraType);
            typ_ptr.* = field.typ.*;
            owned[i] = .{ .name = name_copy, .typ = typ_ptr };
        }
        return owned;
    }

    pub fn iterAnonymousStructs(self: *const TypeMapper) std.StringHashMap([]const lib.ast.type_info.AnonymousStructFieldType).Iterator {
        return self.anon_structs.iterator();
    }

    /// Convert module type
    pub fn mapModuleType(self: *const TypeMapper, module_info: anytype) c.MlirType {
        _ = module_info; // Module information
        // for now, use i256 as placeholder for module type
        // in the future, this could be a proper MLIR module type or custom type
        return c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);
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
            .struct_type, .bitfield_type, .enum_type, .contract_type, .array, .slice, .Map, .tuple, .function, .error_union, ._union, .anonymous_struct, .module => true,
            else => false,
        };
    }

    /// Create memref type with memory space for arrays `[T; N]` -> `memref<NxT, space>`
    pub fn createMemRefType(self: *const TypeMapper, element_type: c.MlirType, size: i64, memory_space: u32) c.MlirType {
        // create shaped type with dimensions [size]
        const shape: [1]i64 = .{size};

        // create memory space attribute for EVM regions:
        // 0 = memory (default)
        // 1 = storage (persistent)
        // 2 = tstore (transient storage)
        const space_attr = if (memory_space > 0)
            c.oraIntegerAttrCreateI64FromType(c.oraIntegerTypeCreate(self.ctx, 64), @intCast(memory_space))
        else
            c.oraNullAttrCreate();

        // use ranked tensor for now (memref requires layout attributes which are complex)
        // for proper memref: c.oraMemRefTypeCreate(element_type, 1, &shape, space_attr)
        return h.rankedTensorType(self.ctx, 1, &shape, element_type, space_attr);
    }

    /// Create Ora dialect type (foundation for custom dialect types)
    /// This function will create proper dialect types when TableGen is integrated
    /// For now, it returns EVM-compatible representations
    pub fn createOraDialectType(self: *const TypeMapper, type_name: []const u8, param_types: []const c.MlirType) c.MlirType {
        // ora dialect types (future implementation with TableGen):
        // - !ora.slice<T>         → dynamic array with length
        // - !ora.map<K, V>        → storage mapping with keccak256
        // - !ora.enum<name, repr> → named enumeration with integer repr
        // - !ora.error<T>         → result type with error handling
        // - !ora.error_union<Ts>  → sum type for multiple error kinds
        // - !ora.contract<name>   → contract type reference
        //
        // current strategy (pre-TableGen):
        // - All dialect types map to i256 for EVM compatibility
        // - Type information is preserved in symbol table
        // - Operations use dialect ops (ora.map_get, ora.sload, etc.)
        // - Target code generation handles the actual EVM semantics
        //
        // integration path:
        // 1. Define types in OraDialect.td using TableGen
        // 2. Generate C bindings via mlir-tblgen
        // 3. Link generated types in dialect.zig
        // 4. Update this function to call C bindings
        // 5. Remove i256 fallback

        _ = type_name; // Type name for dialect type creation
        _ = param_types; // Type parameters for parameterized types

        // return i256 for EVM compatibility until TableGen integration
        // all type semantics are enforced through operations and target code generation
        return c.oraIntegerTypeCreate(self.ctx, constants.DEFAULT_INTEGER_BITS);
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
        // generic instantiation process:
        // 1. Resolve type variables in generic_type using type_args
        // 2. Substitute concrete types for type parameters
        // 3. Generate MLIR type for the instantiated generic
        //
        // examples:
        // - Array<u256> → tensor<Nxi256>
        // - Map<address, u256> → i256 (storage slot)
        // - Result<bool, Error> → i256 (tagged union)
        //
        // for now, we apply type arguments through recursive type conversion
        // the generic base type is converted with its concrete type arguments

        if (type_args.len > 0) {
            // store type arguments in inference context for substitution
            for (type_args) |type_arg| {
                // convert each type argument to MLIR type
                // this ensures type arguments are validated and available
                _ = self.toMlirType(.{ .ora_type = type_arg });
            }
        }

        // convert the generic type with type arguments resolved
        // the base type conversion will use the resolved type arguments
        return self.toMlirType(.{ .ora_type = generic_type });
    }

    /// Check if a type conversion is valid
    pub fn isValidConversion(self: *const TypeMapper, from_type: lib.ast.type_info.OraType, to_type: lib.ast.type_info.OraType) bool {
        return self.inference_ctx.isTypeCompatible(from_type, to_type);
    }

    /// Get the most specific common type between two types
    pub fn getCommonType(self: *const TypeMapper, type1: lib.ast.type_info.OraType, type2: lib.ast.type_info.OraType) ?lib.ast.type_info.OraType {
        // if types are equal, return either one
        if (lib.ast.type_info.OraType.equals(type1, type2)) {
            return type1;
        }

        // handle integer type promotion
        if (type1.isInteger() and type2.isInteger()) {
            // both signed or both unsigned
            if ((type1.isSignedInteger() and type2.isSignedInteger()) or
                (type1.isUnsignedInteger() and type2.isUnsignedInteger()))
            {

                // get bit widths and return the larger type
                const width1 = self.getIntegerBitWidth(type1) orelse return null;
                const width2 = self.getIntegerBitWidth(type2) orelse return null;

                if (width1 >= width2) return type1;
                return type2;
            }

            // mixed signed/unsigned - promote to signed with larger width
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

        // no common type found
        return null;
    }

    /// Create a type conversion operation if needed
    /// Note: This function works with Ora types directly in the Ora dialect.
    /// Conversions between Ora types (e.g., u64 -> u256) are handled here.
    /// Convert between builtin MLIR types (now using iN types everywhere)
    pub fn createConversionOp(self: *const TypeMapper, block: c.MlirBlock, value: c.MlirValue, target_type: c.MlirType, _: ?lib.ast.SourceSpan) c.MlirValue {
        const value_type = c.oraValueGetType(value);
        const types_equal = c.oraTypeEqual(value_type, target_type);

        // if types are already the same, no conversion needed
        if (types_equal) {
            return value;
        }

        // refinement -> base conversion
        const refinement_base = c.oraRefinementTypeGetBaseType(value_type);
        if (refinement_base.ptr != null and c.oraTypeEqual(refinement_base, target_type)) {
            const loc = c.oraLocationUnknownGet(self.ctx);
            const convert_op = c.oraRefinementToBaseOpCreate(self.ctx, loc, value, block);
            if (convert_op.ptr != null) {
                return h.getResult(convert_op, 0);
            }
        }

        // base -> refinement conversion
        const target_ref_base = c.oraRefinementTypeGetBaseType(target_type);
        if (target_ref_base.ptr != null and c.oraTypeEqual(target_ref_base, value_type)) {
            const loc = c.oraLocationUnknownGet(self.ctx);
            const convert_op = c.oraBaseToRefinementOpCreate(self.ctx, loc, value, target_type, block);
            if (convert_op.ptr != null) {
                return h.getResult(convert_op, 0);
            }
        }

        // refinement -> refinement conversion
        if (refinement_base.ptr != null and target_ref_base.ptr != null) {
            const loc = c.oraLocationUnknownGet(self.ctx);
            const to_base_op = c.oraRefinementToBaseOpCreate(self.ctx, loc, value, block);
            if (to_base_op.ptr != null) {
                const base_val = h.getResult(to_base_op, 0);
                const base_converted = self.createConversionOp(block, base_val, target_ref_base, null);
                const to_ref_op = c.oraBaseToRefinementOpCreate(self.ctx, loc, base_converted, target_type, block);
                if (to_ref_op.ptr != null) {
                    return h.getResult(to_ref_op, 0);
                }
            }
        }

        // check if types are integers or index types
        // mlir index type can be converted to/from integers
        const value_is_int = c.oraTypeIsAInteger(value_type);
        const target_is_int = c.oraTypeIsAInteger(target_type);
        const index_ty = c.oraIndexTypeCreate(self.ctx);
        const value_is_index = c.oraTypeEqual(value_type, index_ty);
        const target_is_index = c.oraTypeEqual(target_type, index_ty);

        // handle index <-> integer conversions
        if (value_is_index and target_is_int) {
            // convert index to integer: use arith.index_castui to convert index to target integer type
            const loc = c.oraLocationUnknownGet(self.ctx);
            const cast_op = c.oraArithIndexCastUIOpCreate(self.ctx, loc, value, target_type);
            h.appendOp(block, cast_op);
            return h.getResult(cast_op, 0);
        } else if (value_is_int and target_is_index) {
            // convert integer to index: use arith.index_castui to convert integer to index
            const loc = c.oraLocationUnknownGet(self.ctx);
            const cast_op = c.oraArithIndexCastUIOpCreate(self.ctx, loc, value, target_type);
            h.appendOp(block, cast_op);
            return h.getResult(cast_op, 0);
        } else if (c.oraTypeIsAddressType(target_type) and value_is_int) {
            // convert i160 -> !ora.address using ora.i160.to.addr
            if (c.oraIntegerTypeGetWidth(value_type) == 160) {
                const loc = c.oraLocationUnknownGet(self.ctx);
                const addr_op = c.oraI160ToAddrOpCreate(self.ctx, loc, value);
                h.appendOp(block, addr_op);
                return h.getResult(addr_op, 0);
            }
        } else if (c.oraTypeIsAddressType(value_type) and c.oraTypeIsAddressType(target_type)) {
            // convert between address-like types (address <-> non_zero_address)
            const loc = c.oraLocationUnknownGet(self.ctx);
            const cast_op = c.oraUnrealizedConversionCastOpCreate(self.ctx, loc, value, target_type);
            h.appendOp(block, cast_op);
            return h.getResult(cast_op, 0);
        } else if (c.oraTypeIsAddressType(value_type) and target_is_int) {
            // convert !ora.address -> i160 using ora.addr.to.i160 when requested
            if (c.oraIntegerTypeGetWidth(target_type) == 160) {
                const loc = c.oraLocationUnknownGet(self.ctx);
                const addr_to_i160 = c.oraAddrToI160OpCreate(self.ctx, loc, value);
                h.appendOp(block, addr_to_i160);
                return h.getResult(addr_to_i160, 0);
            }
        } else if (value_is_int and target_is_int) {
            const value_width = c.oraIntegerTypeGetWidth(value_type);
            const target_width = c.oraIntegerTypeGetWidth(target_type);

            // use unknown location for conversions (span info not available in TypeMapper)
            const loc = c.oraLocationUnknownGet(self.ctx);

            if (value_width == target_width) {
                // same width - use bitcast to ensure exact type match
                const cast_op = c.oraArithBitcastOpCreate(self.ctx, loc, value, target_type);
                h.appendOp(block, cast_op);
                return h.getResult(cast_op, 0);
            } else if (value_width < target_width) {
                // extend - use arith.extui (zero-extend for unsigned semantics)
                const ext_op = c.oraArithExtUIOpCreate(self.ctx, loc, value, target_type);
                h.appendOp(block, ext_op);
                return h.getResult(ext_op, 0);
            } else {
                // truncate - use arith.trunci
                const trunc_op = c.oraArithTruncIOpCreate(self.ctx, loc, value, target_type);
                h.appendOp(block, trunc_op);
                return h.getResult(trunc_op, 0);
            }
        }

        // for non-integer types or if conversion fails, return value as-is
        // this should not happen with our current type system
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

        return c.oraIntegerAttrCreateI64FromType(c.oraIntegerTypeCreate(self.ctx, 64), space_value);
    }

    /// Create region attribute for attaching `ora.region` attributes
    pub fn createRegionAttribute(self: *const TypeMapper, region: []const u8) c.MlirAttribute {
        return h.stringAttr(self.ctx, region);
    }
};
