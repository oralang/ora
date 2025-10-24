// ============================================================================
// Ora MLIR Dialect
// ============================================================================
//
// Defines and registers the Ora-specific MLIR dialect.
//
// FEATURES:
//   • Runtime dialect registration
//   • Operation creation (ora.constant, ora.storage_load, etc.)
//   • Type system integration
//   • Attribute handling
//
// ============================================================================

const std = @import("std");
const c = @import("c.zig").c;
const h = @import("helpers.zig");

/// Ora MLIR Dialect implementation
/// This provides runtime registration and operation creation for the Ora dialect
pub const OraDialect = struct {
    ctx: c.MlirContext,
    dialect_handle: ?c.MlirDialect,
    allocator: std.mem.Allocator,

    pub fn init(ctx: c.MlirContext, allocator: std.mem.Allocator) OraDialect {
        return OraDialect{
            .ctx = ctx,
            .dialect_handle = null,
            .allocator = allocator,
        };
    }

    /// Register the Ora dialect with MLIR context
    /// Currently using unregistered mode due to TableGen compatibility issues
    pub fn register(self: *OraDialect) !void {
        // For now, we use unregistered mode due to TableGen compatibility issues
        // This still produces clean MLIR output with ora.* operations
        // std.log.info("Ora dialect using unregistered mode (produces clean MLIR output)", .{});
        self.dialect_handle = null;
    }

    /// Check if the Ora dialect is properly registered
    pub fn isRegistered(self: *const OraDialect) bool {
        return self.dialect_handle != null;
    }

    /// Get the dialect namespace
    pub fn getNamespace() []const u8 {
        return "ora";
    }

    // Helper function to create ora.global operation
    pub fn createGlobal(
        self: *OraDialect,
        name: []const u8,
        value_type: c.MlirType,
        init_value: c.MlirAttribute,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // Use unregistered mode (produces clean MLIR output)
        return self.createGlobalUnregistered(name, value_type, init_value, loc);
    }

    // Create ora.global using unregistered approach (current implementation)
    fn createGlobalUnregistered(
        self: *OraDialect,
        name: []const u8,
        value_type: c.MlirType,
        init_value: c.MlirAttribute,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // Create a global variable declaration
        // This will be equivalent to ora.global @name : type = init_value
        var state = h.opState("ora.global", loc);

        // Add the global name as a symbol attribute
        const name_attr = h.stringAttr(self.ctx, name);
        var attrs = [_]c.MlirNamedAttribute{h.namedAttr(self.ctx, "sym_name", name_attr)};
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        // Add the type and initial value
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&value_type));

        // Add the initial value attribute
        var init_attrs = [_]c.MlirNamedAttribute{h.namedAttr(self.ctx, "init", init_value)};
        c.mlirOperationStateAddAttributes(&state, init_attrs.len, &init_attrs);

        const op = c.mlirOperationCreate(&state);
        return op;
    }

    // Helper function to create ora.sload operation
    pub fn createSLoad(
        self: *OraDialect,
        global_name: []const u8,
        result_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // Use unregistered mode (produces clean MLIR output)
        return self.createSLoadUnregistered(global_name, result_type, loc);
    }

    // Create ora.sload using unregistered approach (current implementation)
    fn createSLoadUnregistered(
        self: *OraDialect,
        global_name: []const u8,
        result_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        var state = h.opState("ora.sload", loc);

        // Add the global name attribute
        const global_attr = h.stringAttr(self.ctx, global_name);
        const global_id = h.identifier(self.ctx, "global");
        var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(global_id, global_attr)};
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        // Add result type
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_type));

        const op = c.mlirOperationCreate(&state);
        return op;
    }

    // Helper function to create ora.sstore operation
    pub fn createSStore(
        self: *OraDialect,
        value: c.MlirValue,
        global_name: []const u8,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // Use unregistered mode (produces clean MLIR output)
        return self.createSStoreUnregistered(value, global_name, loc);
    }

    // Create ora.sstore using unregistered approach (current implementation)
    fn createSStoreUnregistered(
        self: *OraDialect,
        value: c.MlirValue,
        global_name: []const u8,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        var state = h.opState("ora.sstore", loc);

        // Add operands
        var operands = [_]c.MlirValue{value};
        c.mlirOperationStateAddOperands(&state, operands.len, &operands);

        // Add the global name attribute
        const global_attr = h.stringAttr(self.ctx, global_name);
        const global_id = h.identifier(self.ctx, "global");
        var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(global_id, global_attr)};
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        const op = c.mlirOperationCreate(&state);
        return op;
    }

    // Helper function to create ora.contract operation (dual-mode)
    pub fn createContract(
        self: *OraDialect,
        name: []const u8,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        if (self.isRegistered()) {
            return self.createContractRegistered(name, loc);
        } else {
            std.log.warn("Registered ora.contract creation failed, falling back to unregistered", .{});
        }
        return self.createContractUnregistered(name, loc);
    }

    // Create ora.contract using registered dialect
    fn createContractRegistered(
        self: *OraDialect,
        name: []const u8,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // TODO: Once C++ dialect is linked, use:
        // const name_ref = c.mlirStringRefCreateFromCString(name.ptr);
        // const region = c.mlirRegionCreate();
        // return c.oraContractCreate(loc, name_ref, region);

        // For now, fallback to unregistered
        return self.createContractUnregistered(name, loc);
    }

    // Create ora.contract using unregistered approach (current implementation)
    fn createContractUnregistered(
        self: *OraDialect,
        name: []const u8,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        var state = h.opState("ora.contract", loc);

        // Add the contract name as a symbol attribute
        const name_attr = h.stringAttr(self.ctx, name);
        const name_id = h.identifier(self.ctx, "sym_name");
        var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(name_id, name_attr)};
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        // Add a single region for the contract body
        const region = c.mlirRegionCreate();
        c.mlirOperationStateAddOwnedRegions(&state, 1, @ptrCast(&region));

        const op = c.mlirOperationCreate(&state);
        return op;
    }

    /// Create ora.cast operation for type conversions
    pub fn createCast(
        self: *OraDialect,
        input: c.MlirValue,
        result_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        _ = self; // Mark as used
        var state = h.opState("ora.cast", loc);

        // Add operands
        var operands = [_]c.MlirValue{input};
        c.mlirOperationStateAddOperands(&state, operands.len, &operands);

        // Add result type
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_type));

        const op = c.mlirOperationCreate(&state);
        return op;
    }

    /// Create ora.requires operation for preconditions
    pub fn createRequires(
        _: *OraDialect,
        condition: c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        var state = h.opState("ora.requires", loc);

        // Add operands
        var operands = [_]c.MlirValue{condition};
        c.mlirOperationStateAddOperands(&state, operands.len, &operands);

        const op = c.mlirOperationCreate(&state);
        return op;
    }

    /// Create ora.ensures operation for postconditions
    pub fn createEnsures(
        _: *OraDialect,
        condition: c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        var state = h.opState("ora.ensures", loc);

        // Add operands
        var operands = [_]c.MlirValue{condition};
        c.mlirOperationStateAddOperands(&state, operands.len, &operands);

        const op = c.mlirOperationCreate(&state);
        return op;
    }

    /// Create ora.invariant operation
    pub fn createInvariant(
        _: *OraDialect,
        condition: c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        var state = h.opState("ora.invariant", loc);

        // Add operands
        var operands = [_]c.MlirValue{condition};
        c.mlirOperationStateAddOperands(&state, operands.len, &operands);

        const op = c.mlirOperationCreate(&state);
        return op;
    }

    /// Create ora.old operation for old value references
    pub fn createOld(
        _: *OraDialect,
        value: c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        var state = h.opState("ora.old", loc);

        // Add operands
        var operands = [_]c.MlirValue{value};
        c.mlirOperationStateAddOperands(&state, operands.len, &operands);

        // Result type is same as input type
        const input_type = c.mlirValueGetType(value);
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&input_type));

        const op = c.mlirOperationCreate(&state);
        return op;
    }

    //===----------------------------------------------------------------------===//
    // Phase 1 Operations - Constants
    //===----------------------------------------------------------------------===//

    /// Create ora.string.constant operation
    pub fn createStringConstant(
        self: *OraDialect,
        value: []const u8,
        result_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        var state = h.opState("ora.string.constant", loc);

        // Add the string value attribute
        const value_attr = h.stringAttr(self.ctx, value);
        const value_id = h.identifier(self.ctx, "value");
        var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(value_id, value_attr)};
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        // Add result type
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_type));

        const op = c.mlirOperationCreate(&state);
        return op;
    }

    /// Create ora.power operation
    pub fn createPower(
        _: *OraDialect,
        base: c.MlirValue,
        exponent: c.MlirValue,
        result_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        var state = h.opState("ora.power", loc);

        // Add operands
        var operands = [_]c.MlirValue{ base, exponent };
        c.mlirOperationStateAddOperands(&state, operands.len, &operands);

        // Add result type
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_type));

        const op = c.mlirOperationCreate(&state);
        return op;
    }

    /// Create ora.yield operation
    pub fn createYield(
        self: *OraDialect,
        operands: []c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        _ = self; // Mark as used
        var state = h.opState("ora.yield", loc);

        // Add operands
        if (operands.len > 0) {
            c.mlirOperationStateAddOperands(&state, operands.len, operands.ptr);
        }

        const op = c.mlirOperationCreate(&state);
        return op;
    }

    //===----------------------------------------------------------------------===//
    // Destructuring Operations
    //===----------------------------------------------------------------------===//

    /// Create ora.destructure operation
    pub fn createDestructure(
        self: *OraDialect,
        value: c.MlirValue,
        pattern_type: []const u8,
        result_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        var state = h.opState("ora.destructure", loc);

        // Add operands
        var operands = [_]c.MlirValue{value};
        c.mlirOperationStateAddOperands(&state, operands.len, &operands);

        // Add pattern type attribute
        const pattern_attr = h.stringAttr(self.ctx, pattern_type);
        const pattern_id = h.identifier(self.ctx, "pattern_type");
        var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(pattern_id, pattern_attr)};
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        // Add result type
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_type));

        const op = c.mlirOperationCreate(&state);
        return op;
    }

    //===----------------------------------------------------------------------===//
    // Enum Operations
    //===----------------------------------------------------------------------===//

    /// Create ora.enum.decl operation
    pub fn createEnumDecl(
        self: *OraDialect,
        name: []const u8,
        repr_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        var state = h.opState("ora.enum.decl", loc);

        // Add enum name attribute
        const name_attr = h.stringAttr(self.ctx, name);
        const name_id = h.identifier(self.ctx, "name");
        var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(name_id, name_attr)};
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        // Add representation type attribute
        const type_attr = c.mlirTypeAttrGet(repr_type);
        const type_id = h.identifier(self.ctx, "repr_type");
        var type_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(type_id, type_attr)};
        c.mlirOperationStateAddAttributes(&state, type_attrs.len, &type_attrs);

        // Add variants region
        const region = c.mlirRegionCreate();
        c.mlirOperationStateAddOwnedRegions(&state, 1, @ptrCast(&region));

        const op = c.mlirOperationCreate(&state);
        return op;
    }

    /// Create ora.enum_constant operation
    pub fn createEnumConstant(
        self: *OraDialect,
        enum_name: []const u8,
        variant_name: []const u8,
        result_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        var state = h.opState("ora.enum_constant", loc);

        // Add enum name attribute
        const enum_attr = h.stringAttr(self.ctx, enum_name);
        const enum_id = h.identifier(self.ctx, "enum_name");
        const enum_named_attr = c.mlirNamedAttributeGet(enum_id, enum_attr);

        // Add variant name attribute
        const variant_attr = h.stringAttr(self.ctx, variant_name);
        const variant_id = h.identifier(self.ctx, "variant_name");
        const variant_named_attr = c.mlirNamedAttributeGet(variant_id, variant_attr);

        var all_attrs = [_]c.MlirNamedAttribute{ enum_named_attr, variant_named_attr };
        c.mlirOperationStateAddAttributes(&state, all_attrs.len, &all_attrs);

        // Add result type
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_type));

        const op = c.mlirOperationCreate(&state);
        return op;
    }

    //===----------------------------------------------------------------------===//
    // Struct Operations
    //===----------------------------------------------------------------------===//

    /// Create ora.struct.decl operation
    pub fn createStructDecl(
        self: *OraDialect,
        name: []const u8,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        var state = h.opState("ora.struct.decl", loc);

        // Add struct name attribute
        const name_attr = h.stringAttr(self.ctx, name);
        const name_id = h.identifier(self.ctx, "name");
        var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(name_id, name_attr)};
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        // Add fields region
        const region = c.mlirRegionCreate();
        c.mlirOperationStateAddOwnedRegions(&state, 1, @ptrCast(&region));

        const op = c.mlirOperationCreate(&state);
        return op;
    }

    /// Create ora.struct_field_store operation
    pub fn createStructFieldStore(
        self: *OraDialect,
        struct_value: c.MlirValue,
        field_name: []const u8,
        value: c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        var state = h.opState("ora.struct_field_store", loc);

        // Add operands
        var operands = [_]c.MlirValue{ struct_value, value };
        c.mlirOperationStateAddOperands(&state, operands.len, &operands);

        // Add field name attribute
        const field_attr = h.stringAttr(self.ctx, field_name);
        const field_id = h.identifier(self.ctx, "field_name");
        var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(field_id, field_attr)};
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        const op = c.mlirOperationCreate(&state);
        return op;
    }

    /// Create ora.struct_instantiate operation
    pub fn createStructInstantiate(
        self: *OraDialect,
        struct_name: []const u8,
        field_values: []c.MlirValue,
        result_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        var state = h.opState("ora.struct_instantiate", loc);

        // Add operands
        if (field_values.len > 0) {
            c.mlirOperationStateAddOperands(&state, field_values.len, field_values.ptr);
        }

        // Add struct name attribute
        const name_attr = h.stringAttr(self.ctx, struct_name);
        const name_id = h.identifier(self.ctx, "struct_name");
        var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(name_id, name_attr)};
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        // Add result type
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_type));

        const op = c.mlirOperationCreate(&state);
        return op;
    }

    /// Create ora.struct_init operation
    pub fn createStructInit(
        _: *OraDialect,
        field_values: []c.MlirValue,
        result_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        var state = h.opState("ora.struct_init", loc);

        // Add operands
        if (field_values.len > 0) {
            c.mlirOperationStateAddOperands(&state, field_values.len, field_values.ptr);
        }

        // Add result type
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_type));

        const op = c.mlirOperationCreate(&state);
        return op;
    }

    //===----------------------------------------------------------------------===//
    // Map Operations
    //===----------------------------------------------------------------------===//

    /// Create ora.map_get operation
    pub fn createMapGet(
        _: *OraDialect,
        map: c.MlirValue,
        key: c.MlirValue,
        result_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        var state = h.opState("ora.map_get", loc);

        // Add operands
        var operands = [_]c.MlirValue{ map, key };
        c.mlirOperationStateAddOperands(&state, operands.len, &operands);

        // Add result type
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_type));

        const op = c.mlirOperationCreate(&state);
        return op;
    }

    /// Create ora.map_store operation
    pub fn createMapStore(
        _: *OraDialect,
        map: c.MlirValue,
        key: c.MlirValue,
        value: c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        var state = h.opState("ora.map_store", loc);

        // Add operands
        var operands = [_]c.MlirValue{ map, key, value };
        c.mlirOperationStateAddOperands(&state, operands.len, &operands);

        const op = c.mlirOperationCreate(&state);
        return op;
    }

    // ============================================================================
    // Standard MLIR Operations (arith.*, func.*, etc.)
    // ============================================================================

    /// Create arith.constant operation for integer values
    pub fn createArithConstant(
        self: *OraDialect,
        value: i64,
        value_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        var state = h.opState("arith.constant", loc);

        // Add result type
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&value_type));

        // Add value attribute
        const attr = c.mlirIntegerAttrGet(value_type, value);
        const value_id = h.identifier(self.ctx, "value");
        const attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(value_id, attr)};
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        const op = c.mlirOperationCreate(&state);
        return op;
    }

    /// Create arith.constant operation for boolean values
    pub fn createArithConstantBool(
        self: *OraDialect,
        value: bool,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        const bool_type = c.mlirIntegerTypeGet(self.ctx, 1);
        const int_value: i64 = if (value) 1 else 0;
        return self.createArithConstant(int_value, bool_type, loc);
    }

    /// Create arith.constant operation with custom attributes
    pub fn createArithConstantWithAttrs(
        self: *OraDialect,
        value: i64,
        value_type: c.MlirType,
        custom_attrs: []const c.MlirNamedAttribute,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        var state = h.opState("arith.constant", loc);

        // Add result type
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&value_type));

        // Add value attribute
        const attr = c.mlirIntegerAttrGet(value_type, value);
        const value_id = h.identifier(self.ctx, "value");
        const attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(value_id, attr)};

        // Combine with custom attributes
        var all_attrs: [16]c.MlirNamedAttribute = undefined;
        var attr_count: usize = 1;
        all_attrs[0] = attrs[0];

        for (custom_attrs) |custom_attr| {
            if (attr_count < all_attrs.len) {
                all_attrs[attr_count] = custom_attr;
                attr_count += 1;
            }
        }

        c.mlirOperationStateAddAttributes(&state, @intCast(attr_count), &all_attrs);

        const op = c.mlirOperationCreate(&state);
        return op;
    }

    /// Create func.return operation with no return value
    pub fn createFuncReturn(
        self: *OraDialect,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        _ = self;
        var state = h.opState("func.return", loc);
        const op = c.mlirOperationCreate(&state);
        return op;
    }

    /// Create func.return operation with return value
    pub fn createFuncReturnWithValue(
        self: *OraDialect,
        return_value: c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        _ = self;
        var state = h.opState("func.return", loc);
        c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&return_value));
        const op = c.mlirOperationCreate(&state);
        return op;
    }

    /// Create scf.yield operation with no values
    pub fn createScfYield(
        self: *OraDialect,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        _ = self;
        var state = h.opState("scf.yield", loc);
        const op = c.mlirOperationCreate(&state);
        return op;
    }

    /// Create scf.yield operation with return values
    pub fn createScfYieldWithValues(
        self: *OraDialect,
        values: []const c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        _ = self;
        var state = h.opState("scf.yield", loc);
        if (values.len > 0) {
            c.mlirOperationStateAddOperands(&state, @intCast(values.len), values.ptr);
        }
        const op = c.mlirOperationCreate(&state);
        return op;
    }

    /// Create cf.br operation (unconditional branch)
    pub fn createCfBr(
        self: *OraDialect,
        dest_block: c.MlirBlock,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        _ = self;
        var state = h.opState("cf.br", loc);
        c.mlirOperationStateAddSuccessors(&state, 1, @ptrCast(&dest_block));
        const op = c.mlirOperationCreate(&state);
        return op;
    }

    /// Create cf.cond_br operation (conditional branch)
    pub fn createCfCondBr(
        self: *OraDialect,
        condition: c.MlirValue,
        true_block: c.MlirBlock,
        false_block: c.MlirBlock,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        _ = self;
        var state = h.opState("cf.cond_br", loc);
        c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&condition));
        const successors = [_]c.MlirBlock{ true_block, false_block };
        c.mlirOperationStateAddSuccessors(&state, 2, successors.ptr);
        const op = c.mlirOperationCreate(&state);
        return op;
    }

    /// Create memref.alloca operation
    pub fn createMemrefAlloca(
        self: *OraDialect,
        memref_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        _ = self;
        var state = h.opState("memref.alloca", loc);
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&memref_type));
        const op = c.mlirOperationCreate(&state);
        return op;
    }

    /// Create llvm.extractvalue operation
    pub fn createLlvmExtractvalue(
        self: *OraDialect,
        aggregate: c.MlirValue,
        indices: []const u32,
        result_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        var state = h.opState("llvm.extractvalue", loc);
        c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&aggregate));
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_type));

        // Add indices attribute - create array of integer attributes
        var index_attrs = std.ArrayList(c.MlirAttribute){};
        defer index_attrs.deinit(self.allocator);
        for (indices) |index| {
            const index_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 32), @intCast(index));
            index_attrs.append(self.allocator, index_attr) catch unreachable;
        }
        const indices_attr = c.mlirArrayAttrGet(self.ctx, @intCast(index_attrs.items.len), index_attrs.items.ptr);
        const indices_id = h.identifier(self.ctx, "position");
        const attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(indices_id, indices_attr)};
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        const op = c.mlirOperationCreate(&state);
        return op;
    }

    /// Create func.call operation
    pub fn createFuncCall(
        self: *OraDialect,
        callee: []const u8,
        operands: []const c.MlirValue,
        result_types: []const c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        var state = h.opState("func.call", loc);

        // Add operands
        if (operands.len > 0) {
            c.mlirOperationStateAddOperands(&state, @intCast(operands.len), operands.ptr);
        }

        // Add result types
        if (result_types.len > 0) {
            c.mlirOperationStateAddResults(&state, @intCast(result_types.len), result_types.ptr);
        }

        // Add callee attribute
        const callee_ref = c.mlirStringRefCreate(callee.ptr, callee.len);
        const callee_attr = c.mlirFlatSymbolRefAttrGet(self.ctx, callee_ref);
        const callee_id = h.identifier(self.ctx, "callee");
        const attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(callee_id, callee_attr)};
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        const op = c.mlirOperationCreate(&state);
        return op;
    }

    /// Create arith.bitcast operation
    pub fn createArithBitcast(
        self: *OraDialect,
        operand: c.MlirValue,
        result_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        _ = self;
        var state = h.opState("arith.bitcast", loc);
        c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&operand));
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_type));
        const op = c.mlirOperationCreate(&state);
        return op;
    }

    /// Create scf.if operation
    pub fn createScfIf(
        self: *OraDialect,
        condition: c.MlirValue,
        result_types: []const c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        _ = self;
        var state = h.opState("scf.if", loc);
        c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&condition));
        if (result_types.len > 0) {
            c.mlirOperationStateAddResults(&state, @intCast(result_types.len), result_types.ptr);
        }
        const op = c.mlirOperationCreate(&state);
        return op;
    }

    /// Create scf.while operation
    pub fn createScfWhile(
        self: *OraDialect,
        operands: []const c.MlirValue,
        result_types: []const c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        _ = self;
        var state = h.opState("scf.while", loc);
        if (operands.len > 0) {
            c.mlirOperationStateAddOperands(&state, @intCast(operands.len), operands.ptr);
        }
        if (result_types.len > 0) {
            c.mlirOperationStateAddResults(&state, @intCast(result_types.len), result_types.ptr);
        }
        const op = c.mlirOperationCreate(&state);
        return op;
    }

    /// Create scf.for operation
    pub fn createScfFor(
        self: *OraDialect,
        lower_bound: c.MlirValue,
        upper_bound: c.MlirValue,
        step: c.MlirValue,
        init_args: []const c.MlirValue,
        result_types: []const c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        _ = self;
        var state = h.opState("scf.for", loc);

        // Add bounds and step operands
        const bounds_operands = [_]c.MlirValue{ lower_bound, upper_bound, step };
        c.mlirOperationStateAddOperands(&state, bounds_operands.len, &bounds_operands);

        // Add init args if any
        if (init_args.len > 0) {
            c.mlirOperationStateAddOperands(&state, @intCast(init_args.len), init_args.ptr);
        }

        // Add result types if any
        if (result_types.len > 0) {
            c.mlirOperationStateAddResults(&state, @intCast(result_types.len), result_types.ptr);
        }

        const op = c.mlirOperationCreate(&state);
        return op;
    }

    /// Create scf.condition operation
    pub fn createScfCondition(
        self: *OraDialect,
        condition: c.MlirValue,
        args: []const c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        _ = self;
        var state = h.opState("scf.condition", loc);

        // Add condition operand
        c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&condition));

        // Add args if any
        if (args.len > 0) {
            c.mlirOperationStateAddOperands(&state, @intCast(args.len), args.ptr);
        }

        const op = c.mlirOperationCreate(&state);
        return op;
    }

    /// Create memref.load operation
    pub fn createMemrefLoad(
        self: *OraDialect,
        memref: c.MlirValue,
        indices: []const c.MlirValue,
        result_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        var state = h.opState("memref.load", loc);

        // Add operands: memref + indices
        var operands = std.ArrayList(c.MlirValue){};
        defer operands.deinit(self.allocator);
        operands.append(self.allocator, memref) catch unreachable;
        for (indices) |index| {
            operands.append(self.allocator, index) catch unreachable;
        }
        c.mlirOperationStateAddOperands(&state, @intCast(operands.items.len), operands.items.ptr);

        // Add result type
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_type));

        const op = c.mlirOperationCreate(&state);
        return op;
    }

    /// Create memref.store operation
    pub fn createMemrefStore(
        self: *OraDialect,
        value: c.MlirValue,
        memref: c.MlirValue,
        indices: []const c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        var state = h.opState("memref.store", loc);

        // Add operands: value + memref + indices
        var operands = std.ArrayList(c.MlirValue){};
        defer operands.deinit(self.allocator);
        operands.append(self.allocator, value) catch unreachable;
        operands.append(self.allocator, memref) catch unreachable;
        for (indices) |index| {
            operands.append(self.allocator, index) catch unreachable;
        }
        c.mlirOperationStateAddOperands(&state, @intCast(operands.items.len), operands.items.ptr);

        const op = c.mlirOperationCreate(&state);
        return op;
    }

    /// Create scf.break operation
    pub fn createScfBreak(
        self: *OraDialect,
        operands: []const c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        _ = self;
        var state = h.opState("scf.break", loc);
        if (operands.len > 0) {
            c.mlirOperationStateAddOperands(&state, @intCast(operands.len), operands.ptr);
        }
        const op = c.mlirOperationCreate(&state);
        return op;
    }

    /// Create scf.continue operation
    pub fn createScfContinue(
        self: *OraDialect,
        operands: []const c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        _ = self;
        var state = h.opState("scf.continue", loc);
        if (operands.len > 0) {
            c.mlirOperationStateAddOperands(&state, @intCast(operands.len), operands.ptr);
        }
        const op = c.mlirOperationCreate(&state);
        return op;
    }

    /// Create arith.addi operation
    pub fn createArithAddi(
        self: *OraDialect,
        lhs: c.MlirValue,
        rhs: c.MlirValue,
        result_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        _ = self;
        var state = h.opState("arith.addi", loc);
        const operands = [_]c.MlirValue{ lhs, rhs };
        c.mlirOperationStateAddOperands(&state, operands.len, &operands);
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_type));
        const op = c.mlirOperationCreate(&state);
        return op;
    }

    /// Create arith.subi operation
    pub fn createArithSubi(
        self: *OraDialect,
        lhs: c.MlirValue,
        rhs: c.MlirValue,
        result_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        _ = self;
        var state = h.opState("arith.subi", loc);
        const operands = [_]c.MlirValue{ lhs, rhs };
        c.mlirOperationStateAddOperands(&state, operands.len, &operands);
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_type));
        const op = c.mlirOperationCreate(&state);
        return op;
    }

    /// Create arith.muli operation
    pub fn createArithMuli(
        self: *OraDialect,
        lhs: c.MlirValue,
        rhs: c.MlirValue,
        result_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        _ = self;
        var state = h.opState("arith.muli", loc);
        const operands = [_]c.MlirValue{ lhs, rhs };
        c.mlirOperationStateAddOperands(&state, operands.len, &operands);
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_type));
        const op = c.mlirOperationCreate(&state);
        return op;
    }

    /// Create arith.divsi operation
    pub fn createArithDivsi(
        self: *OraDialect,
        lhs: c.MlirValue,
        rhs: c.MlirValue,
        result_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        _ = self;
        var state = h.opState("arith.divsi", loc);
        const operands = [_]c.MlirValue{ lhs, rhs };
        c.mlirOperationStateAddOperands(&state, operands.len, &operands);
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_type));
        const op = c.mlirOperationCreate(&state);
        return op;
    }

    /// Create arith.remsi operation
    pub fn createArithRemsi(
        self: *OraDialect,
        lhs: c.MlirValue,
        rhs: c.MlirValue,
        result_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        _ = self;
        var state = h.opState("arith.remsi", loc);
        const operands = [_]c.MlirValue{ lhs, rhs };
        c.mlirOperationStateAddOperands(&state, operands.len, &operands);
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_type));
        const op = c.mlirOperationCreate(&state);
        return op;
    }
};

// Legacy alias for backward compatibility
pub const Dialect = OraDialect;
