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
    pub fn register(self: *OraDialect) !void {
        const success = c.oraDialectRegister(self.ctx);
        if (!success) {
            std.log.warn("Failed to register Ora dialect, falling back to unregistered mode", .{});
            self.dialect_handle = null;
            return;
        }
        const handle = c.oraDialectGet(self.ctx);
        if (handle.ptr == null) {
            std.log.warn("Ora dialect registered but handle is null", .{});
            self.dialect_handle = null;
            return;
        }
        self.dialect_handle = handle;
        std.log.info("Ora dialect successfully registered", .{});
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
        // Always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const name_ref = h.strRef(name);
        const op = c.oraGlobalOpCreate(self.ctx, loc, name_ref, value_type, init_value);
        if (op.ptr == null) {
            @panic("Failed to create ora.global operation");
        }
        return op;
    }

    // Helper function to create ora.sload operation
    pub fn createSLoad(
        self: *OraDialect,
        global_name: []const u8,
        result_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        return self.createSLoadWithName(global_name, result_type, loc, null);
    }

    // Helper function to create ora.sload operation with named result
    pub fn createSLoadWithName(
        self: *OraDialect,
        global_name: []const u8,
        result_type: c.MlirType,
        loc: c.MlirLocation,
        result_name: ?[]const u8,
    ) c.MlirOperation {
        // Always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const name_ref = h.strRef(global_name);
        const result_name_ref = if (result_name) |name| h.strRef(name) else c.MlirStringRef{ .data = null, .length = 0 };
        const op = c.oraSLoadOpCreateWithName(self.ctx, loc, name_ref, result_type, result_name_ref);
        if (op.ptr == null) {
            @panic("Failed to create ora.sload operation");
        }
        return op;
    }

    // Helper function to create ora.sstore operation
    pub fn createSStore(
        self: *OraDialect,
        value: c.MlirValue,
        global_name: []const u8,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // Always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const name_ref = h.strRef(global_name);
        const op = c.oraSStoreOpCreate(self.ctx, loc, value, name_ref);
        if (op.ptr == null) {
            @panic("Failed to create ora.sstore operation");
        }
        return op;
    }

    // Helper function to create ora.contract operation
    pub fn createContract(
        self: *OraDialect,
        name: []const u8,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // Always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const name_ref = h.strRef(name);
        const op = c.oraContractOpCreate(self.ctx, loc, name_ref);
        if (op.ptr == null) {
            @panic("Failed to create ora.contract operation");
        }
        return op;
    }

    /// Create ora.requires operation for preconditions
    pub fn createRequires(
        self: *OraDialect,
        condition: c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // Always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        return c.oraRequiresOpCreate(self.ctx, loc, condition);
    }

    /// Create ora.ensures operation for postconditions
    pub fn createEnsures(
        self: *OraDialect,
        condition: c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // Always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        return c.oraEnsuresOpCreate(self.ctx, loc, condition);
    }

    /// Create ora.invariant operation
    pub fn createInvariant(
        self: *OraDialect,
        condition: c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // Always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        return c.oraInvariantOpCreate(self.ctx, loc, condition);
    }

    /// Create ora.assert operation for assertions
    pub fn createAssert(
        self: *OraDialect,
        condition: c.MlirValue,
        loc: c.MlirLocation,
        message: ?[]const u8,
    ) c.MlirOperation {
        // Always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const message_ref = if (message) |msg| h.strRef(msg) else c.MlirStringRef{ .data = null, .length = 0 };
        return c.oraAssertOpCreate(self.ctx, loc, condition, message_ref);
    }

    /// Create ora.decreases operation for loop termination measures
    pub fn createDecreases(
        self: *OraDialect,
        measure: c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // Always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const op = c.oraDecreasesOpCreate(self.ctx, loc, measure);
        if (op.ptr == null) {
            @panic("Failed to create ora.decreases operation");
        }
        return op;
    }

    /// Create ora.increases operation for loop progress measures
    pub fn createIncreases(
        self: *OraDialect,
        measure: c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // Always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const op = c.oraIncreasesOpCreate(self.ctx, loc, measure);
        if (op.ptr == null) {
            @panic("Failed to create ora.increases operation");
        }
        return op;
    }

    /// Create ora.assume operation for assumptions
    pub fn createAssume(
        self: *OraDialect,
        condition: c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // Always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const op = c.oraAssumeOpCreate(self.ctx, loc, condition);
        if (op.ptr == null) {
            @panic("Failed to create ora.assume operation");
        }
        return op;
    }

    /// Create ora.havoc operation for havoc statements
    pub fn createHavoc(
        self: *OraDialect,
        variable_name: []const u8,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // Always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const name_ref = h.strRef(variable_name);
        const op = c.oraHavocOpCreate(self.ctx, loc, name_ref);
        if (op.ptr == null) {
            @panic("Failed to create ora.havoc operation");
        }
        return op;
    }

    /// Create ora.old operation for old value references
    pub fn createOld(
        self: *OraDialect,
        value: c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // Always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        // Result type is same as input type
        const input_type = c.mlirValueGetType(value);
        const op = c.oraOldOpCreate(self.ctx, loc, value, input_type);
        if (op.ptr == null) {
            @panic("Failed to create ora.old operation");
        }
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
        // Always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const value_ref = h.strRef(value);
        const op = c.oraStringConstantOpCreate(self.ctx, loc, value_ref, result_type);
        if (op.ptr == null) {
            @panic("Failed to create ora.string.constant operation");
        }
        return op;
    }

    /// Create ora.hex.constant operation
    pub fn createHexConstant(
        self: *OraDialect,
        value: []const u8,
        result_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // Always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const value_ref = h.strRef(value);
        const op = c.oraHexConstantOpCreate(self.ctx, loc, value_ref, result_type);
        if (op.ptr == null) {
            @panic("Failed to create ora.hex.constant operation");
        }
        return op;
    }

    /// Create ora.power operation
    pub fn createPower(
        self: *OraDialect,
        base: c.MlirValue,
        exponent: c.MlirValue,
        result_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // Always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const op = c.oraPowerOpCreate(self.ctx, loc, base, exponent, result_type);
        if (op.ptr == null) {
            @panic("Failed to create ora.power operation");
        }
        return op;
    }

    /// Create ora.yield operation
    pub fn createYield(
        self: *OraDialect,
        operands: []const c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // Always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const operands_ptr = if (operands.len > 0) operands.ptr else null;
        return c.oraYieldOpCreate(self.ctx, loc, operands_ptr, operands.len);
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
        // Always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const pattern_type_ref = h.strRef(pattern_type);
        const op = c.oraDestructureOpCreate(self.ctx, loc, value, pattern_type_ref, result_type);
        if (op.ptr == null) {
            @panic("Failed to create ora.destructure operation");
        }
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
        // Always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const name_ref = h.strRef(name);
        const op = c.oraEnumDeclOpCreate(self.ctx, loc, name_ref, repr_type);
        if (op.ptr == null) {
            @panic("Failed to create ora.enum.decl operation");
        }
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
        // Always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const enum_name_ref = h.strRef(enum_name);
        const variant_name_ref = h.strRef(variant_name);
        const op = c.oraEnumConstantOpCreate(self.ctx, loc, enum_name_ref, variant_name_ref, result_type);
        if (op.ptr == null) {
            @panic("Failed to create ora.enum_constant operation");
        }
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
        // Always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const name_ref = h.strRef(name);
        const op = c.oraStructDeclOpCreate(self.ctx, loc, name_ref);
        if (op.ptr == null) {
            @panic("Failed to create ora.struct.decl operation");
        }
        return op;
    }

    /// Create ora.struct.decl operation (old implementation kept for reference)
    pub fn createStructDeclOld(
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
        // Always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const field_name_ref = h.strRef(field_name);
        const op = c.oraStructFieldStoreOpCreate(self.ctx, loc, struct_value, field_name_ref, value);
        if (op.ptr == null) {
            @panic("Failed to create ora.struct_field_store operation");
        }
        return op;
    }

    /// Create ora.struct_field_extract operation (pure, returns field value)
    pub fn createStructFieldExtract(
        self: *OraDialect,
        struct_value: c.MlirValue,
        field_name: []const u8,
        result_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // Always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const field_name_ref = h.strRef(field_name);
        const op = c.oraStructFieldExtractOpCreate(self.ctx, loc, struct_value, field_name_ref, result_type);
        if (op.ptr == null) {
            @panic("Failed to create ora.struct_field_extract operation");
        }
        return op;
    }

    /// Create ora.struct_field_update operation (pure, returns new struct with updated field)
    pub fn createStructFieldUpdate(
        self: *OraDialect,
        struct_value: c.MlirValue,
        field_name: []const u8,
        value: c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // Always use C++ API (dialect must be registered)
        // Result type is inferred from struct_value (enforced by SameOperandsAndResultType trait)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const field_name_ref = h.strRef(field_name);
        const op = c.oraStructFieldUpdateOpCreate(self.ctx, loc, struct_value, field_name_ref, value);
        if (op.ptr == null) {
            std.debug.print("ERROR: Failed to create ora.struct_field_update operation for field: {s}\n", .{field_name});
            @panic("Failed to create ora.struct_field_update operation");
        }
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
        // Always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const struct_name_ref = h.strRef(struct_name);
        const field_values_ptr = if (field_values.len > 0) field_values.ptr else null;
        const op = c.oraStructInstantiateOpCreate(self.ctx, loc, struct_name_ref, field_values_ptr, @intCast(field_values.len), result_type);
        if (op.ptr == null) {
            @panic("Failed to create ora.struct_instantiate operation");
        }
        return op;
    }

    /// Create ora.struct_init operation
    pub fn createStructInit(
        self: *OraDialect,
        field_values: []const c.MlirValue,
        result_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // Always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const field_values_ptr = if (field_values.len > 0) field_values.ptr else null;
        const op = c.oraStructInitOpCreate(self.ctx, loc, field_values_ptr, @intCast(field_values.len), result_type);
        if (op.ptr == null) {
            @panic("Failed to create ora.struct_init operation");
        }
        return op;
    }

    //===----------------------------------------------------------------------===//
    // Map Operations
    //===----------------------------------------------------------------------===//

    /// Create ora.map_get operation
    pub fn createMapGet(
        self: *OraDialect,
        map: c.MlirValue,
        key: c.MlirValue,
        result_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // Always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const op = c.oraMapGetOpCreate(self.ctx, loc, map, key, result_type);
        if (op.ptr == null) {
            @panic("Failed to create ora.map_get operation");
        }
        return op;
    }

    /// Create ora.map_store operation
    pub fn createMapStore(
        self: *OraDialect,
        map: c.MlirValue,
        key: c.MlirValue,
        value: c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // Always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const op = c.oraMapStoreOpCreate(self.ctx, loc, map, key, value);
        if (op.ptr == null) {
            @panic("Failed to create ora.map_store operation");
        }
        return op;
    }

    /// Create ora.mload operation
    pub fn createMLoad(
        self: *OraDialect,
        variable_name: []const u8,
        result_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // Always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const name_ref = h.strRef(variable_name);
        const op = c.oraMLoadOpCreate(self.ctx, loc, name_ref, result_type);
        if (op.ptr == null) {
            @panic("Failed to create ora.mload operation");
        }
        return op;
    }

    /// Create ora.mstore operation
    pub fn createMStore(
        self: *OraDialect,
        value: c.MlirValue,
        variable_name: []const u8,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // Always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const name_ref = h.strRef(variable_name);
        const op = c.oraMStoreOpCreate(self.ctx, loc, value, name_ref);
        if (op.ptr == null) {
            @panic("Failed to create ora.mstore operation");
        }
        return op;
    }

    /// Create ora.tload operation
    pub fn createTLoad(
        self: *OraDialect,
        key: []const u8,
        result_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // Always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const key_ref = h.strRef(key);
        const op = c.oraTLoadOpCreate(self.ctx, loc, key_ref, result_type);
        if (op.ptr == null) {
            @panic("Failed to create ora.tload operation");
        }
        return op;
    }

    /// Create ora.tstore operation
    pub fn createTStore(
        self: *OraDialect,
        value: c.MlirValue,
        key: []const u8,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // Always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const key_ref = h.strRef(key);
        const op = c.oraTStoreOpCreate(self.ctx, loc, value, key_ref);
        if (op.ptr == null) {
            @panic("Failed to create ora.tstore operation");
        }
        return op;
    }

    /// Create ora.continue operation
    pub fn createContinue(
        self: *OraDialect,
        label: ?[]const u8,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // Always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const label_ref = if (label) |l| h.strRef(l) else c.MlirStringRef{ .data = null, .length = 0 };
        const op = c.oraContinueOpCreate(self.ctx, loc, label_ref);
        if (op.ptr == null) {
            @panic("Failed to create ora.continue operation");
        }
        return op;
    }

    /// Create ora.return operation
    pub fn createReturn(
        self: *OraDialect,
        operands: []const c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // Always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const op = c.oraReturnOpCreate(self.ctx, loc, operands.ptr, @intCast(operands.len));
        if (op.ptr == null) {
            @panic("Failed to create ora.return operation");
        }
        return op;
    }

    /// Create ora.const operation
    pub fn createConst(
        self: *OraDialect,
        name: []const u8,
        value: c.MlirAttribute,
        result_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // Always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const name_ref = h.strRef(name);
        const op = c.oraConstOpCreate(self.ctx, loc, name_ref, value, result_type);
        if (op.ptr == null) {
            @panic("Failed to create ora.const operation");
        }
        return op;
    }

    /// Create ora.immutable operation
    pub fn createImmutable(
        self: *OraDialect,
        name: []const u8,
        value: c.MlirValue,
        result_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // Always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const name_ref = h.strRef(name);
        const op = c.oraImmutableOpCreate(self.ctx, loc, name_ref, value, result_type);
        if (op.ptr == null) {
            @panic("Failed to create ora.immutable operation");
        }
        return op;
    }

    /// Create ora.lock operation
    pub fn createLock(
        self: *OraDialect,
        resource: c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // Always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const op = c.oraLockOpCreate(self.ctx, loc, resource);
        if (op.ptr == null) {
            @panic("Failed to create ora.lock operation");
        }
        return op;
    }

    /// Create ora.unlock operation
    pub fn createUnlock(
        self: *OraDialect,
        resource: c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // Always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const op = c.oraUnlockOpCreate(self.ctx, loc, resource);
        if (op.ptr == null) {
            @panic("Failed to create ora.unlock operation");
        }
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

        // Add gas cost attribute (constants are free = 0)
        const gas_cost_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), 0);
        const gas_cost_id = h.identifier(self.ctx, "gas_cost");
        var attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(value_id, attr),
            c.mlirNamedAttributeGet(gas_cost_id, gas_cost_attr),
        };
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

    /// Create ora.return operation with no return value
    pub fn createFuncReturn(
        self: *OraDialect,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        var state = h.opState("ora.return", loc);

        // Add gas cost attribute (JUMP = 8, return is similar to jump)
        const gas_cost_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), 8);
        const gas_cost_id = h.identifier(self.ctx, "gas_cost");
        var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(gas_cost_id, gas_cost_attr)};
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        const op = c.mlirOperationCreate(&state);
        return op;
    }

    /// Create ora.return operation with return value
    pub fn createFuncReturnWithValue(
        self: *OraDialect,
        return_value: c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        var state = h.opState("ora.return", loc);
        c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&return_value));

        // Add gas cost attribute (JUMP = 8, return is similar to jump)
        const gas_cost_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), 8);
        const gas_cost_id = h.identifier(self.ctx, "gas_cost");
        var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(gas_cost_id, gas_cost_attr)};
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        const op = c.mlirOperationCreate(&state);
        return op;
    }

    /// Create scf.yield operation with no values
    pub fn createScfYield(
        self: *OraDialect,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        var state = h.opState("scf.yield", loc);

        // Add gas cost attribute (yield itself has no cost, but JUMPDEST = 1 at destination)
        const gas_cost_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), 1);
        const gas_cost_id = h.identifier(self.ctx, "gas_cost");
        var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(gas_cost_id, gas_cost_attr)};
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        const op = c.mlirOperationCreate(&state);
        return op;
    }

    /// Create scf.yield operation with return values
    pub fn createScfYieldWithValues(
        self: *OraDialect,
        values: []const c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        var state = h.opState("scf.yield", loc);
        if (values.len > 0) {
            c.mlirOperationStateAddOperands(&state, @intCast(values.len), values.ptr);
        }

        // Add gas cost attribute (yield itself has no cost, but JUMPDEST = 1 at destination)
        const gas_cost_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), 1);
        const gas_cost_id = h.identifier(self.ctx, "gas_cost");
        var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(gas_cost_id, gas_cost_attr)};
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

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

        // Add gas cost attribute (internal function call - minimal cost, similar to JUMPI = 10)
        // For external calls, this would be CALL_BASE = 700, but internal calls are cheaper
        const gas_cost_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), 10);
        const gas_cost_id = h.identifier(self.ctx, "gas_cost");
        const gas_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(gas_cost_id, gas_cost_attr)};

        // Add callee attribute
        const callee_ref = c.mlirStringRefCreate(callee.ptr, callee.len);
        const callee_attr = c.mlirFlatSymbolRefAttrGet(self.ctx, callee_ref);
        const callee_id = h.identifier(self.ctx, "callee");
        var attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(callee_id, callee_attr),
            gas_attrs[0],
        };
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

    /// Create ora.if operation using C++ API (enables custom assembly formats)
    pub fn createIf(
        self: *OraDialect,
        condition: c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // Always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const op = c.oraIfOpCreate(self.ctx, loc, condition);
        if (op.ptr == null) {
            @panic("Failed to create ora.if operation");
        }
        return op;
    }

    /// Create scf.if operation (legacy - for compatibility)
    pub fn createScfIf(
        self: *OraDialect,
        condition: c.MlirValue,
        result_types: []const c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        var state = h.opState("scf.if", loc);
        c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&condition));
        if (result_types.len > 0) {
            c.mlirOperationStateAddResults(&state, @intCast(result_types.len), result_types.ptr);
        }

        // Add gas cost attribute (JUMPI = 10, conditional branch)
        const gas_cost_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), 10);
        const gas_cost_id = h.identifier(self.ctx, "gas_cost");
        var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(gas_cost_id, gas_cost_attr)};
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        const op = c.mlirOperationCreate(&state);
        return op;
    }

    /// Create ora.while operation using C++ API (enables custom assembly formats)
    pub fn createWhile(
        self: *OraDialect,
        condition: c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // Always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const op = c.oraWhileOpCreate(self.ctx, loc, condition);
        if (op.ptr == null) {
            @panic("Failed to create ora.while operation");
        }
        return op;
    }

    /// Create ora.test operation (simple test for custom printer)
    pub fn createTest(
        self: *OraDialect,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // Always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const op = c.oraTestOpCreate(self.ctx, loc);
        if (op.ptr == null) {
            @panic("Failed to create ora.test operation");
        }
        return op;
    }

    /// Create scf.while operation (legacy - for compatibility)
    pub fn createScfWhile(
        self: *OraDialect,
        operands: []const c.MlirValue,
        result_types: []const c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        var state = h.opState("scf.while", loc);
        if (operands.len > 0) {
            c.mlirOperationStateAddOperands(&state, @intCast(operands.len), operands.ptr);
        }
        if (result_types.len > 0) {
            c.mlirOperationStateAddResults(&state, @intCast(result_types.len), result_types.ptr);
        }

        // Add gas cost attribute (JUMPI = 10 per loop iteration for the conditional jump)
        // Note: This is the cost per iteration, actual total depends on loop execution
        const gas_cost_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), 10);
        const gas_cost_id = h.identifier(self.ctx, "gas_cost");
        var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(gas_cost_id, gas_cost_attr)};
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        const op = c.mlirOperationCreate(&state);
        return op;
    }

    /// Create scf.for operation
    pub fn createScfFor(
        self: *const OraDialect,
        lower_bound: c.MlirValue,
        upper_bound: c.MlirValue,
        step: c.MlirValue,
        init_args: []const c.MlirValue,
        result_types: []const c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
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

        // Add gas cost attribute (JUMPI = 10 per loop iteration for the conditional jump)
        const gas_cost_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), 10);
        const gas_cost_id = h.identifier(self.ctx, "gas_cost");
        var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(gas_cost_id, gas_cost_attr)};
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

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
        var state = h.opState("arith.addi", loc);
        const operands = [_]c.MlirValue{ lhs, rhs };
        c.mlirOperationStateAddOperands(&state, operands.len, &operands);
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_type));

        // Add gas cost attribute (ADD = 3)
        const gas_cost_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), 3);
        const gas_cost_id = h.identifier(self.ctx, "gas_cost");
        var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(gas_cost_id, gas_cost_attr)};
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

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
        var state = h.opState("arith.subi", loc);
        const operands = [_]c.MlirValue{ lhs, rhs };
        c.mlirOperationStateAddOperands(&state, operands.len, &operands);
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_type));

        // Add gas cost attribute (SUB = 3)
        const gas_cost_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), 3);
        const gas_cost_id = h.identifier(self.ctx, "gas_cost");
        var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(gas_cost_id, gas_cost_attr)};
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

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
        var state = h.opState("arith.muli", loc);
        const operands = [_]c.MlirValue{ lhs, rhs };
        c.mlirOperationStateAddOperands(&state, operands.len, &operands);
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_type));

        // Add gas cost attribute (MUL = 5)
        const gas_cost_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), 5);
        const gas_cost_id = h.identifier(self.ctx, "gas_cost");
        var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(gas_cost_id, gas_cost_attr)};
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

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
        var state = h.opState("arith.divsi", loc);
        const operands = [_]c.MlirValue{ lhs, rhs };
        c.mlirOperationStateAddOperands(&state, operands.len, &operands);
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_type));

        // Add gas cost attribute (DIV = 5)
        const gas_cost_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), 5);
        const gas_cost_id = h.identifier(self.ctx, "gas_cost");
        var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(gas_cost_id, gas_cost_attr)};
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

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
        var state = h.opState("arith.remsi", loc);
        const operands = [_]c.MlirValue{ lhs, rhs };
        c.mlirOperationStateAddOperands(&state, operands.len, &operands);
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_type));

        // Add gas cost attribute (MOD = 5)
        const gas_cost_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), 5);
        const gas_cost_id = h.identifier(self.ctx, "gas_cost");
        var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(gas_cost_id, gas_cost_attr)};
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        const op = c.mlirOperationCreate(&state);
        return op;
    }

    //===----------------------------------------------------------------------===//
    // Financial Operations
    //===----------------------------------------------------------------------===//

    /// Create ora.move operation
    pub fn createMove(
        self: *OraDialect,
        amount: c.MlirValue,
        source: c.MlirValue,
        destination: c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // Always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const op = c.oraMoveOpCreate(self.ctx, loc, amount, source, destination);
        if (op.ptr == null) {
            @panic("Failed to create ora.move operation");
        }
        return op;
    }

    //===----------------------------------------------------------------------===//
    // Event Operations
    //===----------------------------------------------------------------------===//

    /// Create ora.log operation
    pub fn createLog(
        self: *OraDialect,
        event_name: []const u8,
        parameters: []const c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // Always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const event_name_ref = h.strRef(event_name);
        const parameters_ptr = if (parameters.len > 0) parameters.ptr else null;
        const op = c.oraLogOpCreate(self.ctx, loc, event_name_ref, parameters_ptr, @intCast(parameters.len));
        if (op.ptr == null) {
            @panic("Failed to create ora.log operation");
        }
        return op;
    }

    //===----------------------------------------------------------------------===//
    // Error Handling Operations
    //===----------------------------------------------------------------------===//

    /// Create ora.try_catch operation
    pub fn createTry(
        self: *OraDialect,
        try_operation: c.MlirValue,
        result_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // Always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const op = c.oraTryOpCreate(self.ctx, loc, try_operation, result_type);
        if (op.ptr == null) {
            @panic("Failed to create ora.try_catch operation");
        }
        return op;
    }

    //===----------------------------------------------------------------------===//
    // Loop Operations
    //===----------------------------------------------------------------------===//

    /// Create ora.for operation
    pub fn createFor(
        self: *OraDialect,
        collection: c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // Always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const op = c.oraForOpCreate(self.ctx, loc, collection);
        if (op.ptr == null) {
            @panic("Failed to create ora.for operation");
        }
        return op;
    }

    /// Create ora.break operation
    pub fn createBreak(
        self: *OraDialect,
        label: ?[]const u8,
        values: []const c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // Always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const label_ref = if (label) |l| h.strRef(l) else c.MlirStringRef{ .data = null, .length = 0 };
        const values_ptr = if (values.len > 0) values.ptr else null;
        const op = c.oraBreakOpCreate(self.ctx, loc, label_ref, values_ptr, @intCast(values.len));
        if (op.ptr == null) {
            @panic("Failed to create ora.break operation");
        }
        return op;
    }

    /// Create ora.switch operation
    pub fn createSwitch(
        self: *OraDialect,
        value: c.MlirValue,
        result_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // Always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const op = c.oraSwitchOpCreate(self.ctx, loc, value, result_type);
        if (op.ptr == null) {
            @panic("Failed to create ora.switch operation");
        }
        return op;
    }
};

// Legacy alias for backward compatibility
pub const Dialect = OraDialect;
