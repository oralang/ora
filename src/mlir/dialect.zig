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
const c = @import("mlir_c_api").c;
const h = @import("helpers.zig");
const log = @import("log");

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
            log.warn("Failed to register Ora dialect, falling back to unregistered mode\n", .{});
            self.dialect_handle = null;
            return;
        }
        const handle = c.oraDialectGet(self.ctx);
        if (handle.ptr == null) {
            log.warn("Ora dialect registered but handle is null\n", .{});
            self.dialect_handle = null;
            return;
        }
        self.dialect_handle = handle;
        log.debug("Ora dialect successfully registered\n", .{});
    }

    /// Check if the Ora dialect is properly registered
    pub fn isRegistered(self: *const OraDialect) bool {
        return self.dialect_handle != null;
    }

    /// Get the dialect namespace
    pub fn getNamespace() []const u8 {
        return "ora";
    }

    // helper function to create ora.global operation
    pub fn createGlobal(
        self: *OraDialect,
        name: []const u8,
        value_type: c.MlirType,
        init_value: c.MlirAttribute,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // always use C++ API (dialect must be registered)
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

    // helper function to create ora.sload operation
    pub fn createSLoad(
        self: *OraDialect,
        global_name: []const u8,
        result_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        return self.createSLoadWithName(global_name, result_type, loc, null);
    }

    // helper function to create ora.sload operation with named result
    pub fn createSLoadWithName(
        self: *OraDialect,
        global_name: []const u8,
        result_type: c.MlirType,
        loc: c.MlirLocation,
        result_name: ?[]const u8,
    ) c.MlirOperation {
        // always use C++ API (dialect must be registered)
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

    // helper function to create ora.sstore operation
    pub fn createSStore(
        self: *OraDialect,
        value: c.MlirValue,
        global_name: []const u8,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // always use C++ API (dialect must be registered)
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

    // helper function to create ora.contract operation
    pub fn createContract(
        self: *OraDialect,
        name: []const u8,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // always use C++ API (dialect must be registered)
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
        // always use C++ API (dialect must be registered)
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
        // always use C++ API (dialect must be registered)
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
        // always use C++ API (dialect must be registered)
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
        // always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const message_ref = if (message) |msg| h.strRef(msg) else c.MlirStringRef{ .data = null, .length = 0 };
        return c.oraAssertOpCreate(self.ctx, loc, condition, message_ref);
    }

    /// Create ora.refinement_guard operation
    pub fn createRefinementGuard(
        self: *OraDialect,
        condition: c.MlirValue,
        loc: c.MlirLocation,
        message: ?[]const u8,
    ) c.MlirOperation {
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const message_ref = if (message) |msg| h.strRef(msg) else c.MlirStringRef{ .data = null, .length = 0 };
        return c.oraRefinementGuardOpCreate(self.ctx, loc, condition, message_ref);
    }

    /// Create ora.decreases operation for loop termination measures
    pub fn createDecreases(
        self: *OraDialect,
        measure: c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // always use C++ API (dialect must be registered)
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
        // always use C++ API (dialect must be registered)
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
        // always use C++ API (dialect must be registered)
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
        // always use C++ API (dialect must be registered)
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
        // always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        // result type is same as input type
        const input_type = c.oraValueGetType(value);
        const op = c.oraOldOpCreate(self.ctx, loc, value, input_type);
        if (op.ptr == null) {
            @panic("Failed to create ora.old operation");
        }
        return op;
    }

    //===----------------------------------------------------------------------===//
    // phase 1 Operations - Constants
    //===----------------------------------------------------------------------===//

    /// Create ora.string.constant operation
    pub fn createStringConstant(
        self: *OraDialect,
        value: []const u8,
        result_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // always use C++ API (dialect must be registered)
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
        // always use C++ API (dialect must be registered)
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

    /// Create ora.bytes.constant operation
    pub fn createBytesConstant(
        self: *OraDialect,
        value: []const u8,
        result_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const value_ref = h.strRef(value);
        const op = c.oraBytesConstantOpCreate(self.ctx, loc, value_ref, result_type);
        if (op.ptr == null) {
            @panic("Failed to create ora.bytes.constant operation");
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
        // always use C++ API (dialect must be registered)
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
        // always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const operands_ptr = if (operands.len > 0) operands.ptr else null;
        return c.oraYieldOpCreate(self.ctx, loc, operands_ptr, operands.len);
    }

    //===----------------------------------------------------------------------===//
    // destructuring Operations
    //===----------------------------------------------------------------------===//

    /// Create ora.destructure operation
    pub fn createDestructure(
        self: *OraDialect,
        value: c.MlirValue,
        pattern_type: []const u8,
        result_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // always use C++ API (dialect must be registered)
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
    // enum Operations
    //===----------------------------------------------------------------------===//

    /// Create ora.enum.decl operation
    pub fn createEnumDecl(
        self: *OraDialect,
        name: []const u8,
        repr_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // always use C++ API (dialect must be registered)
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
        // always use C++ API (dialect must be registered)
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
    // struct Operations
    //===----------------------------------------------------------------------===//

    /// Create ora.struct.decl operation
    pub fn createStructDecl(
        self: *OraDialect,
        name: []const u8,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // always use C++ API (dialect must be registered)
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

    /// Create ora.struct_field_store operation
    pub fn createStructFieldStore(
        self: *OraDialect,
        struct_value: c.MlirValue,
        field_name: []const u8,
        value: c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // always use C++ API (dialect must be registered)
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
        // always use C++ API (dialect must be registered)
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
        // always use C++ API (dialect must be registered)
        // result type is inferred from struct_value (enforced by SameOperandsAndResultType trait)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const field_name_ref = h.strRef(field_name);
        const op = c.oraStructFieldUpdateOpCreate(self.ctx, loc, struct_value, field_name_ref, value);
        if (op.ptr == null) {
            log.err("Failed to create ora.struct_field_update operation for field: {s}\n", .{field_name});
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
        // always use C++ API (dialect must be registered)
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
        // always use C++ API (dialect must be registered)
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
    // map Operations
    //===----------------------------------------------------------------------===//

    /// Create ora.map_get operation
    pub fn createMapGet(
        self: *OraDialect,
        map: c.MlirValue,
        key: c.MlirValue,
        result_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // always use C++ API (dialect must be registered)
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
        // always use C++ API (dialect must be registered)
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
        // always use C++ API (dialect must be registered)
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
        // always use C++ API (dialect must be registered)
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
        // always use C++ API (dialect must be registered)
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
        // always use C++ API (dialect must be registered)
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

    /// Create ora.tstore.guard (revert at runtime if TStore slot key is locked)
    pub fn createTStoreGuard(self: *OraDialect, resource: c.MlirValue, key: []const u8, loc: c.MlirLocation) c.MlirOperation {
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const key_ref = h.strRef(key);
        const op = c.oraTStoreGuardOpCreateWithResource(self.ctx, loc, resource, key_ref);
        if (op.ptr == null) {
            @panic("Failed to create ora.tstore.guard operation");
        }
        return op;
    }

    /// Create ora.continue operation
    pub fn createContinue(
        self: *OraDialect,
        label: ?[]const u8,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // always use C++ API (dialect must be registered)
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
        // always use C++ API (dialect must be registered)
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
        // always use C++ API (dialect must be registered)
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
        // always use C++ API (dialect must be registered)
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

    /// Create ora.lock operation (optionally with key for OraToSIR slot lookup)
    pub fn createLockWithKey(
        self: *OraDialect,
        resource: c.MlirValue,
        key: []const u8,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const key_ref = if (key.len > 0) h.strRef(key) else c.MlirStringRef{ .data = null, .length = 0 };
        const op = c.oraLockOpCreateWithKey(self.ctx, loc, resource, key_ref);
        if (op.ptr == null) {
            @panic("Failed to create ora.lock operation");
        }
        return op;
    }

    pub fn createLock(self: *OraDialect, resource: c.MlirValue, loc: c.MlirLocation) c.MlirOperation {
        return self.createLockWithKey(resource, "", loc);
    }

    /// Create ora.unlock operation (optionally with key for OraToSIR slot lookup)
    pub fn createUnlockWithKey(
        self: *OraDialect,
        resource: c.MlirValue,
        key: []const u8,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const key_ref = if (key.len > 0) h.strRef(key) else c.MlirStringRef{ .data = null, .length = 0 };
        const op = c.oraUnlockOpCreateWithKey(self.ctx, loc, resource, key_ref);
        if (op.ptr == null) {
            @panic("Failed to create ora.unlock operation");
        }
        return op;
    }

    pub fn createUnlock(self: *OraDialect, resource: c.MlirValue, loc: c.MlirLocation) c.MlirOperation {
        return self.createUnlockWithKey(resource, "", loc);
    }

    // ============================================================================
    // standard MLIR Operations (arith.*, func.*, etc.)
    // ============================================================================

    /// Create arith.constant operation for integer values
    pub fn createArithConstant(
        self: *OraDialect,
        value: i64,
        value_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        const attr = c.oraIntegerAttrCreateI64FromType(value_type, value);
        const op = c.oraArithConstantOpCreate(self.ctx, loc, value_type, attr);
        if (op.ptr == null) {
            @panic("Failed to create arith.constant operation");
        }
        return op;
    }

    /// Create arith.constant operation for boolean values
    pub fn createArithConstantBool(
        self: *OraDialect,
        value: bool,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        const bool_type = c.oraIntegerTypeCreate(self.ctx, 1);
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
        // add value attribute
        const attr = c.oraIntegerAttrCreateI64FromType(value_type, value);
        if (attr.ptr == null) {
            @panic("Failed to create integer attribute");
        }
        const op = c.oraArithConstantOpCreate(self.ctx, loc, value_type, attr);
        if (op.ptr == null) {
            @panic("Failed to create arith.constant operation");
        }
        for (custom_attrs) |custom_attr| {
            c.oraOperationSetAttributeByName(op, c.oraIdentifierStr(custom_attr.name), custom_attr.attribute);
        }
        return op;
    }

    /// Create ora.return operation with no return value
    pub fn createFuncReturn(
        self: *OraDialect,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        const op = c.oraReturnOpCreate(self.ctx, loc, null, 0);
        if (op.ptr == null) {
            @panic("Failed to create ora.return operation");
        }
        return op;
    }

    /// Create ora.return operation with return value
    pub fn createFuncReturnWithValue(
        self: *OraDialect,
        return_value: c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        const op = c.oraReturnOpCreate(self.ctx, loc, &[_]c.MlirValue{return_value}, 1);
        if (op.ptr == null) {
            @panic("Failed to create ora.return operation");
        }
        return op;
    }

    /// Create scf.yield operation with no values
    pub fn createScfYield(
        self: *OraDialect,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        const op = c.oraScfYieldOpCreate(self.ctx, loc, null, 0);
        if (op.ptr == null) {
            @panic("Failed to create scf.yield operation");
        }
        return op;
    }

    /// Create scf.yield operation with return values
    pub fn createScfYieldWithValues(
        self: *OraDialect,
        values: []const c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        const values_ptr = if (values.len == 0) null else values.ptr;
        const op = c.oraScfYieldOpCreate(self.ctx, loc, values_ptr, values.len);
        if (op.ptr == null) {
            @panic("Failed to create scf.yield operation");
        }
        return op;
    }

    /// Create cf.br operation (unconditional branch)
    pub fn createCfBr(
        self: *OraDialect,
        dest_block: c.MlirBlock,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        const op = c.oraCfBrOpCreate(self.ctx, loc, dest_block);
        if (op.ptr == null) {
            @panic("Failed to create cf.br operation");
        }
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
        const op = c.oraCfCondBrOpCreate(self.ctx, loc, condition, true_block, false_block);
        if (op.ptr == null) {
            @panic("Failed to create cf.cond_br operation");
        }
        return op;
    }

    /// Create cf.assert operation
    pub fn createCfAssert(
        self: *OraDialect,
        condition: c.MlirValue,
        message: []const u8,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        const msg_ref = c.oraStringRefCreate(message.ptr, message.len);
        const op = c.oraCfAssertOpCreate(self.ctx, loc, condition, msg_ref);
        if (op.ptr == null) {
            @panic("Failed to create cf.assert operation");
        }
        return op;
    }

    /// Create cf.assert operation with custom attributes
    pub fn createCfAssertWithAttrs(
        self: *OraDialect,
        condition: c.MlirValue,
        attrs: []const c.MlirNamedAttribute,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        const op = c.oraCfAssertOpCreateWithAttrs(self.ctx, loc, condition, attrs.ptr, attrs.len);
        if (op.ptr == null) {
            @panic("Failed to create cf.assert operation");
        }
        return op;
    }

    /// Create memref.alloca operation
    pub fn createMemrefAlloca(
        self: *OraDialect,
        memref_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        const op = c.oraMemrefAllocaOpCreate(self.ctx, loc, memref_type);
        if (op.ptr == null) {
            @panic("Failed to create memref.alloca operation");
        }
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
        var positions = std.ArrayList(i64){};
        defer positions.deinit(self.allocator);
        for (indices) |index| {
            positions.append(self.allocator, @intCast(index)) catch unreachable;
        }

        const op = c.oraLlvmExtractValueOpCreate(
            self.ctx,
            loc,
            result_type,
            aggregate,
            if (positions.items.len > 0) positions.items.ptr else null,
            positions.items.len,
        );
        if (op.ptr == null) {
            @panic("Failed to create llvm.extractvalue operation");
        }
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
        const callee_ref = c.oraStringRefCreate(callee.ptr, callee.len);
        const op = c.oraFuncCallOpCreate(
            self.ctx,
            loc,
            callee_ref,
            if (operands.len > 0) operands.ptr else null,
            operands.len,
            if (result_types.len > 0) result_types.ptr else null,
            result_types.len,
        );
        if (op.ptr == null) {
            @panic("Failed to create func.call operation");
        }
        return op;
    }

    /// Create arith.bitcast operation
    pub fn createArithBitcast(
        self: *OraDialect,
        operand: c.MlirValue,
        result_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        const op = c.oraArithBitcastOpCreate(self.ctx, loc, operand, result_type);
        if (op.ptr == null) {
            @panic("Failed to create arith.bitcast operation");
        }
        return op;
    }

    /// Create ora.if operation using C++ API (enables custom assembly formats)
    pub fn createIf(
        self: *OraDialect,
        condition: c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const op = c.oraIfOpCreate(self.ctx, loc, condition);
        if (op.ptr == null) {
            @panic("Failed to create ora.if operation");
        }
        return op;
    }

    // Note: createIsolatedIf removed — unused.

    /// Create scf.if operation
    pub fn createScfIf(
        self: *OraDialect,
        condition: c.MlirValue,
        result_types: []const c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        const result_ptr = if (result_types.len == 0) null else result_types.ptr;
        const op = c.oraScfIfOpCreate(self.ctx, loc, condition, result_ptr, result_types.len, true);
        if (op.ptr == null) {
            @panic("Failed to create scf.if operation");
        }
        return op;
    }

    /// Create ora.while operation using C++ API (enables custom assembly formats)
    pub fn createWhile(
        self: *OraDialect,
        condition: c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const op = c.oraWhileOpCreate(self.ctx, loc, condition);
        if (op.ptr == null) {
            @panic("Failed to create ora.while operation");
        }
        return op;
    }

    // Note: createTest (ora.test) removed — unused test op.

    /// Create scf.while operation (legacy - for compatibility)
    pub fn createScfWhile(
        self: *OraDialect,
        operands: []const c.MlirValue,
        result_types: []const c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        const operands_ptr = if (operands.len == 0) null else operands.ptr;
        const result_ptr = if (result_types.len == 0) null else result_types.ptr;
        const op = c.oraScfWhileOpCreate(self.ctx, loc, operands_ptr, operands.len, result_ptr, result_types.len);
        if (op.ptr == null) {
            @panic("Failed to create scf.while operation");
        }
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
        const init_ptr = if (init_args.len == 0) null else init_args.ptr;
        _ = result_types;
        const op = c.oraScfForOpCreate(self.ctx, loc, lower_bound, upper_bound, step, init_ptr, init_args.len, false);
        if (op.ptr == null) {
            @panic("Failed to create scf.for operation");
        }
        return op;
    }

    /// Create scf.condition operation
    pub fn createScfCondition(
        self: *OraDialect,
        condition: c.MlirValue,
        args: []const c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        const args_ptr = if (args.len == 0) null else args.ptr;
        const op = c.oraScfConditionOpCreate(self.ctx, loc, condition, args_ptr, args.len);
        if (op.ptr == null) {
            @panic("Failed to create scf.condition operation");
        }
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
        const indices_ptr = if (indices.len == 0) @as([*]const c.MlirValue, undefined) else indices.ptr;
        const op = c.oraMemrefLoadOpCreate(self.ctx, loc, memref, indices_ptr, indices.len, result_type);
        if (op.ptr == null) {
            @panic("Failed to create memref.load operation");
        }
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
        const indices_ptr = if (indices.len == 0) @as([*]const c.MlirValue, undefined) else indices.ptr;
        const op = c.oraMemrefStoreOpCreate(self.ctx, loc, value, memref, indices_ptr, indices.len);
        if (op.ptr == null) {
            @panic("Failed to create memref.store operation");
        }
        return op;
    }

    /// Create scf.break operation
    pub fn createScfBreak(
        self: *OraDialect,
        operands: []const c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        const operands_ptr = if (operands.len == 0) null else operands.ptr;
        const op = c.oraScfBreakOpCreate(self.ctx, loc, operands_ptr, operands.len);
        if (op.ptr == null) {
            @panic("Failed to create scf.break operation");
        }
        return op;
    }

    /// Create scf.continue operation
    pub fn createScfContinue(
        self: *OraDialect,
        operands: []const c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        const operands_ptr = if (operands.len == 0) null else operands.ptr;
        const op = c.oraScfContinueOpCreate(self.ctx, loc, operands_ptr, operands.len);
        if (op.ptr == null) {
            @panic("Failed to create scf.continue operation");
        }
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
        _ = result_type;
        const op = c.oraArithAddIOpCreate(self.ctx, loc, lhs, rhs);
        if (op.ptr == null) {
            @panic("Failed to create arith.addi operation");
        }
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
        _ = result_type;
        const op = c.oraArithSubIOpCreate(self.ctx, loc, lhs, rhs);
        if (op.ptr == null) {
            @panic("Failed to create arith.subi operation");
        }
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
        _ = result_type;
        const op = c.oraArithMulIOpCreate(self.ctx, loc, lhs, rhs);
        if (op.ptr == null) {
            @panic("Failed to create arith.muli operation");
        }
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
        _ = result_type;
        const op = c.oraArithDivSIOpCreate(self.ctx, loc, lhs, rhs);
        if (op.ptr == null) {
            @panic("Failed to create arith.divsi operation");
        }
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
        _ = result_type;
        const op = c.oraArithRemSIOpCreate(self.ctx, loc, lhs, rhs);
        if (op.ptr == null) {
            @panic("Failed to create arith.remsi operation");
        }
        return op;
    }

    //===----------------------------------------------------------------------===//
    // financial Operations
    //===----------------------------------------------------------------------===//

    /// Create ora.move operation
    pub fn createMove(
        self: *OraDialect,
        amount: c.MlirValue,
        source: c.MlirValue,
        destination: c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // always use C++ API (dialect must be registered)
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
    // event Operations
    //===----------------------------------------------------------------------===//

    /// Create ora.log operation
    pub fn createLog(
        self: *OraDialect,
        event_name: []const u8,
        parameters: []const c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // always use C++ API (dialect must be registered)
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
    // error Handling Operations
    //===----------------------------------------------------------------------===//

    /// Create ora.error.ok operation
    pub fn createErrorOk(
        self: *OraDialect,
        value: c.MlirValue,
        result_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const op = c.oraErrorOkOpCreate(self.ctx, loc, value, result_type);
        if (op.ptr == null) {
            @panic("Failed to create ora.error.ok operation");
        }
        return op;
    }

    /// Create ora.error.err operation
    pub fn createErrorErr(
        self: *OraDialect,
        value: c.MlirValue,
        result_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const op = c.oraErrorErrOpCreate(self.ctx, loc, value, result_type);
        if (op.ptr == null) {
            @panic("Failed to create ora.error.err operation");
        }
        return op;
    }

    /// Create ora.error.is_error operation
    pub fn createErrorIsError(
        self: *OraDialect,
        value: c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const op = c.oraErrorIsErrorOpCreate(self.ctx, loc, value);
        if (op.ptr == null) {
            @panic("Failed to create ora.error.is_error operation");
        }
        return op;
    }

    /// Create ora.error.unwrap operation
    pub fn createErrorUnwrap(
        self: *OraDialect,
        value: c.MlirValue,
        result_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const op = c.oraErrorUnwrapOpCreate(self.ctx, loc, value, result_type);
        if (op.ptr == null) {
            @panic("Failed to create ora.error.unwrap operation");
        }
        return op;
    }

    /// Create ora.error.get_error operation
    pub fn createErrorGetError(
        self: *OraDialect,
        value: c.MlirValue,
        result_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const op = c.oraErrorGetErrorOpCreate(self.ctx, loc, value, result_type);
        if (op.ptr == null) {
            @panic("Failed to create ora.error.get_error operation");
        }
        return op;
    }

    /// Create ora.try_catch operation
    pub fn createTry(
        self: *OraDialect,
        try_operation: c.MlirValue,
        result_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // always use C++ API (dialect must be registered)
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const op = c.oraTryOpCreate(self.ctx, loc, try_operation, result_type);
        if (op.ptr == null) {
            @panic("Failed to create ora.try_catch operation");
        }
        return op;
    }

    /// Create ora.try_stmt operation
    pub fn createTryStmt(
        self: *OraDialect,
        result_types: []const c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        if (!self.isRegistered()) {
            @panic("Ora dialect must be registered before creating operations");
        }
        const op = c.oraTryStmtOpCreate(self.ctx, loc, result_types.ptr, result_types.len);
        if (op.ptr == null) {
            @panic("Failed to create ora.try_stmt operation");
        }
        return op;
    }

    //===----------------------------------------------------------------------===//
    // loop Operations
    //===----------------------------------------------------------------------===//

    // Note: ora.for removed — all for-loops use scf.for directly.

    /// Create ora.break operation
    pub fn createBreak(
        self: *OraDialect,
        label: ?[]const u8,
        values: []const c.MlirValue,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // always use C++ API (dialect must be registered)
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
        // always use C++ API (dialect must be registered)
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
