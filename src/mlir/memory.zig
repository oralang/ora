// ============================================================================
// Memory Management
// ============================================================================
//
// Manages memory regions and storage allocation for MLIR lowering.
//
// KEY COMPONENTS:
//   • StorageMap: Contract storage variable → address mapping
//   • MemoryManager: Memory region tracking (storage/memory/stack)
//   • Allocation strategies for different memory types
//
// MEMORY REGIONS:
//   • Storage: Persistent contract state (blockchain storage)
//   • Memory: Temporary heap-like memory
//   • Stack: Function-local variables
//
// ============================================================================

const std = @import("std");
const c = @import("c.zig").c;
const lib = @import("ora_lib");
const constants = @import("lower.zig");
const h = @import("helpers.zig"); // Import helpers

// Storage variable mapping for contract storage
pub const StorageMap = struct {
    variables: std.StringHashMap(usize), // variable name -> storage address
    next_address: usize,

    pub fn init(allocator: std.mem.Allocator) StorageMap {
        return .{
            .variables = std.StringHashMap(usize).init(allocator),
            .next_address = 0,
        };
    }

    pub fn deinit(self: *StorageMap) void {
        self.variables.deinit();
    }

    pub fn getOrCreateAddress(self: *StorageMap, name: []const u8) !usize {
        if (self.variables.get(name)) |addr| {
            return addr;
        }
        const addr = self.next_address;
        try self.variables.put(name, addr);
        self.next_address += 1;
        return addr;
    }

    pub fn hasStorageVariable(self: *const StorageMap, name: []const u8) bool {
        return self.variables.contains(name);
    }

    pub fn addStorageVariable(self: *StorageMap, name: []const u8, _: lib.ast.SourceSpan) !usize {
        const addr = try self.getOrCreateAddress(name);
        return addr;
    }
};

/// Memory region management system for Ora storage types
pub const MemoryManager = struct {
    ctx: c.MlirContext,
    ora_dialect: *@import("dialect.zig").OraDialect,

    pub fn init(ctx: c.MlirContext, ora_dialect: *@import("dialect.zig").OraDialect) MemoryManager {
        return .{
            .ctx = ctx,
            .ora_dialect = ora_dialect,
        };
    }

    /// Get memory space mapping: storage=1, memory=0, tstore=2
    pub fn getMemorySpace(self: *const MemoryManager, storage_type: lib.ast.Statements.MemoryRegion) u32 {
        _ = self; // Context not needed for this function
        return switch (storage_type) {
            .Storage => 1, // storage=1
            .Memory => 0, // memory=0
            .TStore => 2, // tstore=2
            .Stack => 0, // stack=0 (default to memory space)
        };
    }

    /// Get memory space as MLIR attribute
    pub fn getMemorySpaceAttribute(self: *const MemoryManager, storage_type: lib.ast.Statements.MemoryRegion) c.MlirAttribute {
        const space_value = self.getMemorySpace(storage_type);
        return c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), @intCast(space_value));
    }

    /// Create region attribute for attaching `ora.region` attributes
    pub fn createRegionAttribute(self: *const MemoryManager, storage_type: lib.ast.Statements.MemoryRegion) c.MlirAttribute {
        const region_str = switch (storage_type) {
            .Storage => "storage",
            .Memory => "memory",
            .TStore => "tstore",
            .Stack => "stack",
        };
        const region_ref = c.mlirStringRefCreate(region_str.ptr, region_str.len);
        return c.mlirStringAttrGet(self.ctx, region_ref);
    }

    /// Create allocation operation for variables in correct memory spaces
    pub fn createAllocaOp(self: *const MemoryManager, var_type: c.MlirType, storage_type: lib.ast.Statements.MemoryRegion, var_name: []const u8, loc: c.MlirLocation) c.MlirOperation {
        switch (storage_type) {
            .Storage => {
                // Storage variables use ora.global operations, not alloca
                return self.createGlobalStorageDeclaration(var_name, var_type, loc);
            },
            .Memory => {
                // Memory variables use memref.alloca with memory space 0
                // TODO: Create proper memref type with memory space attribute
                // For now, we'll use the simple helper without attributes
                // TODO: Add support for attributes in the dialect helper
                return self.ora_dialect.createMemrefAlloca(var_type, loc);
            },
            .TStore => {
                // Transient storage variables use ora.tstore.global operations
                return self.createGlobalTStoreDeclaration(var_name, var_type, loc);
            },
            .Stack => {
                // Stack variables use regular memref.alloca
                return self.ora_dialect.createMemrefAlloca(var_type, loc);
            },
        }
    }

    /// Create store operation with memory space semantics
    pub fn createStoreOp(self: *const MemoryManager, value: c.MlirValue, address: c.MlirValue, storage_type: lib.ast.Statements.MemoryRegion, loc: c.MlirLocation) c.MlirOperation {
        switch (storage_type) {
            .Storage => {
                // Storage uses ora.sstore - address should be variable name
                std.debug.print("ERROR: Use createStorageStore for storage variables\n", .{});
                // Create a placeholder error operation
                var state = h.opState("ora.error", loc);
                const error_ty = c.mlirIntegerTypeGet(self.ctx, 32);
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&error_ty));
                return c.mlirOperationCreate(&state);
            },
            .Memory => {
                // Memory uses memref.store with memory space 0
                var state = h.opState("memref.store", loc);
                c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ value, address }));

                // Add memory space attribute
                const space_attr = self.getMemorySpaceAttribute(storage_type);
                const space_id = h.identifier(self.ctx, "memspace");
                var attrs = [_]c.MlirNamedAttribute{
                    c.mlirNamedAttributeGet(space_id, space_attr),
                };
                c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

                return c.mlirOperationCreate(&state);
            },
            .TStore => {
                // Transient storage uses ora.tstore
                std.debug.print("ERROR: Use createTStoreStore for transient storage variables\n", .{});
                // Create a placeholder error operation
                var state = h.opState("ora.error", loc);
                const error_ty = c.mlirIntegerTypeGet(self.ctx, 32);
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&error_ty));
                return c.mlirOperationCreate(&state);
            },
            .Stack => {
                // Stack uses regular memref.store
                const indices = [_]c.MlirValue{};
                return self.ora_dialect.createMemrefStore(value, address, &indices, loc);
            },
        }
    }

    /// Create load operation with memory space semantics
    pub fn createLoadOp(self: *const MemoryManager, address: c.MlirValue, storage_type: lib.ast.Statements.MemoryRegion, result_type: c.MlirType, loc: c.MlirLocation) c.MlirOperation {
        switch (storage_type) {
            .Storage => {
                // Storage uses ora.sload - address should be variable name
                std.debug.print("ERROR: Use createStorageLoad for storage variables\n", .{});
                // Create a placeholder error operation
                var state = h.opState("ora.error", loc);
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_type));
                return c.mlirOperationCreate(&state);
            },
            .Memory => {
                // Memory uses memref.load with memory space 0
                var state = h.opState("memref.load", loc);
                c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&address));
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_type));

                // Add memory space attribute
                const space_attr = self.getMemorySpaceAttribute(storage_type);
                const space_id = h.identifier(self.ctx, "memspace");
                var attrs = [_]c.MlirNamedAttribute{
                    c.mlirNamedAttributeGet(space_id, space_attr),
                };
                c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

                return c.mlirOperationCreate(&state);
            },
            .TStore => {
                // Transient storage uses ora.tload
                std.debug.print("ERROR: Use createTStoreLoad for transient storage variables\n", .{});
                // Create a placeholder error operation
                var state = h.opState("ora.error", loc);
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_type));
                return c.mlirOperationCreate(&state);
            },
            .Stack => {
                // Stack uses regular memref.load
                const indices = [_]c.MlirValue{};
                return self.ora_dialect.createMemrefLoad(address, &indices, result_type, loc);
            },
        }
    }

    /// Create storage load operation (ora.sload)
    pub fn createStorageLoad(self: *const MemoryManager, var_name: []const u8, result_type: c.MlirType, loc: c.MlirLocation) c.MlirOperation {
        return self.ora_dialect.createSLoad(var_name, result_type, loc);
    }

    /// Create memory load operation (ora.mload)
    pub fn createMemoryLoad(self: *const MemoryManager, var_name: []const u8, loc: c.MlirLocation) c.MlirOperation {
        var state = h.opState("ora.mload", loc);

        // Add the variable name as an attribute
        const name_ref = c.mlirStringRefCreate(var_name.ptr, var_name.len);
        const name_attr = c.mlirStringAttrGet(self.ctx, name_ref);
        const name_id = h.identifier(self.ctx, "name");
        var attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(name_id, name_attr),
        };
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        // Add result type (default to i256 for now)
        const result_ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));

        return c.mlirOperationCreate(&state);
    }

    /// Create transient storage load operation (ora.tload)
    pub fn createTStoreLoad(self: *const MemoryManager, var_name: []const u8, loc: c.MlirLocation) c.MlirOperation {
        var state = h.opState("ora.tload", loc);

        // Add the global name as a symbol reference
        var name_buffer: [256]u8 = undefined;
        for (0..var_name.len) |i| {
            name_buffer[i] = var_name[i];
        }
        name_buffer[var_name.len] = 0; // null-terminate
        const name_str = c.mlirStringRefCreateFromCString(&name_buffer[0]);
        const name_attr = c.mlirStringAttrGet(self.ctx, name_str);
        const name_id = h.identifier(self.ctx, "global");
        var attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(name_id, name_attr),
        };
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        // Add result type (default to i256 for now)
        const result_ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));

        return c.mlirOperationCreate(&state);
    }

    /// Create storage store operation (ora.sstore)
    pub fn createStorageStore(self: *const MemoryManager, value: c.MlirValue, var_name: []const u8, loc: c.MlirLocation) c.MlirOperation {
        return self.ora_dialect.createSStore(value, var_name, loc);
    }

    /// Create memory store operation (ora.mstore)
    pub fn createMemoryStore(self: *const MemoryManager, value: c.MlirValue, var_name: []const u8, loc: c.MlirLocation) c.MlirOperation {
        var state = h.opState("ora.mstore", loc);
        c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&value));

        // Add the variable name as an attribute
        const name_ref = c.mlirStringRefCreate(var_name.ptr, var_name.len);
        const name_attr = c.mlirStringAttrGet(self.ctx, name_ref);
        const name_id = h.identifier(self.ctx, "name");
        var attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(name_id, name_attr),
        };
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        return c.mlirOperationCreate(&state);
    }

    /// Create transient storage store operation (ora.tstore)
    pub fn createTStoreStore(self: *const MemoryManager, value: c.MlirValue, var_name: []const u8, loc: c.MlirLocation) c.MlirOperation {
        var state = h.opState("ora.tstore", loc);
        c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&value));

        // Add the global name as a symbol reference
        var name_buffer: [256]u8 = undefined;
        for (0..var_name.len) |i| {
            name_buffer[i] = var_name[i];
        }
        name_buffer[var_name.len] = 0; // null-terminate
        const name_str = c.mlirStringRefCreateFromCString(&name_buffer[0]);
        const name_attr = c.mlirStringAttrGet(self.ctx, name_str);
        const name_id = h.identifier(self.ctx, "global");
        var attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(name_id, name_attr),
        };
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        return c.mlirOperationCreate(&state);
    }

    /// Create memref type with appropriate memory space
    fn createMemRefType(self: *const MemoryManager, element_type: c.MlirType, storage_type: lib.ast.Statements.MemoryRegion) c.MlirType {
        _ = self; // Context not used in this simplified implementation
        _ = storage_type; // Storage type not used in this simplified implementation
        // For now, create a simple memref type
        // In the future, this could handle dynamic shapes and strides
        // Note: This is a simplified implementation - actual memref type creation
        // would require more complex MLIR API calls
        return element_type;
    }

    /// Get element type from memref type
    fn getMemRefElementType(self: *const MemoryManager, memref_type: c.MlirType) c.MlirType {
        _ = self; // Context not used in this simplified implementation
        // For now, return the type as-is
        // In the future, this would extract the element type from the memref
        return memref_type;
    }

    /// Create load operation for different storage types
    pub fn createLoadOperation(self: *const MemoryManager, var_name: []const u8, storage_type: lib.ast.Statements.MemoryRegion, span: lib.ast.SourceSpan) c.MlirOperation {
        const loc = self.createFileLocation(span);

        switch (storage_type) {
            .Storage => {
                // Generate ora.sload for storage variables
                var state = h.opState("ora.sload", loc);

                // Add the global name as a symbol reference
                var name_buffer: [256]u8 = undefined;
                for (0..var_name.len) |i| {
                    name_buffer[i] = var_name[i];
                }
                name_buffer[var_name.len] = 0; // null-terminate
                const name_str = c.mlirStringRefCreateFromCString(&name_buffer[0]);
                const name_attr = c.mlirStringAttrGet(self.ctx, name_str);
                const name_id = h.identifier(self.ctx, "global");
                var attrs = [_]c.MlirNamedAttribute{
                    c.mlirNamedAttributeGet(name_id, name_attr),
                };
                c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

                // Add result type (default to i256 for now)
                const result_ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));

                return c.mlirOperationCreate(&state);
            },
            .Memory => {
                // Generate ora.mload for memory variables
                var state = h.opState("ora.mload", loc);

                // Add the variable name as an attribute
                const name_ref = c.mlirStringRefCreate(var_name.ptr, var_name.len);
                const name_attr = c.mlirStringAttrGet(self.ctx, name_ref);
                const name_id = h.identifier(self.ctx, "name");
                var attrs = [_]c.MlirNamedAttribute{
                    c.mlirNamedAttributeGet(name_id, name_attr),
                };
                c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

                // Add result type (default to i256 for now)
                const result_ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));

                return c.mlirOperationCreate(&state);
            },
            .TStore => {
                // Generate ora.tload for transient storage variables
                var state = h.opState("ora.tload", loc);

                // Add the global name as a symbol reference
                var name_buffer: [256]u8 = undefined;
                for (0..var_name.len) |i| {
                    name_buffer[i] = var_name[i];
                }
                name_buffer[var_name.len] = 0; // null-terminate
                const name_str = c.mlirStringRefCreateFromCString(&name_buffer[0]);
                const name_attr = c.mlirStringAttrGet(self.ctx, name_str);
                const name_id = h.identifier(self.ctx, "global");
                var attrs = [_]c.MlirNamedAttribute{
                    c.mlirNamedAttributeGet(name_id, name_attr),
                };
                c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

                // Add result type (default to i256 for now)
                const result_ty = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS);
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));

                return c.mlirOperationCreate(&state);
            },
            .Stack => {
                // For stack variables, we return the value directly from our local variable map
                // This is handled differently in the identifier lowering
                std.debug.print("ERROR: Stack variables should not use createLoadOperation\n", .{});
                // Create a placeholder error operation
                var state = h.opState("ora.error", loc);
                const error_ty = c.mlirIntegerTypeGet(self.ctx, 32);
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&error_ty));
                return c.mlirOperationCreate(&state);
            },
        }
    }

    /// Create store operation for different storage types
    pub fn createStoreOperation(self: *const MemoryManager, value: c.MlirValue, var_name: []const u8, storage_type: lib.ast.Statements.MemoryRegion, span: lib.ast.SourceSpan) c.MlirOperation {
        const loc = self.createFileLocation(span);

        switch (storage_type) {
            .Storage => {
                // Generate ora.sstore for storage variables
                var state = h.opState("ora.sstore", loc);
                c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&value));

                // Add the global name as a symbol reference
                var name_buffer: [256]u8 = undefined;
                for (0..var_name.len) |i| {
                    name_buffer[i] = var_name[i];
                }
                name_buffer[var_name.len] = 0; // null-terminate
                const name_str = c.mlirStringRefCreateFromCString(&name_buffer[0]);
                const name_attr = c.mlirStringAttrGet(self.ctx, name_str);
                const name_id = h.identifier(self.ctx, "global");
                var attrs = [_]c.MlirNamedAttribute{
                    c.mlirNamedAttributeGet(name_id, name_attr),
                };
                c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

                return c.mlirOperationCreate(&state);
            },
            .Memory => {
                // Generate ora.mstore for memory variables
                var state = h.opState("ora.mstore", loc);
                c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&value));

                // Add the variable name as an attribute
                const name_ref = c.mlirStringRefCreate(var_name.ptr, var_name.len);
                const name_attr = c.mlirStringAttrGet(self.ctx, name_ref);
                const name_id = h.identifier(self.ctx, "name");
                var attrs = [_]c.MlirNamedAttribute{
                    c.mlirNamedAttributeGet(name_id, name_attr),
                };
                c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

                return c.mlirOperationCreate(&state);
            },
            .TStore => {
                // Generate ora.tstore for transient storage variables
                var state = h.opState("ora.tstore", loc);
                c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&value));

                // Add the global name as a symbol reference
                var name_buffer: [256]u8 = undefined;
                for (0..var_name.len) |i| {
                    name_buffer[i] = var_name[i];
                }
                name_buffer[var_name.len] = 0; // null-terminate
                const name_str = c.mlirStringRefCreateFromCString(&name_buffer[0]);
                const name_attr = c.mlirStringAttrGet(self.ctx, name_str);
                const name_id = h.identifier(self.ctx, "global");
                var attrs = [_]c.MlirNamedAttribute{
                    c.mlirNamedAttributeGet(name_id, name_attr),
                };
                c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

                return c.mlirOperationCreate(&state);
            },
            .Stack => {
                // For stack variables, we store the value directly in our local variable map
                // This is handled differently in the assignment lowering
                std.debug.print("ERROR: Stack variables should not use createStoreOperation\n", .{});
                // Create a placeholder error operation
                var state = h.opState("ora.error", loc);
                const error_ty = c.mlirIntegerTypeGet(self.ctx, 32);
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&error_ty));
                return c.mlirOperationCreate(&state);
            },
        }
    }

    /// Create file location for operations
    fn createFileLocation(self: *const MemoryManager, span: lib.ast.SourceSpan) c.MlirLocation {
        return c.mlirLocationFileLineColGet(self.ctx, h.strRefLit("input.ora"), span.line, span.column);
    }

    /// Create global storage declaration
    fn createGlobalStorageDeclaration(self: *const MemoryManager, var_name: []const u8, var_type: c.MlirType, loc: c.MlirLocation) c.MlirOperation {
        var state = h.opState("ora.global", loc);

        // Add variable name as symbol attribute
        const name_ref = c.mlirStringRefCreate(var_name.ptr, var_name.len);
        const name_attr = c.mlirStringAttrGet(self.ctx, name_ref);
        const name_id = h.identifier(self.ctx, "sym_name");

        // Add type attribute
        const type_attr = c.mlirTypeAttrGet(var_type);
        const type_id = h.identifier(self.ctx, "type");

        var attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(name_id, name_attr),
            c.mlirNamedAttributeGet(type_id, type_attr),
        };
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        return c.mlirOperationCreate(&state);
    }

    /// Create global transient storage declaration
    fn createGlobalTStoreDeclaration(self: *const MemoryManager, var_name: []const u8, var_type: c.MlirType, loc: c.MlirLocation) c.MlirOperation {
        var state = h.opState("ora.tstore.global", loc);

        // Add variable name as symbol attribute
        const name_ref = c.mlirStringRefCreate(var_name.ptr, var_name.len);
        const name_attr = c.mlirStringAttrGet(self.ctx, name_ref);
        const name_id = h.identifier(self.ctx, "sym_name");

        // Add type attribute
        const type_attr = c.mlirTypeAttrGet(var_type);
        const type_id = h.identifier(self.ctx, "type");

        var attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(name_id, name_attr),
            c.mlirNamedAttributeGet(type_id, type_attr),
        };
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        return c.mlirOperationCreate(&state);
    }

    /// Validate memory region constraints
    pub fn validateMemoryAccess(self: *const MemoryManager, region: lib.ast.Statements.MemoryRegion, access_type: AccessType) bool {
        _ = self; // Context not needed for validation

        switch (region) {
            .Storage => {
                // Storage variables can be read and written
                return access_type == .Read or access_type == .Write;
            },
            .Memory => {
                // Memory variables can be read and written
                return access_type == .Read or access_type == .Write;
            },
            .TStore => {
                // Transient storage variables can be read and written
                return access_type == .Read or access_type == .Write;
            },
            .Stack => {
                // Stack variables can be read and written
                return access_type == .Read or access_type == .Write;
            },
        }
    }

    /// Check if a memory region is persistent
    pub fn isPersistent(self: *const MemoryManager, region: lib.ast.Statements.MemoryRegion) bool {
        _ = self;
        return switch (region) {
            .Storage => true, // Storage is persistent across transactions
            .TStore => true, // Transient storage is persistent within transaction
            .Memory => false, // Memory is cleared between calls
            .Stack => false, // Stack is function-local
        };
    }

    /// Check if a memory region requires gas for access
    pub fn requiresGas(self: *const MemoryManager, region: lib.ast.Statements.MemoryRegion) bool {
        _ = self;
        return switch (region) {
            .Storage => true, // Storage access costs gas
            .TStore => true, // Transient storage access costs gas
            .Memory => false, // Memory access is free
            .Stack => false, // Stack access is free
        };
    }
};

/// Access type for memory validation
pub const AccessType = enum {
    Read,
    Write,
};
