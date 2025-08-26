const std = @import("std");
const c = @import("c.zig").c;
const lib = @import("ora_lib");

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
};

/// Memory region management system for Ora storage types
pub const MemoryManager = struct {
    ctx: c.MlirContext,

    pub fn init(ctx: c.MlirContext) MemoryManager {
        return .{ .ctx = ctx };
    }

    /// Get memory space for different storage types
    pub fn getMemorySpace(self: *const MemoryManager, storage_type: lib.ast.Statements.MemoryRegion) c.MlirAttribute {
        return switch (storage_type) {
            .Storage => c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), 1), // storage=1
            .Memory => c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), 0), // memory=0
            .TStore => c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), 2), // tstore=2
            .Stack => c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 64), 0), // stack=0 (default to memory)
        };
    }

    /// Create region attribute for attaching to operations
    pub fn createRegionAttribute(self: *const MemoryManager, storage_type: lib.ast.Statements.MemoryRegion) c.MlirAttribute {
        const space = self.getMemorySpace(storage_type);
        // For now, return the memory space directly
        // In the future, this could create a proper region attribute
        return space;
    }

    /// Create allocation operation for variables
    pub fn createAllocaOp(self: *const MemoryManager, var_type: c.MlirType, storage_type: []const u8, var_name: []const u8) c.MlirOperation {
        _ = var_type;
        _ = storage_type;
        _ = var_name;
        // TODO: Implement allocation operation creation
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("memref.alloca"), c.mlirLocationUnknownGet(self.ctx));
        return c.mlirOperationCreate(&state);
    }

    /// Create store operation with memory space semantics
    pub fn createStoreOp(self: *const MemoryManager, value: c.MlirValue, address: c.MlirValue, storage_type: []const u8) c.MlirOperation {
        _ = value;
        _ = address;
        _ = storage_type;
        // TODO: Implement store operation creation with memory space
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("memref.store"), c.mlirLocationUnknownGet(self.ctx));
        return c.mlirOperationCreate(&state);
    }

    /// Create load operation with memory space semantics
    pub fn createLoadOp(self: *const MemoryManager, address: c.MlirValue, storage_type: []const u8) c.MlirOperation {
        _ = address;
        _ = storage_type;
        // TODO: Implement load operation creation with memory space
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("memref.load"), c.mlirLocationUnknownGet(self.ctx));
        return c.mlirOperationCreate(&state);
    }

    /// Create storage load operation (ora.sload)
    pub fn createStorageLoad(self: *const MemoryManager, global_name: []const u8, result_type: c.MlirType, loc: c.MlirLocation) c.MlirOperation {
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("ora.sload"), loc);

        // Add the result type
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_type));

        // Add the global name as a symbol reference
        const name_ref = c.mlirStringRefCreate(global_name.ptr, global_name.len);
        const name_attr = c.mlirStringAttrGet(self.ctx, name_ref);
        const name_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("global"));
        var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(name_id, name_attr)};
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        return c.mlirOperationCreate(&state);
    }

    /// Create storage store operation (ora.sstore)
    pub fn createStorageStore(self: *const MemoryManager, value: c.MlirValue, global_name: []const u8, loc: c.MlirLocation) c.MlirOperation {
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("ora.sstore"), loc);

        // Add the value operand
        c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&value));

        // Add the global name as a symbol reference
        const name_ref = c.mlirStringRefCreate(global_name.ptr, global_name.len);
        const name_attr = c.mlirStringAttrGet(self.ctx, name_ref);
        const name_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("global"));
        var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(name_id, name_attr)};
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

    /// Create storage-type-aware load operations
    pub fn createLoadOperation(self: *const MemoryManager, var_name: []const u8, storage_type: lib.ast.Statements.MemoryRegion, span: lib.ast.SourceSpan) c.MlirOperation {
        switch (storage_type) {
            .Storage => {
                // Generate ora.sload for storage variables
                var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("ora.sload"), self.fileLoc(span));

                // Add the global name as a symbol reference
                var name_buffer: [256]u8 = undefined;
                for (0..var_name.len) |i| {
                    name_buffer[i] = var_name[i];
                }
                name_buffer[var_name.len] = 0; // null-terminate
                const name_str = c.mlirStringRefCreateFromCString(&name_buffer[0]);
                const name_attr = c.mlirStringAttrGet(self.ctx, name_str);
                const name_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("global"));
                const attrs = [_]c.MlirNamedAttribute{
                    c.mlirNamedAttributeGet(name_id, name_attr),
                };
                c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

                // Add result type (default to i256 for now)
                const result_ty = c.mlirIntegerTypeGet(self.ctx, 256);
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));

                return c.mlirOperationCreate(&state);
            },
            .Memory => {
                // Generate ora.mload for memory variables
                var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("ora.mload"), self.fileLoc(span));

                // Add the variable name as an attribute
                const name_ref = c.mlirStringRefCreate(var_name.ptr, var_name.len);
                const name_attr = c.mlirStringAttrGet(self.ctx, name_ref);
                const name_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("name"));
                const attrs = [_]c.MlirNamedAttribute{
                    c.mlirNamedAttributeGet(name_id, name_attr),
                };
                c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

                // Add result type (default to i256 for now)
                const result_ty = c.mlirIntegerTypeGet(self.ctx, 256);
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));

                return c.mlirOperationCreate(&state);
            },
            .TStore => {
                // Generate ora.tload for transient storage variables
                var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("ora.tload"), self.fileLoc(span));

                // Add the global name as a symbol reference
                var name_buffer: [256]u8 = undefined;
                for (0..var_name.len) |i| {
                    name_buffer[i] = var_name[i];
                }
                name_buffer[var_name.len] = 0; // null-terminate
                const name_str = c.mlirStringRefCreateFromCString(&name_buffer[0]);
                const name_attr = c.mlirStringAttrGet(self.ctx, name_str);
                const name_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("global"));
                const attrs = [_]c.MlirNamedAttribute{
                    c.mlirNamedAttributeGet(name_id, name_attr),
                };
                c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

                // Add result type (default to i256 for now)
                const result_ty = c.mlirIntegerTypeGet(self.ctx, 256);
                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));

                return c.mlirOperationCreate(&state);
            },
            .Stack => {
                // For stack variables, we return the value directly from our local variable map
                // This is handled differently in the identifier lowering
                @panic("Stack variables should not use createLoadOperation");
            },
        }
    }

    /// Helper function to create file location
    fn fileLoc(self: *const MemoryManager, span: lib.ast.SourceSpan) c.MlirLocation {
        const fname = c.mlirStringRefCreateFromCString("input.ora");
        return c.mlirLocationFileLineColGet(self.ctx, fname, span.line, span.column);
    }
};
