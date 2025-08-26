const std = @import("std");
const c = @import("c.zig");

pub const Dialect = struct {
    ctx: c.MlirContext,

    pub fn init(ctx: c.MlirContext) Dialect {
        return Dialect{ .ctx = ctx };
    }

    pub fn register(self: *Dialect) void {
        // Register the Ora dialect with MLIR
        // For now, we'll use the existing MLIR operations but structure them as Ora dialect
        _ = self;
    }

    // Helper function to create ora.global operation
    pub fn createGlobal(
        self: *Dialect,
        name: []const u8,
        value_type: c.MlirType,
        init_value: c.MlirAttribute,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // Create a global variable declaration
        // This will be equivalent to ora.global @name : type = init_value
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("ora.global"), loc);

        // Add the global name as a symbol attribute
        const name_attr = c.mlirStringAttrGet(self.ctx, c.mlirStringRefCreateFromCString(name.ptr));
        const name_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("sym_name"));
        var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(name_id, name_attr)};
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        // Add the type and initial value
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&value_type));

        // Add the initial value attribute
        const init_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("init"));
        var init_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(init_id, init_value)};
        c.mlirOperationStateAddAttributes(&state, init_attrs.len, &init_attrs);

        const op = c.mlirOperationCreate(&state);
        return op;
    }

    // Helper function to create ora.load operation
    pub fn createLoad(
        self: *Dialect,
        global_name: []const u8,
        result_type: c.MlirType,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // Create a load operation from a global
        // This will be equivalent to ora.load @global_name : result_type
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("ora.load"), loc);

        // Add the global name as a symbol reference
        const name_attr = c.mlirStringAttrGet(self.ctx, c.mlirStringRefCreateFromCString(global_name.ptr));
        const name_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("global"));
        var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(name_id, name_attr)};
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        // Add the result type
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_type));

        const op = c.mlirOperationCreate(&state);
        return op;
    }

    // Helper function to create ora.store operation
    pub fn createStore(
        self: *Dialect,
        value: c.MlirValue,
        global_name: []const u8,
        loc: c.MlirLocation,
    ) c.MlirOperation {
        // Create a store operation to a global
        // This will be equivalent to ora.store %value, @global_name
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("ora.store"), loc);

        // Add the value operand
        c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&value));

        // Add the global name as a symbol reference
        const name_attr = c.mlirStringAttrGet(self.ctx, c.mlirStringRefCreateFromCString(global_name.ptr));
        const name_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("global"));
        var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(name_id, name_attr)};
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        const op = c.mlirOperationCreate(&state);
        return op;
    }
};
