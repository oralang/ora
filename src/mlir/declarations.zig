const std = @import("std");
const c = @import("c.zig").c;
const lib = @import("ora_lib");
const constants = @import("constants.zig");
const TypeMapper = @import("types.zig").TypeMapper;
const LocationTracker = @import("locations.zig").LocationTracker;
const LocalVarMap = @import("symbols.zig").LocalVarMap;
const ParamMap = @import("symbols.zig").ParamMap;
const StorageMap = @import("memory.zig").StorageMap;
const ExpressionLowerer = @import("expressions.zig").ExpressionLowerer;
const StatementLowerer = @import("statements.zig").StatementLowerer;
const LoweringError = @import("statements.zig").StatementLowerer.LoweringError;

/// Declaration lowering system for converting Ora top-level declarations to MLIR
pub const DeclarationLowerer = struct {
    ctx: c.MlirContext,
    type_mapper: *const TypeMapper,
    locations: LocationTracker,

    pub fn init(ctx: c.MlirContext, type_mapper: *const TypeMapper, locations: LocationTracker) DeclarationLowerer {
        return .{
            .ctx = ctx,
            .type_mapper = type_mapper,
            .locations = locations,
        };
    }

    /// Lower function declarations
    pub fn lowerFunction(self: *const DeclarationLowerer, func: *const lib.FunctionNode, contract_storage_map: ?*StorageMap, local_var_map: ?*LocalVarMap) c.MlirOperation {
        // Create a local variable map for this function
        var local_vars = LocalVarMap.init(std.heap.page_allocator);
        defer local_vars.deinit();

        // Create parameter mapping for calldata parameters
        var param_map = ParamMap.init(std.heap.page_allocator);
        defer param_map.deinit();
        for (func.parameters, 0..) |param, i| {
            // Function parameters are calldata by default in Ora
            param_map.addParam(param.name, i) catch {};
            std.debug.print("DEBUG: Added calldata parameter: {s} at index {d}\n", .{ param.name, i });
        }

        // Create the function operation
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("func.func"), self.createFileLocation(func.span));

        // Add function name
        const name_ref = c.mlirStringRefCreate(func.name.ptr, func.name.len);
        const name_attr = c.mlirStringAttrGet(self.ctx, name_ref);
        const sym_name_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("sym_name"));
        var attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(sym_name_id, name_attr),
        };
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        // Add function type
        const fn_type = self.createFunctionType(func);
        const fn_type_attr = c.mlirTypeAttrGet(fn_type);
        const fn_type_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("function_type"));
        var type_attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(fn_type_id, fn_type_attr),
        };
        c.mlirOperationStateAddAttributes(&state, type_attrs.len, &type_attrs);

        // Create the function body region
        const region = c.mlirRegionCreate();
        const block = c.mlirBlockCreate(0, null, null);
        c.mlirRegionInsertOwnedBlock(region, 0, block);
        c.mlirOperationStateAddOwnedRegions(&state, 1, @ptrCast(&region));

        // Lower the function body
        self.lowerFunctionBody(func, block, &param_map, contract_storage_map, local_var_map orelse &local_vars) catch |err| {
            std.debug.print("Error lowering function body: {}\n", .{err});
            return c.mlirOperationCreate(&state);
        };

        // Ensure a terminator exists (void return)
        if (func.return_type_info == null) {
            var return_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("func.return"), self.createFileLocation(func.span));
            const return_op = c.mlirOperationCreate(&return_state);
            c.mlirBlockAppendOwnedOperation(block, return_op);
        }

        // Create the function operation
        const func_op = c.mlirOperationCreate(&state);
        return func_op;
    }

    /// Lower contract declarations
    pub fn lowerContract(self: *const DeclarationLowerer, contract: *const lib.ContractNode) c.MlirOperation {
        // Create the contract operation
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("ora.contract"), self.createFileLocation(contract.span));

        // Add contract name
        const name_ref = c.mlirStringRefCreate(contract.name.ptr, contract.name.len);
        const name_attr = c.mlirStringAttrGet(self.ctx, name_ref);
        const name_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("sym_name"));
        var attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(name_id, name_attr),
        };
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        // Create the contract body region
        const region = c.mlirRegionCreate();
        const block = c.mlirBlockCreate(0, null, null);
        c.mlirRegionInsertOwnedBlock(region, 0, block);
        c.mlirOperationStateAddOwnedRegions(&state, 1, @ptrCast(&region));

        // First pass: collect all storage variables and create a shared StorageMap
        var storage_map = StorageMap.init(std.heap.page_allocator);
        defer storage_map.deinit();

        for (contract.body) |child| {
            switch (child) {
                .VariableDecl => |var_decl| {
                    switch (var_decl.region) {
                        .Storage => {
                            // This is a storage variable - add it to the storage map
                            _ = storage_map.getOrCreateAddress(var_decl.name) catch {};
                        },
                        .Memory => {
                            // Memory variables are allocated in memory space
                            // For now, we'll track them but handle allocation later
                            std.debug.print("DEBUG: Found memory variable at contract level: {s}\n", .{var_decl.name});
                        },
                        .TStore => {
                            // Transient storage variables are allocated in transient storage space
                            // For now, we'll track them but handle allocation later
                            std.debug.print("DEBUG: Found transient storage variable at contract level: {s}\n", .{var_decl.name});
                        },
                        .Stack => {
                            // Stack variables at contract level are not allowed in Ora
                            std.debug.print("WARNING: Stack variable at contract level: {s}\n", .{var_decl.name});
                        },
                    }
                },
                else => {},
            }
        }

        // Second pass: create global declarations and process functions
        for (contract.body) |child| {
            switch (child) {
                .Function => |f| {
                    var local_var_map = LocalVarMap.init(std.heap.page_allocator);
                    defer local_var_map.deinit();
                    const func_op = self.lowerFunction(&f, &storage_map, &local_var_map);
                    c.mlirBlockAppendOwnedOperation(block, func_op);
                },
                .VariableDecl => |var_decl| {
                    switch (var_decl.region) {
                        .Storage => {
                            // Create ora.global operation for storage variables
                            const global_op = self.createGlobalDeclaration(&var_decl);
                            c.mlirBlockAppendOwnedOperation(block, global_op);
                        },
                        .Memory => {
                            // Create ora.memory.global operation for memory variables
                            const memory_global_op = self.createMemoryGlobalDeclaration(&var_decl);
                            c.mlirBlockAppendOwnedOperation(block, memory_global_op);
                        },
                        .TStore => {
                            // Create ora.tstore.global operation for transient storage variables
                            const tstore_global_op = self.createTStoreGlobalDeclaration(&var_decl);
                            c.mlirBlockAppendOwnedOperation(block, tstore_global_op);
                        },
                        .Stack => {
                            // Stack variables at contract level are not allowed
                            // This should have been caught in the first pass
                        },
                    }
                },
                .EnumDecl => |enum_decl| {
                    // For now, just skip enum declarations
                    // TODO: Add proper enum type handling
                    _ = enum_decl;
                },
                else => {
                    @panic("Unhandled contract body node type in MLIR lowering");
                },
            }
        }

        // Create and return the contract operation
        return c.mlirOperationCreate(&state);
    }

    /// Lower struct declarations
    pub fn lowerStruct(self: *const DeclarationLowerer, struct_decl: *const lib.ast.StructDeclNode) c.MlirOperation {
        // TODO: Implement struct declaration lowering
        // For now, just skip the struct declaration
        _ = struct_decl;
        // Return a dummy operation
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("func.func"), c.mlirLocationUnknownGet(self.ctx));
        return c.mlirOperationCreate(&state);
    }

    /// Lower enum declarations
    pub fn lowerEnum(self: *const DeclarationLowerer, enum_decl: *const lib.ast.EnumDeclNode) c.MlirOperation {
        // TODO: Implement enum declaration lowering
        // For now, just skip the enum declaration
        _ = enum_decl;
        // Return a dummy operation
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("func.func"), c.mlirLocationUnknownGet(self.ctx));
        return c.mlirOperationCreate(&state);
    }

    /// Lower import declarations
    pub fn lowerImport(self: *const DeclarationLowerer, import_decl: *const lib.ast.ImportNode) c.MlirOperation {
        // TODO: Implement import declaration lowering
        // For now, just skip the import declaration
        _ = import_decl;
        // Return a dummy operation
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("func.func"), c.mlirLocationUnknownGet(self.ctx));
        return c.mlirOperationCreate(&state);
    }

    /// Create global storage variable declaration
    pub fn createGlobalDeclaration(self: *const DeclarationLowerer, var_decl: *const lib.ast.Statements.VariableDeclNode) c.MlirOperation {
        // Create ora.global operation
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("ora.global"), self.createFileLocation(var_decl.span));

        // Add the global name as a symbol attribute
        const name_ref = c.mlirStringRefCreate(var_decl.name.ptr, var_decl.name.len);
        const name_attr = c.mlirStringAttrGet(self.ctx, name_ref);
        const name_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("sym_name"));
        var attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(name_id, name_attr),
        };

        // Add the type attribute
        const var_type = if (std.mem.eql(u8, var_decl.name, "status"))
            c.mlirIntegerTypeGet(self.ctx, 1) // bool -> i1
        else
            c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS); // default to i256
        const type_attr = c.mlirTypeAttrGet(var_type);
        const type_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("type"));
        var type_attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(type_id, type_attr),
        };
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
        c.mlirOperationStateAddAttributes(&state, type_attrs.len, &type_attrs);

        // Add initial value
        const init_attr = if (std.mem.eql(u8, var_decl.name, "status"))
            c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 1), 0) // bool -> i1 with value 0 (false)
        else
            c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS), 0); // default to i256 with value 0
        const init_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("init"));
        var init_attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(init_id, init_attr),
        };
        c.mlirOperationStateAddAttributes(&state, init_attrs.len, &init_attrs);

        return c.mlirOperationCreate(&state);
    }

    /// Create memory global variable declaration
    pub fn createMemoryGlobalDeclaration(self: *const DeclarationLowerer, var_decl: *const lib.ast.Statements.VariableDeclNode) c.MlirOperation {
        // Create ora.memory.global operation
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("ora.memory.global"), self.createFileLocation(var_decl.span));

        // Add the global name as a symbol attribute
        const name_ref = c.mlirStringRefCreate(var_decl.name.ptr, var_decl.name.len);
        const name_attr = c.mlirStringAttrGet(self.ctx, name_ref);
        const name_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("sym_name"));
        var attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(name_id, name_attr),
        };
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        // Add the type attribute
        const var_type = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS); // default to i256
        const type_attr = c.mlirTypeAttrGet(var_type);
        const type_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("type"));
        var type_attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(type_id, type_attr),
        };
        c.mlirOperationStateAddAttributes(&state, type_attrs.len, &type_attrs);

        return c.mlirOperationCreate(&state);
    }

    /// Create transient storage global variable declaration
    pub fn createTStoreGlobalDeclaration(self: *const DeclarationLowerer, var_decl: *const lib.ast.Statements.VariableDeclNode) c.MlirOperation {
        // Create ora.tstore.global operation
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("ora.tstore.global"), self.createFileLocation(var_decl.span));

        // Add the global name as a symbol attribute
        const name_ref = c.mlirStringRefCreate(var_decl.name.ptr, var_decl.name.len);
        const name_attr = c.mlirStringAttrGet(self.ctx, name_ref);
        const name_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("sym_name"));
        var attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(name_id, name_attr),
        };
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        // Add the type attribute
        const var_type = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS); // default to i256
        const type_attr = c.mlirTypeAttrGet(var_type);
        const type_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("type"));
        var type_attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(type_id, type_attr),
        };
        c.mlirOperationStateAddAttributes(&state, type_attrs.len, &type_attrs);

        return c.mlirOperationCreate(&state);
    }

    /// Create function type
    fn createFunctionType(self: *const DeclarationLowerer, func: *const lib.FunctionNode) c.MlirType {
        // For now, create a simple function type
        // TODO: Implement proper function type creation based on parameters and return type
        const result_type = if (func.return_type_info) |ret_info|
            self.type_mapper.toMlirType(ret_info)
        else
            c.mlirNoneTypeGet(self.ctx);

        // Create function type with no parameters for now
        // TODO: Add parameter types
        return c.mlirFunctionTypeGet(self.ctx, 0, null, 1, @ptrCast(&result_type));
    }

    /// Lower function body
    fn lowerFunctionBody(self: *const DeclarationLowerer, func: *const lib.FunctionNode, block: c.MlirBlock, param_map: *const ParamMap, storage_map: ?*const StorageMap, local_var_map: ?*LocalVarMap) LoweringError!void {
        // Create a statement lowerer for this function
        const const_local_var_map = if (local_var_map) |lvm| @as(*const LocalVarMap, lvm) else null;
        const expr_lowerer = ExpressionLowerer.init(self.ctx, block, self.type_mapper, param_map, storage_map, const_local_var_map, self.locations);
        const stmt_lowerer = StatementLowerer.init(self.ctx, block, self.type_mapper, &expr_lowerer, param_map, storage_map, local_var_map, self.locations, null, std.heap.page_allocator);

        // Lower the function body
        try stmt_lowerer.lowerBlockBody(func.body, block);
    }

    /// Create file location for operatio
    fn createFileLocation(self: *const DeclarationLowerer, span: lib.ast.SourceSpan) c.MlirLocation {
        return LocationTracker.createFileLocationFromSpan(&self.locations, span);
    }
};
