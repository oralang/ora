const std = @import("std");
const c = @import("c.zig").c;
const lib = @import("ora_lib");

/// Declaration lowering system for converting Ora top-level declarations to MLIR
pub const DeclarationLowerer = struct {
    ctx: c.MlirContext,
    module: c.MlirModule,
    type_mapper: *const @import("types.zig").TypeMapper,

    pub fn init(ctx: c.MlirContext, module: c.MlirModule, type_mapper: *const @import("types.zig").TypeMapper) DeclarationLowerer {
        return .{
            .ctx = ctx,
            .module = module,
            .type_mapper = type_mapper,
        };
    }

    /// Lower function declarations
    pub fn lowerFunction(self: *const DeclarationLowerer, func: *const lib.FunctionNode) c.MlirOperation {
        // TODO: Implement function declaration lowering with visibility modifiers
        // For now, just skip the function declaration
        _ = func;
        // Return a dummy operation
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("func.func"), c.mlirLocationUnknownGet(self.ctx));
        return c.mlirOperationCreate(&state);
    }

    /// Lower contract declarations
    pub fn lowerContract(self: *const DeclarationLowerer, contract: *const lib.ContractNode) c.MlirOperation {
        // TODO: Implement contract declaration lowering
        // For now, just skip the contract declaration
        _ = contract;
        // Return a dummy operation
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("func.func"), c.mlirLocationUnknownGet(self.ctx));
        return c.mlirOperationCreate(&state);
    }

    /// Lower struct declarations
    pub fn lowerStruct(self: *const DeclarationLowerer, struct_decl: *const lib.ast.Declarations.StructDeclNode) c.MlirOperation {
        // TODO: Implement struct declaration lowering
        // For now, just skip the struct declaration
        _ = struct_decl;
        // Return a dummy operation
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("func.func"), c.mlirLocationUnknownGet(self.ctx));
        return c.mlirOperationCreate(&state);
    }

    /// Lower enum declarations
    pub fn lowerEnum(self: *const DeclarationLowerer, enum_decl: *const lib.ast.Declarations.EnumDeclNode) c.MlirOperation {
        // TODO: Implement enum declaration lowering
        // For now, just skip the enum declaration
        _ = enum_decl;
        // Return a dummy operation
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("func.func"), c.mlirLocationUnknownGet(self.ctx));
        return c.mlirOperationCreate(&state);
    }

    /// Lower import declarations
    pub fn lowerImport(self: *const DeclarationLowerer, import_decl: *const lib.ast.Declarations.ImportDeclNode) c.MlirOperation {
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
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("ora.global"), c.mlirLocationUnknownGet(self.ctx));

        // Add the global name as a symbol attribute
        const name_ref = c.mlirStringRefCreate(var_decl.name.ptr, var_decl.name.len);
        const name_attr = c.mlirStringAttrGet(self.ctx, name_ref);
        const name_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("sym_name"));
        var attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(name_id, name_attr),
        };
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        // Add the type attribute
        const var_type = if (std.mem.eql(u8, var_decl.name, "status"))
            c.mlirIntegerTypeGet(self.ctx, 1) // bool -> i1
        else
            c.mlirIntegerTypeGet(self.ctx, 256); // default to i256
        const type_attr = c.mlirTypeAttrGet(var_type);
        const type_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("type"));
        var type_attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(type_id, type_attr),
        };
        c.mlirOperationStateAddAttributes(&state, type_attrs.len, &type_attrs);

        // Add initial value if present
        if (var_decl.value) |_| {
            const init_attr = if (std.mem.eql(u8, var_decl.name, "status"))
                c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 1), 0) // bool -> i1 with value 0 (false)
            else
                c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 256), 0); // default to i256 with value 0
            const init_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("init"));
            var init_attrs = [_]c.MlirNamedAttribute{
                c.mlirNamedAttributeGet(init_id, init_attr),
            };
            c.mlirOperationStateAddAttributes(&state, init_attrs.len, &init_attrs);
        }

        return c.mlirOperationCreate(&state);
    }

    /// Create global memory variable declaration
    pub fn createMemoryGlobalDeclaration(self: *const DeclarationLowerer, var_decl: *const lib.ast.Statements.VariableDeclNode) c.MlirOperation {
        // Create ora.memory.global operation
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("ora.memory.global"), c.mlirLocationUnknownGet(self.ctx));

        // Add the global name as a symbol attribute
        const name_ref = c.mlirStringRefCreate(var_decl.name.ptr, var_decl.name.len);
        const name_attr = c.mlirStringAttrGet(self.ctx, name_ref);
        const name_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("sym_name"));
        var attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(name_id, name_attr),
        };
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        // Add the type attribute
        const var_type = c.mlirIntegerTypeGet(self.ctx, 256); // default to i256
        const type_attr = c.mlirTypeAttrGet(var_type);
        const type_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("type"));
        var type_attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(type_id, type_attr),
        };
        c.mlirOperationStateAddAttributes(&state, type_attrs.len, &type_attrs);

        return c.mlirOperationCreate(&state);
    }

    /// Create global transient storage variable declaration
    pub fn createTStoreGlobalDeclaration(self: *const DeclarationLowerer, var_decl: *const lib.ast.Statements.VariableDeclNode) c.MlirOperation {
        // Create ora.tstore.global operation
        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("ora.tstore.global"), c.mlirLocationUnknownGet(self.ctx));

        // Add the global name as a symbol attribute
        const name_ref = c.mlirStringRefCreate(var_decl.name.ptr, var_decl.name.len);
        const name_attr = c.mlirStringAttrGet(self.ctx, name_ref);
        const name_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("sym_name"));
        var attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(name_id, name_attr),
        };
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        // Add the type attribute
        const var_type = c.mlirIntegerTypeGet(self.ctx, 256); // default to i256
        const type_attr = c.mlirTypeAttrGet(var_type);
        const type_id = c.mlirIdentifierGet(self.ctx, c.mlirStringRefCreateFromCString("type"));
        var type_attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(type_id, type_attr),
        };
        c.mlirOperationStateAddAttributes(&state, type_attrs.len, &type_attrs);

        return c.mlirOperationCreate(&state);
    }
};
