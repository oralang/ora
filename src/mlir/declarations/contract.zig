// ============================================================================
// Declaration Lowering - Contracts
// ============================================================================

const std = @import("std");
const c = @import("mlir_c_api").c;
const lib = @import("ora_lib");
const h = @import("../helpers.zig");
const LocalVarMap = @import("../symbols.zig").LocalVarMap;
const SymbolTable = @import("../symbols.zig").SymbolTable;
const StorageMap = @import("../memory.zig").StorageMap;
const DeclarationLowerer = @import("mod.zig").DeclarationLowerer;
const helpers = @import("helpers.zig");
const error_handling = @import("../error_handling.zig");

fn lowerContractTypes(self: *const DeclarationLowerer, block: c.MlirBlock, contract: *const lib.ContractNode) void {
    for (contract.body) |child| {
        switch (child) {
            .StructDecl => |struct_decl| {
                const struct_op = self.lowerStruct(&struct_decl);
                h.appendOp(block, struct_op);

                if (self.symbol_table) |st| {
                    if (st.lookupType(struct_decl.name)) |_| {
                        std.debug.print("WARNING: Duplicate struct type: {s}, skipping\n", .{struct_decl.name});
                    } else {
                        const struct_type = self.createStructType(&struct_decl);

                        const allocator = std.heap.page_allocator;
                        if (allocator.alloc(@import("../lower.zig").TypeSymbol.FieldInfo, struct_decl.fields.len)) |fields_slice| {
                            var current_offset: usize = 0;
                            for (struct_decl.fields, 0..) |field, i| {
                                const field_type = self.type_mapper.toMlirType(field.type_info);
                                fields_slice[i] = .{
                                    .name = field.name,
                                    .field_type = field_type,
                                    .offset = current_offset,
                                };
                                current_offset += 32;
                            }

                            const type_symbol = @import("../lower.zig").TypeSymbol{
                                .name = struct_decl.name,
                                .type_kind = .Struct,
                                .mlir_type = struct_type,
                                .fields = fields_slice,
                                .variants = null,
                                .allocator = allocator,
                            };

                            st.addType(struct_decl.name, type_symbol) catch {
                                allocator.free(fields_slice);
                                std.debug.print("ERROR: Failed to register struct type: {s}\n", .{struct_decl.name});
                            };
                        } else |_| {
                            std.debug.print("ERROR: Failed to allocate fields slice for struct: {s}\n", .{struct_decl.name});
                        }
                    }
                }
            },
            .EnumDecl => |enum_decl| {
                const enum_op = self.lowerEnum(&enum_decl);
                h.appendOp(block, enum_op);
            },
            else => {},
        }
    }
}

/// Lower contract declarations with enhanced metadata and inheritance support
pub fn lowerContract(self: *const DeclarationLowerer, contract: *const lib.ContractNode) c.MlirOperation {
    // create the contract operation using C++ API
    const name_ref = c.mlirStringRefCreate(contract.name.ptr, contract.name.len);
    const contract_op = c.oraContractOpCreate(self.ctx, helpers.createFileLocation(self, contract.span), name_ref);
    if (contract_op.ptr == null) {
        @panic("Failed to create ora.contract operation");
    }

    // set additional attributes using C API (attributes are just metadata)
    if (contract.extends) |base_contract| {
        const extends_ref = c.mlirStringRefCreate(base_contract.ptr, base_contract.len);
        const extends_attr = c.mlirStringAttrGet(self.ctx, extends_ref);
        const extends_name = h.strRef("ora.extends");
        c.mlirOperationSetAttributeByName(contract_op, extends_name, extends_attr);
    }

    if (contract.implements.len > 0) {
        // create array attribute for implemented interfaces
        var interface_attrs = std.ArrayList(c.MlirAttribute){};
        defer interface_attrs.deinit(std.heap.page_allocator);

        for (contract.implements) |interface_name| {
            const interface_ref = c.mlirStringRefCreate(interface_name.ptr, interface_name.len);
            const interface_attr = c.mlirStringAttrGet(self.ctx, interface_ref);
            interface_attrs.append(std.heap.page_allocator, interface_attr) catch {};
        }

        const implements_array = c.mlirArrayAttrGet(self.ctx, @intCast(interface_attrs.items.len), interface_attrs.items.ptr);
        const implements_name = h.strRef("ora.implements");
        c.mlirOperationSetAttributeByName(contract_op, implements_name, implements_array);
    }

    // add contract metadata attribute
    const contract_attr = c.mlirBoolAttrGet(self.ctx, 1);
    const contract_name = h.strRef("ora.contract_decl");
    c.mlirOperationSetAttributeByName(contract_op, contract_name, contract_attr);

    // get the body region from the created operation
    const region = c.mlirOperationGetRegion(contract_op, 0);
    const block = c.mlirRegionGetFirstBlock(region);

    // create contract-level symbol management
    var contract_symbol_table = SymbolTable.init(std.heap.page_allocator);
    defer contract_symbol_table.deinit();

    // first pass: collect all storage variables and create a shared StorageMap
    var storage_map = StorageMap.init(std.heap.page_allocator);
    defer storage_map.deinit();

    for (contract.body) |child| {
        switch (child) {
            .VariableDecl => |var_decl| {
                // include ghost variables in storage map for verification
                // they will be filtered out during target code generation (not in bytecode)
                switch (var_decl.region) {
                    .Storage => {
                        // this is a storage variable - add it to the storage map and symbol table
                        // ghost variables are included for verification purposes
                        _ = storage_map.getOrCreateAddress(var_decl.name) catch {};
                        // add to contract symbol table for member variable tracking
                        const var_type = self.type_mapper.toMlirType(var_decl.type_info);
                        contract_symbol_table.addSymbol(var_decl.name, var_type, var_decl.region, null, var_decl.kind) catch {};
                        // also populate the actual symbol table used by expression/statement lowerers
                        if (self.symbol_table) |st| {
                            st.addSymbol(var_decl.name, var_type, var_decl.region, null, var_decl.kind) catch {};
                        }
                    },
                    .Memory => {
                        // memory variables are allocated in memory space
                    },
                    .TStore => {
                        // transient storage variables are allocated in transient storage space
                    },
                    .Stack => {
                        // stack variables at contract level are not allowed in Ora
                        std.debug.print("WARNING: Stack variable at contract level: {s}\n", .{var_decl.name});
                    },
                }
            },
            .Function => |_| {
                // functions are processed in the second pass - skip in first pass
                // this avoids creating operations before the state is fully configured
            },
            .ContractInvariant => {
                // skip contract invariants (specification-only)
            },
            else => {},
        }
    }

    // second pass: first register all struct/enum types, then process functions and variables
    // this ensures types are available when functions use them
    lowerContractTypes(self, block, contract);

    // third pass: process functions and variables (after types are registered)
    for (contract.body) |child| {
        switch (child) {
            .Function => |f| {
                // include ghost functions in MLIR for verification
                // they will be filtered out during target code generation (not in bytecode)
                var local_var_map = LocalVarMap.init(std.heap.page_allocator);
                defer local_var_map.deinit();
                const func_op = self.lowerFunction(&f, &storage_map, &local_var_map);
                h.appendOp(block, func_op);
            },
            .VariableDecl => |var_decl| {
                // include ghost variables in MLIR for verification
                // they will be filtered out during target code generation (not in bytecode)

                switch (var_decl.region) {
                    .Storage => {
                        // create ora.global operation for storage variables (directly in contract body)
                        const global_op = self.createGlobalDeclaration(&var_decl);
                        h.appendOp(block, global_op);
                    },
                    .Memory => {
                        // create ora.memory.global operation for memory variables
                        const memory_global_op = self.createMemoryGlobalDeclaration(&var_decl);
                        h.appendOp(block, memory_global_op);
                    },
                    .TStore => {
                        // create ora.tstore.global operation for transient storage variables
                        const tstore_global_op = self.createTStoreGlobalDeclaration(&var_decl);
                        h.appendOp(block, tstore_global_op);
                    },
                    .Stack => {
                        // stack variables at contract level are not allowed
                        // this should have been caught in the first pass
                    },
                }
            },
            .StructDecl, .EnumDecl => {
                // structs/enums are lowered and registered in the second pass.
            },
            .LogDecl => |log_decl| {
                // lower log declarations within contract
                const log_op = self.lowerLogDecl(&log_decl);
                h.appendOp(block, log_op);
            },
            .ErrorDecl => |error_decl| {
                // lower error declarations within contract
                const error_op = self.lowerErrorDecl(&error_decl);
                h.appendOp(block, error_op);

                // register error in symbol table so it can be referenced
                if (self.symbol_table) |st| {
                    const error_type = self.createErrorType(&error_decl);
                    st.addError(error_decl.name, error_type, null) catch {
                        std.debug.print("ERROR: Failed to add error to symbol table: {s}\n", .{error_decl.name});
                    };
                }
            },
            .Constant => |const_decl| {
                // lower constant declarations within contract
                // lower the constant declaration operation (for metadata/documentation)
                const const_op = self.lowerConstDecl(&const_decl);
                h.appendOp(block, const_op);

                // register constant declaration for lazy value creation
                // constants are created lazily when referenced to avoid cross-region issues
                if (self.symbol_table) |st| {
                    const const_type = self.type_mapper.toMlirType(const_decl.typ);
                    // register the constant declaration so we can create its value when referenced
                    st.registerConstantDecl(const_decl.name, &const_decl) catch {
                        std.debug.print("ERROR: Failed to register constant declaration: {s}\n", .{const_decl.name});
                    };
                    // add to symbol table with null value - will be created lazily when referenced
                    st.addConstant(const_decl.name, const_type, null, null) catch {
                        std.debug.print("ERROR: Failed to add constant to symbol table: {s}\n", .{const_decl.name});
                    };
                }
            },
            .ContractInvariant => {
                // skip contract invariants (specification-only)
            },
            else => {
                // report missing node type with context and continue processing
                if (self.error_handler) |eh| {
                    eh.reportMissingNodeType(@tagName(child), error_handling.getSpanFromAstNode(&child), "contract body") catch {};
                } else {
                    std.debug.print("WARNING: Unhandled contract body node type in MLIR lowering: {s}\n", .{@tagName(child)});
                }
            },
        }
    }

    // return the contract operation (already created via C++ API)
    return contract_op;
}
