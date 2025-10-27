// ============================================================================
// Declaration Lowering
// ============================================================================
//
// Converts Ora top-level declarations to MLIR module-level constructs.
//
// SUPPORTED DECLARATIONS:
//   • Functions with parameters, return types, and bodies
//   • Contracts with member functions and storage variables
//   • Structs with field definitions
//   • Enums with variant definitions
//   • Constants and global variables
//   • Logs (event declarations)
//   • Error type declarations
//
// FEATURES:
//   • Symbol table construction
//   • Function signature generation
//   • ABI generation for public functions
//   • Contract storage layout
//   • Type definition registration
//
// ============================================================================

const std = @import("std");
const c = @import("c.zig").c;
const lib = @import("ora_lib");
const constants = @import("lower.zig");
const h = @import("helpers.zig");
const TypeMapper = @import("types.zig").TypeMapper;
const LocationTracker = @import("locations.zig").LocationTracker;
const LocalVarMap = @import("symbols.zig").LocalVarMap;
const ParamMap = @import("symbols.zig").ParamMap;
const SymbolTable = @import("symbols.zig").SymbolTable;
const StorageMap = @import("memory.zig").StorageMap;
const ExpressionLowerer = @import("expressions.zig").ExpressionLowerer;
const StatementLowerer = @import("statements.zig").StatementLowerer;
const LoweringError = @import("statements.zig").StatementLowerer.LoweringError;
const error_handling = @import("error_handling.zig");

/// Declaration lowering system for converting Ora top-level declarations to MLIR
pub const DeclarationLowerer = struct {
    ctx: c.MlirContext,
    type_mapper: *const TypeMapper,
    locations: LocationTracker,
    error_handler: ?*const @import("error_handling.zig").ErrorHandler,
    ora_dialect: *@import("dialect.zig").OraDialect,
    symbol_table: ?*@import("lower.zig").SymbolTable = null,
    builtin_registry: ?*const lib.semantics.builtins.BuiltinRegistry = null,

    pub fn init(ctx: c.MlirContext, type_mapper: *const TypeMapper, locations: LocationTracker, ora_dialect: *@import("dialect.zig").OraDialect) DeclarationLowerer {
        return .{
            .ctx = ctx,
            .type_mapper = type_mapper,
            .locations = locations,
            .error_handler = null,
            .ora_dialect = ora_dialect,
        };
    }

    pub fn withErrorHandler(ctx: c.MlirContext, type_mapper: *const TypeMapper, locations: LocationTracker, error_handler: *const @import("error_handling.zig").ErrorHandler, ora_dialect: *@import("dialect.zig").OraDialect) DeclarationLowerer {
        return .{
            .ctx = ctx,
            .type_mapper = type_mapper,
            .locations = locations,
            .error_handler = error_handler,
            .ora_dialect = ora_dialect,
        };
    }

    pub fn withErrorHandlerAndDialect(ctx: c.MlirContext, type_mapper: *const TypeMapper, locations: LocationTracker, error_handler: *const @import("error_handling.zig").ErrorHandler, ora_dialect: *@import("dialect.zig").OraDialect) DeclarationLowerer {
        return .{
            .ctx = ctx,
            .type_mapper = type_mapper,
            .locations = locations,
            .error_handler = error_handler,
            .ora_dialect = ora_dialect,
        };
    }

    pub fn withErrorHandlerAndDialectAndSymbolTable(ctx: c.MlirContext, type_mapper: *const TypeMapper, locations: LocationTracker, error_handler: *const @import("error_handling.zig").ErrorHandler, ora_dialect: *@import("dialect.zig").OraDialect, symbol_table: *@import("lower.zig").SymbolTable, builtin_registry: ?*const lib.semantics.builtins.BuiltinRegistry) DeclarationLowerer {
        return .{
            .ctx = ctx,
            .type_mapper = type_mapper,
            .locations = locations,
            .error_handler = error_handler,
            .ora_dialect = ora_dialect,
            .symbol_table = symbol_table,
            .builtin_registry = builtin_registry,
        };
    }

    /// Lower function declarations with enhanced features
    pub fn lowerFunction(self: *const DeclarationLowerer, func: *const lib.FunctionNode, contract_storage_map: ?*StorageMap, local_var_map: ?*LocalVarMap) c.MlirOperation {
        // Create a local variable map for this function
        var local_vars = LocalVarMap.init(std.heap.page_allocator);
        defer local_vars.deinit();

        // Create parameter mapping for calldata parameters
        var param_map = ParamMap.init(std.heap.page_allocator);
        defer param_map.deinit();

        // Collect parameter types for MLIR block arguments
        var param_types_buf: [16]c.MlirType = undefined; // Support up to 16 parameters
        const param_types = if (func.parameters.len <= 16) param_types_buf[0..func.parameters.len] else blk: {
            break :blk std.heap.page_allocator.alloc(c.MlirType, func.parameters.len) catch {
                std.debug.print("FATAL: Failed to allocate parameter types\n", .{});
                @panic("Allocation failure");
            };
        };
        defer if (func.parameters.len > 16) std.heap.page_allocator.free(param_types);

        for (func.parameters, 0..) |param, i| {
            // Function parameters are calldata by default in Ora
            param_map.addParam(param.name, i) catch {};

            // Get MLIR type for parameter
            const param_type = self.type_mapper.toMlirType(param.type_info);
            param_types[i] = param_type;
        }

        // Create the function operation
        var state = h.opState("func.func", self.createFileLocation(func.span));

        // Add function name
        const name_ref = c.mlirStringRefCreate(func.name.ptr, func.name.len);
        const name_attr = c.mlirStringAttrGet(self.ctx, name_ref);
        const sym_name_id = h.identifier(self.ctx, "sym_name");

        // Collect all function attributes
        var attributes = std.ArrayList(c.MlirNamedAttribute){};
        defer attributes.deinit(std.heap.page_allocator);

        // Add function name attribute
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(sym_name_id, name_attr)) catch {};

        // Add visibility modifier attribute (Requirements 6.1)
        const visibility_attr = switch (func.visibility) {
            .Public => c.mlirStringAttrGet(self.ctx, h.strRef("pub")),
            .Private => c.mlirStringAttrGet(self.ctx, h.strRef("private")),
        };
        const visibility_id = h.identifier(self.ctx, "ora.visibility");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(visibility_id, visibility_attr)) catch {};

        // Add inline function attribute (Requirements 6.2)
        if (func.is_inline) {
            const inline_attr = c.mlirBoolAttrGet(self.ctx, 1);
            const inline_id = h.identifier(self.ctx, "ora.inline");
            attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(inline_id, inline_attr)) catch {};
        }

        // Add special function name attributes (Requirements 6.8)
        if (std.mem.eql(u8, func.name, "init")) {
            const init_attr = c.mlirBoolAttrGet(self.ctx, 1);
            const init_id = h.identifier(self.ctx, "ora.init");
            attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(init_id, init_attr)) catch {};
        }

        // Add comprehensive verification metadata for function contracts (Requirements 6.4, 6.5)
        if (func.requires_clauses.len > 0 or func.ensures_clauses.len > 0) {
            // Add verification marker for formal verification tools
            const verification_attr = c.mlirBoolAttrGet(self.ctx, 1);
            const verification_id = h.identifier(self.ctx, "ora.verification");
            attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(verification_id, verification_attr)) catch {};

            // Add formal verification marker
            const formal_attr = c.mlirBoolAttrGet(self.ctx, 1);
            const formal_id = h.identifier(self.ctx, "ora.formal");
            attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(formal_id, formal_attr)) catch {};

            // Add verification context attribute
            const context_attr = c.mlirStringAttrGet(self.ctx, h.strRef("function_contract"));
            const context_id = h.identifier(self.ctx, "ora.verification_context");
            attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(context_id, context_attr)) catch {};

            // Add requires clauses count
            if (func.requires_clauses.len > 0) {
                const requires_count_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 32), @intCast(func.requires_clauses.len));
                const requires_count_id = h.identifier(self.ctx, "ora.requires_count");
                attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(requires_count_id, requires_count_attr)) catch {};
            }

            // Add ensures clauses count
            if (func.ensures_clauses.len > 0) {
                const ensures_count_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 32), @intCast(func.ensures_clauses.len));
                const ensures_count_id = h.identifier(self.ctx, "ora.ensures_count");
                attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(ensures_count_id, ensures_count_attr)) catch {};
            }

            // Add contract verification level
            const contract_level_attr = c.mlirStringAttrGet(self.ctx, h.strRef("full"));
            const contract_level_id = h.identifier(self.ctx, "ora.contract_level");
            attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(contract_level_id, contract_level_attr)) catch {};
        }

        // Add function type
        const fn_type = self.createFunctionType(func);
        const fn_type_attr = c.mlirTypeAttrGet(fn_type);
        const fn_type_id = h.identifier(self.ctx, "function_type");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(fn_type_id, fn_type_attr)) catch {};

        // Apply all attributes to the operation state
        c.mlirOperationStateAddAttributes(&state, @intCast(attributes.items.len), attributes.items.ptr);

        // Create the function body region with block arguments for parameters
        const region = c.mlirRegionCreate();

        // Create locations for block arguments
        var param_locs_buf: [16]c.MlirLocation = undefined;
        const param_locs = if (func.parameters.len <= 16) param_locs_buf[0..func.parameters.len] else blk: {
            break :blk std.heap.page_allocator.alloc(c.MlirLocation, func.parameters.len) catch {
                std.debug.print("FATAL: Failed to allocate parameter locations\n", .{});
                @panic("Allocation failure");
            };
        };
        defer if (func.parameters.len > 16) std.heap.page_allocator.free(param_locs);

        for (func.parameters, 0..) |param, i| {
            param_locs[i] = self.createFileLocation(param.span);
        }

        const block = c.mlirBlockCreate(@intCast(param_types.len), param_types.ptr, param_locs.ptr);
        c.mlirRegionInsertOwnedBlock(region, 0, block);
        c.mlirOperationStateAddOwnedRegions(&state, 1, @ptrCast(&region));

        // Map parameter names to block arguments
        for (func.parameters, 0..) |param, i| {
            const block_arg = c.mlirBlockGetArgument(block, @intCast(i));
            param_map.setBlockArgument(param.name, block_arg) catch {};
        }

        // Add precondition assertions for requires clauses (Requirements 6.4)
        if (func.requires_clauses.len > 0) {
            self.lowerRequiresClauses(func.requires_clauses, block, &param_map, contract_storage_map, local_var_map orelse &local_vars) catch |err| {
                std.debug.print("Error lowering requires clauses: {s}\n", .{@errorName(err)});
            };
        }

        // Lower the function body
        self.lowerFunctionBody(func, block, &param_map, contract_storage_map, local_var_map orelse &local_vars) catch |err| {
            std.debug.print("Error lowering function body: {s}\n", .{@errorName(err)});

            // Report to error handler if available
            if (self.error_handler) |eh| {
                var error_handler = @constCast(eh);
                error_handler.reportError(.MlirOperationFailed, func.span, "Failed to lower function body", "check function implementation for type errors and unsupported operations") catch {};
            }

            return c.mlirOperationCreate(&state);
        };

        // Add postcondition assertions for ensures clauses (Requirements 6.5)
        if (func.ensures_clauses.len > 0) {
            self.lowerEnsuresClauses(func.ensures_clauses, block, &param_map, contract_storage_map, local_var_map orelse &local_vars) catch |err| {
                std.debug.print("Error lowering ensures clauses: {s}\n", .{@errorName(err)});
            };
        }

        // Ensure a terminator exists (void return)
        if (func.return_type_info == null) {
            const return_op = self.ora_dialect.createFuncReturn(self.createFileLocation(func.span));
            h.appendOp(block, return_op);
        }

        // Create the function operation
        const func_op = c.mlirOperationCreate(&state);
        return func_op;
    }

    /// Lower contract declarations with enhanced metadata and inheritance support (Requirements 6.7)
    pub fn lowerContract(self: *const DeclarationLowerer, contract: *const lib.ContractNode) c.MlirOperation {
        // Create the contract operation
        var state = h.opState("ora.contract", self.createFileLocation(contract.span));

        // Collect contract attributes
        var attributes = std.ArrayList(c.MlirNamedAttribute){};
        defer attributes.deinit(std.heap.page_allocator);

        // Add contract name
        const name_ref = c.mlirStringRefCreate(contract.name.ptr, contract.name.len);
        const name_attr = c.mlirStringAttrGet(self.ctx, name_ref);
        const name_id = h.identifier(self.ctx, "sym_name");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(name_id, name_attr)) catch {};

        // Add inheritance information if present
        if (contract.extends) |base_contract| {
            const extends_ref = c.mlirStringRefCreate(base_contract.ptr, base_contract.len);
            const extends_attr = c.mlirStringAttrGet(self.ctx, extends_ref);
            const extends_id = h.identifier(self.ctx, "ora.extends");
            attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(extends_id, extends_attr)) catch {};
        }

        // Add interface implementation information
        if (contract.implements.len > 0) {
            // Create array attribute for implemented interfaces
            var interface_attrs = std.ArrayList(c.MlirAttribute){};
            defer interface_attrs.deinit(std.heap.page_allocator);

            for (contract.implements) |interface_name| {
                const interface_ref = c.mlirStringRefCreate(interface_name.ptr, interface_name.len);
                const interface_attr = c.mlirStringAttrGet(self.ctx, interface_ref);
                interface_attrs.append(std.heap.page_allocator, interface_attr) catch {};
            }

            const implements_array = c.mlirArrayAttrGet(self.ctx, @intCast(interface_attrs.items.len), interface_attrs.items.ptr);
            const implements_id = h.identifier(self.ctx, "ora.implements");
            attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(implements_id, implements_array)) catch {};
        }

        // Add contract metadata attributes
        const contract_attr = c.mlirBoolAttrGet(self.ctx, 1);
        const contract_id = h.identifier(self.ctx, "ora.contract_decl");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(contract_id, contract_attr)) catch {};

        // Apply all attributes
        c.mlirOperationStateAddAttributes(&state, @intCast(attributes.items.len), attributes.items.ptr);

        // Create the contract body region
        const region = c.mlirRegionCreate();
        const block = c.mlirBlockCreate(0, null, null);
        c.mlirRegionInsertOwnedBlock(region, 0, block);

        // Create ora.storage region for storage variables
        const storage_region = c.mlirRegionCreate();
        const storage_block = c.mlirBlockCreate(0, null, null);
        c.mlirRegionInsertOwnedBlock(storage_region, 0, storage_block);

        // Create ora.storage operation
        var storage_state = h.opState("ora.storage", self.createFileLocation(contract.span));
        const storage_region_for_op = c.mlirRegionCreate();
        const storage_block_for_op = c.mlirBlockCreate(0, null, null);
        c.mlirRegionInsertOwnedBlock(storage_region_for_op, 0, storage_block_for_op);
        c.mlirOperationStateAddOwnedRegions(&storage_state, 1, @ptrCast(&storage_region_for_op));
        const storage_op = c.mlirOperationCreate(&storage_state);

        // Add both regions to the contract operation
        const regions = [_]c.MlirRegion{ region, storage_region };
        c.mlirOperationStateAddOwnedRegions(&state, 2, @ptrCast(&regions));

        // Add storage operation to the main contract block
        h.appendOp(block, storage_op);

        // Create contract-level symbol management
        var contract_symbol_table = SymbolTable.init(std.heap.page_allocator);
        defer contract_symbol_table.deinit();

        // First pass: collect all storage variables and create a shared StorageMap
        var storage_map = StorageMap.init(std.heap.page_allocator);
        defer storage_map.deinit();

        for (contract.body) |child| {
            switch (child) {
                .VariableDecl => |var_decl| {
                    // Skip ghost variables (specification-only)
                    if (var_decl.is_ghost) continue;

                    switch (var_decl.region) {
                        .Storage => {
                            // This is a storage variable - add it to the storage map and symbol table
                            _ = storage_map.getOrCreateAddress(var_decl.name) catch {};
                            // Add to contract symbol table for member variable tracking
                            const var_type = self.type_mapper.toMlirType(var_decl.type_info);
                            contract_symbol_table.addSymbol(var_decl.name, var_type, var_decl.region, null) catch {};
                        },
                        .Memory => {
                            // Memory variables are allocated in memory space

                        },
                        .TStore => {
                            // Transient storage variables are allocated in transient storage space

                        },
                        .Stack => {
                            // Stack variables at contract level are not allowed in Ora
                            std.debug.print("WARNING: Stack variable at contract level: {s}\n", .{var_decl.name});
                        },
                    }
                },
                .Function => |_| {
                    // Functions are processed in the second pass - skip in first pass
                    // This avoids creating operations before the state is fully configured
                },
                .ContractInvariant => {
                    // Skip contract invariants (specification-only)
                },
                else => {},
            }
        }

        // Second pass: create global declarations and process functions
        for (contract.body) |child| {
            switch (child) {
                .Function => |f| {
                    // Skip ghost functions (specification-only)
                    if (f.is_ghost) continue;

                    var local_var_map = LocalVarMap.init(std.heap.page_allocator);
                    defer local_var_map.deinit();
                    const func_op = self.lowerFunction(&f, &storage_map, &local_var_map);
                    h.appendOp(block, func_op);
                },
                .VariableDecl => |var_decl| {
                    // Skip ghost variables (specification-only)
                    if (var_decl.is_ghost) continue;

                    switch (var_decl.region) {
                        .Storage => {
                            // Create ora.global operation for storage variables in storage region
                            const global_op = self.createGlobalDeclaration(&var_decl);
                            h.appendOp(storage_block_for_op, global_op);
                        },
                        .Memory => {
                            // Create ora.memory.global operation for memory variables
                            const memory_global_op = self.createMemoryGlobalDeclaration(&var_decl);
                            h.appendOp(block, memory_global_op);
                        },
                        .TStore => {
                            // Create ora.tstore.global operation for transient storage variables
                            const tstore_global_op = self.createTStoreGlobalDeclaration(&var_decl);
                            h.appendOp(block, tstore_global_op);
                        },
                        .Stack => {
                            // Stack variables at contract level are not allowed
                            // This should have been caught in the first pass
                        },
                    }
                },
                .StructDecl => |struct_decl| {
                    // Lower struct declarations within contract
                    const struct_op = self.lowerStruct(&struct_decl);
                    h.appendOp(block, struct_op);
                },
                .EnumDecl => |enum_decl| {
                    // Lower enum declarations within contract
                    const enum_op = self.lowerEnum(&enum_decl);
                    h.appendOp(block, enum_op);
                },
                .LogDecl => |log_decl| {
                    // Lower log declarations within contract
                    const log_op = self.lowerLogDecl(&log_decl);
                    h.appendOp(block, log_op);
                },
                .ErrorDecl => |error_decl| {
                    // Lower error declarations within contract
                    const error_op = self.lowerErrorDecl(&error_decl);
                    h.appendOp(block, error_op);
                },
                .ContractInvariant => {
                    // Skip contract invariants (specification-only)
                },
                else => {
                    // Report missing node type with context and continue processing
                    if (self.error_handler) |eh| {
                        // Create a mutable copy for error reporting
                        var error_handler = @constCast(eh);
                        error_handler.reportMissingNodeType(@tagName(child), error_handling.getSpanFromAstNode(&child), "contract body") catch {};
                    } else {
                        std.debug.print("WARNING: Unhandled contract body node type in MLIR lowering: {s}\n", .{@tagName(child)});
                    }
                },
            }
        }

        // Create and return the contract operation
        return c.mlirOperationCreate(&state);
    }

    /// Lower struct declarations with type definitions and field information (Requirements 7.1)
    pub fn lowerStruct(self: *const DeclarationLowerer, struct_decl: *const lib.ast.StructDeclNode) c.MlirOperation {
        // Create ora.struct.decl operation
        var state = h.opState("ora.struct.decl", self.createFileLocation(struct_decl.span));

        // Collect struct attributes
        var attributes = std.ArrayList(c.MlirNamedAttribute){};
        defer attributes.deinit(std.heap.page_allocator);

        // Add struct name
        const name_ref = c.mlirStringRefCreate(struct_decl.name.ptr, struct_decl.name.len);
        const name_attr = c.mlirStringAttrGet(self.ctx, name_ref);
        const name_id = h.identifier(self.ctx, "sym_name");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(name_id, name_attr)) catch {};

        // Create field information as attributes
        var field_names = std.ArrayList(c.MlirAttribute){};
        defer field_names.deinit(std.heap.page_allocator);
        var field_types = std.ArrayList(c.MlirAttribute){};
        defer field_types.deinit(std.heap.page_allocator);

        for (struct_decl.fields) |field| {
            // Add field name
            const field_name_ref = c.mlirStringRefCreate(field.name.ptr, field.name.len);
            const field_name_attr = c.mlirStringAttrGet(self.ctx, field_name_ref);
            field_names.append(std.heap.page_allocator, field_name_attr) catch {};

            // Add field type
            const field_type = self.type_mapper.toMlirType(field.type_info);
            const field_type_attr = c.mlirTypeAttrGet(field_type);
            field_types.append(std.heap.page_allocator, field_type_attr) catch {};
        }

        // Add field names array attribute
        if (field_names.items.len > 0) {
            const field_names_array = c.mlirArrayAttrGet(self.ctx, @intCast(field_names.items.len), field_names.items.ptr);
            const field_names_id = h.identifier(self.ctx, "ora.field_names");
            attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(field_names_id, field_names_array)) catch {};

            // Add field types array attribute
            const field_types_array = c.mlirArrayAttrGet(self.ctx, @intCast(field_types.items.len), field_types.items.ptr);
            const field_types_id = h.identifier(self.ctx, "ora.field_types");
            attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(field_types_id, field_types_array)) catch {};
        }

        // Add struct declaration marker
        const struct_decl_attr = c.mlirBoolAttrGet(self.ctx, 1);
        const struct_decl_id = h.identifier(self.ctx, "ora.struct_decl");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(struct_decl_id, struct_decl_attr)) catch {};

        // Apply all attributes
        c.mlirOperationStateAddAttributes(&state, @intCast(attributes.items.len), attributes.items.ptr);

        // Create the struct type and add it as a result
        const struct_type = self.createStructType(struct_decl);
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&struct_type));

        return c.mlirOperationCreate(&state);
    }

    /// Lower enum declarations with enum type definitions and variant information (Requirements 7.2)
    pub fn lowerEnum(self: *const DeclarationLowerer, enum_decl: *const lib.ast.EnumDeclNode) c.MlirOperation {
        // Create ora.enum.decl operation
        var state = h.opState("ora.enum.decl", self.createFileLocation(enum_decl.span));

        // Collect enum attributes
        var attributes = std.ArrayList(c.MlirNamedAttribute){};
        defer attributes.deinit(std.heap.page_allocator);

        // Add enum name
        const name_ref = c.mlirStringRefCreate(enum_decl.name.ptr, enum_decl.name.len);
        const name_attr = c.mlirStringAttrGet(self.ctx, name_ref);
        const name_id = h.identifier(self.ctx, "sym_name");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(name_id, name_attr)) catch {};

        // Add underlying type information
        const underlying_type = if (enum_decl.underlying_type_info) |type_info|
            self.type_mapper.toMlirType(type_info)
        else
            c.mlirIntegerTypeGet(self.ctx, 32); // Default to i32
        const underlying_type_attr = c.mlirTypeAttrGet(underlying_type);
        const underlying_type_id = h.identifier(self.ctx, "ora.underlying_type");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(underlying_type_id, underlying_type_attr)) catch {};

        // Create variant information as attributes
        var variant_names = std.ArrayList(c.MlirAttribute){};
        defer variant_names.deinit(std.heap.page_allocator);
        var variant_values = std.ArrayList(c.MlirAttribute){};
        defer variant_values.deinit(std.heap.page_allocator);

        for (enum_decl.variants, 0..) |variant, i| {
            // Add variant name
            const variant_name_ref = c.mlirStringRefCreate(variant.name.ptr, variant.name.len);
            const variant_name_attr = c.mlirStringAttrGet(self.ctx, variant_name_ref);
            variant_names.append(std.heap.page_allocator, variant_name_attr) catch {};

            // Add variant value (use resolved value if available, otherwise use index)
            const variant_value = if (variant.resolved_value) |resolved|
                @as(i64, @intCast(resolved))
            else
                @as(i64, @intCast(i));
            const variant_value_attr = c.mlirIntegerAttrGet(underlying_type, variant_value);
            variant_values.append(std.heap.page_allocator, variant_value_attr) catch {};
        }

        // Add variant names array attribute
        if (variant_names.items.len > 0) {
            const variant_names_array = c.mlirArrayAttrGet(self.ctx, @intCast(variant_names.items.len), variant_names.items.ptr);
            const variant_names_id = h.identifier(self.ctx, "ora.variant_names");
            attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(variant_names_id, variant_names_array)) catch {};

            // Add variant values array attribute
            const variant_values_array = c.mlirArrayAttrGet(self.ctx, @intCast(variant_values.items.len), variant_values.items.ptr);
            const variant_values_id = h.identifier(self.ctx, "ora.variant_values");
            attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(variant_values_id, variant_values_array)) catch {};
        }

        // Add explicit values flag
        const has_explicit_values_attr = c.mlirBoolAttrGet(self.ctx, if (enum_decl.has_explicit_values) 1 else 0);
        const has_explicit_values_id = h.identifier(self.ctx, "ora.has_explicit_values");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(has_explicit_values_id, has_explicit_values_attr)) catch {};

        // Add enum declaration marker
        const enum_decl_attr = c.mlirBoolAttrGet(self.ctx, 1);
        const enum_decl_id = h.identifier(self.ctx, "ora.enum_decl");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(enum_decl_id, enum_decl_attr)) catch {};

        // Apply all attributes
        c.mlirOperationStateAddAttributes(&state, @intCast(attributes.items.len), attributes.items.ptr);

        // Create the enum type and add it as a result
        const enum_type = self.createEnumType(enum_decl);
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&enum_type));

        return c.mlirOperationCreate(&state);
    }

    /// Lower module declarations for top-level program structure
    pub fn lowerModule(self: *const DeclarationLowerer, module: *const lib.ast.ModuleNode) c.MlirOperation {
        // Create ora.module operation
        var state = h.opState("ora.module", self.createFileLocation(module.span));

        // Collect module attributes
        var attributes = std.ArrayList(c.MlirNamedAttribute){};
        defer attributes.deinit(std.heap.page_allocator);

        // Add module name if present
        if (module.name) |name| {
            const name_ref = c.mlirStringRefCreate(name.ptr, name.len);
            const name_attr = c.mlirStringAttrGet(self.ctx, name_ref);
            const name_id = h.identifier(self.ctx, "sym_name");
            attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(name_id, name_attr)) catch {};
        }

        // Add module declaration marker
        const module_decl_attr = c.mlirBoolAttrGet(self.ctx, 1);
        const module_decl_id = h.identifier(self.ctx, "ora.module_decl");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(module_decl_id, module_decl_attr)) catch {};

        // Add import count attribute
        const import_count_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 32), @intCast(module.imports.len));
        const import_count_id = h.identifier(self.ctx, "ora.import_count");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(import_count_id, import_count_attr)) catch {};

        // Add declaration count attribute
        const decl_count_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 32), @intCast(module.declarations.len));
        const decl_count_id = h.identifier(self.ctx, "ora.declaration_count");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(decl_count_id, decl_count_attr)) catch {};

        // Apply all attributes
        c.mlirOperationStateAddAttributes(&state, @intCast(attributes.items.len), attributes.items.ptr);

        // Create a region for the module body
        const region = c.mlirRegionCreate();
        c.mlirOperationStateAddOwnedRegions(&state, 1, @ptrCast(&region));

        return c.mlirOperationCreate(&state);
    }

    /// Lower block declarations for block constructs
    pub fn lowerBlock(self: *const DeclarationLowerer, block_decl: *const lib.ast.Statements.BlockNode) c.MlirOperation {
        // Create ora.block operation
        var state = h.opState("ora.block", self.createFileLocation(block_decl.span));

        // Collect block attributes
        var attributes = std.ArrayList(c.MlirNamedAttribute){};
        defer attributes.deinit(std.heap.page_allocator);

        // Add block declaration marker
        const block_decl_attr = c.mlirBoolAttrGet(self.ctx, 1);
        const block_decl_id = h.identifier(self.ctx, "ora.block_decl");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(block_decl_id, block_decl_attr)) catch {};

        // Add statement count attribute
        const stmt_count_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 32), @intCast(block_decl.statements.len));
        const stmt_count_id = h.identifier(self.ctx, "ora.statement_count");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(stmt_count_id, stmt_count_attr)) catch {};

        // Apply all attributes
        c.mlirOperationStateAddAttributes(&state, @intCast(attributes.items.len), attributes.items.ptr);

        // Create a region for the block body
        const region = c.mlirRegionCreate();
        const block = c.mlirBlockCreate(0, null, null);
        c.mlirRegionInsertOwnedBlock(region, 0, block);
        c.mlirOperationStateAddOwnedRegions(&state, 1, @ptrCast(&region));

        return c.mlirOperationCreate(&state);
    }

    /// Lower try-block declarations for try-catch blocks
    pub fn lowerTryBlock(self: *const DeclarationLowerer, try_block: *const lib.ast.Statements.TryBlockNode) c.MlirOperation {
        // Create ora.try_block operation
        var state = h.opState("ora.try_block", self.createFileLocation(try_block.span));

        // Collect try-block attributes
        var attributes = std.ArrayList(c.MlirNamedAttribute){};
        defer attributes.deinit(std.heap.page_allocator);

        // Add try-block declaration marker
        const try_block_decl_attr = c.mlirBoolAttrGet(self.ctx, 1);
        const try_block_decl_id = h.identifier(self.ctx, "ora.try_block_decl");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(try_block_decl_id, try_block_decl_attr)) catch {};

        // Add error handling marker
        const error_handling_attr = c.mlirBoolAttrGet(self.ctx, 1);
        const error_handling_id = h.identifier(self.ctx, "ora.error_handling");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(error_handling_id, error_handling_attr)) catch {};

        // Add catch block presence attribute
        const has_catch_attr = c.mlirBoolAttrGet(self.ctx, if (try_block.catch_block != null) 1 else 0);
        const has_catch_id = h.identifier(self.ctx, "ora.has_catch");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(has_catch_id, has_catch_attr)) catch {};

        // Apply all attributes
        c.mlirOperationStateAddAttributes(&state, @intCast(attributes.items.len), attributes.items.ptr);

        // Create regions for try and catch blocks
        const try_region = c.mlirRegionCreate();
        const try_block_mlir = c.mlirBlockCreate(0, null, null);
        c.mlirRegionInsertOwnedBlock(try_region, 0, try_block_mlir);
        c.mlirOperationStateAddOwnedRegions(&state, 1, @ptrCast(&try_region));

        // Add catch region if present
        if (try_block.catch_block != null) {
            const catch_region = c.mlirRegionCreate();
            const catch_block_mlir = c.mlirBlockCreate(0, null, null);
            c.mlirRegionInsertOwnedBlock(catch_region, 0, catch_block_mlir);
            c.mlirOperationStateAddOwnedRegions(&state, 1, @ptrCast(&catch_region));
        }

        return c.mlirOperationCreate(&state);
    }

    /// Lower import declarations with module import constructs (Requirements 7.5)
    pub fn lowerImport(self: *const DeclarationLowerer, import_decl: *const lib.ast.ImportNode) c.MlirOperation {
        // Create ora.import operation
        var state = h.opState("ora.import", self.createFileLocation(import_decl.span));

        // Collect import attributes
        var attributes = std.ArrayList(c.MlirNamedAttribute){};
        defer attributes.deinit(std.heap.page_allocator);

        // Add import path
        const path_ref = c.mlirStringRefCreate(import_decl.path.ptr, import_decl.path.len);
        const path_attr = c.mlirStringAttrGet(self.ctx, path_ref);
        const path_id = h.identifier(self.ctx, "ora.import_path");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(path_id, path_attr)) catch {};

        // Add alias if present
        if (import_decl.alias) |alias| {
            const alias_ref = c.mlirStringRefCreate(alias.ptr, alias.len);
            const alias_attr = c.mlirStringAttrGet(self.ctx, alias_ref);
            const alias_id = h.identifier(self.ctx, "ora.import_alias");
            attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(alias_id, alias_attr)) catch {};
        }

        // Add import declaration marker
        const import_decl_attr = c.mlirBoolAttrGet(self.ctx, 1);
        const import_decl_id = h.identifier(self.ctx, "ora.import_decl");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(import_decl_id, import_decl_attr)) catch {};

        // Apply all attributes
        c.mlirOperationStateAddAttributes(&state, @intCast(attributes.items.len), attributes.items.ptr);

        return c.mlirOperationCreate(&state);
    }

    /// Lower const declarations with global constant definitions (Requirements 7.6)
    pub fn lowerConstDecl(self: *const DeclarationLowerer, const_decl: *const lib.ast.ConstantNode) c.MlirOperation {
        // Create ora.const operation
        var state = h.opState("ora.const", self.createFileLocation(const_decl.span));

        // Collect const attributes
        var attributes = std.ArrayList(c.MlirNamedAttribute){};
        defer attributes.deinit(std.heap.page_allocator);

        // Add constant name
        const name_ref = c.mlirStringRefCreate(const_decl.name.ptr, const_decl.name.len);
        const name_attr = c.mlirStringAttrGet(self.ctx, name_ref);
        const name_id = h.identifier(self.ctx, "sym_name");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(name_id, name_attr)) catch {};

        // Add constant type
        const const_type = self.type_mapper.toMlirType(const_decl.typ);
        const type_attr = c.mlirTypeAttrGet(const_type);
        const type_id = h.identifier(self.ctx, "type");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(type_id, type_attr)) catch {};

        // Add visibility modifier
        const visibility_attr = switch (const_decl.visibility) {
            .Public => c.mlirStringAttrGet(self.ctx, h.strRef("pub")),
            .Private => c.mlirStringAttrGet(self.ctx, h.strRef("private")),
        };
        const visibility_id = h.identifier(self.ctx, "ora.visibility");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(visibility_id, visibility_attr)) catch {};

        // Add constant declaration marker
        const const_decl_attr = c.mlirBoolAttrGet(self.ctx, 1);
        const const_decl_id = h.identifier(self.ctx, "ora.const_decl");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(const_decl_id, const_decl_attr)) catch {};

        // Apply all attributes
        c.mlirOperationStateAddAttributes(&state, @intCast(attributes.items.len), attributes.items.ptr);

        // Add the constant type as a result
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&const_type));

        // Create a region for the constant value initialization
        const region = c.mlirRegionCreate();
        const block = c.mlirBlockCreate(0, null, null);
        c.mlirRegionInsertOwnedBlock(region, 0, block);
        c.mlirOperationStateAddOwnedRegions(&state, 1, @ptrCast(&region));

        // Lower the constant value expression
        // Create a temporary expression lowerer to lower the constant value
        const expr_lowerer = ExpressionLowerer.init(self.ctx, block, self.type_mapper, null, null, null, null, self.builtin_registry, self.locations, self.ora_dialect);
        _ = expr_lowerer.lowerExpression(const_decl.value);

        return c.mlirOperationCreate(&state);
    }

    /// Lower immutable declarations with immutable global definitions and initialization constraints (Requirements 7.7)
    pub fn lowerImmutableDecl(self: *const DeclarationLowerer, var_decl: *const lib.ast.Statements.VariableDeclNode) c.MlirOperation {
        // Create ora.immutable operation for immutable global variables
        var state = h.opState("ora.immutable", self.createFileLocation(var_decl.span));

        // Collect immutable attributes
        var attributes = std.ArrayList(c.MlirNamedAttribute){};
        defer attributes.deinit(std.heap.page_allocator);

        // Add variable name
        const name_ref = c.mlirStringRefCreate(var_decl.name.ptr, var_decl.name.len);
        const name_attr = c.mlirStringAttrGet(self.ctx, name_ref);
        const name_id = h.identifier(self.ctx, "sym_name");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(name_id, name_attr)) catch {};

        // Add variable type
        const var_type = self.type_mapper.toMlirType(var_decl.type_info);
        const type_attr = c.mlirTypeAttrGet(var_type);
        const type_id = h.identifier(self.ctx, "type");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(type_id, type_attr)) catch {};

        // Add immutable constraint marker
        const immutable_attr = c.mlirBoolAttrGet(self.ctx, 1);
        const immutable_id = h.identifier(self.ctx, "ora.immutable");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(immutable_id, immutable_attr)) catch {};

        // Add initialization constraint - immutable variables must be initialized
        if (var_decl.value == null) {
            const requires_init_attr = c.mlirBoolAttrGet(self.ctx, 1);
            const requires_init_id = h.identifier(self.ctx, "ora.requires_init");
            attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(requires_init_id, requires_init_attr)) catch {};
        }

        // Apply all attributes
        c.mlirOperationStateAddAttributes(&state, @intCast(attributes.items.len), attributes.items.ptr);

        // Add the variable type as a result
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&var_type));

        // Create a region for initialization if there's an initial value
        if (var_decl.value != null) {
            const region = c.mlirRegionCreate();
            const block = c.mlirBlockCreate(0, null, null);
            c.mlirRegionInsertOwnedBlock(region, 0, block);
            c.mlirOperationStateAddOwnedRegions(&state, 1, @ptrCast(&region));

            // Initialization expression lowering handled by expression lowerer
            // Body block created for variable init code
        }

        return c.mlirOperationCreate(&state);
    }

    /// Create global storage variable declaration
    pub fn createGlobalDeclaration(self: *const DeclarationLowerer, var_decl: *const lib.ast.Statements.VariableDeclNode) c.MlirOperation {
        // Determine the variable type
        const var_type = if (std.mem.eql(u8, var_decl.name, "status"))
            h.boolType(self.ctx) // bool -> i1
        else
            c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS); // default to i256

        // Determine the initial value
        const init_attr = if (std.mem.eql(u8, var_decl.name, "status"))
            c.mlirIntegerAttrGet(h.boolType(self.ctx), 0) // bool -> i1 with value 0 (false)
        else
            c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS), 0); // default to i256 with value 0

        // Use the dialect helper function to create the global operation
        return self.ora_dialect.createGlobal(var_decl.name, var_type, init_attr, self.createFileLocation(var_decl.span));
    }

    /// Create memory global variable declaration
    pub fn createMemoryGlobalDeclaration(self: *const DeclarationLowerer, var_decl: *const lib.ast.Statements.VariableDeclNode) c.MlirOperation {
        // Create ora.memory.global operation
        var state = h.opState("ora.memory.global", self.createFileLocation(var_decl.span));

        // Add the global name as a symbol attribute
        const name_ref = c.mlirStringRefCreate(var_decl.name.ptr, var_decl.name.len);
        const name_attr = c.mlirStringAttrGet(self.ctx, name_ref);
        const name_id = h.identifier(self.ctx, "sym_name");
        var attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(name_id, name_attr),
        };
        c.mlirOperationStateAddAttributes(&state, @intCast(attrs.len), &attrs);

        // Add the type attribute
        const var_type = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS); // default to i256
        const type_attr = c.mlirTypeAttrGet(var_type);
        const type_id = h.identifier(self.ctx, "type");
        var type_attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(type_id, type_attr),
        };
        c.mlirOperationStateAddAttributes(&state, @intCast(type_attrs.len), &type_attrs);

        return c.mlirOperationCreate(&state);
    }

    /// Create transient storage global variable declaration
    pub fn createTStoreGlobalDeclaration(self: *const DeclarationLowerer, var_decl: *const lib.ast.Statements.VariableDeclNode) c.MlirOperation {
        // Create ora.tstore.global operation
        var state = h.opState("ora.tstore.global", self.createFileLocation(var_decl.span));

        // Add the global name as a symbol attribute
        const name_ref = c.mlirStringRefCreate(var_decl.name.ptr, var_decl.name.len);
        const name_attr = c.mlirStringAttrGet(self.ctx, name_ref);
        const name_id = h.identifier(self.ctx, "sym_name");
        var attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(name_id, name_attr),
        };
        c.mlirOperationStateAddAttributes(&state, @intCast(attrs.len), &attrs);

        // Add the type attribute
        const var_type = c.mlirIntegerTypeGet(self.ctx, constants.DEFAULT_INTEGER_BITS); // default to i256
        const type_attr = c.mlirTypeAttrGet(var_type);
        const type_id = h.identifier(self.ctx, "type");
        var type_attrs = [_]c.MlirNamedAttribute{
            c.mlirNamedAttributeGet(type_id, type_attr),
        };
        c.mlirOperationStateAddAttributes(&state, @intCast(type_attrs.len), &type_attrs);

        return c.mlirOperationCreate(&state);
    }

    /// Lower function body
    fn lowerFunctionBody(self: *const DeclarationLowerer, func: *const lib.FunctionNode, block: c.MlirBlock, param_map: *const ParamMap, storage_map: ?*const StorageMap, local_var_map: ?*LocalVarMap) LoweringError!void {
        // Create a statement lowerer for this function
        const const_local_var_map = if (local_var_map) |lvm| @as(*const LocalVarMap, lvm) else null;
        const expr_lowerer = ExpressionLowerer.init(self.ctx, block, self.type_mapper, param_map, storage_map, const_local_var_map, self.symbol_table, self.builtin_registry, self.locations, self.ora_dialect);

        // Get the function's return type
        const function_return_type = if (func.return_type_info) |ret_info|
            self.type_mapper.toMlirType(ret_info)
        else
            null;

        const stmt_lowerer = StatementLowerer.init(self.ctx, block, self.type_mapper, &expr_lowerer, param_map, storage_map, local_var_map, self.locations, self.symbol_table, self.builtin_registry, std.heap.page_allocator, function_return_type, self.ora_dialect);

        // Lower the function body
        try stmt_lowerer.lowerBlockBody(func.body, block);
    }

    /// Lower requires clauses as precondition assertions with enhanced verification metadata (Requirements 6.4)
    fn lowerRequiresClauses(self: *const DeclarationLowerer, requires_clauses: []*lib.ast.Expressions.ExprNode, block: c.MlirBlock, param_map: *const ParamMap, storage_map: ?*const StorageMap, local_var_map: ?*LocalVarMap) LoweringError!void {
        const const_local_var_map = if (local_var_map) |lvm| @as(*const LocalVarMap, lvm) else null;
        const expr_lowerer = ExpressionLowerer.init(self.ctx, block, self.type_mapper, param_map, storage_map, const_local_var_map, self.symbol_table, self.builtin_registry, self.locations, self.ora_dialect);

        for (requires_clauses, 0..) |clause, i| {
            // Lower the requires expression
            const condition_value = expr_lowerer.lowerExpression(clause);

            // Create an assertion operation with comprehensive verification attributes
            var assert_state = h.opState("cf.assert", self.createFileLocation(self.getExpressionSpan(clause)));

            // Add the condition as an operand
            c.mlirOperationStateAddOperands(&assert_state, 1, @ptrCast(&condition_value));

            // Collect verification attributes
            var attributes = std.ArrayList(c.MlirNamedAttribute){};
            defer attributes.deinit(std.heap.page_allocator);

            // Add ora.requires attribute to mark this as a precondition
            const requires_attr = c.mlirBoolAttrGet(self.ctx, 1);
            const requires_id = h.identifier(self.ctx, "ora.requires");
            attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(requires_id, requires_attr)) catch {};

            // Add verification context attribute
            const context_attr = c.mlirStringAttrGet(self.ctx, h.strRef("function_precondition"));
            const context_id = h.identifier(self.ctx, "ora.verification_context");
            attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(context_id, context_attr)) catch {};

            // Add verification marker for formal verification tools
            const verification_attr = c.mlirBoolAttrGet(self.ctx, 1);
            const verification_id = h.identifier(self.ctx, "ora.verification");
            attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(verification_id, verification_attr)) catch {};

            // Add precondition index for multiple requires clauses
            const index_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 32), @intCast(i));
            const index_id = h.identifier(self.ctx, "ora.precondition_index");
            attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(index_id, index_attr)) catch {};

            // Add formal verification marker
            const formal_attr = c.mlirBoolAttrGet(self.ctx, 1);
            const formal_id = h.identifier(self.ctx, "ora.formal");
            attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(formal_id, formal_attr)) catch {};

            // Apply all attributes
            c.mlirOperationStateAddAttributes(&assert_state, @intCast(attributes.items.len), attributes.items.ptr);

            const assert_op = c.mlirOperationCreate(&assert_state);
            h.appendOp(block, assert_op);
        }
    }

    /// Lower ensures clauses as postcondition assertions with enhanced verification metadata (Requirements 6.5)
    fn lowerEnsuresClauses(self: *const DeclarationLowerer, ensures_clauses: []*lib.ast.Expressions.ExprNode, block: c.MlirBlock, param_map: *const ParamMap, storage_map: ?*const StorageMap, local_var_map: ?*LocalVarMap) LoweringError!void {
        const const_local_var_map = if (local_var_map) |lvm| @as(*const LocalVarMap, lvm) else null;
        const expr_lowerer = ExpressionLowerer.init(self.ctx, block, self.type_mapper, param_map, storage_map, const_local_var_map, self.symbol_table, self.builtin_registry, self.locations, self.ora_dialect);

        for (ensures_clauses, 0..) |clause, i| {
            // Lower the ensures expression
            const condition_value = expr_lowerer.lowerExpression(clause);

            // Create an assertion operation with comprehensive verification attributes
            var assert_state = h.opState("cf.assert", self.createFileLocation(self.getExpressionSpan(clause)));

            // Add the condition as an operand
            c.mlirOperationStateAddOperands(&assert_state, 1, @ptrCast(&condition_value));

            // Collect verification attributes
            var attributes = std.ArrayList(c.MlirNamedAttribute){};
            defer attributes.deinit(std.heap.page_allocator);

            // Add ora.ensures attribute to mark this as a postcondition
            const ensures_attr = c.mlirBoolAttrGet(self.ctx, 1);
            const ensures_id = h.identifier(self.ctx, "ora.ensures");
            attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(ensures_id, ensures_attr)) catch {};

            // Add verification context attribute
            const context_attr = c.mlirStringAttrGet(self.ctx, h.strRef("function_postcondition"));
            const context_id = h.identifier(self.ctx, "ora.verification_context");
            attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(context_id, context_attr)) catch {};

            // Add verification marker for formal verification tools
            const verification_attr = c.mlirBoolAttrGet(self.ctx, 1);
            const verification_id = h.identifier(self.ctx, "ora.verification");
            attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(verification_id, verification_attr)) catch {};

            // Add postcondition index for multiple ensures clauses
            const index_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 32), @intCast(i));
            const index_id = h.identifier(self.ctx, "ora.postcondition_index");
            attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(index_id, index_attr)) catch {};

            // Add formal verification marker
            const formal_attr = c.mlirBoolAttrGet(self.ctx, 1);
            const formal_id = h.identifier(self.ctx, "ora.formal");
            attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(formal_id, formal_attr)) catch {};

            // Add return value reference for postconditions
            const return_ref_attr = c.mlirStringAttrGet(self.ctx, h.strRef("return_value"));
            const return_ref_id = h.identifier(self.ctx, "ora.return_reference");
            attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(return_ref_id, return_ref_attr)) catch {};

            // Apply all attributes
            c.mlirOperationStateAddAttributes(&assert_state, @intCast(attributes.items.len), attributes.items.ptr);

            const assert_op = c.mlirOperationCreate(&assert_state);
            h.appendOp(block, assert_op);
        }
    }

    /// Enhanced function type creation with parameter default values (Requirements 6.3)
    fn createFunctionType(self: *const DeclarationLowerer, func: *const lib.FunctionNode) c.MlirType {
        // Create parameter types array
        var param_types = std.ArrayList(c.MlirType){};
        defer param_types.deinit(std.heap.page_allocator);

        for (func.parameters) |param| {
            const param_type = self.type_mapper.toMlirType(param.type_info);
            param_types.append(std.heap.page_allocator, param_type) catch {};
        }

        // Create function type
        if (func.return_type_info) |ret_info| {
            const result_type = self.type_mapper.toMlirType(ret_info);
            return c.mlirFunctionTypeGet(self.ctx, @intCast(param_types.items.len), if (param_types.items.len > 0) param_types.items.ptr else null, 1, @ptrCast(&result_type));
        } else {
            // Functions with no return type should have 0 result types, not a 'none' type
            return c.mlirFunctionTypeGet(self.ctx, @intCast(param_types.items.len), if (param_types.items.len > 0) param_types.items.ptr else null, 0, null);
        }
    }

    /// Create struct type from struct declaration
    pub fn createStructType(self: *const DeclarationLowerer, struct_decl: *const lib.ast.StructDeclNode) c.MlirType {
        // Create field types array
        var field_types = std.ArrayList(c.MlirType){};
        defer field_types.deinit(std.heap.page_allocator);

        for (struct_decl.fields) |field| {
            const field_type = self.type_mapper.toMlirType(field.type_info);
            field_types.append(std.heap.page_allocator, field_type) catch {};
        }

        // Struct type using field types from symbol table
        // Migration to !ora.struct<fields> planned for TableGen integration
        if (field_types.items.len > 0) {
            return field_types.items[0];
        } else {
            return c.mlirIntegerTypeGet(self.ctx, 32); // Default to i32 if no fields
        }
    }

    /// Create enum type from enum declaration
    pub fn createEnumType(self: *const DeclarationLowerer, enum_decl: *const lib.ast.EnumDeclNode) c.MlirType {
        // Enum type uses underlying integer representation
        // Migration to !ora.enum<name, repr> planned for TableGen integration
        return if (enum_decl.underlying_type_info) |type_info|
            self.type_mapper.toMlirType(type_info)
        else
            c.mlirIntegerTypeGet(self.ctx, 32); // Default to i32
    }

    /// Lower log declarations with event type definitions and indexed field information (Requirements 7.3)
    pub fn lowerLogDecl(self: *const DeclarationLowerer, log_decl: *const lib.ast.LogDeclNode) c.MlirOperation {
        // Create ora.log.decl operation
        var state = h.opState("ora.log.decl", self.createFileLocation(log_decl.span));

        // Collect log attributes
        var attributes = std.ArrayList(c.MlirNamedAttribute){};
        defer attributes.deinit(std.heap.page_allocator);

        // Add log name
        const name_ref = c.mlirStringRefCreate(log_decl.name.ptr, log_decl.name.len);
        const name_attr = c.mlirStringAttrGet(self.ctx, name_ref);
        const name_id = h.identifier(self.ctx, "sym_name");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(name_id, name_attr)) catch {};

        // Create field information as attributes
        var field_names = std.ArrayList(c.MlirAttribute){};
        defer field_names.deinit(std.heap.page_allocator);
        var field_types = std.ArrayList(c.MlirAttribute){};
        defer field_types.deinit(std.heap.page_allocator);
        var field_indexed = std.ArrayList(c.MlirAttribute){};
        defer field_indexed.deinit(std.heap.page_allocator);

        for (log_decl.fields) |field| {
            // Add field name
            const field_name_ref = c.mlirStringRefCreate(field.name.ptr, field.name.len);
            const field_name_attr = c.mlirStringAttrGet(self.ctx, field_name_ref);
            field_names.append(std.heap.page_allocator, field_name_attr) catch {};

            // Add field type
            const field_type = self.type_mapper.toMlirType(field.type_info);
            const field_type_attr = c.mlirTypeAttrGet(field_type);
            field_types.append(std.heap.page_allocator, field_type_attr) catch {};

            // Add indexed flag
            const indexed_attr = c.mlirBoolAttrGet(self.ctx, if (field.indexed) 1 else 0);
            field_indexed.append(std.heap.page_allocator, indexed_attr) catch {};
        }

        // Add field arrays as attributes
        if (field_names.items.len > 0) {
            const field_names_array = c.mlirArrayAttrGet(self.ctx, @intCast(field_names.items.len), field_names.items.ptr);
            const field_names_id = h.identifier(self.ctx, "ora.field_names");
            attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(field_names_id, field_names_array)) catch {};

            const field_types_array = c.mlirArrayAttrGet(self.ctx, @intCast(field_types.items.len), field_types.items.ptr);
            const field_types_id = h.identifier(self.ctx, "ora.field_types");
            attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(field_types_id, field_types_array)) catch {};

            const field_indexed_array = c.mlirArrayAttrGet(self.ctx, @intCast(field_indexed.items.len), field_indexed.items.ptr);
            const field_indexed_id = h.identifier(self.ctx, "ora.field_indexed");
            attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(field_indexed_id, field_indexed_array)) catch {};
        }

        // Add log declaration marker
        const log_decl_attr = c.mlirBoolAttrGet(self.ctx, 1);
        const log_decl_id = h.identifier(self.ctx, "ora.log_decl");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(log_decl_id, log_decl_attr)) catch {};

        // Apply all attributes
        c.mlirOperationStateAddAttributes(&state, @intCast(attributes.items.len), attributes.items.ptr);

        return c.mlirOperationCreate(&state);
    }

    /// Lower error declarations with error type definitions (Requirements 7.4)
    pub fn lowerErrorDecl(self: *const DeclarationLowerer, error_decl: *const lib.ast.Statements.ErrorDeclNode) c.MlirOperation {
        // Create ora.error.decl operation
        var state = h.opState("ora.error.decl", self.createFileLocation(error_decl.span));

        // Collect error attributes
        var attributes = std.ArrayList(c.MlirNamedAttribute){};
        defer attributes.deinit(std.heap.page_allocator);

        // Add error name
        const name_ref = c.mlirStringRefCreate(error_decl.name.ptr, error_decl.name.len);
        const name_attr = c.mlirStringAttrGet(self.ctx, name_ref);
        const name_id = h.identifier(self.ctx, "sym_name");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(name_id, name_attr)) catch {};

        // Add error parameters if present
        if (error_decl.parameters) |params| {
            var param_names = std.ArrayList(c.MlirAttribute){};
            defer param_names.deinit(std.heap.page_allocator);
            var param_types = std.ArrayList(c.MlirAttribute){};
            defer param_types.deinit(std.heap.page_allocator);

            for (params) |param| {
                // Add parameter name
                const param_name_ref = c.mlirStringRefCreate(param.name.ptr, param.name.len);
                const param_name_attr = c.mlirStringAttrGet(self.ctx, param_name_ref);
                param_names.append(std.heap.page_allocator, param_name_attr) catch {};

                // Add parameter type
                const param_type = self.type_mapper.toMlirType(param.type_info);
                const param_type_attr = c.mlirTypeAttrGet(param_type);
                param_types.append(std.heap.page_allocator, param_type_attr) catch {};
            }

            // Add parameter arrays as attributes
            if (param_names.items.len > 0) {
                const param_names_array = c.mlirArrayAttrGet(self.ctx, @intCast(param_names.items.len), param_names.items.ptr);
                const param_names_id = h.identifier(self.ctx, "ora.param_names");
                attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(param_names_id, param_names_array)) catch {};

                const param_types_array = c.mlirArrayAttrGet(self.ctx, @intCast(param_types.items.len), param_types.items.ptr);
                const param_types_id = h.identifier(self.ctx, "ora.param_types");
                attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(param_types_id, param_types_array)) catch {};
            }
        }

        // Add error declaration marker
        const error_decl_attr = c.mlirBoolAttrGet(self.ctx, 1);
        const error_decl_id = h.identifier(self.ctx, "ora.error_decl");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(error_decl_id, error_decl_attr)) catch {};

        // Apply all attributes
        c.mlirOperationStateAddAttributes(&state, @intCast(attributes.items.len), attributes.items.ptr);

        // Create the error type and add it as a result
        const error_type = self.createErrorType(error_decl);
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&error_type));

        return c.mlirOperationCreate(&state);
    }

    /// Create error type from error declaration
    fn createErrorType(self: *const DeclarationLowerer, error_decl: *const lib.ast.Statements.ErrorDeclNode) c.MlirType {
        // Error type uses i32 representation for error codes
        // Migration to !ora.error<T> planned for TableGen integration
        _ = error_decl;
        return c.mlirIntegerTypeGet(self.ctx, 32); // Placeholder error type
    }

    /// Lower quantified expressions (forall, exists) with verification constructs and ora.quantified attributes (Requirements 6.6)
    pub fn lowerQuantifiedExpression(self: *const DeclarationLowerer, quantified: *const lib.ast.Expressions.QuantifiedExpr, block: c.MlirBlock, param_map: *const ParamMap, storage_map: ?*const StorageMap, local_var_map: ?*LocalVarMap) !c.MlirValue {
        // Create ora.quantified operation
        var state = h.opState("ora.quantified", self.createFileLocation(quantified.span));

        // Collect quantified attributes
        var attributes = std.ArrayList(c.MlirNamedAttribute){};
        defer attributes.deinit(std.heap.page_allocator);

        // Add quantifier type (forall or exists)
        const quantifier_str = switch (quantified.quantifier) {
            .Forall => "forall",
            .Exists => "exists",
        };
        const quantifier_attr = h.stringAttr(self.ctx, quantifier_str);
        const quantifier_id = h.identifier(self.ctx, "ora.quantifier");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(quantifier_id, quantifier_attr)) catch {};

        // Add bound variable name
        const var_name_ref = c.mlirStringRefCreate(quantified.variable.ptr, quantified.variable.len);
        const var_name_attr = c.mlirStringAttrGet(self.ctx, var_name_ref);
        const var_name_id = h.identifier(self.ctx, "ora.bound_variable");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(var_name_id, var_name_attr)) catch {};

        // Add bound variable type
        const var_type = self.type_mapper.toMlirType(quantified.variable_type);
        const var_type_attr = c.mlirTypeAttrGet(var_type);
        const var_type_id = h.identifier(self.ctx, "ora.bound_variable_type");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(var_type_id, var_type_attr)) catch {};

        // Add quantified expression marker
        const quantified_attr = c.mlirBoolAttrGet(self.ctx, 1);
        const quantified_id = h.identifier(self.ctx, "ora.quantified");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(quantified_id, quantified_attr)) catch {};

        // Apply all attributes
        c.mlirOperationStateAddAttributes(&state, @intCast(attributes.items.len), attributes.items.ptr);

        // Create result type (quantified expressions return boolean)
        const result_type = h.boolType(self.ctx); // i1 for boolean
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_type));

        // Create regions for condition and body
        const condition_region = c.mlirRegionCreate();
        const body_region = c.mlirRegionCreate();

        // Create blocks for condition and body
        const condition_block = c.mlirBlockCreate(0, null, null);
        const body_block = c.mlirBlockCreate(0, null, null);

        c.mlirRegionInsertOwnedBlock(condition_region, 0, condition_block);
        c.mlirRegionInsertOwnedBlock(body_region, 0, body_block);

        // Add regions to the operation
        var regions = [_]c.MlirRegion{ condition_region, body_region };
        c.mlirOperationStateAddOwnedRegions(&state, regions.len, &regions);

        // Lower the condition if present
        if (quantified.condition) |condition| {
            const const_local_var_map = if (local_var_map) |lvm| @as(*const LocalVarMap, lvm) else null;
            const expr_lowerer = ExpressionLowerer.init(self.ctx, condition_block, self.type_mapper, param_map, storage_map, const_local_var_map, self.symbol_table, self.locations, self.ora_dialect);
            _ = expr_lowerer.lowerExpression(condition);
        }

        // Lower the body expression
        const const_local_var_map = if (local_var_map) |lvm| @as(*const LocalVarMap, lvm) else null;
        const expr_lowerer = ExpressionLowerer.init(self.ctx, body_block, self.type_mapper, param_map, storage_map, const_local_var_map, self.symbol_table, self.locations, self.ora_dialect);
        _ = expr_lowerer.lowerExpression(quantified.body);

        // Create the quantified operation
        const quantified_op = c.mlirOperationCreate(&state);
        h.appendOp(block, quantified_op);

        return c.mlirValueFromOpResult(h.getResult(quantified_op, 0));
    }

    /// Add verification-related attributes and metadata support
    pub fn addVerificationAttributes(self: *const DeclarationLowerer, operation: c.MlirOperation, verification_type: []const u8, metadata: ?[]const u8) void {
        // Add verification type attribute
        const verification_attr = h.stringAttr(self.ctx, verification_type);
        const verification_id = h.identifier(self.ctx, "ora.verification_type");
        c.mlirOperationSetAttribute(operation, verification_id, verification_attr);

        // Add metadata if provided
        if (metadata) |meta| {
            const metadata_attr = h.stringAttr(self.ctx, meta);
            const metadata_id = h.identifier(self.ctx, "ora.verification_metadata");
            c.mlirOperationSetAttribute(operation, metadata_id, metadata_attr);
        }

        // Add verification marker
        const verification_marker = c.mlirBoolAttrGet(self.ctx, 1);
        const verification_marker_id = h.identifier(self.ctx, "ora.formal_verification");
        c.mlirOperationSetAttribute(operation, verification_marker_id, verification_marker);
    }

    /// Handle formal verification constructs in function contracts
    pub fn lowerFormalVerificationConstructs(self: *const DeclarationLowerer, func: *const lib.FunctionNode, func_op: c.MlirOperation) void {
        // Add verification attributes for functions with requires/ensures clauses
        if (func.requires_clauses.len > 0 or func.ensures_clauses.len > 0) {
            self.addVerificationAttributes(func_op, "function_contract", null);
        }

        // Add specific attributes for preconditions
        if (func.requires_clauses.len > 0) {
            const precondition_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 32), @intCast(func.requires_clauses.len));
            const precondition_id = h.identifier(self.ctx, "ora.precondition_count");
            c.mlirOperationSetAttribute(func_op, precondition_id, precondition_attr);
        }

        // Add specific attributes for postconditions
        if (func.ensures_clauses.len > 0) {
            const postcondition_attr = c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(self.ctx, 32), @intCast(func.ensures_clauses.len));
            const postcondition_id = h.identifier(self.ctx, "ora.postcondition_count");
            c.mlirOperationSetAttribute(func_op, postcondition_id, postcondition_attr);
        }
    }

    /// Create file location for operation
    fn createFileLocation(self: *const DeclarationLowerer, span: lib.ast.SourceSpan) c.MlirLocation {
        return self.locations.createLocation(span);
    }

    /// Get the source span for any expression type
    fn getExpressionSpan(_: *const DeclarationLowerer, expr: *const lib.ast.Expressions.ExprNode) lib.ast.SourceSpan {
        return switch (expr.*) {
            .Identifier => |ident| ident.span,
            .Literal => |lit| switch (lit) {
                .Integer => |int| int.span,
                .String => |str| str.span,
                .Bool => |bool_lit| bool_lit.span,
                .Address => |addr| addr.span,
                .Hex => |hex| hex.span,
                .Binary => |bin| bin.span,
                .Character => |char| char.span,
                .Bytes => |bytes| bytes.span,
            },
            .Binary => |bin| bin.span,
            .Unary => |unary| unary.span,
            .Assignment => |assign| assign.span,
            .CompoundAssignment => |comp_assign| comp_assign.span,
            .Call => |call| call.span,
            .Index => |index| index.span,
            .FieldAccess => |field| field.span,
            .Cast => |cast| cast.span,
            .Comptime => |comptime_expr| comptime_expr.span,
            .Old => |old| old.span,
            .Tuple => |tuple| tuple.span,
            .SwitchExpression => |switch_expr| switch_expr.span,
            .Quantified => |quantified| quantified.span,
            .Try => |try_expr| try_expr.span,
            .ErrorReturn => |error_ret| error_ret.span,
            .ErrorCast => |error_cast| error_cast.span,
            .Shift => |shift| shift.span,
            .StructInstantiation => |struct_inst| struct_inst.span,
            .AnonymousStruct => |anon_struct| anon_struct.span,
            .Range => |range| range.span,
            .LabeledBlock => |labeled_block| labeled_block.span,
            .Destructuring => |destructuring| destructuring.span,
            .EnumLiteral => |enum_lit| enum_lit.span,
            .ArrayLiteral => |array_lit| array_lit.span,
        };
    }

    /// Create a placeholder operation for unsupported variable declarations
    pub fn createVariablePlaceholder(self: *const DeclarationLowerer, var_decl: *const lib.ast.Statements.VariableDeclNode) c.MlirOperation {
        const loc = self.createFileLocation(var_decl.span);
        var state = h.opState("ora.variable_placeholder", loc);

        // Add variable name as attribute
        const name_ref = c.mlirStringRefCreate(var_decl.name.ptr, var_decl.name.len);
        const name_attr = c.mlirStringAttrGet(self.ctx, name_ref);
        const name_id = h.identifier(self.ctx, "name");
        var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(name_id, name_attr)};
        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

        // Add placeholder type
        const placeholder_ty = c.mlirIntegerTypeGet(self.ctx, 32);
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&placeholder_ty));

        return c.mlirOperationCreate(&state);
    }

    /// Create a placeholder operation for unsupported nested modules
    pub fn createModulePlaceholder(self: *const DeclarationLowerer, module_decl: *const lib.ast.ModuleNode) c.MlirOperation {
        const loc = self.createFileLocation(module_decl.span);
        var state = h.opState("ora.module_placeholder", loc);

        // Add module name as attribute if available
        if (module_decl.name) |name| {
            const name_ref = c.mlirStringRefCreate(name.ptr, name.len);
            const name_attr = c.mlirStringAttrGet(self.ctx, name_ref);
            const name_id = h.identifier(self.ctx, "name");
            var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(name_id, name_attr)};
            c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
        }

        // Add placeholder type
        const placeholder_ty = c.mlirIntegerTypeGet(self.ctx, 32);
        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&placeholder_ty));

        return c.mlirOperationCreate(&state);
    }
};
