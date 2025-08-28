// TODO: This file contains duplicated code that should be moved to modular files
// - ParamMap, LocalVarMap -> symbols.zig
// - StorageMap, createLoadOperation, createStoreOperation -> memory.zig
// - lowerExpr, createConstant -> expressions.zig
// - lowerStmt, lowerBlockBody -> statements.zig
// - createGlobalDeclaration, createMemoryGlobalDeclaration, createTStoreGlobalDeclaration, Emit -> declarations.zig
// - fileLoc -> locations.zig
//
// After moving all code, this file should only contain the main lowerFunctionsToModule function
// and orchestration logic, not the actual MLIR operation creation.

const std = @import("std");
const lib = @import("ora_lib");
const c = @import("c.zig").c;
const tmap = @import("types.zig");

pub fn lowerFunctionsToModule(ctx: c.MlirContext, nodes: []lib.AstNode) c.MlirModule {
    const loc = c.mlirLocationUnknownGet(ctx);
    const module = c.mlirModuleCreateEmpty(loc);
    const body = c.mlirModuleGetBody(module);

    // Initialize the variable namer for generating descriptive names

    // Function type building is now handled by the modular type system
    const sym_name_id = c.mlirIdentifierGet(ctx, c.mlirStringRefCreateFromCString("sym_name"));
    const fn_type_id = c.mlirIdentifierGet(ctx, c.mlirStringRefCreateFromCString("function_type"));

    const Lower = struct {
        // TODO: Move ParamMap to symbols.zig - this is duplicated code
        const ParamMap = struct {
            names: std.StringHashMap(usize), // parameter name -> block argument index
            block_args: std.StringHashMap(c.MlirValue), // parameter name -> block argument value

            fn init(allocator: std.mem.Allocator) ParamMap {
                return .{
                    .names = std.StringHashMap(usize).init(allocator),
                    .block_args = std.StringHashMap(c.MlirValue).init(allocator),
                };
            }

            fn deinit(self: *ParamMap) void {
                self.names.deinit();
                self.block_args.deinit();
            }

            fn addParam(self: *ParamMap, name: []const u8, index: usize) !void {
                try self.names.put(name, index);
            }

            fn getParamIndex(self: *const ParamMap, name: []const u8) ?usize {
                return self.names.get(name);
            }

            fn setBlockArgument(self: *ParamMap, name: []const u8, block_arg: c.MlirValue) !void {
                try self.block_args.put(name, block_arg);
            }

            fn getBlockArgument(self: *const ParamMap, name: []const u8) ?c.MlirValue {
                return self.block_args.get(name);
            }
        };

        // Use the modular StorageMap from memory.zig
        const StorageMap = @import("memory.zig").StorageMap;

        // TODO: Move createLoadOperation to memory.zig - this is duplicated code
        fn createLoadOperation(ctx_: c.MlirContext, var_name: []const u8, storage_type: lib.ast.Statements.MemoryRegion, span: lib.ast.SourceSpan) c.MlirOperation {
            switch (storage_type) {
                .Storage => {
                    // Generate ora.sload for storage variables
                    var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("ora.sload"), fileLoc(ctx_, span));

                    // Add the global name as a symbol reference
                    var name_buffer: [256]u8 = undefined;
                    for (0..var_name.len) |i| {
                        name_buffer[i] = var_name[i];
                    }
                    name_buffer[var_name.len] = 0; // null-terminate
                    const name_str = c.mlirStringRefCreateFromCString(&name_buffer[0]);
                    const name_attr = c.mlirStringAttrGet(ctx_, name_str);
                    const name_id = c.mlirIdentifierGet(ctx_, c.mlirStringRefCreateFromCString("global"));
                    var attrs = [_]c.MlirNamedAttribute{
                        c.mlirNamedAttributeGet(name_id, name_attr),
                    };
                    c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

                    // Add result type (default to i256 for now)
                    const result_ty = c.mlirIntegerTypeGet(ctx_, 256);
                    c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));

                    return c.mlirOperationCreate(&state);
                },
                .Memory => {
                    // Generate ora.mload for memory variables
                    var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("ora.mload"), fileLoc(ctx_, span));

                    // Add the variable name as an attribute
                    const name_ref = c.mlirStringRefCreate(var_name.ptr, var_name.len);
                    const name_attr = c.mlirStringAttrGet(ctx_, name_ref);
                    const name_id = c.mlirIdentifierGet(ctx_, c.mlirStringRefCreateFromCString("name"));
                    var attrs = [_]c.MlirNamedAttribute{
                        c.mlirNamedAttributeGet(name_id, name_attr),
                    };
                    c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

                    // Add result type (default to i256 for now)
                    const result_ty = c.mlirIntegerTypeGet(ctx_, 256);
                    c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));

                    return c.mlirOperationCreate(&state);
                },
                .TStore => {
                    // Generate ora.tload for transient storage variables
                    var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("ora.tload"), fileLoc(ctx_, span));

                    // Add the global name as a symbol reference
                    var name_buffer: [256]u8 = undefined;
                    for (0..var_name.len) |i| {
                        name_buffer[i] = var_name[i];
                    }
                    name_buffer[var_name.len] = 0; // null-terminate
                    const name_str = c.mlirStringRefCreateFromCString(&name_buffer[0]);
                    const name_attr = c.mlirStringAttrGet(ctx_, name_str);
                    const name_id = c.mlirIdentifierGet(ctx_, c.mlirStringRefCreateFromCString("global"));
                    var attrs = [_]c.MlirNamedAttribute{
                        c.mlirNamedAttributeGet(name_id, name_attr),
                    };
                    c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

                    // Add result type (default to i256 for now)
                    const result_ty = c.mlirIntegerTypeGet(ctx_, 256);
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

        // TODO: Move createStoreOperation to memory.zig - this is duplicated code
        fn createStoreOperation(ctx_: c.MlirContext, value: c.MlirValue, var_name: []const u8, storage_type: lib.ast.Statements.MemoryRegion, span: lib.ast.SourceSpan) c.MlirOperation {
            switch (storage_type) {
                .Storage => {
                    // Generate ora.sstore for storage variables
                    var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("ora.sstore"), fileLoc(ctx_, span));
                    c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&value));

                    // Add the global name as a symbol reference
                    var name_buffer: [256]u8 = undefined;
                    for (0..var_name.len) |i| {
                        name_buffer[i] = var_name[i];
                    }
                    name_buffer[var_name.len] = 0; // null-terminate
                    const name_str = c.mlirStringRefCreateFromCString(&name_buffer[0]);
                    const name_attr = c.mlirStringAttrGet(ctx_, name_str);
                    const name_id = c.mlirIdentifierGet(ctx_, c.mlirStringRefCreateFromCString("global"));
                    var attrs = [_]c.MlirNamedAttribute{
                        c.mlirNamedAttributeGet(name_id, name_attr),
                    };
                    c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

                    return c.mlirOperationCreate(&state);
                },
                .Memory => {
                    // Generate ora.mstore for memory variables
                    var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("ora.mstore"), fileLoc(ctx_, span));
                    c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&value));

                    // Add the variable name as an attribute
                    const name_ref = c.mlirStringRefCreate(var_name.ptr, var_name.len);
                    const name_attr = c.mlirStringAttrGet(ctx_, name_ref);
                    const name_id = c.mlirIdentifierGet(ctx_, c.mlirStringRefCreateFromCString("name"));
                    var attrs = [_]c.MlirNamedAttribute{
                        c.mlirNamedAttributeGet(name_id, name_attr),
                    };
                    c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

                    return c.mlirOperationCreate(&state);
                },
                .TStore => {
                    // Generate ora.tstore for transient storage variables
                    var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("ora.tstore"), fileLoc(ctx_, span));
                    c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&value));

                    // Add the global name as a symbol reference
                    var name_buffer: [256]u8 = undefined;
                    for (0..var_name.len) |i| {
                        name_buffer[i] = var_name[i];
                    }
                    name_buffer[var_name.len] = 0; // null-terminate
                    const name_str = c.mlirStringRefCreateFromCString(&name_buffer[0]);
                    const name_attr = c.mlirStringAttrGet(ctx_, name_str);
                    const name_id = c.mlirIdentifierGet(ctx_, c.mlirStringRefCreateFromCString("global"));
                    var attrs = [_]c.MlirNamedAttribute{
                        c.mlirNamedAttributeGet(name_id, name_attr),
                    };
                    c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

                    return c.mlirOperationCreate(&state);
                },
                .Stack => {
                    // For stack variables, we store in our local variable map
                    // This is handled differently in the variable declaration
                    @panic("Stack variables should not use createStoreOperation");
                },
            }
        }

        // Use the modular LocalVarMap from symbols.zig
        const LocalVarMap = @import("symbols.zig").LocalVarMap;

        // TODO: Move lowerExpr to expressions.zig - this is duplicated code
        fn lowerExpr(ctx_: c.MlirContext, block: c.MlirBlock, expr: *const lib.ast.Expressions.ExprNode, param_map: ?*const ParamMap, storage_map: ?*StorageMap, local_var_map: ?*LocalVarMap) c.MlirValue {
            return switch (expr.*) {
                .Literal => |lit| switch (lit) {
                    .Integer => |int| blk_int: {
                        const ty = c.mlirIntegerTypeGet(ctx_, 256);
                        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), fileLoc(ctx_, int.span));
                        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ty));

                        // Parse the string value to an integer
                        const parsed: i64 = std.fmt.parseInt(i64, int.value, 0) catch 0;
                        const attr = c.mlirIntegerAttrGet(ty, parsed);

                        const value_id = c.mlirIdentifierGet(ctx_, c.mlirStringRefCreateFromCString("value"));
                        var attrs = [_]c.MlirNamedAttribute{
                            c.mlirNamedAttributeGet(value_id, attr),
                        };
                        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
                        const op = c.mlirOperationCreate(&state);

                        // Note: MLIR operations get their names from the operation state
                        // We can't set names after creation, but the variable naming system
                        // helps with debugging and understanding the generated IR

                        c.mlirBlockAppendOwnedOperation(block, op);
                        break :blk_int c.mlirOperationGetResult(op, 0);
                    },
                    .Bool => |bool_lit| blk_bool: {
                        const ty = c.mlirIntegerTypeGet(ctx_, 1);
                        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), fileLoc(ctx_, bool_lit.span));
                        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ty));
                        const default_value: i64 = if (bool_lit.value) 1 else 0;
                        const attr = c.mlirIntegerAttrGet(ty, default_value);
                        const value_id = c.mlirIdentifierGet(ctx_, c.mlirStringRefCreateFromCString("value"));
                        var attrs = [_]c.MlirNamedAttribute{
                            c.mlirNamedAttributeGet(value_id, attr),
                        };
                        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
                        const op = c.mlirOperationCreate(&state);

                        // Note: MLIR operations get their names from the operation state
                        // We can't set names after creation, but the variable naming system
                        // helps with debugging and understanding the generated IR

                        c.mlirBlockAppendOwnedOperation(block, op);
                        break :blk_bool c.mlirOperationGetResult(op, 0);
                    },
                    .String => |string_lit| blk_string: {
                        // For now, create a placeholder constant for strings
                        // TODO: Implement proper string handling with string attributes
                        const ty = c.mlirIntegerTypeGet(ctx_, 256);
                        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), fileLoc(ctx_, string_lit.span));
                        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ty));
                        const attr = c.mlirIntegerAttrGet(ty, 0); // Placeholder value
                        const value_id = c.mlirIdentifierGet(ctx_, c.mlirStringRefCreateFromCString("value"));
                        var attrs = [_]c.MlirNamedAttribute{
                            c.mlirNamedAttributeGet(value_id, attr),
                        };
                        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
                        const op = c.mlirOperationCreate(&state);
                        c.mlirBlockAppendOwnedOperation(block, op);
                        break :blk_string c.mlirOperationGetResult(op, 0);
                    },
                    .Address => |addr_lit| blk_address: {
                        // Parse address as hex and create integer constant
                        const ty = c.mlirIntegerTypeGet(ctx_, 256);
                        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), fileLoc(ctx_, addr_lit.span));
                        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ty));

                        // Parse hex address (remove 0x prefix if present)
                        const addr_str = if (std.mem.startsWith(u8, addr_lit.value, "0x"))
                            addr_lit.value[2..]
                        else
                            addr_lit.value;
                        const parsed: i64 = std.fmt.parseInt(i64, addr_str, 16) catch 0;
                        const attr = c.mlirIntegerAttrGet(ty, parsed);

                        const value_id = c.mlirIdentifierGet(ctx_, c.mlirStringRefCreateFromCString("value"));
                        var attrs = [_]c.MlirNamedAttribute{
                            c.mlirNamedAttributeGet(value_id, attr),
                        };
                        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
                        const op = c.mlirOperationCreate(&state);
                        c.mlirBlockAppendOwnedOperation(block, op);
                        break :blk_address c.mlirOperationGetResult(op, 0);
                    },
                    .Hex => |hex_lit| blk_hex: {
                        // Parse hex literal and create integer constant
                        const ty = c.mlirIntegerTypeGet(ctx_, 256);
                        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), fileLoc(ctx_, hex_lit.span));
                        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ty));

                        // Parse hex value (remove 0x prefix if present)
                        const hex_str = if (std.mem.startsWith(u8, hex_lit.value, "0x"))
                            hex_lit.value[2..]
                        else
                            hex_lit.value;
                        const parsed: i64 = std.fmt.parseInt(i64, hex_str, 16) catch 0;
                        const attr = c.mlirIntegerAttrGet(ty, parsed);

                        const value_id = c.mlirIdentifierGet(ctx_, c.mlirStringRefCreateFromCString("value"));
                        var attrs = [_]c.MlirNamedAttribute{
                            c.mlirNamedAttributeGet(value_id, attr),
                        };
                        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
                        const op = c.mlirOperationCreate(&state);
                        c.mlirBlockAppendOwnedOperation(block, op);
                        break :blk_hex c.mlirOperationGetResult(op, 0);
                    },
                    .Binary => |bin_lit| blk_binary: {
                        // Parse binary literal and create integer constant
                        const ty = c.mlirIntegerTypeGet(ctx_, 256);
                        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), fileLoc(ctx_, bin_lit.span));
                        c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ty));

                        // Parse binary value (remove 0b prefix if present)
                        const bin_str = if (std.mem.startsWith(u8, bin_lit.value, "0b"))
                            bin_lit.value[2..]
                        else
                            bin_lit.value;
                        const parsed: i64 = std.fmt.parseInt(i64, bin_str, 2) catch 0;
                        const attr = c.mlirIntegerAttrGet(ty, parsed);

                        const value_id = c.mlirIdentifierGet(ctx_, c.mlirStringRefCreateFromCString("value"));
                        var attrs = [_]c.MlirNamedAttribute{
                            c.mlirNamedAttributeGet(value_id, attr),
                        };
                        c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
                        const op = c.mlirOperationCreate(&state);
                        c.mlirBlockAppendOwnedOperation(block, op);
                        break :blk_binary c.mlirOperationGetResult(op, 0);
                    },
                },
                .Binary => |bin| {
                    const lhs = lowerExpr(ctx_, block, bin.lhs, param_map, storage_map, local_var_map);
                    const rhs = lowerExpr(ctx_, block, bin.rhs, param_map, storage_map, local_var_map);
                    const result_ty = c.mlirIntegerTypeGet(ctx_, 256);

                    switch (bin.operator) {
                        // Arithmetic operators
                        .Plus => {
                            var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.addi"), fileLoc(ctx_, bin.span));
                            c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ lhs, rhs }));
                            c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));
                            const op = c.mlirOperationCreate(&state);

                            // Note: MLIR operations get their names from the operation state
                            // We can't set names after creation, but the variable naming system
                            // helps with debugging and understanding the generated IR

                            c.mlirBlockAppendOwnedOperation(block, op);
                            return c.mlirOperationGetResult(op, 0);
                        },
                        .Minus => {
                            var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.subi"), fileLoc(ctx_, bin.span));
                            c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ lhs, rhs }));
                            c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));
                            const op = c.mlirOperationCreate(&state);

                            // Note: MLIR operations get their names from the operation state
                            // We can't set names after creation, but the variable naming system
                            // helps with debugging and understanding the generated IR

                            c.mlirBlockAppendOwnedOperation(block, op);
                            return c.mlirOperationGetResult(op, 0);
                        },
                        .Star => {
                            var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.muli"), fileLoc(ctx_, bin.span));
                            c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ lhs, rhs }));
                            c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));
                            const op = c.mlirOperationCreate(&state);

                            // Note: MLIR operations get their names from the operation state
                            // We can't set names after creation, but the variable naming system
                            // helps with debugging and understanding the generated IR

                            c.mlirBlockAppendOwnedOperation(block, op);
                            return c.mlirOperationGetResult(op, 0);
                        },
                        .Slash => {
                            var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.divsi"), fileLoc(ctx_, bin.span));
                            c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ lhs, rhs }));
                            c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));
                            const op = c.mlirOperationCreate(&state);
                            c.mlirBlockAppendOwnedOperation(block, op);
                            return c.mlirOperationGetResult(op, 0);
                        },
                        .Percent => {
                            var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.remsi"), fileLoc(ctx_, bin.span));
                            c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ lhs, rhs }));
                            c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));
                            const op = c.mlirOperationCreate(&state);
                            c.mlirBlockAppendOwnedOperation(block, op);
                            return c.mlirOperationGetResult(op, 0);
                        },
                        .StarStar => {
                            // Power operation - for now use multiplication as placeholder
                            // TODO: Implement proper power operation
                            var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.muli"), fileLoc(ctx_, bin.span));
                            c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ lhs, rhs }));
                            c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));
                            const op = c.mlirOperationCreate(&state);
                            c.mlirBlockAppendOwnedOperation(block, op);
                            return c.mlirOperationGetResult(op, 0);
                        },

                        // Comparison operators
                        .EqualEqual => {
                            var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.cmpi"), fileLoc(ctx_, bin.span));
                            c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ lhs, rhs }));
                            c.mlirOperationStateAddResults(&state, 1, @ptrCast(&c.mlirIntegerTypeGet(ctx_, 1)));
                            const eq_attr = c.mlirStringRefCreateFromCString("eq");
                            const predicate_id = c.mlirIdentifierGet(ctx_, c.mlirStringRefCreateFromCString("predicate"));
                            const eq_attr_value = c.mlirStringAttrGet(ctx_, eq_attr);
                            var attrs = [_]c.MlirNamedAttribute{
                                c.mlirNamedAttributeGet(predicate_id, eq_attr_value),
                            };
                            c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
                            const op = c.mlirOperationCreate(&state);
                            c.mlirBlockAppendOwnedOperation(block, op);
                            return c.mlirOperationGetResult(op, 0);
                        },
                        .BangEqual => {
                            var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.cmpi"), fileLoc(ctx_, bin.span));
                            c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ lhs, rhs }));
                            c.mlirOperationStateAddResults(&state, 1, @ptrCast(&c.mlirIntegerTypeGet(ctx_, 1)));
                            const ne_attr = c.mlirStringRefCreateFromCString("ne");
                            const predicate_id = c.mlirIdentifierGet(ctx_, c.mlirStringRefCreateFromCString("predicate"));
                            const ne_attr_value = c.mlirStringAttrGet(ctx_, ne_attr);
                            var attrs = [_]c.MlirNamedAttribute{
                                c.mlirNamedAttributeGet(predicate_id, ne_attr_value),
                            };
                            c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
                            const op = c.mlirOperationCreate(&state);
                            c.mlirBlockAppendOwnedOperation(block, op);
                            return c.mlirOperationGetResult(op, 0);
                        },
                        .Less => {
                            var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.cmpi"), fileLoc(ctx_, bin.span));
                            c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ lhs, rhs }));
                            c.mlirOperationStateAddResults(&state, 1, @ptrCast(&c.mlirIntegerTypeGet(ctx_, 1)));
                            const ult_attr = c.mlirStringRefCreateFromCString("ult");
                            const predicate_id = c.mlirIdentifierGet(ctx_, c.mlirStringRefCreateFromCString("predicate"));
                            const ult_attr_value = c.mlirStringAttrGet(ctx_, ult_attr);
                            var attrs = [_]c.MlirNamedAttribute{
                                c.mlirNamedAttributeGet(predicate_id, ult_attr_value),
                            };
                            c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
                            const op = c.mlirOperationCreate(&state);
                            c.mlirBlockAppendOwnedOperation(block, op);
                            return c.mlirOperationGetResult(op, 0);
                        },
                        .LessEqual => {
                            var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.cmpi"), fileLoc(ctx_, bin.span));
                            c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ lhs, rhs }));
                            c.mlirOperationStateAddResults(&state, 1, @ptrCast(&c.mlirIntegerTypeGet(ctx_, 1)));
                            const ule_attr = c.mlirStringRefCreateFromCString("ule");
                            const predicate_id = c.mlirIdentifierGet(ctx_, c.mlirStringRefCreateFromCString("predicate"));
                            const ule_attr_value = c.mlirStringAttrGet(ctx_, ule_attr);
                            var attrs = [_]c.MlirNamedAttribute{
                                c.mlirNamedAttributeGet(predicate_id, ule_attr_value),
                            };
                            c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
                            const op = c.mlirOperationCreate(&state);
                            c.mlirBlockAppendOwnedOperation(block, op);
                            return c.mlirOperationGetResult(op, 0);
                        },
                        .Greater => {
                            var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.cmpi"), fileLoc(ctx_, bin.span));
                            c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ lhs, rhs }));
                            c.mlirOperationStateAddResults(&state, 1, @ptrCast(&c.mlirIntegerTypeGet(ctx_, 1)));
                            const ugt_attr = c.mlirStringRefCreateFromCString("ugt");
                            const predicate_id = c.mlirIdentifierGet(ctx_, c.mlirStringRefCreateFromCString("predicate"));
                            const ugt_attr_value = c.mlirStringAttrGet(ctx_, ugt_attr);
                            var attrs = [_]c.MlirNamedAttribute{
                                c.mlirNamedAttributeGet(predicate_id, ugt_attr_value),
                            };
                            c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
                            const op = c.mlirOperationCreate(&state);

                            // Note: MLIR operations get their names from the operation state
                            // We can't set names after creation, but the variable naming system
                            // helps with debugging and understanding the generated IR

                            c.mlirBlockAppendOwnedOperation(block, op);
                            return c.mlirOperationGetResult(op, 0);
                        },
                        .GreaterEqual => {
                            var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.cmpi"), fileLoc(ctx_, bin.span));
                            c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ lhs, rhs }));
                            c.mlirOperationStateAddResults(&state, 1, @ptrCast(&c.mlirIntegerTypeGet(ctx_, 1)));
                            const uge_attr = c.mlirStringRefCreateFromCString("uge");
                            const predicate_id = c.mlirIdentifierGet(ctx_, c.mlirStringRefCreateFromCString("predicate"));
                            const uge_attr_value = c.mlirStringAttrGet(ctx_, uge_attr);
                            var attrs = [_]c.MlirNamedAttribute{
                                c.mlirNamedAttributeGet(predicate_id, uge_attr_value),
                            };
                            c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
                            const op = c.mlirOperationCreate(&state);
                            c.mlirBlockAppendOwnedOperation(block, op);
                            return c.mlirOperationGetResult(op, 0);
                        },

                        // Logical operators
                        .And => {
                            var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.andi"), fileLoc(ctx_, bin.span));
                            c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ lhs, rhs }));
                            c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));
                            const op = c.mlirOperationCreate(&state);
                            c.mlirBlockAppendOwnedOperation(block, op);
                            return c.mlirOperationGetResult(op, 0);
                        },
                        .Or => {
                            var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.ori"), fileLoc(ctx_, bin.span));
                            c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ lhs, rhs }));
                            c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));
                            const op = c.mlirOperationCreate(&state);
                            c.mlirBlockAppendOwnedOperation(block, op);
                            return c.mlirOperationGetResult(op, 0);
                        },

                        // Bitwise operators
                        .BitwiseAnd => {
                            var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.andi"), fileLoc(ctx_, bin.span));
                            c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ lhs, rhs }));
                            c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));
                            const op = c.mlirOperationCreate(&state);
                            c.mlirBlockAppendOwnedOperation(block, op);
                            return c.mlirOperationGetResult(op, 0);
                        },
                        .BitwiseOr => {
                            var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.ori"), fileLoc(ctx_, bin.span));
                            c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ lhs, rhs }));
                            c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));
                            const op = c.mlirOperationCreate(&state);
                            c.mlirBlockAppendOwnedOperation(block, op);
                            return c.mlirOperationGetResult(op, 0);
                        },
                        .BitwiseXor => {
                            var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.xori"), fileLoc(ctx_, bin.span));
                            c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ lhs, rhs }));
                            c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));
                            const op = c.mlirOperationCreate(&state);
                            c.mlirBlockAppendOwnedOperation(block, op);
                            return c.mlirOperationGetResult(op, 0);
                        },
                        .LeftShift => {
                            var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.shli"), fileLoc(ctx_, bin.span));
                            c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ lhs, rhs }));
                            c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));
                            const op = c.mlirOperationCreate(&state);
                            c.mlirBlockAppendOwnedOperation(block, op);
                            return c.mlirOperationGetResult(op, 0);
                        },
                        .RightShift => {
                            var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.shrsi"), fileLoc(ctx_, bin.span));
                            c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{ lhs, rhs }));
                            c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));
                            const op = c.mlirOperationCreate(&state);
                            c.mlirBlockAppendOwnedOperation(block, op);
                            return c.mlirOperationGetResult(op, 0);
                        },

                        // Comma operator - just return the right operand
                        .Comma => {
                            return rhs;
                        },
                    }
                },
                .Unary => |unary| {
                    const operand = lowerExpr(ctx_, block, unary.operand, param_map, storage_map, local_var_map);
                    const result_ty = c.mlirIntegerTypeGet(ctx_, 256);

                    switch (unary.operator) {
                        .Minus => {
                            // Unary minus: -x
                            var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.subi"), fileLoc(ctx_, unary.span));
                            c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{
                                // Subtract from zero: 0 - x = -x
                                c.mlirOperationGetResult(createConstant(ctx_, block, 0, unary.span), 0),
                                operand,
                            }));
                            c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));
                            const op = c.mlirOperationCreate(&state);
                            c.mlirBlockAppendOwnedOperation(block, op);
                            return c.mlirOperationGetResult(op, 0);
                        },
                        .Bang => {
                            // Logical NOT: !x
                            var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.xori"), fileLoc(ctx_, unary.span));
                            c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{
                                operand,
                                // XOR with 1: x ^ 1 = !x (for boolean values)
                                c.mlirOperationGetResult(createConstant(ctx_, block, 1, unary.span), 0),
                            }));
                            c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));
                            const op = c.mlirOperationCreate(&state);
                            c.mlirBlockAppendOwnedOperation(block, op);
                            return c.mlirOperationGetResult(op, 0);
                        },
                        .BitNot => {
                            // Bitwise NOT: ~x
                            var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.xori"), fileLoc(ctx_, unary.span));
                            c.mlirOperationStateAddOperands(&state, 2, @ptrCast(&[_]c.MlirValue{
                                operand,
                                // XOR with -1: x ^ (-1) = ~x
                                c.mlirOperationGetResult(createConstant(ctx_, block, -1, unary.span), 0),
                            }));
                            c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));
                            const op = c.mlirOperationCreate(&state);
                            c.mlirBlockAppendOwnedOperation(block, op);
                            return c.mlirOperationGetResult(op, 0);
                        },
                    }
                },
                .Call => |call| {
                    // Lower all arguments first
                    var args = std.ArrayList(c.MlirValue).init(std.heap.page_allocator);
                    defer args.deinit();

                    for (call.arguments) |arg| {
                        const arg_value = lowerExpr(ctx_, block, arg, param_map, storage_map, local_var_map);
                        args.append(arg_value) catch @panic("Failed to append argument");
                    }

                    // For now, assume the callee is an identifier (function name)
                    // TODO: Handle more complex callee expressions
                    switch (call.callee.*) {
                        .Identifier => |ident| {
                            // Create a function call operation
                            // Note: This is a simplified approach - in a real implementation,
                            // we'd need to look up the function signature and handle types properly
                            const result_ty = c.mlirIntegerTypeGet(ctx_, 256); // Default to i256 for now

                            var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("func.call"), fileLoc(ctx_, call.span));
                            c.mlirOperationStateAddOperands(&state, @intCast(args.items.len), args.items.ptr);
                            c.mlirOperationStateAddResults(&state, 1, @ptrCast(&result_ty));

                            // Add the callee name as a string attribute
                            // Create a null-terminated string for the callee name
                            // Create a proper C string from the slice
                            var callee_buffer: [256]u8 = undefined;
                            for (0..ident.name.len) |i| {
                                callee_buffer[i] = ident.name[i];
                            }
                            callee_buffer[ident.name.len] = 0; // null-terminate
                            const callee_str = c.mlirStringRefCreateFromCString(&callee_buffer[0]);
                            const callee_attr = c.mlirStringAttrGet(ctx_, callee_str);
                            const callee_id = c.mlirIdentifierGet(ctx_, c.mlirStringRefCreateFromCString("callee"));
                            var attrs = [_]c.MlirNamedAttribute{
                                c.mlirNamedAttributeGet(callee_id, callee_attr),
                            };
                            c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

                            const op = c.mlirOperationCreate(&state);
                            c.mlirBlockAppendOwnedOperation(block, op);
                            return c.mlirOperationGetResult(op, 0);
                        },
                        else => {
                            // For now, panic on complex callee expressions
                            std.debug.print("DEBUG: Unhandled callee type: {s}\n", .{@tagName(call.callee.*)});
                            @panic("Complex callee expressions not yet supported");
                        },
                    }
                },
                .Identifier => |ident| {
                    // First check if this is a function parameter
                    if (param_map) |pm| {
                        if (pm.getParamIndex(ident.name)) |param_index| {
                            // This is a function parameter - get the actual block argument
                            if (pm.getBlockArgument(ident.name)) |block_arg| {
                                std.debug.print("DEBUG: Function parameter {s} at index {d} - using block argument\n", .{ ident.name, param_index });
                                return block_arg;
                            } else {
                                // Fallback to dummy value if block argument not found
                                std.debug.print("DEBUG: Function parameter {s} at index {d} - block argument not found, using dummy value\n", .{ ident.name, param_index });
                                const ty = c.mlirIntegerTypeGet(ctx_, 256);
                                var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), fileLoc(ctx_, ident.span));
                                c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ty));
                                const attr = c.mlirIntegerAttrGet(ty, 0);
                                const value_id = c.mlirIdentifierGet(ctx_, c.mlirStringRefCreateFromCString("value"));
                                var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(value_id, attr)};
                                c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
                                const op = c.mlirOperationCreate(&state);
                                c.mlirBlockAppendOwnedOperation(block, op);
                                return c.mlirOperationGetResult(op, 0);
                            }
                        }
                    }

                    // Check if this is a local variable
                    if (local_var_map) |lvm| {
                        if (lvm.hasLocalVar(ident.name)) {
                            // This is a local variable - return the stored value directly
                            std.debug.print("DEBUG: Loading local variable: {s}\n", .{ident.name});
                            return lvm.getLocalVar(ident.name).?;
                        }
                    }

                    // Check if we have a storage map and if this variable exists in storage
                    var is_storage_variable = false;
                    if (storage_map) |sm| {
                        if (sm.hasStorageVariable(ident.name)) {
                            is_storage_variable = true;
                            // Ensure the variable exists in storage (create if needed)
                            _ = sm.getOrCreateAddress(ident.name) catch 0;
                        }
                    }

                    if (is_storage_variable) {
                        // This is a storage variable - use ora.sload
                        std.debug.print("DEBUG: Loading storage variable: {s}\n", .{ident.name});

                        // Use our new storage-type-aware load operation
                        const load_op = createLoadOperation(ctx_, ident.name, .Storage, ident.span);
                        c.mlirBlockAppendOwnedOperation(block, load_op);
                        return c.mlirOperationGetResult(load_op, 0);
                    } else {
                        // This is a local variable - load from the allocated memory
                        std.debug.print("DEBUG: Loading local variable: {s}\n", .{ident.name});

                        // Get the local variable reference from our map
                        if (local_var_map) |lvm| {
                            if (lvm.getLocalVar(ident.name)) |local_var_ref| {
                                // Load the value from the allocated memory
                                var load_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("scf.load"), fileLoc(ctx_, ident.span));

                                // Add the local variable reference as operand
                                c.mlirOperationStateAddOperands(&load_state, 1, @ptrCast(&local_var_ref));

                                // Add the result type (the type of the stored value)
                                const var_type = c.mlirValueGetType(local_var_ref);
                                const memref_type = c.mlirShapedTypeGetElementType(var_type);
                                c.mlirOperationStateAddResults(&load_state, 1, @ptrCast(&memref_type));

                                const load_op = c.mlirOperationCreate(&load_state);
                                c.mlirBlockAppendOwnedOperation(block, load_op);
                                return c.mlirOperationGetResult(load_op, 0);
                            }
                        }

                        // If we can't find the local variable, this is an error
                        std.debug.print("ERROR: Local variable not found: {s}\n", .{ident.name});
                        // For now, return a dummy value to avoid crashes
                        return c.mlirBlockGetArgument(block, 0);
                    }
                },
                .SwitchExpression => |switch_expr| blk_switch: {
                    // For now, just lower the condition and return a placeholder
                    // TODO: Implement proper switch expression lowering
                    _ = lowerExpr(ctx_, block, switch_expr.condition, param_map, storage_map, local_var_map);
                    const ty = c.mlirIntegerTypeGet(ctx_, 256); // Default to i256
                    var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), fileLoc(ctx_, switch_expr.span));
                    c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ty));
                    const attr = c.mlirIntegerAttrGet(ty, 0);
                    const value_id = c.mlirIdentifierGet(ctx_, c.mlirStringRefCreateFromCString("value"));
                    var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(value_id, attr)};
                    c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
                    const op = c.mlirOperationCreate(&state);
                    c.mlirBlockAppendOwnedOperation(block, op);
                    break :blk_switch c.mlirOperationGetResult(op, 0);
                },
                .Index => |index_expr| blk_index: {
                    // Lower the target (array/map) and index expressions
                    const target_value = lowerExpr(ctx_, block, index_expr.target, param_map, storage_map, local_var_map);
                    const index_value = lowerExpr(ctx_, block, index_expr.index, param_map, storage_map, local_var_map);

                    // Calculate the memory address: base_address + (index * element_size)
                    // For now, assume element_size is 32 bytes (256 bits) for most types
                    const element_size = c.mlirIntegerTypeGet(ctx_, 256);
                    const element_size_const = c.mlirIntegerAttrGet(element_size, 32);
                    const element_size_id = c.mlirIdentifierGet(ctx_, c.mlirStringRefCreateFromCString("value"));
                    var element_size_attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(element_size_id, element_size_const)};

                    var element_size_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), fileLoc(ctx_, index_expr.span));
                    c.mlirOperationStateAddResults(&element_size_state, 1, @ptrCast(&element_size));
                    c.mlirOperationStateAddAttributes(&element_size_state, element_size_attrs.len, &element_size_attrs);
                    const element_size_op = c.mlirOperationCreate(&element_size_state);
                    c.mlirBlockAppendOwnedOperation(block, element_size_op);
                    const element_size_value = c.mlirOperationGetResult(element_size_op, 0);

                    // Multiply index by element size: index * element_size
                    var mul_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.muli"), fileLoc(ctx_, index_expr.span));
                    c.mlirOperationStateAddResults(&mul_state, 1, @ptrCast(&element_size));
                    c.mlirOperationStateAddOperands(&mul_state, 2, @ptrCast(&[_]c.MlirValue{ index_value, element_size_value }));
                    const mul_op = c.mlirOperationCreate(&mul_state);
                    c.mlirBlockAppendOwnedOperation(block, mul_op);
                    const offset_value = c.mlirOperationGetResult(mul_op, 0);

                    // Add base address to offset: base_address + offset
                    var add_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.addi"), fileLoc(ctx_, index_expr.span));
                    c.mlirOperationStateAddResults(&add_state, 1, @ptrCast(&element_size));
                    c.mlirOperationStateAddOperands(&add_state, 2, @ptrCast(&[_]c.MlirValue{ target_value, offset_value }));
                    const add_op = c.mlirOperationCreate(&add_state);
                    c.mlirBlockAppendOwnedOperation(block, add_op);
                    const final_address = c.mlirOperationGetResult(add_op, 0);

                    // Load from the calculated address using memref.load
                    var load_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("memref.load"), fileLoc(ctx_, index_expr.span));
                    c.mlirOperationStateAddResults(&load_state, 1, @ptrCast(&element_size));
                    c.mlirOperationStateAddOperands(&load_state, 1, @ptrCast(&final_address));
                    const load_op = c.mlirOperationCreate(&load_state);
                    c.mlirBlockAppendOwnedOperation(block, load_op);
                    break :blk_index c.mlirOperationGetResult(load_op, 0);
                },
                .FieldAccess => |field_access| blk_field: {
                    // For now, just lower the target expression and return a placeholder
                    // TODO: Add proper field access handling with struct.extract
                    _ = lowerExpr(ctx_, block, field_access.target, param_map, storage_map, local_var_map);
                    const ty = c.mlirIntegerTypeGet(ctx_, 256); // Default to i256
                    var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), fileLoc(ctx_, field_access.span));
                    c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ty));
                    const attr = c.mlirIntegerAttrGet(ty, 0);
                    const value_id = c.mlirIdentifierGet(ctx_, c.mlirStringRefCreateFromCString("value"));
                    var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(value_id, attr)};
                    c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
                    const op = c.mlirOperationCreate(&state);
                    c.mlirBlockAppendOwnedOperation(block, op);
                    break :blk_field c.mlirOperationGetResult(op, 0);
                },
                else => {
                    // Debug: print the unhandled expression type
                    std.debug.print("Unhandled expression type: {s}\n", .{@tagName(expr.*)});
                    @panic("Unhandled expression type in MLIR lowering");
                },
            };
        }

        // TODO: Move fileLoc to locations.zig - this is duplicated code
        fn fileLoc(ctx_: c.MlirContext, span: lib.ast.SourceSpan) c.MlirLocation {
            const fname = c.mlirStringRefCreateFromCString("input.ora");
            return c.mlirLocationFileLineColGet(ctx_, fname, span.line, span.column);
        }

        // TODO: Move createConstant to expressions.zig - this is duplicated code
        fn createConstant(ctx_: c.MlirContext, block: c.MlirBlock, value: i64, span: lib.ast.SourceSpan) c.MlirOperation {
            const ty = c.mlirIntegerTypeGet(ctx_, 256);
            var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), fileLoc(ctx_, span));
            c.mlirOperationStateAddResults(&state, 1, @ptrCast(&ty));
            const attr = c.mlirIntegerAttrGet(ty, @intCast(value));
            const value_id = c.mlirIdentifierGet(ctx_, c.mlirStringRefCreateFromCString("value"));
            var attrs = [_]c.MlirNamedAttribute{
                c.mlirNamedAttributeGet(value_id, attr),
            };
            c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
            const op = c.mlirOperationCreate(&state);
            c.mlirBlockAppendOwnedOperation(block, op);
            return op;
        }

        // Use the modular statement lowerer instead of the duplicated code
        fn lowerStmt(ctx_: c.MlirContext, block: c.MlirBlock, stmt: *const lib.ast.Statements.StmtNode, param_map: ?*const ParamMap, storage_map: ?*StorageMap, local_var_map: ?*LocalVarMap) void {
            const type_mapper = @import("types.zig").TypeMapper.init(ctx_);
            const expr_lowerer = @import("expressions.zig").ExpressionLowerer.init(ctx_, block, &type_mapper, param_map, storage_map, local_var_map);
            const stmt_lowerer = @import("statements.zig").StatementLowerer.init(ctx_, block, &type_mapper, &expr_lowerer, param_map, storage_map, local_var_map);
            stmt_lowerer.lowerStatement(stmt);
        }

        // Use the modular block body lowerer instead of the duplicated code
        fn lowerBlockBody(ctx_: c.MlirContext, b: lib.ast.Statements.BlockNode, block: c.MlirBlock, param_map: ?*const ParamMap, storage_map: ?*StorageMap, local_var_map: ?*LocalVarMap) void {
            const type_mapper = @import("types.zig").TypeMapper.init(ctx_);
            const expr_lowerer = @import("expressions.zig").ExpressionLowerer.init(ctx_, block, &type_mapper, param_map, storage_map, local_var_map);
            const stmt_lowerer = @import("statements.zig").StatementLowerer.init(ctx_, block, &type_mapper, &expr_lowerer, param_map, storage_map, local_var_map);
            stmt_lowerer.lowerBlockBody(b, block);
        }
    };

    // Use the modular declaration lowerer instead of the duplicated code
    const createGlobalDeclaration = struct {
        fn create(ctx_: c.MlirContext, loc_: c.MlirLocation, var_decl: lib.ast.Statements.VariableDeclNode) c.MlirOperation {
            _ = loc_; // Not used in the modular version
            const type_mapper = @import("types.zig").TypeMapper.init(ctx_);
            const locations = @import("locations.zig").LocationTracker.init(ctx_);
            const decl_lowerer = @import("declarations.zig").DeclarationLowerer.init(ctx_, &type_mapper, locations);
            return decl_lowerer.createGlobalDeclaration(&var_decl);
        }
    };

    // Use the modular declaration lowerer instead of the duplicated code
    const createMemoryGlobalDeclaration = struct {
        fn create(ctx_: c.MlirContext, loc_: c.MlirLocation, var_decl: lib.ast.Statements.VariableDeclNode) c.MlirOperation {
            _ = loc_; // Not used in the modular version
            const type_mapper = @import("types.zig").TypeMapper.init(ctx_);
            const locations = @import("locations.zig").LocationTracker.init(ctx_);
            const decl_lowerer = @import("declarations.zig").DeclarationLowerer.init(ctx_, &type_mapper, locations);
            return decl_lowerer.createMemoryGlobalDeclaration(&var_decl);
        }
    };

    // Use the modular declaration lowerer instead of the duplicated code
    const createTStoreGlobalDeclaration = struct {
        fn create(ctx_: c.MlirContext, loc_: c.MlirLocation, var_decl: lib.ast.Statements.VariableDeclNode) c.MlirOperation {
            _ = loc_; // Not used in the modular version
            const type_mapper = @import("types.zig").TypeMapper.init(ctx_);
            const locations = @import("locations.zig").LocationTracker.init(ctx_);
            const decl_lowerer = @import("declarations.zig").DeclarationLowerer.init(ctx_, &type_mapper, locations);
            return decl_lowerer.createTStoreGlobalDeclaration(&var_decl);
        }
    };

    // Use the modular declaration lowerer instead of the duplicated code
    const Emit = struct {
        fn create(ctx_: c.MlirContext, loc_: c.MlirLocation, sym_id: c.MlirIdentifier, type_id: c.MlirIdentifier, f: lib.FunctionNode, contract_storage_map: ?*Lower.StorageMap, local_var_map: ?*Lower.LocalVarMap) c.MlirOperation {
            _ = loc_; // Not used in the modular version
            _ = sym_id; // Not used in the modular version
            _ = type_id; // Not used in the modular version
            const type_mapper = @import("types.zig").TypeMapper.init(ctx_);
            const locations = @import("locations.zig").LocationTracker.init(ctx_);
            const decl_lowerer = @import("declarations.zig").DeclarationLowerer.init(ctx_, &type_mapper, locations);
            return decl_lowerer.lowerFunction(&f, contract_storage_map, local_var_map);
        }
    };

    // end helpers

    for (nodes) |node| {
        switch (node) {
            .Function => |f| {
                var local_var_map = Lower.LocalVarMap.init(std.heap.page_allocator);
                defer local_var_map.deinit();
                const func_op = Emit.create(ctx, loc, sym_name_id, fn_type_id, f, null, &local_var_map);
                c.mlirBlockAppendOwnedOperation(body, func_op);
            },
            .Contract => |contract| {
                // First pass: collect all storage variables and create a shared StorageMap
                var storage_map = Lower.StorageMap.init(std.heap.page_allocator);
                defer storage_map.deinit();

                for (contract.body) |child| {
                    switch (child) {
                        .VariableDecl => |var_decl| {
                            switch (var_decl.region) {
                                .Storage => {
                                    // This is a storage variable - add it to the storage map
                                    _ = storage_map.addStorageVariable(var_decl.name, var_decl.span) catch {};
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
                            var local_var_map = Lower.LocalVarMap.init(std.heap.page_allocator);
                            defer local_var_map.deinit();
                            const func_op = Emit.create(ctx, loc, sym_name_id, fn_type_id, f, &storage_map, &local_var_map);
                            c.mlirBlockAppendOwnedOperation(body, func_op);
                        },
                        .VariableDecl => |var_decl| {
                            switch (var_decl.region) {
                                .Storage => {
                                    // Create ora.global operation for storage variables
                                    const global_op = createGlobalDeclaration.create(ctx, loc, var_decl);
                                    c.mlirBlockAppendOwnedOperation(body, global_op);
                                },
                                .Memory => {
                                    // Create ora.memory.global operation for memory variables
                                    const memory_global_op = createMemoryGlobalDeclaration.create(ctx, loc, var_decl);
                                    c.mlirBlockAppendOwnedOperation(body, memory_global_op);
                                },
                                .TStore => {
                                    // Create ora.tstore.global operation for transient storage variables
                                    const tstore_global_op = createTStoreGlobalDeclaration.create(ctx, loc, var_decl);
                                    c.mlirBlockAppendOwnedOperation(body, tstore_global_op);
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
            },
            else => {
                @panic("Unhandled top-level node type in MLIR lowering");
            },
        }
    }

    return module;
}
