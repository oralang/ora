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

    // Helper to build function type from parameter/return TypeInfo
    const Build = struct {
        fn funcType(ctx_: c.MlirContext, f: lib.FunctionNode) c.MlirType {
            const num_params: usize = f.parameters.len;
            var params_buf: [16]c.MlirType = undefined;
            var dyn_params: []c.MlirType = params_buf[0..0];
            if (num_params > params_buf.len) {
                dyn_params = std.heap.page_allocator.alloc(c.MlirType, num_params) catch unreachable;
            } else {
                dyn_params = params_buf[0..num_params];
            }

            // Create a type mapper for this function
            const type_mapper = @import("types.zig").TypeMapper.init(ctx_);

            for (f.parameters, 0..) |p, i| dyn_params[i] = type_mapper.toMlirType(p.type_info);
            const ret_ti = f.return_type_info;
            var ret_types: [1]c.MlirType = undefined;
            var ret_count: usize = 0;
            if (ret_ti) |r| switch (r.ora_type orelse .void) {
                .void => ret_count = 0,
                else => {
                    ret_types[0] = type_mapper.toMlirType(r);
                    ret_count = 1;
                },
            } else ret_count = 0;
            const in_ptr: [*c]const c.MlirType = if (dyn_params.len == 0) @ptrFromInt(0) else @ptrCast(&dyn_params[0]);
            const out_ptr: [*c]const c.MlirType = if (ret_count == 0) @ptrFromInt(0) else @ptrCast(&ret_types);
            const ty = c.mlirFunctionTypeGet(ctx_, @intCast(dyn_params.len), in_ptr, @intCast(ret_count), out_ptr);
            if (@intFromPtr(dyn_params.ptr) != @intFromPtr(&params_buf[0])) std.heap.page_allocator.free(dyn_params);
            return ty;
        }
    };
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

        // TODO: Move StorageMap to memory.zig - this is duplicated code
        const StorageMap = struct {
            variables: std.StringHashMap(usize), // variable name -> storage address
            next_address: usize,

            fn init(allocator: std.mem.Allocator) StorageMap {
                return .{
                    .variables = std.StringHashMap(usize).init(allocator),
                    .next_address = 0,
                };
            }

            fn deinit(self: *StorageMap) void {
                self.variables.deinit();
            }

            fn getOrCreateAddress(self: *StorageMap, name: []const u8) !usize {
                if (self.variables.get(name)) |addr| {
                    return addr;
                }
                const addr = self.next_address;
                try self.variables.put(name, addr);
                self.next_address += 1;
                return addr;
            }

            fn getStorageAddress(self: *StorageMap, name: []const u8) ?usize {
                return self.variables.get(name);
            }

            fn addStorageVariable(self: *StorageMap, name: []const u8, _: lib.ast.SourceSpan) !usize {
                const addr = try self.getOrCreateAddress(name);
                return addr;
            }

            fn hasStorageVariable(self: *StorageMap, name: []const u8) bool {
                return self.variables.contains(name);
            }
        };

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

        // TODO: Move LocalVarMap to symbols.zig - this is duplicated code
        const LocalVarMap = struct {
            variables: std.StringHashMap(c.MlirValue),
            allocator: std.mem.Allocator,

            fn init(allocator: std.mem.Allocator) LocalVarMap {
                return .{
                    .variables = std.StringHashMap(c.MlirValue).init(allocator),
                    .allocator = allocator,
                };
            }

            fn deinit(self: *LocalVarMap) void {
                self.variables.deinit();
            }

            fn addLocalVar(self: *LocalVarMap, name: []const u8, value: c.MlirValue) !void {
                try self.variables.put(name, value);
            }

            fn getLocalVar(self: *const LocalVarMap, name: []const u8) ?c.MlirValue {
                return self.variables.get(name);
            }

            fn hasLocalVar(self: *const LocalVarMap, name: []const u8) bool {
                return self.variables.contains(name);
            }
        };

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

        // TODO: Move lowerStmt to statements.zig - this is duplicated code
        fn lowerStmt(ctx_: c.MlirContext, block: c.MlirBlock, stmt: *const lib.ast.Statements.StmtNode, param_map: ?*const ParamMap, storage_map: ?*StorageMap, local_var_map: ?*LocalVarMap) void {
            switch (stmt.*) {
                .Return => |ret| {
                    var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("func.return"), fileLoc(ctx_, ret.span));
                    if (ret.value) |e| {
                        const v = lowerExpr(ctx_, block, &e, param_map, storage_map, local_var_map);
                        c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&v));
                    }
                    const op = c.mlirOperationCreate(&state);
                    c.mlirBlockAppendOwnedOperation(block, op);
                },
                .VariableDecl => |var_decl| {
                    std.debug.print("DEBUG: Processing variable declaration: {s} (region: {s})\n", .{ var_decl.name, @tagName(var_decl.region) });
                    // Handle variable declarations based on memory region
                    switch (var_decl.region) {
                        .Stack => {
                            // This is a local variable - we need to handle it properly
                            if (var_decl.value) |init_expr| {
                                // Lower the initializer expression
                                const init_value = lowerExpr(ctx_, block, &init_expr.*, param_map, storage_map, local_var_map);

                                // Store the local variable in our map for later reference
                                if (local_var_map) |lvm| {
                                    lvm.addLocalVar(var_decl.name, init_value) catch {
                                        std.debug.print("WARNING: Failed to add local variable to map: {s}\n", .{var_decl.name});
                                    };
                                }
                            } else {
                                // Local variable without initializer - create a default value and store it
                                if (local_var_map) |lvm| {
                                    // Create a default value (0 for now)
                                    const default_ty = c.mlirIntegerTypeGet(ctx_, 256);
                                    var const_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), fileLoc(ctx_, var_decl.span));
                                    c.mlirOperationStateAddResults(&const_state, 1, @ptrCast(&default_ty));
                                    const attr = c.mlirIntegerAttrGet(default_ty, 0);
                                    const value_id = c.mlirIdentifierGet(ctx_, c.mlirStringRefCreateFromCString("value"));
                                    var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(value_id, attr)};
                                    c.mlirOperationStateAddAttributes(&const_state, attrs.len, &attrs);
                                    const const_op = c.mlirOperationCreate(&const_state);
                                    c.mlirBlockAppendOwnedOperation(block, const_op);
                                    const default_value = c.mlirOperationGetResult(const_op, 0);

                                    lvm.addLocalVar(var_decl.name, default_value) catch {
                                        std.debug.print("WARNING: Failed to add local variable to map: {s}\n", .{var_decl.name});
                                    };
                                    std.debug.print("DEBUG: Added local variable to map: {s}\n", .{var_decl.name});
                                }
                            }
                        },
                        .Storage => {
                            // Storage variables are handled at the contract level
                            // Just lower the initializer if present
                            if (var_decl.value) |init_expr| {
                                _ = lowerExpr(ctx_, block, &init_expr.*, param_map, storage_map, local_var_map);
                            }
                        },
                        .Memory => {
                            // Memory variables are temporary and should be handled like local variables
                            if (var_decl.value) |init_expr| {
                                const init_value = lowerExpr(ctx_, block, &init_expr.*, param_map, storage_map, local_var_map);

                                // Store the memory variable in our local variable map for now
                                // In a full implementation, we'd allocate memory with scf.alloca
                                if (local_var_map) |lvm| {
                                    lvm.addLocalVar(var_decl.name, init_value) catch {
                                        std.debug.print("WARNING: Failed to add memory variable to map: {s}\n", .{var_decl.name});
                                    };
                                }
                            } else {
                                // Memory variable without initializer - create a default value and store it
                                if (local_var_map) |lvm| {
                                    // Create a default value (0 for now)
                                    const default_ty = c.mlirIntegerTypeGet(ctx_, 256);
                                    var const_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.constant"), fileLoc(ctx_, var_decl.span));
                                    c.mlirOperationStateAddResults(&const_state, 1, @ptrCast(&default_ty));
                                    const attr = c.mlirIntegerAttrGet(default_ty, 0);
                                    const value_id = c.mlirIdentifierGet(ctx_, c.mlirStringRefCreateFromCString("value"));
                                    var attrs = [_]c.MlirNamedAttribute{c.mlirNamedAttributeGet(value_id, attr)};
                                    c.mlirOperationStateAddAttributes(&const_state, attrs.len, &attrs);
                                    const const_op = c.mlirOperationCreate(&const_state);
                                    c.mlirBlockAppendOwnedOperation(block, const_op);
                                    const default_value = c.mlirOperationGetResult(const_op, 0);

                                    lvm.addLocalVar(var_decl.name, default_value) catch {
                                        std.debug.print("WARNING: Failed to add memory variable to map: {s}\n", .{var_decl.name});
                                    };
                                    std.debug.print("DEBUG: Added memory variable to map: {s}\n", .{var_decl.name});
                                }
                            }
                        },
                        .TStore => {
                            // Transient storage variables are persistent across calls but temporary
                            // For now, treat them like storage variables
                            if (var_decl.value) |init_expr| {
                                _ = lowerExpr(ctx_, block, &init_expr.*, param_map, storage_map, local_var_map);
                            }
                        },
                    }
                },
                .Switch => |switch_stmt| {
                    _ = lowerExpr(ctx_, block, &switch_stmt.condition, param_map, storage_map, local_var_map);
                    if (switch_stmt.default_case) |default_case| {
                        lowerBlockBody(ctx_, default_case, block, param_map, storage_map, local_var_map);
                    }
                },
                .Expr => |expr| {
                    switch (expr) {
                        .Assignment => |assign| {
                            // Debug: print what we're assigning to
                            std.debug.print("DEBUG: Assignment to: {s}\n", .{@tagName(assign.target.*)});

                            // Lower the value expression
                            const value_result = lowerExpr(ctx_, block, assign.value, param_map, storage_map, local_var_map);

                            // Handle assignment to variables
                            switch (assign.target.*) {
                                .Identifier => |ident| {
                                    std.debug.print("DEBUG: Assignment to identifier: {s}\n", .{ident.name});

                                    // Check if this is a storage variable
                                    if (storage_map) |sm| {
                                        if (sm.hasStorageVariable(ident.name)) {
                                            // This is a storage variable - use ora.sstore
                                            const store_op = createStoreOperation(ctx_, value_result, ident.name, .Storage, ident.span);
                                            c.mlirBlockAppendOwnedOperation(block, store_op);
                                        } else {
                                            // This is a local/memory variable - update it in our map
                                            if (local_var_map) |lvm| {
                                                if (lvm.hasLocalVar(ident.name)) {
                                                    // Update existing local/memory variable
                                                    lvm.addLocalVar(ident.name, value_result) catch {
                                                        std.debug.print("WARNING: Failed to update local variable: {s}\n", .{ident.name});
                                                    };
                                                } else {
                                                    // Add new local/memory variable
                                                    lvm.addLocalVar(ident.name, value_result) catch {
                                                        std.debug.print("WARNING: Failed to add new local variable: {s}\n", .{ident.name});
                                                    };
                                                }
                                            }
                                        }
                                    } else {
                                        // No storage map - check if it's a local/memory variable
                                        if (local_var_map) |lvm| {
                                            if (lvm.hasLocalVar(ident.name)) {
                                                // This is a local/memory variable - update it in our map
                                                lvm.addLocalVar(ident.name, value_result) catch {
                                                    std.debug.print("WARNING: Failed to update local variable: {s}\n", .{ident.name});
                                                };
                                            } else {
                                                // This is a new local variable - add it to our map
                                                lvm.addLocalVar(ident.name, value_result) catch {
                                                    std.debug.print("WARNING: Failed to add new local variable: {s}\n", .{ident.name});
                                                };
                                            }
                                        }
                                    }
                                },
                                else => {
                                    std.debug.print("DEBUG: Would assign to: {s}\n", .{@tagName(assign.target.*)});
                                    // For now, skip non-identifier assignments
                                },
                            }
                        },
                        .CompoundAssignment => |compound| {
                            // Debug: print what we're compound assigning to
                            std.debug.print("DEBUG: Compound assignment to: {s}\n", .{@tagName(compound.target.*)});

                            // Handle compound assignment to storage variables
                            switch (compound.target.*) {
                                .Identifier => |ident| {
                                    std.debug.print("DEBUG: Would compound assign to storage variable: {s}\n", .{ident.name});

                                    if (storage_map) |sm| {
                                        // Ensure the variable exists in storage (create if needed)
                                        _ = sm.getOrCreateAddress(ident.name) catch 0;

                                        // Load current value from storage using ora.sload
                                        const load_op = createLoadOperation(ctx_, ident.name, .Storage, ident.span);
                                        c.mlirBlockAppendOwnedOperation(block, load_op);
                                        const current_value = c.mlirOperationGetResult(load_op, 0);

                                        // Lower the right-hand side expression
                                        const rhs_value = lowerExpr(ctx_, block, compound.value, param_map, storage_map, local_var_map);

                                        // Define result type for arithmetic operations
                                        const result_ty = c.mlirIntegerTypeGet(ctx_, 256);

                                        // Perform the compound operation
                                        var new_value: c.MlirValue = undefined;
                                        switch (compound.operator) {
                                            .PlusEqual => {
                                                // current_value + rhs_value
                                                var add_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.addi"), fileLoc(ctx_, ident.span));
                                                c.mlirOperationStateAddOperands(&add_state, 2, @ptrCast(&[_]c.MlirValue{ current_value, rhs_value }));
                                                c.mlirOperationStateAddResults(&add_state, 1, @ptrCast(&result_ty));
                                                const add_op = c.mlirOperationCreate(&add_state);
                                                c.mlirBlockAppendOwnedOperation(block, add_op);
                                                new_value = c.mlirOperationGetResult(add_op, 0);
                                            },
                                            .MinusEqual => {
                                                // current_value - rhs_value
                                                var sub_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.subi"), fileLoc(ctx_, ident.span));
                                                c.mlirOperationStateAddOperands(&sub_state, 2, @ptrCast(&[_]c.MlirValue{ current_value, rhs_value }));
                                                c.mlirOperationStateAddResults(&sub_state, 1, @ptrCast(&result_ty));
                                                const sub_op = c.mlirOperationCreate(&sub_state);
                                                c.mlirBlockAppendOwnedOperation(block, sub_op);
                                                new_value = c.mlirOperationGetResult(sub_op, 0);
                                            },
                                            .StarEqual => {
                                                // current_value * rhs_value
                                                var mul_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.muli"), fileLoc(ctx_, ident.span));
                                                c.mlirOperationStateAddOperands(&mul_state, 2, @ptrCast(&[_]c.MlirValue{ current_value, rhs_value }));
                                                c.mlirOperationStateAddResults(&mul_state, 1, @ptrCast(&result_ty));
                                                const mul_op = c.mlirOperationCreate(&mul_state);
                                                c.mlirBlockAppendOwnedOperation(block, mul_op);
                                                new_value = c.mlirOperationGetResult(mul_op, 0);
                                            },
                                            .SlashEqual => {
                                                // current_value / rhs_value
                                                var div_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.divsi"), fileLoc(ctx_, ident.span));
                                                c.mlirOperationStateAddOperands(&div_state, 2, @ptrCast(&[_]c.MlirValue{ current_value, rhs_value }));
                                                c.mlirOperationStateAddResults(&div_state, 1, @ptrCast(&result_ty));
                                                const div_op = c.mlirOperationCreate(&div_state);
                                                c.mlirBlockAppendOwnedOperation(block, div_op);
                                                new_value = c.mlirOperationGetResult(div_op, 0);
                                            },
                                            .PercentEqual => {
                                                // current_value % rhs_value
                                                var rem_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("arith.remsi"), fileLoc(ctx_, ident.span));
                                                c.mlirOperationStateAddOperands(&rem_state, 2, @ptrCast(&[_]c.MlirValue{ current_value, rhs_value }));
                                                c.mlirOperationStateAddResults(&rem_state, 1, @ptrCast(&result_ty));
                                                const rem_op = c.mlirOperationCreate(&rem_state);
                                                c.mlirBlockAppendOwnedOperation(block, rem_op);
                                                new_value = c.mlirOperationGetResult(rem_op, 0);
                                            },
                                        }

                                        // Store the result back to storage using ora.sstore
                                        const store_op = createStoreOperation(ctx_, new_value, ident.name, .Storage, ident.span);
                                        c.mlirBlockAppendOwnedOperation(block, store_op);
                                    } else {
                                        // No storage map - fall back to placeholder
                                        var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("ora.compound_assign"), fileLoc(ctx_, ident.span));
                                        const op = c.mlirOperationCreate(&state);
                                        c.mlirBlockAppendOwnedOperation(block, op);
                                    }
                                },
                                else => {
                                    std.debug.print("DEBUG: Would compound assign to: {s}\n", .{@tagName(compound.target.*)});
                                    // For now, skip non-identifier compound assignments
                                },
                            }
                        },
                        else => {
                            // Lower other expression statements
                            _ = lowerExpr(ctx_, block, &expr, param_map, storage_map, local_var_map);
                        },
                    }
                },
                .LabeledBlock => |labeled_block| {
                    // For now, just lower the block body
                    lowerBlockBody(ctx_, labeled_block.block, block, param_map, storage_map, local_var_map);
                    // TODO: Add proper labeled block handling
                },
                .Continue => {
                    // For now, skip continue statements
                    // TODO: Add proper continue statement handling
                },
                .If => |if_stmt| {
                    // Lower the condition expression
                    const condition = lowerExpr(ctx_, block, &if_stmt.condition, param_map, storage_map, local_var_map);

                    // Create the scf.if operation with proper then/else regions
                    var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("scf.if"), fileLoc(ctx_, if_stmt.span));

                    // Add the condition operand
                    c.mlirOperationStateAddOperands(&state, 1, @ptrCast(&condition));

                    // Create then region
                    const then_region = c.mlirRegionCreate();
                    const then_block = c.mlirBlockCreate(0, null, null);
                    c.mlirRegionInsertOwnedBlock(then_region, 0, then_block);
                    c.mlirOperationStateAddOwnedRegions(&state, 1, @ptrCast(&then_region));

                    // Lower then branch
                    lowerBlockBody(ctx_, if_stmt.then_branch, then_block, param_map, storage_map, local_var_map);

                    // Create else region if present
                    if (if_stmt.else_branch) |else_branch| {
                        const else_region = c.mlirRegionCreate();
                        const else_block = c.mlirBlockCreate(0, null, null);
                        c.mlirRegionInsertOwnedBlock(else_region, 0, else_block);
                        c.mlirOperationStateAddOwnedRegions(&state, 1, @ptrCast(&else_region));

                        // Lower else branch
                        lowerBlockBody(ctx_, else_branch, else_block, param_map, storage_map, local_var_map);
                    }

                    const op = c.mlirOperationCreate(&state);
                    c.mlirBlockAppendOwnedOperation(block, op);
                },
                else => @panic("Unhandled statement type"),
            }
        }

        // TODO: Move lowerBlockBody to statements.zig - this is duplicated code
        fn lowerBlockBody(ctx_: c.MlirContext, b: lib.ast.Statements.BlockNode, block: c.MlirBlock, param_map: ?*const ParamMap, storage_map: ?*StorageMap, local_var_map: ?*LocalVarMap) void {
            std.debug.print("DEBUG: Processing block with {d} statements\n", .{b.statements.len});
            for (b.statements) |*s| {
                std.debug.print("DEBUG: Processing statement type: {s}\n", .{@tagName(s.*)});
                lowerStmt(ctx_, block, s, param_map, storage_map, local_var_map);
            }
        }
    };

    // TODO: Move createGlobalDeclaration to declarations.zig - this is duplicated code
    const createGlobalDeclaration = struct {
        fn create(ctx_: c.MlirContext, loc_: c.MlirLocation, var_decl: lib.ast.Statements.VariableDeclNode) c.MlirOperation {
            // Create ora.global operation
            var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("ora.global"), loc_);

            // Add the global name as a symbol attribute
            const name_ref = c.mlirStringRefCreate(var_decl.name.ptr, var_decl.name.len);
            const name_attr = c.mlirStringAttrGet(ctx_, name_ref);
            const name_id = c.mlirIdentifierGet(ctx_, c.mlirStringRefCreateFromCString("sym_name"));
            var attrs = [_]c.MlirNamedAttribute{
                c.mlirNamedAttributeGet(name_id, name_attr),
            };
            c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

            // Add the type attribute
            // TODO: Get the actual type from the variable declaration
            // For now, use a simple heuristic based on variable name
            const var_type = if (std.mem.eql(u8, var_decl.name, "status"))
                c.mlirIntegerTypeGet(ctx_, 1) // bool -> i1
            else
                c.mlirIntegerTypeGet(ctx_, 256); // default to i256
            const type_attr = c.mlirTypeAttrGet(var_type);
            const type_id = c.mlirIdentifierGet(ctx_, c.mlirStringRefCreateFromCString("type"));
            var type_attrs = [_]c.MlirNamedAttribute{
                c.mlirNamedAttributeGet(type_id, type_attr),
            };
            c.mlirOperationStateAddAttributes(&state, type_attrs.len, &type_attrs);

            // Add initial value if present
            if (var_decl.value) |_| {
                // For now, create a default value based on the type
                // TODO: Lower the actual initializer expression
                const init_attr = if (std.mem.eql(u8, var_decl.name, "status"))
                    c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(ctx_, 1), 0) // bool -> i1 with value 0 (false)
                else
                    c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(ctx_, 256), 0); // default to i256 with value 0
                const init_id = c.mlirIdentifierGet(ctx_, c.mlirStringRefCreateFromCString("init"));
                var init_attrs = [_]c.MlirNamedAttribute{
                    c.mlirNamedAttributeGet(init_id, init_attr),
                };
                c.mlirOperationStateAddAttributes(&state, init_attrs.len, &init_attrs);
            }

            return c.mlirOperationCreate(&state);
        }
    };

    // TODO: Move createMemoryGlobalDeclaration to declarations.zig - this is duplicated code
    const createMemoryGlobalDeclaration = struct {
        fn create(ctx_: c.MlirContext, loc_: c.MlirLocation, var_decl: lib.ast.Statements.VariableDeclNode) c.MlirOperation {
            // Create ora.memory.global operation
            var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("ora.memory.global"), loc_);

            // Add the global name as a symbol attribute
            const name_ref = c.mlirStringRefCreate(var_decl.name.ptr, var_decl.name.len);
            const name_attr = c.mlirStringAttrGet(ctx_, name_ref);
            const name_id = c.mlirIdentifierGet(ctx_, c.mlirStringRefCreateFromCString("sym_name"));
            var attrs = [_]c.MlirNamedAttribute{
                c.mlirNamedAttributeGet(name_id, name_attr),
            };
            c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

            // Add the type attribute
            const var_type = c.mlirIntegerTypeGet(ctx_, 256); // default to i256
            const type_attr = c.mlirTypeAttrGet(var_type);
            const type_id = c.mlirIdentifierGet(ctx_, c.mlirStringRefCreateFromCString("type"));
            var type_attrs = [_]c.MlirNamedAttribute{
                c.mlirNamedAttributeGet(type_id, type_attr),
            };
            c.mlirOperationStateAddAttributes(&state, type_attrs.len, &type_attrs);

            return c.mlirOperationCreate(&state);
        }
    };

    // TODO: Move createTStoreGlobalDeclaration to declarations.zig - this is duplicated code
    const createTStoreGlobalDeclaration = struct {
        fn create(ctx_: c.MlirContext, loc_: c.MlirLocation, var_decl: lib.ast.Statements.VariableDeclNode) c.MlirOperation {
            // Create ora.tstore.global operation
            var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("ora.tstore.global"), loc_);

            // Add the global name as a symbol attribute
            const name_ref = c.mlirStringRefCreate(var_decl.name.ptr, var_decl.name.len);
            const name_attr = c.mlirStringAttrGet(ctx_, name_ref);
            const name_id = c.mlirIdentifierGet(ctx_, c.mlirStringRefCreateFromCString("sym_name"));
            var attrs = [_]c.MlirNamedAttribute{
                c.mlirNamedAttributeGet(name_id, name_attr),
            };
            c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);

            // Add the type attribute
            const var_type = c.mlirIntegerTypeGet(ctx_, 256); // default to i256
            const type_attr = c.mlirTypeAttrGet(var_type);
            const type_id = c.mlirIdentifierGet(ctx_, c.mlirStringRefCreateFromCString("type"));
            var type_attrs = [_]c.MlirNamedAttribute{
                c.mlirNamedAttributeGet(type_id, type_attr),
            };
            c.mlirOperationStateAddAttributes(&state, type_attrs.len, &type_attrs);

            return c.mlirOperationCreate(&state);
        }
    };

    // TODO: Move Emit to declarations.zig - this is duplicated code
    const Emit = struct {
        fn create(ctx_: c.MlirContext, loc_: c.MlirLocation, sym_id: c.MlirIdentifier, type_id: c.MlirIdentifier, f: lib.FunctionNode, contract_storage_map: ?*Lower.StorageMap, local_var_map: ?*Lower.LocalVarMap) c.MlirOperation {
            // Create a local variable map for this function if one wasn't provided
            var local_vars: Lower.LocalVarMap = undefined;
            if (local_var_map) |lvm| {
                local_vars = lvm.*;
            } else {
                local_vars = Lower.LocalVarMap.init(std.heap.page_allocator);
            }
            defer if (local_var_map == null) local_vars.deinit();
            var state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("func.func"), loc_);
            const name_ref = c.mlirStringRefCreate(f.name.ptr, f.name.len);
            const name_attr = c.mlirStringAttrGet(ctx_, name_ref);
            const fn_type = Build.funcType(ctx_, f);
            const fn_type_attr = c.mlirTypeAttrGet(fn_type);
            var attrs = [_]c.MlirNamedAttribute{
                c.mlirNamedAttributeGet(sym_id, name_attr),
                c.mlirNamedAttributeGet(type_id, fn_type_attr),
            };
            c.mlirOperationStateAddAttributes(&state, attrs.len, &attrs);
            const region = c.mlirRegionCreate();
            const param_count = @as(c_int, @intCast(f.parameters.len));
            std.debug.print("DEBUG: Creating block with {d} parameters\n", .{param_count});

            // Create the block without parameters
            // In MLIR, function parameters are part of the function signature, not block arguments
            const block = c.mlirBlockCreate(0, null, null);
            c.mlirRegionInsertOwnedBlock(region, 0, block);
            c.mlirOperationStateAddOwnedRegions(&state, 1, @ptrCast(&region));

            // Create parameter mapping for calldata parameters
            var param_map = Lower.ParamMap.init(std.heap.page_allocator);
            defer param_map.deinit();
            for (f.parameters, 0..) |param, i| {
                // Function parameters are calldata by default in Ora
                param_map.addParam(param.name, i) catch {};
                std.debug.print("DEBUG: Added calldata parameter: {s} at index {d}\n", .{ param.name, i });
            }

            // Note: Build.funcType(ctx_, f) already creates the function type with parameters
            // Function parameters are implicitly calldata in Ora

            // Use the contract's storage map if provided, otherwise create an empty one
            var local_storage_map = Lower.StorageMap.init(std.heap.page_allocator);
            defer local_storage_map.deinit();

            const storage_map_to_use = if (contract_storage_map) |csm| csm else &local_storage_map;

            // Lower a minimal body: returns, integer constants, and plus
            Lower.lowerBlockBody(ctx_, f.body, block, &param_map, storage_map_to_use, &local_vars);

            // Ensure a terminator exists (void return)
            if (f.return_type_info == null) {
                var return_state = c.mlirOperationStateGet(c.mlirStringRefCreateFromCString("func.return"), loc_);
                const return_op = c.mlirOperationCreate(&return_state);
                c.mlirBlockAppendOwnedOperation(block, return_op);
            }

            // Create the function operation
            const func_op = c.mlirOperationCreate(&state);
            return func_op;
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
