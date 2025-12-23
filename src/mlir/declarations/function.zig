// ============================================================================
// Declaration Lowering - Functions
// ============================================================================

const std = @import("std");
const c = @import("../c.zig").c;
const lib = @import("ora_lib");
const h = @import("../helpers.zig");
const TypeMapper = @import("../types.zig").TypeMapper;
const LocalVarMap = @import("../symbols.zig").LocalVarMap;
const ParamMap = @import("../symbols.zig").ParamMap;
const StorageMap = @import("../memory.zig").StorageMap;
const ExpressionLowerer = @import("../expressions.zig").ExpressionLowerer;
const StatementLowerer = @import("../statements.zig").StatementLowerer;
const LoweringError = StatementLowerer.LoweringError;
const DeclarationLowerer = @import("mod.zig").DeclarationLowerer;
const helpers = @import("helpers.zig");
const refinements = @import("refinements.zig");

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
    var state = h.opState("func.func", helpers.createFileLocation(self, func.span));

    // Add function name
    const name_ref = c.mlirStringRefCreate(func.name.ptr, func.name.len);
    const name_attr = c.mlirStringAttrGet(self.ctx, name_ref);
    const sym_name_id = h.identifier(self.ctx, "sym_name");

    // Collect all function attributes
    var attributes = std.ArrayList(c.MlirNamedAttribute){};
    defer attributes.deinit(std.heap.page_allocator);

    // Add function name attribute
    attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(sym_name_id, name_attr)) catch {};

    // Add visibility modifier attribute
    const visibility_attr = switch (func.visibility) {
        .Public => c.mlirStringAttrGet(self.ctx, h.strRef("pub")),
        .Private => c.mlirStringAttrGet(self.ctx, h.strRef("private")),
    };
    const visibility_id = h.identifier(self.ctx, "ora.visibility");
    attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(visibility_id, visibility_attr)) catch {};

    // Add special function name attributes
    if (std.mem.eql(u8, func.name, "init")) {
        const init_attr = c.mlirBoolAttrGet(self.ctx, 1);
        const init_id = h.identifier(self.ctx, "ora.init");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(init_id, init_attr)) catch {};
    }

    // Add comprehensive verification metadata for function contracts
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
    const fn_type = helpers.createFunctionType(self, func);
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
        param_locs[i] = helpers.createFileLocation(self, param.span);
    }

    const block = c.mlirBlockCreate(@intCast(param_types.len), param_types.ptr, param_locs.ptr);
    c.mlirRegionInsertOwnedBlock(region, 0, block);
    c.mlirOperationStateAddOwnedRegions(&state, 1, @ptrCast(&region));

    // Create the function operation early so we can add it to symbol table before lowering body
    // (needed for call site type conversion)
    const func_op = c.mlirOperationCreate(&state);

    // Add function to symbol table BEFORE lowering body (so calls within body can look it up)
    if (self.symbol_table) |sym_table| {
        const return_type = if (func.return_type_info) |ret_info|
            self.type_mapper.toMlirType(ret_info)
        else
            c.mlirNoneTypeGet(self.ctx);

        // Allocate param_types array for symbol table (it will be owned by FunctionSymbol)
        if (sym_table.allocator.alloc(c.MlirType, param_types.len)) |allocated| {
            @memcpy(allocated, param_types);
            sym_table.addFunction(func.name, func_op, allocated, return_type) catch {
                sym_table.allocator.free(allocated);
                std.debug.print("WARNING: Failed to add function {s} to symbol table\n", .{func.name});
            };
        } else |_| {
            std.debug.print("WARNING: Failed to allocate parameter types for function {s}\n", .{func.name});
            // Continue without adding to symbol table if allocation fails
        }
    }

    // Map parameter names to block arguments
    for (func.parameters, 0..) |param, i| {
        const block_arg = c.mlirBlockGetArgument(block, @intCast(i));
        param_map.setBlockArgument(param.name, block_arg) catch {};

        // Insert refinement type guards for parameters
        if (param.type_info.ora_type) |ora_type| {
            refinements.insertRefinementGuard(self, block, block_arg, ora_type, param.span) catch |err| {
                std.debug.print("Error inserting refinement guard for parameter {s}: {s}\n", .{ param.name, @errorName(err) });
            };
        }
    }

    // Add precondition assertions for requires clauses
    if (func.requires_clauses.len > 0) {
        lowerRequiresClauses(self, func.requires_clauses, block, &param_map, contract_storage_map, local_var_map orelse &local_vars) catch |err| {
            std.debug.print("Error lowering requires clauses: {s}\n", .{@errorName(err)});
        };
    }

    // Lower the function body
    lowerFunctionBody(self, func, block, &param_map, contract_storage_map, local_var_map orelse &local_vars) catch |err| {
        // Format error message based on error type
        const error_message = switch (err) {
            error.InvalidLValue => "Cannot assign to immutable variable",
            error.TypeMismatch => "Type mismatch in function body",
            error.UndefinedSymbol => "Undefined symbol in function body",
            error.InvalidMemoryRegion => "Invalid memory region access",
            error.MalformedExpression => "Malformed expression in function body",
            error.MlirOperationFailed => "MLIR operation failed",
            error.OutOfMemory => "Out of memory during lowering",
            error.InvalidControlFlow => "Invalid control flow",
            error.InvalidSwitch => "Invalid switch statement",
            error.UnsupportedStatement => "Unsupported statement type",
        };

        const suggestion = switch (err) {
            error.InvalidLValue => "Use 'var' instead of 'let' or 'const' for mutable variables",
            error.TypeMismatch => "Check that all types match expected signatures",
            error.UndefinedSymbol => "Ensure all symbols are declared before use",
            error.InvalidMemoryRegion => "Check memory region annotations",
            error.MalformedExpression => "Verify expression syntax",
            error.MlirOperationFailed => "Check function implementation for type errors and unsupported operations",
            error.OutOfMemory => "Reduce memory usage or increase available memory",
            error.InvalidControlFlow => "Check control flow structure",
            error.InvalidSwitch => "Verify switch statement syntax",
            error.UnsupportedStatement => "This statement type is not yet supported",
        };

        // Report to error handler if available
        if (self.error_handler) |eh| {
            var error_handler = @constCast(eh);
            error_handler.reportError(.MlirOperationFailed, func.span, error_message, suggestion) catch {};
        }

        return c.mlirOperationCreate(&state);
    };

    // Ensures clauses are now handled before each return statement (see return.zig)
    // This ensures postconditions are checked at every return point, not just at the end

    // Ensure a terminator exists (void return)
    if (func.return_type_info == null) {
        const return_op = self.ora_dialect.createFuncReturn(helpers.createFileLocation(self, func.span));
        h.appendOp(block, return_op);
    }

    // Function operation was created earlier (before body lowering) for symbol table registration

    // Set ora.type attributes on function arguments and return values
    const ora_type_attr_name = h.strRef("ora.type");

    // Set attributes on function arguments
    for (func.parameters, 0..) |param, i| {
        const type_str = helpers.oraTypeToString(self, param.type_info, std.heap.page_allocator) catch {
            std.debug.print("WARNING: Failed to convert parameter type to string for {s}\n", .{param.name});
            continue;
        };
        defer std.heap.page_allocator.free(type_str);

        const type_attr = c.mlirStringAttrGet(self.ctx, h.strRef(type_str));
        const success = c.oraFuncSetArgAttr(func_op, @intCast(i), ora_type_attr_name, type_attr);
        if (!success) {
            std.debug.print("WARNING: Failed to set ora.type attribute on parameter {s}\n", .{param.name});
        }
    }

    // Set attribute on function return value (if present)
    if (func.return_type_info) |ret_info| {
        const type_str = helpers.oraTypeToString(self, ret_info, std.heap.page_allocator) catch {
            std.debug.print("WARNING: Failed to convert return type to string\n", .{});
            return func_op;
        };
        defer std.heap.page_allocator.free(type_str);

        const type_attr = c.mlirStringAttrGet(self.ctx, h.strRef(type_str));
        const success = c.oraFuncSetResultAttr(func_op, 0, ora_type_attr_name, type_attr);
        if (!success) {
            std.debug.print("WARNING: Failed to set ora.type attribute on return value\n", .{});
        }
    }

    return func_op;
}

/// Lower function body statements
fn lowerFunctionBody(self: *const DeclarationLowerer, func: *const lib.FunctionNode, block: c.MlirBlock, param_map: *const ParamMap, storage_map: ?*const StorageMap, local_var_map: ?*LocalVarMap) LoweringError!void {
    // Create a statement lowerer for this function
    const const_local_var_map = if (local_var_map) |lvm| @as(*const LocalVarMap, lvm) else null;
    const expr_lowerer = ExpressionLowerer.init(self.ctx, block, self.type_mapper, param_map, storage_map, const_local_var_map, self.symbol_table, self.builtin_registry, self.locations, self.ora_dialect);

    // Get the function's return type
    const function_return_type = if (func.return_type_info) |ret_info|
        self.type_mapper.toMlirType(ret_info)
    else
        null;

    const function_return_type_info = if (func.return_type_info) |ret_info| ret_info else null;
    const stmt_lowerer = StatementLowerer.init(self.ctx, block, self.type_mapper, &expr_lowerer, param_map, storage_map, local_var_map, self.locations, self.symbol_table, self.builtin_registry, std.heap.page_allocator, function_return_type, function_return_type_info, self.ora_dialect, func.ensures_clauses);

    // Lower the function body
    _ = try stmt_lowerer.lowerBlockBody(func.body, block);
}

/// Lower requires clauses as precondition assertions with enhanced verification metadata (Requirements 6.4)
fn lowerRequiresClauses(self: *const DeclarationLowerer, requires_clauses: []*lib.ast.Expressions.ExprNode, block: c.MlirBlock, param_map: *const ParamMap, storage_map: ?*const StorageMap, local_var_map: ?*LocalVarMap) LoweringError!void {
    const const_local_var_map = if (local_var_map) |lvm| @as(*const LocalVarMap, lvm) else null;
    const expr_lowerer = ExpressionLowerer.init(self.ctx, block, self.type_mapper, param_map, storage_map, const_local_var_map, self.symbol_table, self.builtin_registry, self.locations, self.ora_dialect);

    for (requires_clauses, 0..) |clause, i| {
        // Lower the requires expression
        const condition_value = expr_lowerer.lowerExpression(clause);

        // Create an assertion operation with comprehensive verification attributes
        var assert_state = h.opState("cf.assert", helpers.createFileLocation(self, helpers.getExpressionSpan(self, clause)));

        // Add the condition as an operand
        c.mlirOperationStateAddOperands(&assert_state, 1, @ptrCast(&condition_value));

        // Collect verification attributes
        var attributes = std.ArrayList(c.MlirNamedAttribute){};
        defer attributes.deinit(std.heap.page_allocator);

        // Add required 'msg' attribute first (cf.assert requires this)
        const msg_text = try std.fmt.allocPrint(std.heap.page_allocator, "Precondition {d} failed", .{i});
        defer std.heap.page_allocator.free(msg_text);
        const msg_attr = h.stringAttr(self.ctx, msg_text);
        const msg_id = h.identifier(self.ctx, "msg");
        attributes.append(std.heap.page_allocator, c.mlirNamedAttributeGet(msg_id, msg_attr)) catch {};

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
        var assert_state = h.opState("cf.assert", helpers.createFileLocation(self, helpers.getExpressionSpan(self, clause)));

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
