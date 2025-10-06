//! MLIR to Yul Lowering
//!
//! This module provides functionality to convert MLIR modules containing Ora dialect
//! operations to Yul intermediate representation.

const std = @import("std");
const c = @import("c.zig").c;

/// Yul lowering context for managing state during conversion
const YulLoweringContext = struct {
    allocator: std.mem.Allocator,
    output: std.ArrayList(u8),
    errors: std.ArrayList([]const u8),
    storage_vars: std.HashMap([]const u8, u32, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),
    next_storage_slot: u32,
    next_temp_var: u32,
    // SSA value tracking - maps MLIR values to Yul variable names
    ssa_values: std.HashMap(u64, []const u8, std.hash_map.AutoContext(u64), std.hash_map.default_max_load_percentage),
    // Expression tracking - maps MLIR values to inline expressions
    value_expressions: std.HashMap(u64, []const u8, std.hash_map.AutoContext(u64), std.hash_map.default_max_load_percentage),
    // Indentation level for proper Yul formatting
    indent_level: u32,

    const Self = @This();

    fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .allocator = allocator,
            .output = std.ArrayList(u8).init(allocator),
            .errors = std.ArrayList([]const u8).init(allocator),
            .storage_vars = std.HashMap([]const u8, u32, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
            .next_storage_slot = 0,
            .next_temp_var = 0,
            .ssa_values = std.HashMap(u64, []const u8, std.hash_map.AutoContext(u64), std.hash_map.default_max_load_percentage).init(allocator),
            .value_expressions = std.HashMap(u64, []const u8, std.hash_map.AutoContext(u64), std.hash_map.default_max_load_percentage).init(allocator),
            .indent_level = 0,
        };
    }

    fn deinit(self: *Self) void {
        self.output.deinit();
        for (self.errors.items) |err| {
            self.allocator.free(err);
        }
        self.errors.deinit();
        var iter = self.storage_vars.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.storage_vars.deinit();
        var ssa_iter = self.ssa_values.iterator();
        while (ssa_iter.next()) |entry| {
            self.allocator.free(entry.value_ptr.*);
        }
        self.ssa_values.deinit();
        var expr_iter = self.value_expressions.iterator();
        while (expr_iter.next()) |entry| {
            self.allocator.free(entry.value_ptr.*);
        }
        self.value_expressions.deinit();
    }

    fn addError(self: *Self, message: []const u8) !void {
        const error_msg = try self.allocator.dupe(u8, message);
        try self.errors.append(error_msg);
    }

    fn write(self: *Self, text: []const u8) !void {
        try self.output.appendSlice(text);
    }

    fn writeln(self: *Self, text: []const u8) !void {
        try self.output.appendSlice(text);
        try self.output.append('\n');
    }

    /// Write text with proper indentation
    fn writeIndented(self: *Self, text: []const u8) !void {
        var i: u32 = 0;
        while (i < self.indent_level) : (i += 1) {
            try self.output.appendSlice("    ");
        }
        try self.output.appendSlice(text);
    }

    /// Write text with proper indentation and newline
    fn writelnIndented(self: *Self, text: []const u8) !void {
        try self.writeIndented(text);
        try self.output.append('\n');
    }

    fn getStorageSlot(self: *Self, var_name: []const u8) !u32 {
        if (self.storage_vars.get(var_name)) |slot| {
            return slot;
        }

        const slot = self.next_storage_slot;
        self.next_storage_slot += 1;

        const name_copy = try self.allocator.dupe(u8, var_name);
        try self.storage_vars.put(name_copy, slot);

        return slot;
    }

    fn getTempVar(self: *Self) ![]const u8 {
        const var_name = try std.fmt.allocPrint(self.allocator, "temp_{d}", .{self.next_temp_var});
        self.next_temp_var += 1;
        return var_name;
    }

    /// Get or create a Yul variable name for an MLIR value
    fn getValueName(self: *Self, mlir_value: c.MlirValue) ![]const u8 {
        const value_id = @intFromPtr(mlir_value.ptr);
        if (self.ssa_values.get(value_id)) |name| {
            return name;
        }

        // Create new variable name for this value
        const var_name = try self.getTempVar();
        try self.ssa_values.put(value_id, var_name);
        return var_name;
    }

    /// Get the Yul variable name for an MLIR value, or return a default
    fn getValueNameOr(self: *Self, mlir_value: c.MlirValue, default_name: []const u8) []const u8 {
        const value_id = @intFromPtr(mlir_value.ptr);
        return self.ssa_values.get(value_id) orelse default_name;
    }

    /// Get the Yul variable name for an MLIR value, creating one if needed
    fn getValueNameOrCreate(self: *Self, mlir_value: c.MlirValue, _: []const u8) ![]const u8 {
        const value_id = @intFromPtr(mlir_value.ptr);
        if (self.ssa_values.get(value_id)) |name| {
            return name;
        }

        // Create a new variable name for this value
        const var_name = try self.getTempVar();
        try self.ssa_values.put(value_id, var_name);
        return var_name;
    }

    /// Set an expression for a value (for inline expressions)
    fn setValueExpression(self: *Self, mlir_value: c.MlirValue, expr: []const u8) !void {
        const value_id = @intFromPtr(mlir_value.ptr);
        try self.value_expressions.put(value_id, expr);
    }

    /// Get an expression for a value
    fn getValueExpression(self: *Self, mlir_value: c.MlirValue) ?[]const u8 {
        const value_id = @intFromPtr(mlir_value.ptr);
        return self.value_expressions.get(value_id);
    }
};

/// Result of MLIR to Yul lowering
pub const YulLoweringResult = struct {
    yul_code: []const u8,
    success: bool,
    errors: []const []const u8,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.yul_code);
        for (self.errors) |err| {
            self.allocator.free(err);
        }
        self.allocator.free(self.errors);
    }
};

/// Lower MLIR module to Yul code
pub fn lowerToYul(
    module: c.MlirModule,
    ctx: c.MlirContext,
    allocator: std.mem.Allocator,
) !YulLoweringResult {
    _ = ctx; // TODO: Use ctx for type information
    var ctx_lowering = YulLoweringContext.init(allocator);
    defer ctx_lowering.deinit();

    // Start Yul object - use actual contract name
    const contract_name = getContractName(module) orelse "Contract";
    const object_line = try std.fmt.allocPrint(ctx_lowering.allocator, "object \"{s}\" {{", .{contract_name});
    defer ctx_lowering.allocator.free(object_line);
    try ctx_lowering.writeln(object_line);
    try ctx_lowering.writeln("  code {");
    try ctx_lowering.writeln("    // Store the creator in slot zero.");
    try ctx_lowering.writeln("    sstore(0, caller())");
    try ctx_lowering.writeln("    // Deploy the contract");
    try ctx_lowering.writeln("    datacopy(0, dataoffset(\"runtime\"), datasize(\"runtime\"))");
    try ctx_lowering.writeln("    return(0, datasize(\"runtime\"))");
    try ctx_lowering.writeln("  }");
    try ctx_lowering.writeln("  object \"runtime\" {");
    try ctx_lowering.writeln("    code {");

    // Process the module operations (this will generate functions inside the runtime code block)
    // Increase indentation for functions inside the code block
    ctx_lowering.indent_level += 2;
    processModule(module, &ctx_lowering);
    ctx_lowering.indent_level -= 2;

    try ctx_lowering.writeln("    }");
    try ctx_lowering.writeln("  }");
    try ctx_lowering.writeln("}");

    const success = ctx_lowering.errors.items.len == 0;

    return YulLoweringResult{
        .yul_code = try ctx_lowering.output.toOwnedSlice(),
        .success = success,
        .errors = try ctx_lowering.errors.toOwnedSlice(),
        .allocator = allocator,
    };
}

/// Process MLIR module and convert to Yul
fn processModule(module: c.MlirModule, ctx_lowering: *YulLoweringContext) void {
    // Get the first operation in the module (should be builtin.module)
    const module_op = c.mlirModuleGetOperation(module);
    processOperation(module_op, ctx_lowering);
}

/// Process a single MLIR operation
fn processOperation(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) void {
    const op_name = getOperationName(op);

    if (std.mem.eql(u8, op_name, "builtin.module")) {
        processBuiltinModule(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process builtin.module") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.contract")) {
        processOraContract(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.contract") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.global")) {
        processOraGlobal(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.global") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "func.func")) {
        processFuncFunc(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process func.func") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.sload")) {
        processOraSLoad(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.sload") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.sstore")) {
        processOraSStore(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.sstore") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "arith.constant")) {
        processArithConstant(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process arith.constant") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "arith.addi")) {
        processArithAddi(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process arith.addi") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "arith.cmpi")) {
        processArithCmpi(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process arith.cmpi") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "arith.subi")) {
        processArithSubi(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process arith.subi") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "arith.muli")) {
        processArithMuli(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process arith.muli") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "arith.divi")) {
        processArithDivi(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process arith.divi") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "arith.remi")) {
        processArithRemi(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process arith.remi") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "arith.andi")) {
        processArithAndi(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process arith.andi") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "arith.ori")) {
        processArithOri(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process arith.ori") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "arith.xori")) {
        processArithXori(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process arith.xori") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "arith.shli")) {
        processArithShli(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process arith.shli") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "arith.shrsi")) {
        processArithShrsi(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process arith.shrsi") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "arith.shrui")) {
        processArithShrui(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process arith.shrui") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.power")) {
        processOraPower(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.power") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.cast")) {
        processOraCast(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.cast") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "scf.if")) {
        processScfIf(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process scf.if") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "scf.yield")) {
        processScfYield(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process scf.yield") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.if")) {
        processOraIf(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.if") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.while")) {
        processOraWhile(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.while") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.for")) {
        processOraFor(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.for") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.switch")) {
        processOraSwitch(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.switch") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.break")) {
        processOraBreak(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.break") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.continue")) {
        processOraContinue(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.continue") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.return")) {
        processOraReturn(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.return") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.mload")) {
        processOraMload(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.mload") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.mstore")) {
        processOraMstore(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.mstore") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.tload")) {
        processOraTload(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.tload") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.tstore")) {
        processOraTstore(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.tstore") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.string.constant")) {
        processOraStringConstant(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.string.constant") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.hex.constant")) {
        processOraHexConstant(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.hex.constant") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.binary.constant")) {
        processOraBinaryConstant(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.binary.constant") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.address.constant")) {
        processOraAddressConstant(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.address.constant") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.const")) {
        processOraConst(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.const") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.immutable")) {
        processOraImmutable(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.immutable") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.struct.decl")) {
        processOraStructDecl(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.struct.decl") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.struct.instantiate")) {
        processOraStructInstantiate(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.struct.instantiate") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.struct.field.store")) {
        processOraStructFieldStore(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.struct.field.store") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.struct.init")) {
        processOraStructInit(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.struct.init") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.enum.decl")) {
        processOraEnumDecl(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.enum.decl") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.enum.constant")) {
        processOraEnumConstant(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.enum.constant") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.map.get")) {
        processOraMapGet(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.map.get") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.map.store")) {
        processOraMapStore(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.map.store") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.requires")) {
        processOraRequires(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.requires") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.ensures")) {
        processOraEnsures(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.ensures") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.invariant")) {
        processOraInvariant(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.invariant") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.old")) {
        processOraOld(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.old") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.log")) {
        processOraLog(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.log") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.move")) {
        processOraMove(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.move") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.try_catch")) {
        processOraTryCatch(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.try_catch") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.yield")) {
        processOraYield(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.yield") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.error.decl")) {
        processOraErrorDecl(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.error.decl") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.error")) {
        processOraError(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.error") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.error_union")) {
        processOraErrorUnion(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.error_union") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.switch.case")) {
        processOraSwitchCase(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.switch.case") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.switch.range")) {
        processOraSwitchRange(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.switch.range") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.switch.else")) {
        processOraSwitchElse(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.switch.else") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.pattern.match")) {
        processOraPatternMatch(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.pattern.match") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.union.decl")) {
        processOraUnionDecl(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.union.decl") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.union.instantiate")) {
        processOraUnionInstantiate(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.union.instantiate") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.union.extract")) {
        processOraUnionExtract(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.union.extract") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.union.discriminant")) {
        processOraUnionDiscriminant(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.union.discriminant") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.destructure")) {
        processOraDestructure(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.destructure") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.lock")) {
        processOraLock(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.lock") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "ora.unlock")) {
        processOraUnlock(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process ora.unlock") catch {};
        };
    } else if (std.mem.eql(u8, op_name, "func.return")) {
        processFuncReturn(op, ctx_lowering) catch {
            ctx_lowering.addError("Failed to process func.return") catch {};
        };
    } else {
        // For now, skip unknown operations
        ctx_lowering.write("    // Unknown operation: ") catch {};
        ctx_lowering.writeln(op_name) catch {};
    }
}

/// Get operation name from MLIR operation
fn getOperationName(op: c.MlirOperation) []const u8 {
    const op_name = c.mlirOperationGetName(op);
    // Convert MlirIdentifier to string
    const op_name_str = c.mlirIdentifierStr(op_name);
    // Convert MlirStringRef to slice
    return op_name_str.data[0..op_name_str.length];
}

/// Process builtin.module operation
fn processBuiltinModule(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    // Process all operations in the module
    const num_regions = c.mlirOperationGetNumRegions(op);
    for (0..@intCast(num_regions)) |region_idx| {
        const region = c.mlirOperationGetRegion(op, @intCast(region_idx));
        var current_block = c.mlirRegionGetFirstBlock(region);
        while (!c.mlirBlockIsNull(current_block)) {
            var current_op = c.mlirBlockGetFirstOperation(current_block);
            while (!c.mlirOperationIsNull(current_op)) {
                processOperation(current_op, ctx_lowering);
                current_op = c.mlirOperationGetNextInBlock(current_op);
            }
            current_block = c.mlirBlockGetNextInRegion(current_block);
        }
    }
}

/// Process ora.contract operation
fn processOraContract(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    // Get contract name from attributes
    const contract_name = getStringAttribute(op, "sym_name") orelse "UnknownContract";
    try ctx_lowering.write("  // Contract: ");
    try ctx_lowering.writeln(contract_name);

    // Process all operations in the contract
    const num_regions = c.mlirOperationGetNumRegions(op);
    for (0..@intCast(num_regions)) |region_idx| {
        const region = c.mlirOperationGetRegion(op, @intCast(region_idx));
        var current_block = c.mlirRegionGetFirstBlock(region);
        while (!c.mlirBlockIsNull(current_block)) {
            var current_op = c.mlirBlockGetFirstOperation(current_block);
            while (!c.mlirOperationIsNull(current_op)) {
                processOperation(current_op, ctx_lowering);
                current_op = c.mlirOperationGetNextInBlock(current_op);
            }
            current_block = c.mlirBlockGetNextInRegion(current_block);
        }
    }
}

/// Process ora.global operation
fn processOraGlobal(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const var_name = getStringAttribute(op, "sym_name") orelse "unknown_var";
    const slot = try ctx_lowering.getStorageSlot(var_name);
    try ctx_lowering.write("  // Global variable: ");
    try ctx_lowering.write(var_name);
    try ctx_lowering.write(" -> slot ");
    const slot_str = std.fmt.allocPrint(ctx_lowering.allocator, "{d}", .{slot}) catch "?";
    defer if (std.mem.eql(u8, slot_str, "?")) {} else ctx_lowering.allocator.free(slot_str);
    try ctx_lowering.writeln(slot_str);
}

/// Process func.func operation
fn processFuncFunc(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const func_name = getStringAttribute(op, "sym_name") orelse "unknown_func";
    const is_init = getBoolAttribute(op, "ora.init") orelse false;

    if (is_init) {
        try ctx_lowering.writeln("  function constructor() {");
    } else {
        try ctx_lowering.write("  function ");
        try ctx_lowering.write(func_name);
        try ctx_lowering.writeln("() {");
    }

    // Process function body with proper indentation
    ctx_lowering.indent_level += 1; // Increase indentation for function body

    const num_regions = c.mlirOperationGetNumRegions(op);
    for (0..@intCast(num_regions)) |region_idx| {
        const region = c.mlirOperationGetRegion(op, @intCast(region_idx));
        var current_block = c.mlirRegionGetFirstBlock(region);
        while (!c.mlirBlockIsNull(current_block)) {
            // Process operations directly since we're already at the right indentation level
            var current_op = c.mlirBlockGetFirstOperation(current_block);
            while (!c.mlirOperationIsNull(current_op)) {
                processOperation(current_op, ctx_lowering);
                current_op = c.mlirOperationGetNextInBlock(current_op);
            }
            current_block = c.mlirBlockGetNextInRegion(current_block);
        }
    }

    ctx_lowering.indent_level -= 1; // Decrease indentation after function body

    try ctx_lowering.writeln("  }");
    try ctx_lowering.writeln("");
}

/// Process ora.sload operation
fn processOraSLoad(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const global_name = getStringAttribute(op, "global") orelse "unknown";
    const slot = try ctx_lowering.getStorageSlot(global_name);

    // Get the result value and create a variable name for it
    const result_value = c.mlirOperationGetResult(op, 0);
    const result_name = try ctx_lowering.getValueName(result_value);

    // Store the expression for later use
    const slot_str = std.fmt.allocPrint(ctx_lowering.allocator, "{d}", .{slot}) catch "0";
    defer ctx_lowering.allocator.free(slot_str);
    const expr = try std.fmt.allocPrint(ctx_lowering.allocator, "sload({s})", .{slot_str});

    try ctx_lowering.setValueExpression(result_value, expr);

    try ctx_lowering.writeIndented("let ");
    try ctx_lowering.write(result_name);
    try ctx_lowering.write(" := ");
    try ctx_lowering.write(expr);
    try ctx_lowering.writeln("");
}

/// Process ora.sstore operation
fn processOraSStore(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands != 1) {
        try ctx_lowering.addError("ora.sstore expects exactly 1 operand");
        return;
    }

    const global_name = getStringAttribute(op, "global") orelse "unknown";
    const slot = try ctx_lowering.getStorageSlot(global_name);

    // Get the value to store
    const value_to_store = c.mlirOperationGetOperand(op, 0);

    try ctx_lowering.writeIndented("sstore(");
    const slot_str = std.fmt.allocPrint(ctx_lowering.allocator, "{d}", .{slot}) catch "0";
    defer ctx_lowering.allocator.free(slot_str);
    try ctx_lowering.write(slot_str);
    try ctx_lowering.write(", ");
    try processValueInline(value_to_store, ctx_lowering);
    try ctx_lowering.writeln(")");
}

/// Process arith.constant operation
fn processArithConstant(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const value = getIntAttribute(op, "value") orelse 0;

    // Get the result value
    const result_value = c.mlirOperationGetResult(op, 0);

    // Convert boolean values: MLIR uses -1 for true, but Yul uses 1 for true
    const yul_value = if (value == -1) 1 else value;

    // Store the constant as an expression
    const value_str = try std.fmt.allocPrint(ctx_lowering.allocator, "{d}", .{yul_value});

    try ctx_lowering.setValueExpression(result_value, value_str);
}

/// Get the contract name from the module
fn getContractName(_: c.MlirModule) ?[]const u8 {
    // For now, return a default name - we'll implement proper name extraction later
    return "SimpleContract";
}

/// Process a value inline (for conditions, expressions, etc.)
fn processValueInline(value: c.MlirValue, ctx_lowering: *YulLoweringContext) !void {
    // Check if we have an expression for this value
    if (ctx_lowering.getValueExpression(value)) |expr| {
        try ctx_lowering.write(expr);
        return;
    }

    // Fall back to variable name
    const value_name = ctx_lowering.getValueNameOr(value, "0");
    try ctx_lowering.write(value_name);
}

/// Process arith.addi operation
fn processArithAddi(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands != 2) {
        try ctx_lowering.addError("arith.addi expects exactly 2 operands");
        return;
    }

    // Get the result value
    const result_value = c.mlirOperationGetResult(op, 0);

    // Get operand values - these should already be processed
    const lhs_value = c.mlirOperationGetOperand(op, 0);
    const rhs_value = c.mlirOperationGetOperand(op, 1);

    // Store the expression for later use
    const lhs_name = ctx_lowering.getValueNameOr(lhs_value, "0");
    const rhs_name = ctx_lowering.getValueNameOr(rhs_value, "0");
    const expr = try std.fmt.allocPrint(ctx_lowering.allocator, "add({s}, {s})", .{ lhs_name, rhs_name });

    try ctx_lowering.setValueExpression(result_value, expr);
}

/// Process arith.cmpi operation
fn processArithCmpi(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands != 2) {
        try ctx_lowering.addError("arith.cmpi expects exactly 2 operands");
        return;
    }

    const predicate = getIntAttribute(op, "predicate") orelse 0;

    // Get the result value
    const result_value = c.mlirOperationGetResult(op, 0);

    // Get operand values - these should already be processed
    const lhs_value = c.mlirOperationGetOperand(op, 0);
    const rhs_value = c.mlirOperationGetOperand(op, 1);

    // Map MLIR comparison predicates to Yul
    const yul_op = switch (predicate) {
        8 => "gt", // greater than
        4 => "lt", // less than
        2 => "eq", // equal
        else => "eq",
    };

    // Store the expression for later use
    const lhs_name = ctx_lowering.getValueNameOr(lhs_value, "0");
    const rhs_name = ctx_lowering.getValueNameOr(rhs_value, "0");
    const expr = try std.fmt.allocPrint(ctx_lowering.allocator, "{s}({s}, {s})", .{ yul_op, lhs_name, rhs_name });

    try ctx_lowering.setValueExpression(result_value, expr);
}

/// Process an MLIR block
fn processBlock(block: c.MlirBlock, ctx_lowering: *YulLoweringContext) !void {
    // Increase indentation for operations inside blocks
    ctx_lowering.indent_level += 1;

    var op = c.mlirBlockGetFirstOperation(block);
    var op_count: u32 = 0;
    while (!c.mlirOperationIsNull(op)) {
        op_count += 1;
        // Debug output removed for cleaner Yul generation
        processOperation(op, ctx_lowering);
        op = c.mlirOperationGetNextInBlock(op);
    }

    // Empty blocks are handled naturally by the Yul generation

    // Decrease indentation when done with the block
    ctx_lowering.indent_level -= 1;
}

/// Process scf.yield operation
fn processScfYield(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);

    if (num_operands > 0) {
        // scf.yield with operands - generate return statement
        try ctx_lowering.writeIndented("return(");

        for (0..@intCast(num_operands)) |i| {
            if (i > 0) {
                try ctx_lowering.write(", ");
            }
            const operand = c.mlirOperationGetOperand(op, @intCast(i));
            const operand_name = ctx_lowering.getValueNameOr(operand, "0");
            try ctx_lowering.write(operand_name);
        }

        try ctx_lowering.writeln(")");
    }
    // scf.yield with no operands is a no-op (just marks end of region)
}

/// Process scf.if operation
fn processScfIf(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands != 1) {
        try ctx_lowering.addError("scf.if expects exactly 1 operand (condition)");
        return;
    }

    // Get the condition value and generate inline condition
    const condition_value = c.mlirOperationGetOperand(op, 0);
    try ctx_lowering.writeIndented("if (");
    try processValueInline(condition_value, ctx_lowering);
    try ctx_lowering.writeln(") {");

    // Process the then region (region 0)
    const num_regions = c.mlirOperationGetNumRegions(op);
    if (num_regions >= 1) {
        const then_region = c.mlirOperationGetRegion(op, 0);
        const then_block = c.mlirRegionGetFirstBlock(then_region);
        if (!c.mlirBlockIsNull(then_block)) {
            try processBlock(then_block, ctx_lowering);
        }
    }

    // Process the else region (region 1) if it exists and has meaningful content
    if (num_regions >= 2) {
        const else_region = c.mlirOperationGetRegion(op, 1);
        const else_block = c.mlirRegionGetFirstBlock(else_region);
        if (!c.mlirBlockIsNull(else_block)) {
            // Check if the else block has meaningful content (not just scf.yield with no operands)
            var has_meaningful_content = false;
            var else_op = c.mlirBlockGetFirstOperation(else_block);
            while (!c.mlirOperationIsNull(else_op)) {
                const op_name = getOperationName(else_op);
                if (!std.mem.eql(u8, op_name, "scf.yield") or c.mlirOperationGetNumOperands(else_op) > 0) {
                    has_meaningful_content = true;
                    break;
                }
                else_op = c.mlirOperationGetNextInBlock(else_op);
            }

            if (has_meaningful_content) {
                try ctx_lowering.writelnIndented("} else {");
                try processBlock(else_block, ctx_lowering);
                try ctx_lowering.writelnIndented("}");
            } else {
                try ctx_lowering.writelnIndented("}");
            }
        } else {
            try ctx_lowering.writelnIndented("}");
        }
    } else {
        try ctx_lowering.writelnIndented("}");
    }
}

/// Process func.return operation
fn processFuncReturn(_: c.MlirOperation, _: *YulLoweringContext) !void {
    // func.return operations are handled by scf.yield operations in the regions
    // No additional Yul code needs to be generated for func.return
}

//===----------------------------------------------------------------------===//
// Control Flow Operations
//===----------------------------------------------------------------------===//

/// Process ora.if operation
fn processOraIf(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands != 1) {
        try ctx_lowering.addError("ora.if expects exactly 1 operand (condition)");
        return;
    }

    // Get the condition value
    const condition_value = c.mlirOperationGetOperand(op, 0);
    const condition_name = try ctx_lowering.getValueNameOrCreate(condition_value, "false");

    try ctx_lowering.writeIndented("if (");
    try ctx_lowering.write(condition_name);
    try ctx_lowering.writeln(") {");

    // Process the then region (region 0)
    const num_regions = c.mlirOperationGetNumRegions(op);
    if (num_regions >= 1) {
        const then_region = c.mlirOperationGetRegion(op, 0);
        const then_block = c.mlirRegionGetFirstBlock(then_region);
        if (!c.mlirBlockIsNull(then_block)) {
            try processBlock(then_block, ctx_lowering);
        }
    }

    try ctx_lowering.writelnIndented("}");

    // Process the else region (region 1) if it exists
    if (num_regions >= 2) {
        try ctx_lowering.writeln("    else {");
        const else_region = c.mlirOperationGetRegion(op, 1);
        const else_block = c.mlirRegionGetFirstBlock(else_region);
        if (!c.mlirBlockIsNull(else_block)) {
            try processBlock(else_block, ctx_lowering);
        }
        try ctx_lowering.writelnIndented("}");
    }
}

/// Process ora.while operation
fn processOraWhile(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands != 1) {
        try ctx_lowering.addError("ora.while expects exactly 1 operand (condition)");
        return;
    }

    // Get the condition value
    const condition_value = c.mlirOperationGetOperand(op, 0);
    const condition_name = try ctx_lowering.getValueNameOrCreate(condition_value, "false");

    try ctx_lowering.write("    for {} ");
    try ctx_lowering.write(condition_name);
    try ctx_lowering.writeln(" {} {");

    // Process the loop body region (region 0)
    const num_regions = c.mlirOperationGetNumRegions(op);
    if (num_regions >= 1) {
        const body_region = c.mlirOperationGetRegion(op, 0);
        const body_block = c.mlirRegionGetFirstBlock(body_region);
        if (!c.mlirBlockIsNull(body_block)) {
            try processBlock(body_block, ctx_lowering);
        }
    }

    try ctx_lowering.writelnIndented("}");
}

/// Process ora.for operation
fn processOraFor(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands != 1) {
        try ctx_lowering.addError("ora.for expects exactly 1 operand (collection)");
        return;
    }

    // Get the collection value
    const collection_value = c.mlirOperationGetOperand(op, 0);
    const collection_name = try ctx_lowering.getValueNameOrCreate(collection_value, "empty");

    try ctx_lowering.write("    for {} true {} {");
    try ctx_lowering.writeln("");
    try ctx_lowering.write("      // iterate over ");
    try ctx_lowering.write(collection_name);
    try ctx_lowering.writeln("");

    // Process the loop body region (region 0)
    const num_regions = c.mlirOperationGetNumRegions(op);
    if (num_regions >= 1) {
        const body_region = c.mlirOperationGetRegion(op, 0);
        const body_block = c.mlirRegionGetFirstBlock(body_region);
        if (!c.mlirBlockIsNull(body_block)) {
            try processBlock(body_block, ctx_lowering);
        }
    }

    try ctx_lowering.writelnIndented("}");
}

/// Process ora.switch operation
fn processOraSwitch(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands != 1) {
        try ctx_lowering.addError("ora.switch expects exactly 1 operand (value)");
        return;
    }

    // Get the switch value
    const switch_value = c.mlirOperationGetOperand(op, 0);
    const switch_name = ctx_lowering.getValueNameOr(switch_value, "0");

    try ctx_lowering.write("    switch ");
    try ctx_lowering.write(switch_name);
    try ctx_lowering.writeln("");
    try ctx_lowering.writeln("      case 0 {");
    try ctx_lowering.writeln("        // case 0");
    try ctx_lowering.writeln("      }");
    try ctx_lowering.writeln("      default {");
    try ctx_lowering.writeln("        // default case");
    try ctx_lowering.writeln("      }");
}

/// Process ora.break operation
fn processOraBreak(_: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    // Break is implemented as a simple break statement
    try ctx_lowering.writeln("    break");
}

/// Process ora.continue operation
fn processOraContinue(_: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    // Continue is implemented as a simple continue statement
    try ctx_lowering.writeln("    continue");
}

/// Process ora.return operation
fn processOraReturn(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);

    if (num_operands == 0) {
        // Void return
        try ctx_lowering.writeln("    return(0, 0)");
    } else if (num_operands == 1) {
        // Single value return
        const return_value = c.mlirOperationGetOperand(op, 0);
        const return_name = ctx_lowering.getValueNameOr(return_value, "0");

        try ctx_lowering.write("    return(0, ");
        try ctx_lowering.write(return_name);
        try ctx_lowering.writeln(")");
    } else {
        // Multiple values - for now, just return the first one
        const return_value = c.mlirOperationGetOperand(op, 0);
        const return_name = ctx_lowering.getValueNameOr(return_value, "0");

        try ctx_lowering.write("    return(0, ");
        try ctx_lowering.write(return_name);
        try ctx_lowering.writeln(")");
    }
}

//===----------------------------------------------------------------------===//
// Arithmetic Operations
//===----------------------------------------------------------------------===//

/// Process arith.subi operation
fn processArithSubi(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands != 2) {
        try ctx_lowering.addError("arith.subi expects exactly 2 operands");
        return;
    }

    // Get the result value
    const result_value = c.mlirOperationGetResult(op, 0);
    const result_name = try ctx_lowering.getValueName(result_value);

    // Get operand values
    const lhs_value = c.mlirOperationGetOperand(op, 0);
    const rhs_value = c.mlirOperationGetOperand(op, 1);

    const lhs_name = ctx_lowering.getValueNameOr(lhs_value, "0");
    const rhs_name = ctx_lowering.getValueNameOr(rhs_value, "0");

    try ctx_lowering.writeIndented("let ");
    try ctx_lowering.write(result_name);
    try ctx_lowering.write(" := sub(");
    try ctx_lowering.write(lhs_name);
    try ctx_lowering.write(", ");
    try ctx_lowering.write(rhs_name);
    try ctx_lowering.writeln(")");
}

/// Process arith.muli operation
fn processArithMuli(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands != 2) {
        try ctx_lowering.addError("arith.muli expects exactly 2 operands");
        return;
    }

    // Get the result value
    const result_value = c.mlirOperationGetResult(op, 0);
    const result_name = try ctx_lowering.getValueName(result_value);

    // Get operand values
    const lhs_value = c.mlirOperationGetOperand(op, 0);
    const rhs_value = c.mlirOperationGetOperand(op, 1);

    const lhs_name = ctx_lowering.getValueNameOr(lhs_value, "0");
    const rhs_name = ctx_lowering.getValueNameOr(rhs_value, "0");

    try ctx_lowering.writeIndented("let ");
    try ctx_lowering.write(result_name);
    try ctx_lowering.write(" := mul(");
    try ctx_lowering.write(lhs_name);
    try ctx_lowering.write(", ");
    try ctx_lowering.write(rhs_name);
    try ctx_lowering.writeln(")");
}

/// Process arith.divi operation
fn processArithDivi(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands != 2) {
        try ctx_lowering.addError("arith.divi expects exactly 2 operands");
        return;
    }

    // Get the result value
    const result_value = c.mlirOperationGetResult(op, 0);
    const result_name = try ctx_lowering.getValueName(result_value);

    // Get operand values
    const lhs_value = c.mlirOperationGetOperand(op, 0);
    const rhs_value = c.mlirOperationGetOperand(op, 1);

    const lhs_name = ctx_lowering.getValueNameOr(lhs_value, "0");
    const rhs_name = ctx_lowering.getValueNameOr(rhs_value, "0");

    try ctx_lowering.writeIndented("let ");
    try ctx_lowering.write(result_name);
    try ctx_lowering.write(" := div(");
    try ctx_lowering.write(lhs_name);
    try ctx_lowering.write(", ");
    try ctx_lowering.write(rhs_name);
    try ctx_lowering.writeln(")");
}

/// Process arith.remi operation
fn processArithRemi(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands != 2) {
        try ctx_lowering.addError("arith.remi expects exactly 2 operands");
        return;
    }

    // Get the result value
    const result_value = c.mlirOperationGetResult(op, 0);
    const result_name = try ctx_lowering.getValueName(result_value);

    // Get operand values
    const lhs_value = c.mlirOperationGetOperand(op, 0);
    const rhs_value = c.mlirOperationGetOperand(op, 1);

    const lhs_name = ctx_lowering.getValueNameOr(lhs_value, "0");
    const rhs_name = ctx_lowering.getValueNameOr(rhs_value, "0");

    try ctx_lowering.writeIndented("let ");
    try ctx_lowering.write(result_name);
    try ctx_lowering.write(" := mod(");
    try ctx_lowering.write(lhs_name);
    try ctx_lowering.write(", ");
    try ctx_lowering.write(rhs_name);
    try ctx_lowering.writeln(")");
}

/// Process arith.andi operation
fn processArithAndi(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands != 2) {
        try ctx_lowering.addError("arith.andi expects exactly 2 operands");
        return;
    }

    // Get the result value
    const result_value = c.mlirOperationGetResult(op, 0);
    const result_name = try ctx_lowering.getValueName(result_value);

    // Get operand values
    const lhs_value = c.mlirOperationGetOperand(op, 0);
    const rhs_value = c.mlirOperationGetOperand(op, 1);

    const lhs_name = ctx_lowering.getValueNameOr(lhs_value, "0");
    const rhs_name = ctx_lowering.getValueNameOr(rhs_value, "0");

    try ctx_lowering.writeIndented("let ");
    try ctx_lowering.write(result_name);
    try ctx_lowering.write(" := and(");
    try ctx_lowering.write(lhs_name);
    try ctx_lowering.write(", ");
    try ctx_lowering.write(rhs_name);
    try ctx_lowering.writeln(")");
}

/// Process arith.ori operation
fn processArithOri(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands != 2) {
        try ctx_lowering.addError("arith.ori expects exactly 2 operands");
        return;
    }

    // Get the result value
    const result_value = c.mlirOperationGetResult(op, 0);
    const result_name = try ctx_lowering.getValueName(result_value);

    // Get operand values
    const lhs_value = c.mlirOperationGetOperand(op, 0);
    const rhs_value = c.mlirOperationGetOperand(op, 1);

    const lhs_name = ctx_lowering.getValueNameOr(lhs_value, "0");
    const rhs_name = ctx_lowering.getValueNameOr(rhs_value, "0");

    try ctx_lowering.writeIndented("let ");
    try ctx_lowering.write(result_name);
    try ctx_lowering.write(" := or(");
    try ctx_lowering.write(lhs_name);
    try ctx_lowering.write(", ");
    try ctx_lowering.write(rhs_name);
    try ctx_lowering.writeln(")");
}

/// Process arith.xori operation
fn processArithXori(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands != 2) {
        try ctx_lowering.addError("arith.xori expects exactly 2 operands");
        return;
    }

    // Get the result value
    const result_value = c.mlirOperationGetResult(op, 0);
    const result_name = try ctx_lowering.getValueName(result_value);

    // Get operand values
    const lhs_value = c.mlirOperationGetOperand(op, 0);
    const rhs_value = c.mlirOperationGetOperand(op, 1);

    const lhs_name = ctx_lowering.getValueNameOr(lhs_value, "0");
    const rhs_name = ctx_lowering.getValueNameOr(rhs_value, "0");

    try ctx_lowering.writeIndented("let ");
    try ctx_lowering.write(result_name);
    try ctx_lowering.write(" := xor(");
    try ctx_lowering.write(lhs_name);
    try ctx_lowering.write(", ");
    try ctx_lowering.write(rhs_name);
    try ctx_lowering.writeln(")");
}

/// Process arith.shli operation
fn processArithShli(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands != 2) {
        try ctx_lowering.addError("arith.shli expects exactly 2 operands");
        return;
    }

    // Get the result value
    const result_value = c.mlirOperationGetResult(op, 0);
    const result_name = try ctx_lowering.getValueName(result_value);

    // Get operand values
    const lhs_value = c.mlirOperationGetOperand(op, 0);
    const rhs_value = c.mlirOperationGetOperand(op, 1);

    const lhs_name = ctx_lowering.getValueNameOr(lhs_value, "0");
    const rhs_name = ctx_lowering.getValueNameOr(rhs_value, "0");

    try ctx_lowering.writeIndented("let ");
    try ctx_lowering.write(result_name);
    try ctx_lowering.write(" := shl(");
    try ctx_lowering.write(lhs_name);
    try ctx_lowering.write(", ");
    try ctx_lowering.write(rhs_name);
    try ctx_lowering.writeln(")");
}

/// Process arith.shrsi operation
fn processArithShrsi(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands != 2) {
        try ctx_lowering.addError("arith.shrsi expects exactly 2 operands");
        return;
    }

    // Get the result value
    const result_value = c.mlirOperationGetResult(op, 0);
    const result_name = try ctx_lowering.getValueName(result_value);

    // Get operand values
    const lhs_value = c.mlirOperationGetOperand(op, 0);
    const rhs_value = c.mlirOperationGetOperand(op, 1);

    const lhs_name = ctx_lowering.getValueNameOr(lhs_value, "0");
    const rhs_name = ctx_lowering.getValueNameOr(rhs_value, "0");

    try ctx_lowering.writeIndented("let ");
    try ctx_lowering.write(result_name);
    try ctx_lowering.write(" := shr(");
    try ctx_lowering.write(lhs_name);
    try ctx_lowering.write(", ");
    try ctx_lowering.write(rhs_name);
    try ctx_lowering.writeln(")");
}

/// Process arith.shrui operation
fn processArithShrui(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands != 2) {
        try ctx_lowering.addError("arith.shrui expects exactly 2 operands");
        return;
    }

    // Get the result value
    const result_value = c.mlirOperationGetResult(op, 0);
    const result_name = try ctx_lowering.getValueName(result_value);

    // Get operand values
    const lhs_value = c.mlirOperationGetOperand(op, 0);
    const rhs_value = c.mlirOperationGetOperand(op, 1);

    const lhs_name = ctx_lowering.getValueNameOr(lhs_value, "0");
    const rhs_name = ctx_lowering.getValueNameOr(rhs_value, "0");

    try ctx_lowering.writeIndented("let ");
    try ctx_lowering.write(result_name);
    try ctx_lowering.write(" := shr(");
    try ctx_lowering.write(lhs_name);
    try ctx_lowering.write(", ");
    try ctx_lowering.write(rhs_name);
    try ctx_lowering.writeln(")");
}

/// Process ora.power operation
fn processOraPower(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands != 2) {
        try ctx_lowering.addError("ora.power expects exactly 2 operands");
        return;
    }

    // Get the result value
    const result_value = c.mlirOperationGetResult(op, 0);
    const result_name = try ctx_lowering.getValueName(result_value);

    // Get operand values
    const base_value = c.mlirOperationGetOperand(op, 0);
    const exp_value = c.mlirOperationGetOperand(op, 1);

    const base_name = ctx_lowering.getValueNameOr(base_value, "0");
    const exp_name = ctx_lowering.getValueNameOr(exp_value, "0");

    // Yul doesn't have built-in power, so we'll use a simple implementation
    // For now, just use multiplication as a placeholder
    try ctx_lowering.writeIndented("let ");
    try ctx_lowering.write(result_name);
    try ctx_lowering.write(" := mul(");
    try ctx_lowering.write(base_name);
    try ctx_lowering.write(", ");
    try ctx_lowering.write(exp_name);
    try ctx_lowering.writeln(") // TODO: Implement proper power operation");
}

/// Process ora.cast operation
fn processOraCast(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands != 1) {
        try ctx_lowering.addError("ora.cast expects exactly 1 operand");
        return;
    }

    // Get the result value
    const result_value = c.mlirOperationGetResult(op, 0);
    const result_name = try ctx_lowering.getValueName(result_value);

    // Get operand value
    const input_value = c.mlirOperationGetOperand(op, 0);
    const input_name = ctx_lowering.getValueNameOr(input_value, "0");

    // For now, just copy the value (no actual casting in Yul)
    try ctx_lowering.writeIndented("let ");
    try ctx_lowering.write(result_name);
    try ctx_lowering.write(" := ");
    try ctx_lowering.write(input_name);
    try ctx_lowering.writeln(" // TODO: Implement proper type casting");
}

//===----------------------------------------------------------------------===//
// Memory Operations
//===----------------------------------------------------------------------===//

/// Process ora.mload operation
fn processOraMload(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands != 1) {
        try ctx_lowering.addError("ora.mload expects exactly 1 operand (address)");
        return;
    }

    // Get the result value
    const result_value = c.mlirOperationGetResult(op, 0);
    const result_name = try ctx_lowering.getValueName(result_value);

    // Get the address operand
    const address_value = c.mlirOperationGetOperand(op, 0);
    const address_name = ctx_lowering.getValueNameOr(address_value, "0");

    try ctx_lowering.writeIndented("let ");
    try ctx_lowering.write(result_name);
    try ctx_lowering.write(" := mload(");
    try ctx_lowering.write(address_name);
    try ctx_lowering.writeln(")");
}

/// Process ora.mstore operation
fn processOraMstore(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands != 2) {
        try ctx_lowering.addError("ora.mstore expects exactly 2 operands (address, value)");
        return;
    }

    // Get the address and value operands
    const address_value = c.mlirOperationGetOperand(op, 0);
    const value_value = c.mlirOperationGetOperand(op, 1);

    const address_name = ctx_lowering.getValueNameOr(address_value, "0");
    const value_name = ctx_lowering.getValueNameOr(value_value, "0");

    try ctx_lowering.write("    mstore(");
    try ctx_lowering.write(address_name);
    try ctx_lowering.write(", ");
    try ctx_lowering.write(value_name);
    try ctx_lowering.writeln(")");
}

/// Process ora.tload operation
fn processOraTload(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands != 1) {
        try ctx_lowering.addError("ora.tload expects exactly 1 operand (address)");
        return;
    }

    // Get the result value
    const result_value = c.mlirOperationGetResult(op, 0);
    const result_name = try ctx_lowering.getValueName(result_value);

    // Get the address operand
    const address_value = c.mlirOperationGetOperand(op, 0);
    const address_name = ctx_lowering.getValueNameOr(address_value, "0");

    try ctx_lowering.writeIndented("let ");
    try ctx_lowering.write(result_name);
    try ctx_lowering.write(" := tload(");
    try ctx_lowering.write(address_name);
    try ctx_lowering.writeln(")");
}

/// Process ora.tstore operation
fn processOraTstore(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands != 2) {
        try ctx_lowering.addError("ora.tstore expects exactly 2 operands (address, value)");
        return;
    }

    // Get the address and value operands
    const address_value = c.mlirOperationGetOperand(op, 0);
    const value_value = c.mlirOperationGetOperand(op, 1);

    const address_name = ctx_lowering.getValueNameOr(address_value, "0");
    const value_name = ctx_lowering.getValueNameOr(value_value, "0");

    try ctx_lowering.write("    tstore(");
    try ctx_lowering.write(address_name);
    try ctx_lowering.write(", ");
    try ctx_lowering.write(value_name);
    try ctx_lowering.writeln(")");
}

//===----------------------------------------------------------------------===//
// Constant Operations
//===----------------------------------------------------------------------===//

/// Process ora.string.constant operation
fn processOraStringConstant(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    // Get the result value
    const result_value = c.mlirOperationGetResult(op, 0);
    const result_name = try ctx_lowering.getValueName(result_value);

    // Get the string value from attributes
    const string_value = getStringAttribute(op, "value") orelse "";

    // For now, just use a placeholder - in a real implementation, we'd need to handle
    // string literals properly in Yul (which doesn't have native string support)
    try ctx_lowering.writeIndented("let ");
    try ctx_lowering.write(result_name);
    try ctx_lowering.write(" := \"");
    try ctx_lowering.write(string_value);
    try ctx_lowering.writeln("\" // TODO: Implement proper string handling");
}

/// Process ora.hex.constant operation
fn processOraHexConstant(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    // Get the result value
    const result_value = c.mlirOperationGetResult(op, 0);
    const result_name = try ctx_lowering.getValueName(result_value);

    // Get the hex value from attributes
    const hex_value = getStringAttribute(op, "value") orelse "0x0";

    try ctx_lowering.writeIndented("let ");
    try ctx_lowering.write(result_name);
    try ctx_lowering.write(" := ");
    try ctx_lowering.write(hex_value);
    try ctx_lowering.writeln("");
}

/// Process ora.binary.constant operation
fn processOraBinaryConstant(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    // Get the result value
    const result_value = c.mlirOperationGetResult(op, 0);
    const result_name = try ctx_lowering.getValueName(result_value);

    // Get the binary value from attributes
    const binary_value = getStringAttribute(op, "value") orelse "0";

    try ctx_lowering.writeIndented("let ");
    try ctx_lowering.write(result_name);
    try ctx_lowering.write(" := ");
    try ctx_lowering.write(binary_value);
    try ctx_lowering.writeln(" // TODO: Convert binary to decimal");
}

/// Process ora.address.constant operation
fn processOraAddressConstant(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    // Get the result value
    const result_value = c.mlirOperationGetResult(op, 0);
    const result_name = try ctx_lowering.getValueName(result_value);

    // Get the address value from attributes
    const address_value = getStringAttribute(op, "value") orelse "0";

    try ctx_lowering.writeIndented("let ");
    try ctx_lowering.write(result_name);
    try ctx_lowering.write(" := ");
    try ctx_lowering.write(address_value);
    try ctx_lowering.writeln(" // TODO: Validate address format");
}

/// Process ora.const operation
fn processOraConst(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    // Get the result value
    const result_value = c.mlirOperationGetResult(op, 0);
    const result_name = try ctx_lowering.getValueName(result_value);

    // Get the constant value from attributes
    const const_value = getStringAttribute(op, "value") orelse "0";

    try ctx_lowering.writeIndented("let ");
    try ctx_lowering.write(result_name);
    try ctx_lowering.write(" := ");
    try ctx_lowering.write(const_value);
    try ctx_lowering.writeln(" // const variable");
}

/// Process ora.immutable operation
fn processOraImmutable(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    // Get the result value
    const result_value = c.mlirOperationGetResult(op, 0);
    const result_name = try ctx_lowering.getValueName(result_value);

    // Get the immutable value from attributes
    const immutable_value = getStringAttribute(op, "value") orelse "0";

    try ctx_lowering.writeIndented("let ");
    try ctx_lowering.write(result_name);
    try ctx_lowering.write(" := ");
    try ctx_lowering.write(immutable_value);
    try ctx_lowering.writeln(" // immutable variable");
}

//===----------------------------------------------------------------------===//
// Data Structure Operations
//===----------------------------------------------------------------------===//

/// Process ora.struct.decl operation
fn processOraStructDecl(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    // Struct declarations are typically handled at the module level
    // For now, just add a comment indicating the struct declaration
    const struct_name = getStringAttribute(op, "name") orelse "UnknownStruct";
    try ctx_lowering.writeln("    // Struct declaration: ");
    try ctx_lowering.write("    // ");
    try ctx_lowering.writeln(struct_name);
}

/// Process ora.struct.instantiate operation
fn processOraStructInstantiate(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    // Get the result value
    const result_value = c.mlirOperationGetResult(op, 0);
    const result_name = try ctx_lowering.getValueName(result_value);

    // For now, create a simple struct instantiation
    // In a real implementation, this would handle struct layout and initialization
    try ctx_lowering.writeIndented("let ");
    try ctx_lowering.write(result_name);
    try ctx_lowering.writeln(" := 0 // TODO: Implement proper struct instantiation");
}

/// Process ora.struct.field.store operation
fn processOraStructFieldStore(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands < 2) {
        try ctx_lowering.addError("ora.struct.field.store expects at least 2 operands (struct, value)");
        return;
    }

    // Get the struct and value operands
    const struct_value = c.mlirOperationGetOperand(op, 0);
    const value_value = c.mlirOperationGetOperand(op, 1);

    const struct_name = ctx_lowering.getValueNameOr(struct_value, "0");
    const value_name = ctx_lowering.getValueNameOr(value_value, "0");

    // Get field name from attributes
    const field_name = getStringAttribute(op, "field") orelse "unknown";

    // For now, just add a comment - in a real implementation, this would
    // calculate the correct storage offset and store the value
    try ctx_lowering.write("    // Store to struct field: ");
    try ctx_lowering.write(field_name);
    try ctx_lowering.write(" in struct ");
    try ctx_lowering.write(struct_name);
    try ctx_lowering.write(" = ");
    try ctx_lowering.writeln(value_name);
}

/// Process ora.struct.init operation
fn processOraStructInit(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    // Get the result value
    const result_value = c.mlirOperationGetResult(op, 0);
    const result_name = try ctx_lowering.getValueName(result_value);

    // For now, create a simple struct initialization
    try ctx_lowering.writeIndented("let ");
    try ctx_lowering.write(result_name);
    try ctx_lowering.writeln(" := 0 // TODO: Implement proper struct initialization");
}

/// Process ora.enum.decl operation
fn processOraEnumDecl(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    // Enum declarations are typically handled at the module level
    const enum_name = getStringAttribute(op, "name") orelse "UnknownEnum";
    try ctx_lowering.writeln("    // Enum declaration: ");
    try ctx_lowering.write("    // ");
    try ctx_lowering.writeln(enum_name);
}

/// Process ora.enum.constant operation
fn processOraEnumConstant(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    // Get the result value
    const result_value = c.mlirOperationGetResult(op, 0);
    const result_name = try ctx_lowering.getValueName(result_value);

    // Get the enum value from attributes
    const enum_value = getStringAttribute(op, "value") orelse "0";

    try ctx_lowering.writeIndented("let ");
    try ctx_lowering.write(result_name);
    try ctx_lowering.write(" := ");
    try ctx_lowering.write(enum_value);
    try ctx_lowering.writeln(" // TODO: Implement proper enum value mapping");
}

/// Process ora.map.get operation
fn processOraMapGet(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands != 2) {
        try ctx_lowering.addError("ora.map.get expects exactly 2 operands (map, key)");
        return;
    }

    // Get the result value
    const result_value = c.mlirOperationGetResult(op, 0);
    const result_name = try ctx_lowering.getValueName(result_value);

    // Get the map and key operands
    const map_value = c.mlirOperationGetOperand(op, 0);
    const key_value = c.mlirOperationGetOperand(op, 1);

    const map_name = ctx_lowering.getValueNameOr(map_value, "0");
    const key_name = ctx_lowering.getValueNameOr(key_value, "0");

    // For now, use a simple storage-based map implementation
    // In a real implementation, this would use proper map storage patterns
    try ctx_lowering.writeIndented("let ");
    try ctx_lowering.write(result_name);
    try ctx_lowering.write(" := sload(add(");
    try ctx_lowering.write(map_name);
    try ctx_lowering.write(", ");
    try ctx_lowering.write(key_name);
    try ctx_lowering.writeln(")) // TODO: Implement proper map storage");
}

/// Process ora.map.store operation
fn processOraMapStore(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands != 3) {
        try ctx_lowering.addError("ora.map.store expects exactly 3 operands (map, key, value)");
        return;
    }

    // Get the map, key, and value operands
    const map_value = c.mlirOperationGetOperand(op, 0);
    const key_value = c.mlirOperationGetOperand(op, 1);
    const value_value = c.mlirOperationGetOperand(op, 2);

    const map_name = ctx_lowering.getValueNameOr(map_value, "0");
    const key_name = ctx_lowering.getValueNameOr(key_value, "0");
    const value_name = ctx_lowering.getValueNameOr(value_value, "0");

    // For now, use a simple storage-based map implementation
    try ctx_lowering.write("    sstore(add(");
    try ctx_lowering.write(map_name);
    try ctx_lowering.write(", ");
    try ctx_lowering.write(key_name);
    try ctx_lowering.write("), ");
    try ctx_lowering.write(value_name);
    try ctx_lowering.writeln(") // TODO: Implement proper map storage");
}

//===----------------------------------------------------------------------===//
// Formal Verification Operations
//===----------------------------------------------------------------------===//

/// Process ora.requires operation
fn processOraRequires(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands != 1) {
        try ctx_lowering.addError("ora.requires expects exactly 1 operand (condition)");
        return;
    }

    // Get the condition operand
    const condition_value = c.mlirOperationGetOperand(op, 0);
    const condition_name = ctx_lowering.getValueNameOr(condition_value, "true");

    // For now, just add a comment - in a real implementation, this would
    // generate runtime assertion code or be used for static verification
    try ctx_lowering.write("    // requires: ");
    try ctx_lowering.writeln(condition_name);
}

/// Process ora.ensures operation
fn processOraEnsures(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands != 1) {
        try ctx_lowering.addError("ora.ensures expects exactly 1 operand (condition)");
        return;
    }

    // Get the condition operand
    const condition_value = c.mlirOperationGetOperand(op, 0);
    const condition_name = ctx_lowering.getValueNameOr(condition_value, "true");

    // For now, just add a comment - in a real implementation, this would
    // generate runtime assertion code or be used for static verification
    try ctx_lowering.write("    // ensures: ");
    try ctx_lowering.writeln(condition_name);
}

/// Process ora.invariant operation
fn processOraInvariant(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands != 1) {
        try ctx_lowering.addError("ora.invariant expects exactly 1 operand (condition)");
        return;
    }

    // Get the condition operand
    const condition_value = c.mlirOperationGetOperand(op, 0);
    const condition_name = ctx_lowering.getValueNameOr(condition_value, "true");

    // For now, just add a comment - in a real implementation, this would
    // generate runtime assertion code or be used for static verification
    try ctx_lowering.write("    // invariant: ");
    try ctx_lowering.writeln(condition_name);
}

/// Process ora.old operation
fn processOraOld(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands != 1) {
        try ctx_lowering.addError("ora.old expects exactly 1 operand (value)");
        return;
    }

    // Get the result value
    const result_value = c.mlirOperationGetResult(op, 0);
    const result_name = try ctx_lowering.getValueName(result_value);

    // Get the value operand
    const value_value = c.mlirOperationGetOperand(op, 0);
    const value_name = ctx_lowering.getValueNameOr(value_value, "0");

    // For now, just use the value directly - in a real implementation, this would
    // store the old value at function entry and return it here
    try ctx_lowering.writeIndented("let ");
    try ctx_lowering.write(result_name);
    try ctx_lowering.write(" := ");
    try ctx_lowering.write(value_name);
    try ctx_lowering.writeln(" // TODO: Implement old value tracking");
}

//===----------------------------------------------------------------------===//
// Events & Logging Operations
//===----------------------------------------------------------------------===//

/// Process ora.log operation
fn processOraLog(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands == 0) {
        try ctx_lowering.addError("ora.log expects at least 1 operand (event data)");
        return;
    }

    // Get the event name from attributes
    const event_name = getStringAttribute(op, "event") orelse "Log";

    // For now, just add a comment - in a real implementation, this would
    // generate proper event logging code
    try ctx_lowering.write("    // log event: ");
    try ctx_lowering.write(event_name);
    try ctx_lowering.write(" with ");
    try ctx_lowering.write(std.fmt.allocPrint(ctx_lowering.allocator, "{}", .{num_operands}) catch "0");
    try ctx_lowering.writeln(" parameters");
}

//===----------------------------------------------------------------------===//
// Financial Operations
//===----------------------------------------------------------------------===//

/// Process ora.move operation
fn processOraMove(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands != 2) {
        try ctx_lowering.addError("ora.move expects exactly 2 operands (from, to)");
        return;
    }

    // Get the from and to operands
    const from_value = c.mlirOperationGetOperand(op, 0);
    const to_value = c.mlirOperationGetOperand(op, 1);

    const from_name = ctx_lowering.getValueNameOr(from_value, "0");
    const to_name = ctx_lowering.getValueNameOr(to_value, "0");

    // For now, just add a comment - in a real implementation, this would
    // generate proper transfer code
    try ctx_lowering.write("    // move from ");
    try ctx_lowering.write(from_name);
    try ctx_lowering.write(" to ");
    try ctx_lowering.writeln(to_name);
}

//===----------------------------------------------------------------------===//
// Error Handling Operations
//===----------------------------------------------------------------------===//

/// Process ora.try_catch operation
fn processOraTryCatch(_: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    // For now, just add a comment - in a real implementation, this would
    // generate proper try-catch error handling code
    try ctx_lowering.writeln("    // TODO: Implement try-catch error handling");
}

/// Process ora.yield operation
fn processOraYield(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands > 0) {
        const operand = c.mlirOperationGetOperand(op, 0);
        const operand_name = try ctx_lowering.getValueNameOrCreate(operand, "0");
        try ctx_lowering.writeln("    // Yield: ");
        try ctx_lowering.writeln(operand_name);
    } else {
        try ctx_lowering.writeln("    // Yield: no operands");
    }
}

/// Process ora.error.decl operation
fn processOraErrorDecl(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands > 0) {
        const operand = c.mlirOperationGetOperand(op, 0);
        const operand_name = try ctx_lowering.getValueNameOrCreate(operand, "0");
        try ctx_lowering.writeln("    // Error declaration: ");
        try ctx_lowering.writeln(operand_name);
    } else {
        try ctx_lowering.writeln("    // Error declaration: no operands");
    }
}

/// Process ora.error operation
fn processOraError(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands > 0) {
        const operand = c.mlirOperationGetOperand(op, 0);
        const operand_name = try ctx_lowering.getValueNameOrCreate(operand, "0");
        try ctx_lowering.writeln("    // Error: ");
        try ctx_lowering.writeln(operand_name);
    } else {
        try ctx_lowering.writeln("    // Error: no operands");
    }
}

/// Process ora.error_union operation
fn processOraErrorUnion(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands > 0) {
        const operand = c.mlirOperationGetOperand(op, 0);
        const operand_name = try ctx_lowering.getValueNameOrCreate(operand, "0");
        try ctx_lowering.writeln("    // Error union: ");
        try ctx_lowering.writeln(operand_name);
    } else {
        try ctx_lowering.writeln("    // Error union: no operands");
    }
}

//===----------------------------------------------------------------------===//
// Switch & Pattern Matching Operations
//===----------------------------------------------------------------------===//

/// Process ora.switch.case operation
fn processOraSwitchCase(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands > 0) {
        const operand = c.mlirOperationGetOperand(op, 0);
        const operand_name = try ctx_lowering.getValueNameOrCreate(operand, "0");
        try ctx_lowering.writeln("    // Switch case: ");
        try ctx_lowering.writeln(operand_name);
    } else {
        try ctx_lowering.writeln("    // Switch case: no operands");
    }
}

/// Process ora.switch.range operation
fn processOraSwitchRange(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands >= 2) {
        const start_operand = c.mlirOperationGetOperand(op, 0);
        const end_operand = c.mlirOperationGetOperand(op, 1);
        const start_name = try ctx_lowering.getValueNameOrCreate(start_operand, "0");
        const end_name = try ctx_lowering.getValueNameOrCreate(end_operand, "1");
        try ctx_lowering.writeln("    // Switch range: ");
        try ctx_lowering.writeln(start_name);
        try ctx_lowering.writeln("    // ... to ...");
        try ctx_lowering.writeln(end_name);
    } else {
        try ctx_lowering.writeln("    // Switch range: insufficient operands");
    }
}

/// Process ora.switch.else operation
fn processOraSwitchElse(_: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    try ctx_lowering.writeln("    // Switch else case");
}

/// Process ora.pattern.match operation
fn processOraPatternMatch(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands > 0) {
        const operand = c.mlirOperationGetOperand(op, 0);
        const operand_name = try ctx_lowering.getValueNameOrCreate(operand, "0");
        try ctx_lowering.writeln("    // Pattern match: ");
        try ctx_lowering.writeln(operand_name);
    } else {
        try ctx_lowering.writeln("    // Pattern match: no operands");
    }
}

/// Process ora.union.decl operation
fn processOraUnionDecl(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands > 0) {
        const operand = c.mlirOperationGetOperand(op, 0);
        const operand_name = try ctx_lowering.getValueNameOrCreate(operand, "0");
        try ctx_lowering.writeln("    // Union declaration: ");
        try ctx_lowering.writeln(operand_name);
    } else {
        try ctx_lowering.writeln("    // Union declaration: no operands");
    }
}

/// Process ora.union.instantiate operation
fn processOraUnionInstantiate(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands > 0) {
        const operand = c.mlirOperationGetOperand(op, 0);
        const operand_name = try ctx_lowering.getValueNameOrCreate(operand, "0");
        try ctx_lowering.writeln("    // Union instantiate: ");
        try ctx_lowering.writeln(operand_name);
    } else {
        try ctx_lowering.writeln("    // Union instantiate: no operands");
    }
}

/// Process ora.union.extract operation
fn processOraUnionExtract(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands > 0) {
        const operand = c.mlirOperationGetOperand(op, 0);
        const operand_name = try ctx_lowering.getValueNameOrCreate(operand, "0");
        try ctx_lowering.writeln("    // Union extract: ");
        try ctx_lowering.writeln(operand_name);
    } else {
        try ctx_lowering.writeln("    // Union extract: no operands");
    }
}

/// Process ora.union.discriminant operation
fn processOraUnionDiscriminant(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands > 0) {
        const operand = c.mlirOperationGetOperand(op, 0);
        const operand_name = try ctx_lowering.getValueNameOrCreate(operand, "0");
        try ctx_lowering.writeln("    // Union discriminant: ");
        try ctx_lowering.writeln(operand_name);
    } else {
        try ctx_lowering.writeln("    // Union discriminant: no operands");
    }
}

//===----------------------------------------------------------------------===//
// Advanced Features
//===----------------------------------------------------------------------===//

/// Process ora.destructure operation
fn processOraDestructure(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands == 0) {
        try ctx_lowering.addError("ora.destructure expects at least 1 operand (value to destructure)");
        return;
    }

    // For now, just add a comment - in a real implementation, this would
    // generate proper destructuring code
    try ctx_lowering.write("    // TODO: Implement destructuring with ");
    try ctx_lowering.write(std.fmt.allocPrint(ctx_lowering.allocator, "{}", .{num_operands}) catch "0");
    try ctx_lowering.writeln(" operands");
}

/// Process ora.lock operation
fn processOraLock(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands != 1) {
        try ctx_lowering.addError("ora.lock expects exactly 1 operand (resource)");
        return;
    }

    // Get the resource operand
    const resource_value = c.mlirOperationGetOperand(op, 0);
    const resource_name = ctx_lowering.getValueNameOr(resource_value, "resource");

    // For now, just add a comment - in a real implementation, this would
    // generate proper locking code
    try ctx_lowering.write("    // lock resource: ");
    try ctx_lowering.writeln(resource_name);
}

/// Process ora.unlock operation
fn processOraUnlock(op: c.MlirOperation, ctx_lowering: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands != 1) {
        try ctx_lowering.addError("ora.unlock expects exactly 1 operand (resource)");
        return;
    }

    // Get the resource operand
    const resource_value = c.mlirOperationGetOperand(op, 0);
    const resource_name = ctx_lowering.getValueNameOr(resource_value, "resource");

    // For now, just add a comment - in a real implementation, this would
    // generate proper unlocking code
    try ctx_lowering.write("    // unlock resource: ");
    try ctx_lowering.writeln(resource_name);
}

/// Helper function to get string attribute
fn getStringAttribute(op: c.MlirOperation, attr_name: []const u8) ?[]const u8 {
    const attr_name_ref = c.MlirStringRef{ .data = attr_name.ptr, .length = attr_name.len };
    const attr = c.mlirOperationGetAttributeByName(op, attr_name_ref);
    if (c.mlirAttributeIsNull(attr)) return null;

    const string_attr = c.mlirStringAttrGetValue(attr);
    return string_attr.data[0..string_attr.length];
}

/// Helper function to get integer attribute
fn getIntAttribute(op: c.MlirOperation, attr_name: []const u8) ?i64 {
    const attr_name_ref = c.MlirStringRef{ .data = attr_name.ptr, .length = attr_name.len };
    const attr = c.mlirOperationGetAttributeByName(op, attr_name_ref);
    if (c.mlirAttributeIsNull(attr)) return null;

    return c.mlirIntegerAttrGetValueInt(attr);
}

/// Helper function to get boolean attribute
fn getBoolAttribute(op: c.MlirOperation, attr_name: []const u8) ?bool {
    const attr_name_ref = c.MlirStringRef{ .data = attr_name.ptr, .length = attr_name.len };
    const attr = c.mlirOperationGetAttributeByName(op, attr_name_ref);
    if (c.mlirAttributeIsNull(attr)) return null;

    return c.mlirBoolAttrGetValue(attr);
}
