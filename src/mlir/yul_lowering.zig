//! Minimal MLIR to Yul Lowering

const std = @import("std");
const c = @import("c.zig").c;

const PublicFunction = struct {
    name: []const u8,
    selector: u32,
    has_return: bool,
};

const YulLoweringContext = struct {
    allocator: std.mem.Allocator,
    output: std.ArrayList(u8),
    errors: std.ArrayList([]const u8),
    storage_vars: std.HashMap([]const u8, u32, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),
    next_storage_slot: u32,
    indent_level: u32,
    // Value tracking for MLIR values to Yul variables
    value_map: std.HashMap(u64, []const u8, std.hash_map.AutoContext(u64), std.hash_map.default_max_load_percentage),
    next_temp_var: u32,
    // Public functions for dispatcher
    public_functions: std.ArrayList(PublicFunction),

    const Self = @This();

    fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .allocator = allocator,
            .output = std.ArrayList(u8){},
            .errors = std.ArrayList([]const u8){},
            .storage_vars = std.HashMap([]const u8, u32, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
            .next_storage_slot = 0,
            .indent_level = 0,
            .value_map = std.HashMap(u64, []const u8, std.hash_map.AutoContext(u64), std.hash_map.default_max_load_percentage).init(allocator),
            .next_temp_var = 0,
            .public_functions = std.ArrayList(PublicFunction){},
        };
    }

    fn deinit(self: *Self) void {
        self.output.deinit(self.allocator);
        for (self.errors.items) |err| {
            self.allocator.free(err);
        }
        self.errors.deinit(self.allocator);
        var iter = self.storage_vars.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.storage_vars.deinit();
        var value_iter = self.value_map.iterator();
        while (value_iter.next()) |entry| {
            self.allocator.free(entry.value_ptr.*);
        }
        self.value_map.deinit();
        for (self.public_functions.items) |func| {
            self.allocator.free(func.name);
        }
        self.public_functions.deinit(self.allocator);
    }

    fn write(self: *Self, text: []const u8) !void {
        try self.output.appendSlice(self.allocator, text);
    }

    fn writeln(self: *Self, text: []const u8) !void {
        try self.output.appendSlice(self.allocator, text);
        try self.output.append(self.allocator, '\n');
    }

    fn writeIndented(self: *Self, text: []const u8) !void {
        var i: u32 = 0;
        while (i < self.indent_level) : (i += 1) {
            try self.output.appendSlice(self.allocator, "  ");
        }
        try self.output.appendSlice(self.allocator, text);
    }

    fn writelnIndented(self: *Self, text: []const u8) !void {
        try self.writeIndented(text);
        try self.output.append(self.allocator, '\n');
    }

    fn addError(self: *Self, message: []const u8) !void {
        const error_msg = try self.allocator.dupe(u8, message);
        try self.errors.append(self.allocator, error_msg);
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

    fn generateTempVar(self: *Self) ![]const u8 {
        const var_name = try std.fmt.allocPrint(self.allocator, "temp_{d}", .{self.next_temp_var});
        self.next_temp_var += 1;
        return var_name;
    }

    fn getValueName(self: *Self, value: c.MlirValue) ![]const u8 {
        const value_ptr = @intFromPtr(value.ptr);
        if (self.value_map.get(value_ptr)) |name| {
            return name;
        }

        // Generate new temp variable
        const temp_name = try self.generateTempVar();
        try self.value_map.put(value_ptr, temp_name);
        return temp_name;
    }

    fn setValueName(self: *Self, value: c.MlirValue, name: []const u8) !void {
        const value_ptr = @intFromPtr(value.ptr);
        const name_copy = try self.allocator.dupe(u8, name);
        try self.value_map.put(value_ptr, name_copy);
    }

    fn addPublicFunction(self: *Self, name: []const u8, has_return: bool) !void {
        const name_copy = try self.allocator.dupe(u8, name);
        const selector = calculateFunctionSelector(name);
        try self.public_functions.append(self.allocator, PublicFunction{
            .name = name_copy,
            .selector = selector,
            .has_return = has_return,
        });
    }
};

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

pub fn lowerToYul(
    module: c.MlirModule,
    ctx: c.MlirContext,
    allocator: std.mem.Allocator,
) !YulLoweringResult {
    _ = ctx;
    var ctx_lowering = YulLoweringContext.init(allocator);
    defer ctx_lowering.deinit();

    // Generate Yul structure by parsing MLIR
    try generateYulFromMLIR(module, &ctx_lowering);

    const success = ctx_lowering.errors.items.len == 0;

    return YulLoweringResult{
        .yul_code = try ctx_lowering.output.toOwnedSlice(ctx_lowering.allocator),
        .success = success,
        .errors = try ctx_lowering.errors.toOwnedSlice(ctx_lowering.allocator),
        .allocator = allocator,
    };
}

fn generateYulFromMLIR(module: c.MlirModule, ctx: *YulLoweringContext) !void {
    // Start Yul object
    try ctx.writeln("object \"Contract\" {");
    try ctx.writeln("  code {");
    try ctx.writeln("    datacopy(0, dataoffset(\"runtime\"), datasize(\"runtime\"))");
    try ctx.writeln("    return(0, datasize(\"runtime\"))");
    try ctx.writeln("  }");
    try ctx.writeln("  object \"runtime\" {");
    try ctx.writeln("    code {");

    ctx.indent_level = 3;

    // Process the MLIR module to collect functions first
    try processModule(module, ctx);

    // Generate dispatcher with collected functions
    try generateDispatcher(ctx);
    try ctx.writeln("");

    ctx.indent_level = 0;
    try ctx.writeln("    }");
    try ctx.writeln("  }");
    try ctx.writeln("}");
}

fn processModule(module: c.MlirModule, ctx: *YulLoweringContext) !void {
    const module_op = c.mlirModuleGetOperation(module);
    processOperation(module_op, ctx);
}

fn generateDispatcher(ctx: *YulLoweringContext) !void {
    try ctx.writelnIndented("switch shr(224, calldataload(0))");

    // Generate case statements for each public function
    for (ctx.public_functions.items) |func| {
        try ctx.writeIndented("case 0x");
        const selector_str = try std.fmt.allocPrint(ctx.allocator, "{x:0>8}", .{func.selector});
        defer ctx.allocator.free(selector_str);
        try ctx.write(selector_str);
        try ctx.writeln(" {");

        if (func.has_return) {
            try ctx.writeIndented("  let result := ");
            try ctx.write(func.name);
            try ctx.writeln("()");
            try ctx.writelnIndented("  mstore(0, result)");
            try ctx.writelnIndented("  return(0, 32)");
        } else {
            try ctx.writeIndented("  ");
            try ctx.write(func.name);
            try ctx.writeln("()");
            try ctx.writelnIndented("  return(0, 0)");
        }
        try ctx.writelnIndented("}");
    }

    try ctx.writelnIndented("default { revert(0, 0) }");
}

fn processOperation(op: c.MlirOperation, ctx: *YulLoweringContext) void {
    const op_name = getOperationName(op);

    if (std.mem.eql(u8, op_name, "builtin.module")) {
        processBuiltinModule(op, ctx) catch |err| {
            ctx.addError("Failed to process builtin.module") catch {};
            std.log.err("Error processing builtin.module: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "ora.contract")) {
        processOraContract(op, ctx) catch |err| {
            ctx.addError("Failed to process ora.contract") catch {};
            std.log.err("Error processing ora.contract: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "ora.global")) {
        processOraGlobal(op, ctx) catch |err| {
            ctx.addError("Failed to process ora.global") catch {};
            std.log.err("Error processing ora.global: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "func.func")) {
        processFuncFunc(op, ctx) catch |err| {
            ctx.addError("Failed to process func.func") catch {};
            std.log.err("Error processing func.func: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "ora.sload")) {
        processOraSload(op, ctx) catch |err| {
            ctx.addError("Failed to process ora.sload") catch {};
            std.log.err("Error processing ora.sload: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "ora.sstore")) {
        processOraSstore(op, ctx) catch |err| {
            ctx.addError("Failed to process ora.sstore") catch {};
            std.log.err("Error processing ora.sstore: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "arith.constant")) {
        processArithConstant(op, ctx) catch |err| {
            ctx.addError("Failed to process arith.constant") catch {};
            std.log.err("Error processing arith.constant: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "arith.addi")) {
        processArithAddi(op, ctx) catch |err| {
            ctx.addError("Failed to process arith.addi") catch {};
            std.log.err("Error processing arith.addi: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "arith.cmpi")) {
        processArithCmpi(op, ctx) catch |err| {
            ctx.addError("Failed to process arith.cmpi") catch {};
            std.log.err("Error processing arith.cmpi: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "scf.if")) {
        processScfIf(op, ctx) catch |err| {
            ctx.addError("Failed to process scf.if") catch {};
            std.log.err("Error processing scf.if: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "scf.yield")) {
        processScfYield(op, ctx) catch |err| {
            ctx.addError("Failed to process scf.yield") catch {};
            std.log.err("Error processing scf.yield: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "func.return")) {
        processFuncReturn(op, ctx) catch |err| {
            ctx.addError("Failed to process func.return") catch {};
            std.log.err("Error processing func.return: {}", .{err});
        };
    }
    // Ignore other operations for now
}

fn getOperationName(op: c.MlirOperation) []const u8 {
    const op_name = c.mlirOperationGetName(op);
    const op_name_str = c.mlirIdentifierStr(op_name);
    return op_name_str.data[0..op_name_str.length];
}

fn processBuiltinModule(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    const num_regions = c.mlirOperationGetNumRegions(op);
    for (0..@intCast(num_regions)) |region_idx| {
        const region = c.mlirOperationGetRegion(op, @intCast(region_idx));
        var current_block = c.mlirRegionGetFirstBlock(region);
        while (!c.mlirBlockIsNull(current_block)) {
            var current_op = c.mlirBlockGetFirstOperation(current_block);
            while (!c.mlirOperationIsNull(current_op)) {
                processOperation(current_op, ctx);
                current_op = c.mlirOperationGetNextInBlock(current_op);
            }
            current_block = c.mlirBlockGetNextInRegion(current_block);
        }
    }
}

fn processOraContract(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    const num_regions = c.mlirOperationGetNumRegions(op);
    for (0..@intCast(num_regions)) |region_idx| {
        const region = c.mlirOperationGetRegion(op, @intCast(region_idx));
        var current_block = c.mlirRegionGetFirstBlock(region);
        while (!c.mlirBlockIsNull(current_block)) {
            var current_op = c.mlirBlockGetFirstOperation(current_block);
            while (!c.mlirOperationIsNull(current_op)) {
                processOperation(current_op, ctx);
                current_op = c.mlirOperationGetNextInBlock(current_op);
            }
            current_block = c.mlirBlockGetNextInRegion(current_block);
        }
    }
}

fn processOraGlobal(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    const var_name = getStringAttribute(op, "sym_name") orelse blk: {
        try ctx.addError("ora.global operation missing required sym_name attribute");
        break :blk "unknown_var";
    };
    const slot = try ctx.getStorageSlot(var_name);

    // Add comment about storage mapping
    try ctx.writeIndented("// Global variable: ");
    try ctx.write(var_name);
    try ctx.write(" -> slot ");
    const slot_str = try std.fmt.allocPrint(ctx.allocator, "{d}", .{slot});
    defer ctx.allocator.free(slot_str);
    try ctx.writeln(slot_str);
}

fn processFuncFunc(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    const func_name = getStringAttribute(op, "sym_name") orelse "unknown_func";
    const is_init = getBoolAttribute(op, "ora.init") orelse false;

    if (is_init) {
        try ctx.writelnIndented("function constructor() {");
    } else {
        // Check if this function actually returns a value by looking at func.return operations in the body
        const has_return_value = checkFunctionHasReturnValue(op);

        // Add to public functions for dispatcher (assume all non-init functions are public for now)
        try ctx.addPublicFunction(func_name, has_return_value);

        try ctx.writeIndented("function ");
        try ctx.write(func_name);
        try ctx.write("()");
        if (has_return_value) {
            try ctx.write(" -> result");
        }
        try ctx.writeln(" {");
    }

    ctx.indent_level += 1;

    // Process function body
    try processFuncBody(op, ctx);

    ctx.indent_level -= 1;
    try ctx.writelnIndented("}");
    try ctx.writeln("");
}

fn getStringAttribute(op: c.MlirOperation, attr_name: []const u8) ?[]const u8 {
    const attr_name_ref = c.MlirStringRef{ .data = attr_name.ptr, .length = attr_name.len };
    const attr = c.mlirOperationGetAttributeByName(op, attr_name_ref);
    if (c.mlirAttributeIsNull(attr)) return null;

    const string_attr = c.mlirStringAttrGetValue(attr);
    return string_attr.data[0..string_attr.length];
}

fn getBoolAttribute(op: c.MlirOperation, attr_name: []const u8) ?bool {
    const attr_name_ref = c.MlirStringRef{ .data = attr_name.ptr, .length = attr_name.len };
    const attr = c.mlirOperationGetAttributeByName(op, attr_name_ref);
    if (c.mlirAttributeIsNull(attr)) return null;

    return c.mlirBoolAttrGetValue(attr);
}

fn checkFunctionHasReturnValue(func_op: c.MlirOperation) bool {
    // Parse the function_type attribute to check return type
    const func_type_attr = c.mlirOperationGetAttributeByName(func_op, c.MlirStringRef{ .data = "function_type".ptr, .length = "function_type".len });
    if (c.mlirAttributeIsNull(func_type_attr)) return false;

    const func_type = c.mlirTypeAttrGetValue(func_type_attr);
    if (c.mlirTypeIsNull(func_type)) return false;

    // Check if it's a function type and get number of results
    if (c.mlirTypeIsAFunction(func_type)) {
        const num_results = c.mlirFunctionTypeGetNumResults(func_type);
        return num_results > 0;
    }

    // Fallback: check function name for known patterns
    const func_name = getStringAttribute(func_op, "sym_name") orelse return false;

    // Common patterns that usually return values
    if (std.mem.indexOf(u8, func_name, "get") != null) return true;
    if (std.mem.indexOf(u8, func_name, "check") != null) return true;
    if (std.mem.indexOf(u8, func_name, "increment") != null) return true;
    if (std.mem.indexOf(u8, func_name, "value") != null) return true;

    // Common patterns that usually don't return values
    if (std.mem.indexOf(u8, func_name, "set") != null and std.mem.indexOf(u8, func_name, "get") == null) return false;
    if (std.mem.indexOf(u8, func_name, "reset") != null) return false;
    if (std.mem.indexOf(u8, func_name, "init") != null) return false;

    // Default: assume no return value for safety
    return false;
}

fn calculateFunctionSelector(function_name: []const u8) u32 {
    // Create function signature: "functionName()"
    // For now, we assume no parameters (can be enhanced later)
    var signature_buf: [256]u8 = undefined;
    const signature = std.fmt.bufPrint(&signature_buf, "{s}()", .{function_name}) catch function_name;

    // Calculate Keccak256 hash
    var hasher = std.crypto.hash.sha3.Keccak256.init(.{});
    hasher.update(signature);
    var hash: [32]u8 = undefined;
    hasher.final(&hash);

    // Take first 4 bytes as selector (big-endian)
    return (@as(u32, hash[0]) << 24) |
        (@as(u32, hash[1]) << 16) |
        (@as(u32, hash[2]) << 8) |
        (@as(u32, hash[3]));
}

fn getIntegerAttribute(op: c.MlirOperation, attr_name: []const u8) ?i64 {
    const attr_name_ref = c.MlirStringRef{ .data = attr_name.ptr, .length = attr_name.len };
    const attr = c.mlirOperationGetAttributeByName(op, attr_name_ref);
    if (c.mlirAttributeIsNull(attr)) return null;

    return c.mlirIntegerAttrGetValueSInt(attr);
}

fn processFuncBody(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    const num_regions = c.mlirOperationGetNumRegions(op);
    for (0..@intCast(num_regions)) |region_idx| {
        const region = c.mlirOperationGetRegion(op, @intCast(region_idx));
        var current_block = c.mlirRegionGetFirstBlock(region);
        while (!c.mlirBlockIsNull(current_block)) {
            var current_op = c.mlirBlockGetFirstOperation(current_block);
            while (!c.mlirOperationIsNull(current_op)) {
                processOperation(current_op, ctx);
                current_op = c.mlirOperationGetNextInBlock(current_op);
            }
            current_block = c.mlirBlockGetNextInRegion(current_block);
        }
    }
}

fn processOraSload(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    const global_name = getStringAttribute(op, "global") orelse {
        try ctx.addError("ora.sload operation missing global attribute");
        return;
    };

    const slot = try ctx.getStorageSlot(global_name);
    const result = c.mlirOperationGetResult(op, 0);
    const var_name = try ctx.getValueName(result);

    try ctx.writeIndented("let ");
    try ctx.write(var_name);
    try ctx.write(" := sload(");
    const slot_str = try std.fmt.allocPrint(ctx.allocator, "{d}", .{slot});
    defer ctx.allocator.free(slot_str);
    try ctx.write(slot_str);
    try ctx.writeln(")");
}

fn processOraSstore(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    const global_name = getStringAttribute(op, "global") orelse {
        try ctx.addError("ora.sstore operation missing global attribute");
        return;
    };

    const slot = try ctx.getStorageSlot(global_name);
    const value = c.mlirOperationGetOperand(op, 0);
    const value_name = try ctx.getValueName(value);

    try ctx.writeIndented("sstore(");
    const slot_str = try std.fmt.allocPrint(ctx.allocator, "{d}", .{slot});
    defer ctx.allocator.free(slot_str);
    try ctx.write(slot_str);
    try ctx.write(", ");
    try ctx.write(value_name);
    try ctx.writeln(")");
}

fn processArithConstant(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    const value_attr = c.mlirOperationGetAttributeByName(op, c.MlirStringRef{
        .data = "value".ptr,
        .length = "value".len,
    });

    if (c.mlirAttributeIsNull(value_attr)) {
        try ctx.addError("arith.constant missing value attribute");
        return;
    }

    const result = c.mlirOperationGetResult(op, 0);
    const var_name = try ctx.getValueName(result);

    // For now, we'll extract the value as a simple integer
    // This is a simplified approach - in a full implementation we'd need proper type checking
    const int_value = c.mlirIntegerAttrGetValueSInt(value_attr);
    try ctx.writeIndented("let ");
    try ctx.write(var_name);
    try ctx.write(" := ");
    // Fix boolean representation: MLIR uses -1 for true, but Ethereum/Yul uses 1
    const ethereum_val = if (int_value == -1) 1 else int_value;
    const value_str = try std.fmt.allocPrint(ctx.allocator, "{d}", .{ethereum_val});
    defer ctx.allocator.free(value_str);
    try ctx.writeln(value_str);
}

fn processArithAddi(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    const lhs = c.mlirOperationGetOperand(op, 0);
    const rhs = c.mlirOperationGetOperand(op, 1);
    const result = c.mlirOperationGetResult(op, 0);

    const lhs_name = try ctx.getValueName(lhs);
    const rhs_name = try ctx.getValueName(rhs);
    const result_name = try ctx.getValueName(result);

    try ctx.writeIndented("let ");
    try ctx.write(result_name);
    try ctx.write(" := add(");
    try ctx.write(lhs_name);
    try ctx.write(", ");
    try ctx.write(rhs_name);
    try ctx.writeln(")");
}

fn processArithCmpi(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    const predicate = getIntegerAttribute(op, "predicate") orelse {
        try ctx.addError("arith.cmpi missing predicate attribute");
        return;
    };

    const lhs = c.mlirOperationGetOperand(op, 0);
    const rhs = c.mlirOperationGetOperand(op, 1);
    const result = c.mlirOperationGetResult(op, 0);

    const lhs_name = try ctx.getValueName(lhs);
    const rhs_name = try ctx.getValueName(rhs);
    const result_name = try ctx.getValueName(result);

    const yul_op = switch (predicate) {
        8 => "gt", // greater than (unsigned)
        2 => "eq", // equal
        1 => "ne", // not equal
        4 => "lt", // less than (unsigned)
        5 => "le", // less than or equal (unsigned)
        9 => "ge", // greater than or equal (unsigned)
        else => {
            try ctx.addError("Unsupported comparison predicate");
            return;
        },
    };

    try ctx.writeIndented("let ");
    try ctx.write(result_name);
    try ctx.write(" := ");
    try ctx.write(yul_op);
    try ctx.write("(");
    try ctx.write(lhs_name);
    try ctx.write(", ");
    try ctx.write(rhs_name);
    try ctx.writeln(")");
}

fn processScfIf(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    const condition = c.mlirOperationGetOperand(op, 0);
    const condition_name = try ctx.getValueName(condition);

    // Check if this scf.if has a result (for conditional expressions)
    const has_result = c.mlirOperationGetNumResults(op) > 0;
    var result_name: []const u8 = "";

    if (has_result) {
        const result = c.mlirOperationGetResult(op, 0);
        result_name = try ctx.getValueName(result);
        try ctx.writeIndented("let ");
        try ctx.write(result_name);
        try ctx.writeln("");
    }

    try ctx.writeIndented("if ");
    try ctx.write(condition_name);
    try ctx.writeln(" {");

    ctx.indent_level += 1;

    // Process then region (region 0)
    const then_region = c.mlirOperationGetRegion(op, 0);
    var then_block = c.mlirRegionGetFirstBlock(then_region);
    while (!c.mlirBlockIsNull(then_block)) {
        var current_op = c.mlirBlockGetFirstOperation(then_block);
        while (!c.mlirOperationIsNull(current_op)) {
            const op_name = getOperationName(current_op);
            if (std.mem.eql(u8, op_name, "scf.yield") and has_result) {
                // Handle yield with result value
                const yield_value = c.mlirOperationGetOperand(current_op, 0);
                const yield_value_name = try ctx.getValueName(yield_value);
                try ctx.writeIndented("");
                try ctx.write(result_name);
                try ctx.write(" := ");
                try ctx.writeln(yield_value_name);
            } else if (!std.mem.eql(u8, op_name, "scf.yield")) {
                processOperation(current_op, ctx);
            }
            current_op = c.mlirOperationGetNextInBlock(current_op);
        }
        then_block = c.mlirBlockGetNextInRegion(then_block);
    }

    ctx.indent_level -= 1;
    try ctx.writelnIndented("}");

    // Process else region if it exists and has content (region 1)
    const num_regions = c.mlirOperationGetNumRegions(op);
    if (num_regions > 1) {
        const else_region = c.mlirOperationGetRegion(op, 1);
        var else_block = c.mlirRegionGetFirstBlock(else_region);

        // Check if else block has operations other than scf.yield
        var has_else_content = false;
        while (!c.mlirBlockIsNull(else_block)) {
            var current_op = c.mlirBlockGetFirstOperation(else_block);
            while (!c.mlirOperationIsNull(current_op)) {
                const op_name = getOperationName(current_op);
                if (!std.mem.eql(u8, op_name, "scf.yield")) {
                    has_else_content = true;
                    break;
                }
                current_op = c.mlirOperationGetNextInBlock(current_op);
            }
            if (has_else_content) break;
            else_block = c.mlirBlockGetNextInRegion(else_block);
        }

        // In Yul, there's no 'else' keyword. We need to use a different pattern.
        // For conditional expressions, we can use nested if statements or switch statements.
        // For now, let's handle the else case by inverting the condition.
        if (has_else_content or has_result) {
            // Generate inverted condition for else case
            const else_condition = try std.fmt.allocPrint(ctx.allocator, "iszero({s})", .{condition_name});
            defer ctx.allocator.free(else_condition);

            try ctx.writeIndented("if ");
            try ctx.write(else_condition);
            try ctx.writeln(" {");
            ctx.indent_level += 1;

            else_block = c.mlirRegionGetFirstBlock(else_region);
            while (!c.mlirBlockIsNull(else_block)) {
                var current_op = c.mlirBlockGetFirstOperation(else_block);
                while (!c.mlirOperationIsNull(current_op)) {
                    const op_name = getOperationName(current_op);
                    if (std.mem.eql(u8, op_name, "scf.yield") and has_result) {
                        // Handle yield with result value
                        if (c.mlirOperationGetNumOperands(current_op) > 0) {
                            const yield_value = c.mlirOperationGetOperand(current_op, 0);
                            const yield_value_name = try ctx.getValueName(yield_value);
                            try ctx.writeIndented("");
                            try ctx.write(result_name);
                            try ctx.write(" := ");
                            try ctx.writeln(yield_value_name);
                        }
                    } else if (!std.mem.eql(u8, op_name, "scf.yield")) {
                        processOperation(current_op, ctx);
                    }
                    current_op = c.mlirOperationGetNextInBlock(current_op);
                }
                else_block = c.mlirBlockGetNextInRegion(else_block);
            }

            ctx.indent_level -= 1;
            try ctx.writelnIndented("}");
        }
    }
}

fn processScfYield(_: c.MlirOperation, _: *YulLoweringContext) !void {
    // scf.yield is handled by the parent scf.if operation
    // No direct Yul generation needed here
}

fn processFuncReturn(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands > 0) {
        const return_value = c.mlirOperationGetOperand(op, 0);
        const return_value_name = try ctx.getValueName(return_value);
        try ctx.writeIndented("result := ");
        try ctx.writeln(return_value_name);
    }
    // Note: In Yul functions, we don't need explicit return statements
    // The result variable is automatically returned
}
