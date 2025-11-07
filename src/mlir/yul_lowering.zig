// ============================================================================
// Yul Lowering
// ============================================================================
//
// Converts MLIR IR to Yul (Solidity assembly) for EVM compilation.
//
// ARCHITECTURE:
//   MLIR operations → Yul statements → EVM bytecode (via solc)
//
// FEATURES:
//   • Operation-by-operation Yul generation
//   • Function dispatcher generation
//   • ABI encoding/decoding
//   • Storage access patterns
//   • Memory management
//   • Public function selectors (4-byte signatures)
//
// SUPPORTED OPERATIONS:
//   • Arithmetic: add, sub, mul, div, mod
//   • Logical: and, or, xor, not
//   • Comparison: eq, ne, lt, gt, le, ge
//   • Memory: load, store, alloca
//   • Control flow: br, cond_br, return
//   • Special: storage_load, storage_store, log
//
// ============================================================================

const std = @import("std");
const c = @import("c.zig").c;
const lib = @import("ora_lib");
const ErrorHandler = @import("error_handling.zig").ErrorHandler;
const ErrorType = @import("error_handling.zig").ErrorType;
const WarningType = @import("error_handling.zig").WarningType;

const PublicFunction = struct {
    name: []const u8,
    signature: []const u8, // Full signature: "transfer(address,uint256)"
    selector: u32,
    has_return: bool,
    has_error_union: bool, // True if return type is an error union (!T | E1 | E2)
    num_params: u32, // Number of parameters
};

/// YulLoweringContext manages the MLIR to Yul conversion
///
/// Memory ownership:
/// - Owns: output ArrayList (accumulated Yul code)
/// - Owns: error_handler and all its errors/warnings
/// - Owns: storage_vars HashMap (variable name keys borrowed from MLIR)
/// - Owns: value_map HashMap (Yul variable name strings are owned)
/// - Owns: public_functions ArrayList
/// - Must call: deinit() to avoid leaks
const YulLoweringContext = struct {
    allocator: std.mem.Allocator,
    output: std.ArrayList(u8),
    error_handler: ErrorHandler,
    storage_vars: std.HashMap([]const u8, u32, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),
    next_storage_slot: u32,
    indent_level: u32,
    // Value tracking for MLIR values to Yul variables
    value_map: std.HashMap(u64, []const u8, std.hash_map.AutoContext(u64), std.hash_map.default_max_load_percentage),
    next_temp_var: u32,
    // Constant value tracking for optimization (SSA value -> constant literal)
    constant_values: std.HashMap(u64, []const u8, std.hash_map.AutoContext(u64), std.hash_map.default_max_load_percentage),
    // Public functions for dispatcher
    public_functions: std.ArrayList(PublicFunction),
    // Error code mapping (error name -> error code)
    error_codes: std.HashMap([]const u8, u32, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),
    next_error_code: u32,
    // Track which functions have error union return types
    functions_with_error_unions: std.HashMap([]const u8, bool, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),
    // Track current function being processed (for return statement context)
    current_function: ?[]const u8,
    // Track values that come from error calls (value pointer -> error name)
    error_call_results: std.HashMap(u64, []const u8, std.hash_map.AutoContext(u64), std.hash_map.default_max_load_percentage),

    const Self = @This();

    fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .allocator = allocator,
            .output = std.ArrayList(u8){},
            .error_handler = ErrorHandler.init(allocator),
            .storage_vars = std.HashMap([]const u8, u32, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
            .next_storage_slot = 0,
            .indent_level = 0,
            .value_map = std.HashMap(u64, []const u8, std.hash_map.AutoContext(u64), std.hash_map.default_max_load_percentage).init(allocator),
            .next_temp_var = 0,
            .constant_values = std.HashMap(u64, []const u8, std.hash_map.AutoContext(u64), std.hash_map.default_max_load_percentage).init(allocator),
            .public_functions = std.ArrayList(PublicFunction){},
            .error_codes = std.HashMap([]const u8, u32, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
            .next_error_code = 1, // Start at 1, 0 is reserved for success
            .functions_with_error_unions = std.HashMap([]const u8, bool, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
            .current_function = null,
            .error_call_results = std.HashMap(u64, []const u8, std.hash_map.AutoContext(u64), std.hash_map.default_max_load_percentage).init(allocator),
        };
    }

    fn deinit(self: *Self) void {
        self.output.deinit(self.allocator);
        self.error_handler.deinit();
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
        var const_iter = self.constant_values.iterator();
        while (const_iter.next()) |entry| {
            self.allocator.free(entry.value_ptr.*);
        }
        self.constant_values.deinit();
        for (self.public_functions.items) |func| {
            self.allocator.free(func.name);
            self.allocator.free(func.signature);
        }
        self.public_functions.deinit(self.allocator);
        var error_iter = self.error_codes.iterator();
        while (error_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.error_codes.deinit();
        var func_iter = self.functions_with_error_unions.iterator();
        while (func_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.functions_with_error_unions.deinit();
        var error_result_iter = self.error_call_results.iterator();
        while (error_result_iter.next()) |entry| {
            self.allocator.free(entry.value_ptr.*);
        }
        self.error_call_results.deinit();
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

    fn addError(self: *Self, error_type: ErrorType, message: []const u8, suggestion: ?[]const u8) !void {
        try self.error_handler.reportError(error_type, null, message, suggestion);
    }

    fn addWarning(self: *Self, warning_type: WarningType, message: []const u8) !void {
        try self.error_handler.reportWarning(warning_type, null, message);
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

        // Check if this is a constant value - if so, return the literal
        if (self.constant_values.get(value_ptr)) |const_val| {
            return const_val;
        }

        // Check if we already have a variable name for this value
        if (self.value_map.get(value_ptr)) |name| {
            return name;
        }

        // Generate new temp variable
        const temp_name = try self.generateTempVar();
        try self.value_map.put(value_ptr, temp_name);
        return temp_name;
    }

    fn setConstantValue(self: *Self, value: c.MlirValue, const_literal: []const u8) !void {
        const value_ptr = @intFromPtr(value.ptr);
        const literal_copy = try self.allocator.dupe(u8, const_literal);
        try self.constant_values.put(value_ptr, literal_copy);
    }

    fn setValueName(self: *Self, value: c.MlirValue, name: []const u8) !void {
        const value_ptr = @intFromPtr(value.ptr);
        const name_copy = try self.allocator.dupe(u8, name);
        try self.value_map.put(value_ptr, name_copy);
    }

    fn addPublicFunction(self: *Self, name: []const u8, signature: []const u8, has_return: bool, has_error_union: bool, num_params: u32) !void {
        const name_copy = try self.allocator.dupe(u8, name);
        const signature_copy = try self.allocator.dupe(u8, signature);
        const selector = calculateFunctionSelector(signature);
        try self.public_functions.append(self.allocator, PublicFunction{
            .name = name_copy,
            .signature = signature_copy,
            .selector = selector,
            .has_return = has_return,
            .has_error_union = has_error_union,
            .num_params = num_params,
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

    // Generate Yul structure by parsing MLIR
    try generateYulFromMLIR(module, &ctx_lowering);

    const success = !ctx_lowering.error_handler.hasErrors();

    // Convert LoweringError structs to strings BEFORE deinitializing the context
    const error_list = ctx_lowering.error_handler.getErrors();
    var error_strings = std.ArrayList([]const u8){};
    defer error_strings.deinit(allocator);
    for (error_list) |err| {
        const error_msg = try std.fmt.allocPrint(allocator, "{s}: {s}", .{ @tagName(err.error_type), err.message });
        try error_strings.append(allocator, error_msg);
    }

    // Create result with error strings
    const result = YulLoweringResult{
        .yul_code = try ctx_lowering.output.toOwnedSlice(ctx_lowering.allocator),
        .success = success,
        .errors = try error_strings.toOwnedSlice(allocator),
        .allocator = allocator,
    };

    // Now it's safe to deinitialize the context since we've copied all the data we need
    ctx_lowering.deinit();

    return result;
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

        // Decode calldata parameters
        if (func.num_params > 0) {
            var i: u32 = 0;
            while (i < func.num_params) : (i += 1) {
                const offset = 4 + (i * 32); // 4-byte selector + 32 bytes per param
                try ctx.writeIndented("  let arg");
                const arg_num = try std.fmt.allocPrint(ctx.allocator, "{d}", .{i});
                defer ctx.allocator.free(arg_num);
                try ctx.write(arg_num);
                try ctx.write(" := calldataload(");
                const offset_str = try std.fmt.allocPrint(ctx.allocator, "{d}", .{offset});
                defer ctx.allocator.free(offset_str);
                try ctx.write(offset_str);
                try ctx.writeln(")");
            }
        }

        // Call function with parameters
        if (func.has_return) {
            if (func.has_error_union) {
                // Error union returns: (value, error_code)
                try ctx.writeIndented("  let value, error_code := ");
                try ctx.write(func.name);
                try ctx.write("(");
                if (func.num_params > 0) {
                    var i: u32 = 0;
                    while (i < func.num_params) : (i += 1) {
                        if (i > 0) try ctx.write(", ");
                        try ctx.write("arg");
                        const arg_num = try std.fmt.allocPrint(ctx.allocator, "{d}", .{i});
                        defer ctx.allocator.free(arg_num);
                        try ctx.write(arg_num);
                    }
                }
                try ctx.writeln(")");
                // Check error_code and return value or revert
                try ctx.writelnIndented("  if error_code {");
                try ctx.writelnIndented("    revert(0, 0) // Error occurred");
                try ctx.writelnIndented("  }");
                try ctx.writelnIndented("  mstore(0, value)");
                try ctx.writelnIndented("  return(0, 32)");
            } else {
                // Normal return: result
                try ctx.writeIndented("  let result := ");
                try ctx.write(func.name);
                try ctx.write("(");
                if (func.num_params > 0) {
                    var i: u32 = 0;
                    while (i < func.num_params) : (i += 1) {
                        if (i > 0) try ctx.write(", ");
                        try ctx.write("arg");
                        const arg_num = try std.fmt.allocPrint(ctx.allocator, "{d}", .{i});
                        defer ctx.allocator.free(arg_num);
                        try ctx.write(arg_num);
                    }
                }
                try ctx.writeln(")");
                try ctx.writelnIndented("  mstore(0, result)");
                try ctx.writelnIndented("  return(0, 32)");
            }
        } else {
            try ctx.writeIndented("  ");
            try ctx.write(func.name);
            try ctx.write("(");
            if (func.num_params > 0) {
                var i: u32 = 0;
                while (i < func.num_params) : (i += 1) {
                    if (i > 0) try ctx.write(", ");
                    try ctx.write("arg");
                    const arg_num = try std.fmt.allocPrint(ctx.allocator, "{d}", .{i});
                    defer ctx.allocator.free(arg_num);
                    try ctx.write(arg_num);
                }
            }
            try ctx.writeln(")");
            try ctx.writelnIndented("  return(0, 0)");
        }
        try ctx.writelnIndented("}");
    }

    try ctx.writelnIndented("default { revert(0, 0) }");
}

fn processOperation(op: c.MlirOperation, ctx: *YulLoweringContext) void {
    const op_name = getOperationName(op);
    // std.log.info("Processing operation: {s}", .{op_name});

    if (std.mem.eql(u8, op_name, "builtin.module")) {
        processBuiltinModule(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process builtin.module", "check module structure") catch {};
            std.log.err("Error processing builtin.module: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "ora.contract")) {
        processOraContract(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process ora.contract", "check contract structure") catch {};
            std.log.err("Error processing ora.contract: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "ora.global")) {
        processOraGlobal(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process ora.global", "check global variable structure") catch {};
            std.log.err("Error processing ora.global: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "func.func")) {
        processFuncFunc(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process func.func", "check function structure") catch {};
            std.log.err("Error processing func.func: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "ora.sload")) {
        processOraSload(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process ora.sload", "check storage load operation") catch {};
            std.log.err("Error processing ora.sload: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "ora.sstore")) {
        processOraSstore(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process ora.sstore", "check storage store operation") catch {};
            std.log.err("Error processing ora.sstore: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "ora.mload")) {
        processOraMload(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process ora.mload", "check memory load operation") catch {};
            std.log.err("Error processing ora.mload: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "ora.mstore")) {
        processOraMstore(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process ora.mstore", "check memory store operation") catch {};
            std.log.err("Error processing ora.mstore: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "ora.tload")) {
        processOraTload(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process ora.tload", "check transient storage load operation") catch {};
            std.log.err("Error processing ora.tload: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "ora.tstore")) {
        processOraTstore(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process ora.tstore", "check transient storage store operation") catch {};
            std.log.err("Error processing ora.tstore: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "ora.tstore.global")) {
        processOraTstoreGlobal(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process ora.tstore.global", "check global transient storage store operation") catch {};
            std.log.err("Error processing ora.tstore.global: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "ora.assign")) {
        processOraAssign(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process ora.assign", "check assignment operation") catch {};
            std.log.err("Error processing ora.assign: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "arith.constant")) {
        processArithConstant(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process arith.constant", "check constant value") catch {};
            std.log.err("Error processing arith.constant: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "ora.string.constant")) {
        processOraStringConstant(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process ora.string.constant", "check string constant value") catch {};
            std.log.err("Error processing ora.string.constant: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "ora.hex.constant")) {
        processOraHexConstant(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process ora.hex.constant", "check hex constant value") catch {};
            std.log.err("Error processing ora.hex.constant: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "ora.address.constant")) {
        processOraAddressConstant(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process ora.address.constant", "check address constant value") catch {};
            std.log.err("Error processing ora.address.constant: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "ora.binary.constant")) {
        processOraBinaryConstant(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process ora.binary.constant", "check binary constant value") catch {};
            std.log.err("Error processing ora.binary.constant: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "memref.alloca")) {
        processMemrefAlloca(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process memref.alloca", "check local variable allocation") catch {};
            std.log.err("Error processing memref.alloca: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "memref.store")) {
        processMemrefStore(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process memref.store", "check local variable store") catch {};
            std.log.err("Error processing memref.store: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "memref.load")) {
        processMemrefLoad(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process memref.load", "check local variable load") catch {};
            std.log.err("Error processing memref.load: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "arith.addi")) {
        processArithAddi(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process arith.addi", "check addition operands") catch {};
            std.log.err("Error processing arith.addi: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "arith.subi")) {
        processArithSubi(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process arith.subi", "check subtraction operands") catch {};
            std.log.err("Error processing arith.subi: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "arith.muli")) {
        processArithMuli(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process arith.muli", "check multiplication operands") catch {};
            std.log.err("Error processing arith.muli: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "arith.divi")) {
        processArithDivi(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process arith.divi", "check division operands") catch {};
            std.log.err("Error processing arith.divi: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "arith.remi")) {
        processArithRemi(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process arith.remi", "check remainder operands") catch {};
            std.log.err("Error processing arith.remi: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "arith.andi")) {
        processArithAndi(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process arith.andi", "check bitwise AND operands") catch {};
            std.log.err("Error processing arith.andi: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "arith.ori")) {
        processArithOri(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process arith.ori", "check bitwise OR operands") catch {};
            std.log.err("Error processing arith.ori: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "arith.xori")) {
        processArithXori(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process arith.xori", "check bitwise XOR operands") catch {};
            std.log.err("Error processing arith.xori: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "arith.shli")) {
        processArithShli(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process arith.shli", "check left shift operands") catch {};
            std.log.err("Error processing arith.shli: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "arith.shri")) {
        // std.log.info("Processing arith.shri operation", .{});
        processArithShri(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process arith.shri", "check right shift operands") catch {};
            std.log.err("Error processing arith.shri: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "arith.shrsi")) {
        // std.log.info("Processing arith.shrsi operation", .{});
        processArithShri(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process arith.shrsi", "check signed right shift operands") catch {};
            std.log.err("Error processing arith.shrsi: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "arith.cmpi")) {
        processArithCmpi(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process arith.cmpi", "check comparison operands") catch {};
            std.log.err("Error processing arith.cmpi: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "arith.trunci")) {
        processArithTrunci(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process arith.trunci", "check truncation operands") catch {};
            std.log.err("Error processing arith.trunci: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "arith.extui")) {
        processArithExtui(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process arith.extui", "check extension operands") catch {};
            std.log.err("Error processing arith.extui: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "scf.if")) {
        processScfIf(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process scf.if", "check conditional structure") catch {};
            std.log.err("Error processing scf.if: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "scf.while")) {
        processScfWhile(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process scf.while", "check while loop structure") catch {};
            std.log.err("Error processing scf.while: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "scf.for")) {
        processScfFor(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process scf.for", "check for loop structure") catch {};
            std.log.err("Error processing scf.for: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "scf.yield")) {
        processScfYield(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process scf.yield", "check yield operation") catch {};
            std.log.err("Error processing scf.yield: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "func.call")) {
        processFuncCall(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process func.call", "check function call arguments") catch {};
            std.log.err("Error processing func.call: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "func.return")) {
        processFuncReturn(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process func.return", "check return operation") catch {};
            std.log.err("Error processing func.return: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "ora.move")) {
        processOraMove(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process ora.move", "check move operation") catch {};
            std.log.err("Error processing ora.move: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "ora.struct_instantiate")) {
        processOraStructInstantiate(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process ora.struct_instantiate", "check struct instantiation") catch {};
            std.log.err("Error processing ora.struct_instantiate: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "ora.struct_field_store")) {
        processOraStructFieldStore(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process ora.struct_field_store", "check struct field store") catch {};
            std.log.err("Error processing ora.struct_field_store: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "ora.map_get")) {
        processOraMapGet(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process ora.map_get", "check map get operation") catch {};
            std.log.err("Error processing ora.map_get: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "ora.map_store")) {
        processOraMapStore(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process ora.map_store", "check map store operation") catch {};
            std.log.err("Error processing ora.map_store: {}", .{err});
        };
    } else if (std.mem.startsWith(u8, op_name, "ora.evm.")) {
        // Handle all EVM builtin operations (e.g., ora.evm.timestamp, ora.evm.caller, etc.)
        processOraEvmBuiltin(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process ora.evm builtin", "check EVM builtin operation") catch {};
            std.log.err("Error processing ora.evm.* operation: {}", .{err});
        };
    } else if (std.mem.eql(u8, op_name, "ora.try")) {
        processOraTry(op, ctx) catch |err| {
            ctx.addError(.MlirOperationFailed, "Failed to process ora.try", "check try-catch structure") catch {};
            std.log.err("Error processing ora.try: {}", .{err});
        };
    } else {
        // Handle unprocessed operations - ensure their results get names and emit placeholder code
        // This prevents "undefined identifier" errors
        const num_results = c.mlirOperationGetNumResults(op);
        for (0..@intCast(num_results)) |i| {
            const result = c.mlirOperationGetResult(op, @intCast(i));
            const result_name = ctx.getValueName(result) catch |err| {
                std.log.warn("Failed to get value name for unprocessed operation result: {}", .{err});
                continue;
            };
            // Emit placeholder code to define the variable
            ctx.writeIndented("let ") catch {
                std.log.warn("Failed to write placeholder for unprocessed operation", .{});
                continue;
            };
            ctx.write(result_name) catch {
                std.log.warn("Failed to write result name for unprocessed operation", .{});
                continue;
            };
            ctx.writeln(" := 0 // Unprocessed operation placeholder") catch {
                std.log.warn("Failed to write placeholder value for unprocessed operation", .{});
                continue;
            };
        }
        // Log unprocessed operations for debugging
        std.log.debug("Unprocessed operation: {s}", .{op_name});
    }
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
        try ctx.addError(.MalformedAst, "ora.global operation missing required sym_name attribute", "add sym_name attribute to ora.global operation");
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
        const has_error_union = checkFunctionHasErrorUnion(op);

        // Build function signature for selector calculation
        var signature_buf: [512]u8 = undefined;
        var signature: []const u8 = func_name;

        // Extract function parameters from MLIR function type
        const func_type_attr = c.mlirOperationGetAttributeByName(op, c.MlirStringRef{ .data = "function_type".ptr, .length = "function_type".len });
        if (!c.mlirAttributeIsNull(func_type_attr)) {
            const func_type = c.mlirTypeAttrGetValue(func_type_attr);
            const num_inputs = c.mlirFunctionTypeGetNumInputs(func_type);

            // Build ABI signature with parameter types
            var param_buf: [256]u8 = undefined;
            var param_offset: usize = 0;

            if (num_inputs > 0) {
                var i: u32 = 0;
                while (i < num_inputs) : (i += 1) {
                    const input_type = c.mlirFunctionTypeGetInput(func_type, i);
                    const abi_type = try mlirTypeToAbiType(ctx.allocator, input_type);
                    defer ctx.allocator.free(abi_type);

                    if (i > 0) {
                        param_buf[param_offset] = ',';
                        param_offset += 1;
                    }
                    @memcpy(param_buf[param_offset..][0..abi_type.len], abi_type);
                    param_offset += abi_type.len;
                }
            }

            // Create full signature: "functionName(type1,type2)"
            signature = try std.fmt.bufPrint(&signature_buf, "{s}({s})", .{ func_name, param_buf[0..param_offset] });
        } else {
            // No parameters
            signature = try std.fmt.bufPrint(&signature_buf, "{s}()", .{func_name});
        }

        // Add to public functions for dispatcher with full signature
        const num_params: u32 = if (!c.mlirAttributeIsNull(func_type_attr))
            @intCast(c.mlirFunctionTypeGetNumInputs(c.mlirTypeAttrGetValue(func_type_attr)))
        else
            0;
        try ctx.addPublicFunction(func_name, signature, has_return_value, has_error_union, num_params);

        // Track if this function has error union for call site decoding
        if (has_error_union) {
            const func_name_copy = try ctx.allocator.dupe(u8, func_name);
            try ctx.functions_with_error_unions.put(func_name_copy, true);
        }

        // Set current function context
        ctx.current_function = func_name;

        try ctx.writeIndented("function ");
        try ctx.write(func_name);

        // Write Yul function parameters
        if (!c.mlirAttributeIsNull(func_type_attr)) {
            const func_type = c.mlirTypeAttrGetValue(func_type_attr);
            const num_inputs = c.mlirFunctionTypeGetNumInputs(func_type);

            if (num_inputs > 0) {
                try ctx.write("(");
                var i: u32 = 0;
                while (i < num_inputs) : (i += 1) {
                    if (i > 0) {
                        try ctx.write(", ");
                    }
                    try ctx.write("arg");
                    const arg_name = try std.fmt.allocPrint(ctx.allocator, "{d}", .{i});
                    defer ctx.allocator.free(arg_name);
                    try ctx.write(arg_name);
                }
                try ctx.write(")");
            } else {
                try ctx.write("()");
            }
        } else {
            try ctx.write("()");
        }

        if (has_return_value) {
            if (has_error_union) {
                // Error union returns: (value, error_code)
                try ctx.write(" -> value, error_code");
            } else {
                // Normal return: result
                try ctx.write(" -> result");
            }
        }
        try ctx.writeln(" {");
    }

    ctx.indent_level += 1;

    // Map MLIR block arguments to Yul argN parameters
    const num_regions = c.mlirOperationGetNumRegions(op);
    if (num_regions > 0) {
        const region = c.mlirOperationGetRegion(op, 0);
        const first_block = c.mlirRegionGetFirstBlock(region);
        if (!c.mlirBlockIsNull(first_block)) {
            const num_block_args = c.mlirBlockGetNumArguments(first_block);
            var arg_idx: u32 = 0;
            while (arg_idx < num_block_args) : (arg_idx += 1) {
                const block_arg = c.mlirBlockGetArgument(first_block, @intCast(arg_idx));
                const yul_arg_name = try std.fmt.allocPrint(ctx.allocator, "arg{d}", .{arg_idx});
                defer ctx.allocator.free(yul_arg_name); // Free after setValueName duplicates it
                try ctx.setValueName(block_arg, yul_arg_name);
            }
        }
    }

    // Process function body
    try processFuncBody(op, ctx);

    ctx.indent_level -= 1;
    try ctx.writelnIndented("}");
    try ctx.writeln("");

    // Clear current function context
    ctx.current_function = null;
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

/// Check if function has error union return type by scanning for error return operations
fn checkFunctionHasErrorUnion(func_op: c.MlirOperation) bool {
    // Scan function body for error return operations
    const num_regions = c.mlirOperationGetNumRegions(func_op);
    if (num_regions == 0) return false;

    const region = c.mlirOperationGetRegion(func_op, 0);
    var current_block = c.mlirRegionGetFirstBlock(region);

    // Common error names to check for (these are typically error types)
    const error_names = [_][]const u8{ "Unauthorized", "InvalidAmount", "InsufficientBalance", "InvalidAddress" };

    // Recursively scan all blocks and operations
    while (!c.mlirBlockIsNull(current_block)) {
        var current_op = c.mlirBlockGetFirstOperation(current_block);
        while (!c.mlirOperationIsNull(current_op)) {
            const op_name = getOperationName(current_op);

            // Check for error return attribute on the operation itself
            // Error returns are created as arith.constant with "ora.error" attribute
            const error_attr = c.mlirOperationGetAttributeByName(current_op, c.MlirStringRef{
                .data = "ora.error".ptr,
                .length = "ora.error".len,
            });
            if (!c.mlirAttributeIsNull(error_attr)) {
                return true;
            }

            // Check if this is a func.call to an error name
            // Error calls like Unauthorized(...) are function calls to error names
            if (std.mem.eql(u8, op_name, "func.call")) {
                const callee_attr = c.mlirOperationGetAttributeByName(current_op, c.MlirStringRef{
                    .data = "callee".ptr,
                    .length = "callee".len,
                });
                if (!c.mlirAttributeIsNull(callee_attr)) {
                    const callee_ref = c.mlirFlatSymbolRefAttrGetValue(callee_attr);
                    const callee_name = callee_ref.data[0..callee_ref.length];

                    // Check if callee is a known error name
                    for (error_names) |error_name| {
                        if (std.mem.eql(u8, callee_name, error_name)) {
                            // Check if this call is used in a return statement
                            // For now, if we see an error call, assume the function has error union
                            return true;
                        }
                    }
                }
            }

            // Recursively check nested regions (for control flow operations)
            const num_op_regions = c.mlirOperationGetNumRegions(current_op);
            for (0..@intCast(num_op_regions)) |region_idx| {
                const op_region = c.mlirOperationGetRegion(current_op, @intCast(region_idx));
                var op_block = c.mlirRegionGetFirstBlock(op_region);
                while (!c.mlirBlockIsNull(op_block)) {
                    var op_op = c.mlirBlockGetFirstOperation(op_block);
                    while (!c.mlirOperationIsNull(op_op)) {
                        const nested_op_name = getOperationName(op_op);

                        const nested_error_attr = c.mlirOperationGetAttributeByName(op_op, c.MlirStringRef{
                            .data = "ora.error".ptr,
                            .length = "ora.error".len,
                        });
                        if (!c.mlirAttributeIsNull(nested_error_attr)) {
                            return true;
                        }

                        // Check nested func.call operations too
                        if (std.mem.eql(u8, nested_op_name, "func.call")) {
                            const nested_callee_attr = c.mlirOperationGetAttributeByName(op_op, c.MlirStringRef{
                                .data = "callee".ptr,
                                .length = "callee".len,
                            });
                            if (!c.mlirAttributeIsNull(nested_callee_attr)) {
                                const nested_callee_ref = c.mlirFlatSymbolRefAttrGetValue(nested_callee_attr);
                                const nested_callee_name = nested_callee_ref.data[0..nested_callee_ref.length];

                                for (error_names) |error_name| {
                                    if (std.mem.eql(u8, nested_callee_name, error_name)) {
                                        return true;
                                    }
                                }
                            }
                        }

                        op_op = c.mlirOperationGetNextInBlock(op_op);
                    }
                    op_block = c.mlirBlockGetNextInRegion(op_block);
                }
            }

            current_op = c.mlirOperationGetNextInBlock(current_op);
        }
        current_block = c.mlirBlockGetNextInRegion(current_block);
    }

    return false;
}

/// Convert MLIR type to Solidity ABI type string
fn mlirTypeToAbiType(allocator: std.mem.Allocator, mlir_type: c.MlirType) ![]const u8 {
    // Check if it's an integer type
    if (c.mlirTypeIsAInteger(mlir_type)) {
        const width = c.mlirIntegerTypeGetWidth(mlir_type);
        // Map common widths to Solidity types
        if (width == 1) return allocator.dupe(u8, "bool");
        if (width == 160) return allocator.dupe(u8, "address"); // address is 160 bits
        if (width % 8 == 0 and width <= 256) {
            return std.fmt.allocPrint(allocator, "uint{d}", .{width});
        }
        // Default to uint256 for other integer types
        return allocator.dupe(u8, "uint256");
    }

    // For other types, use generic uint256 (EVM word size)
    // Complex types (arrays, structs) handled at higher lowering levels
    return allocator.dupe(u8, "uint256");
}

/// Calculate function selector from full signature "functionName(type1,type2)"
fn calculateFunctionSelector(signature: []const u8) u32 {
    // Calculate Keccak256 hash of the signature
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
        try ctx.addError(.MalformedAst, "ora.sload operation missing global attribute", "add global attribute to ora.sload operation");
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
        try ctx.addError(.MalformedAst, "ora.sstore operation missing global attribute", "add global attribute to ora.sstore operation");
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

fn processOraMload(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    const result = c.mlirOperationGetResult(op, 0);
    const result_name = try ctx.getValueName(result);

    const address = c.mlirOperationGetOperand(op, 0);
    const address_name = try ctx.getValueName(address);

    try ctx.writeIndented("let ");
    try ctx.write(result_name);
    try ctx.write(" := mload(");
    try ctx.write(address_name);
    try ctx.writeln(")");
}

fn processOraMstore(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    const address = c.mlirOperationGetOperand(op, 0);
    const value = c.mlirOperationGetOperand(op, 1);

    const address_name = try ctx.getValueName(address);
    const value_name = try ctx.getValueName(value);

    try ctx.writeIndented("mstore(");
    try ctx.write(address_name);
    try ctx.write(", ");
    try ctx.write(value_name);
    try ctx.writeln(")");
}

fn processOraTload(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    const key = getStringAttribute(op, "key") orelse {
        try ctx.addError(.MalformedAst, "ora.tload operation missing key attribute", "add key attribute to ora.tload operation");
        return;
    };

    const result = c.mlirOperationGetResult(op, 0);
    const result_name = try ctx.getValueName(result);

    // Transient storage uses tload() in Yul (EIP-1153)
    try ctx.writeIndented("let ");
    try ctx.write(result_name);
    try ctx.write(" := tload(");
    try ctx.write(key);
    try ctx.writeln(")");
}

fn processOraTstore(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    const key = getStringAttribute(op, "key") orelse {
        try ctx.addError(.MalformedAst, "ora.tstore operation missing key attribute", "add key attribute to ora.tstore operation");
        return;
    };

    const value = c.mlirOperationGetOperand(op, 0);
    const value_name = try ctx.getValueName(value);

    // Transient storage uses tstore() in Yul (EIP-1153)
    try ctx.writeIndented("tstore(");
    try ctx.write(key);
    try ctx.write(", ");
    try ctx.write(value_name);
    try ctx.writeln(")");
}

fn processOraTstoreGlobal(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    const global_name = getStringAttribute(op, "global") orelse {
        try ctx.addError(.MalformedAst, "ora.tstore.global operation missing global attribute", "add global attribute to ora.tstore.global operation");
        return;
    };

    const value = c.mlirOperationGetOperand(op, 0);
    const value_name = try ctx.getValueName(value);

    // Global transient storage uses tstore() in Yul (EIP-1153)
    try ctx.writeIndented("tstore(");
    try ctx.write(global_name);
    try ctx.write(", ");
    try ctx.write(value_name);
    try ctx.writeln(")");
}

fn processOraAssign(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    // ora.assign has:
    // - operand 0: destination (lvalue)
    // - operand 1: source (rvalue)

    const destination = c.mlirOperationGetOperand(op, 0);
    const source = c.mlirOperationGetOperand(op, 1);

    const dest_name = try ctx.getValueName(destination);
    const source_name = try ctx.getValueName(source);

    try ctx.writeIndented("");
    try ctx.write(dest_name);
    try ctx.write(" := ");
    try ctx.write(source_name);
    try ctx.writeln("");
}

fn processArithConstant(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    const value_attr = c.mlirOperationGetAttributeByName(op, c.MlirStringRef{
        .data = "value".ptr,
        .length = "value".len,
    });

    if (c.mlirAttributeIsNull(value_attr)) {
        try ctx.addError(.MalformedAst, "arith.constant missing value attribute", "add value attribute to arith.constant operation");
        return;
    }

    const result = c.mlirOperationGetResult(op, 0);

    // Extract the constant value
    // Fix boolean representation: MLIR uses -1 for true, but Ethereum/Yul uses 1
    const int_value = c.mlirIntegerAttrGetValueSInt(value_attr);
    const ethereum_val = if (int_value == -1) 1 else int_value;
    const value_str = try std.fmt.allocPrint(ctx.allocator, "{d}", .{ethereum_val});
    defer ctx.allocator.free(value_str);

    // Register this as a constant value for inlining
    // This avoids generating unnecessary temporary variables
    try ctx.setConstantValue(result, value_str);
}

fn processOraStringConstant(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    const value_attr = getStringAttribute(op, "value") orelse {
        try ctx.addError(.MalformedAst, "ora.string.constant missing value attribute", "add value attribute to ora.string.constant operation");
        return;
    };

    const result = c.mlirOperationGetResult(op, 0);
    const var_name = try ctx.getValueName(result);

    // For strings, we need to store them in memory and return the pointer
    // This is a simplified approach - in a full implementation we'd need proper string handling
    try ctx.writeIndented("let ");
    try ctx.write(var_name);
    try ctx.write(" := 0x0 // String: ");
    try ctx.write(value_attr);
    try ctx.writeln("");
}

fn processOraHexConstant(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    const value_attr = getStringAttribute(op, "value") orelse {
        try ctx.addError(.MalformedAst, "ora.hex.constant missing value attribute", "add value attribute to ora.hex.constant operation");
        return;
    };

    const result = c.mlirOperationGetResult(op, 0);
    const var_name = try ctx.getValueName(result);

    try ctx.writeIndented("let ");
    try ctx.write(var_name);
    try ctx.write(" := ");
    try ctx.write(value_attr);
    try ctx.writeln("");
}

fn processOraAddressConstant(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    // Address constants have:
    // - "value" as integer attribute (parsed address value)
    // - "ora.address" as string attribute (original address string like "0x...")

    // Try to get the string representation first (preferred for readability)
    const address_str_attr = getStringAttribute(op, "ora.address");

    const result = c.mlirOperationGetResult(op, 0);

    if (address_str_attr) |addr_str| {
        // Use the string representation
        const var_name = try ctx.getValueName(result);
        try ctx.writeIndented("let ");
        try ctx.write(var_name);
        try ctx.write(" := ");
        try ctx.write(addr_str);
        try ctx.writeln("");
    } else {
        // Fall back to integer value attribute
        const value_attr = c.mlirOperationGetAttributeByName(op, c.MlirStringRef{
            .data = "value".ptr,
            .length = "value".len,
        });

        if (c.mlirAttributeIsNull(value_attr)) {
            try ctx.addError(.MalformedAst, "ora.address.constant missing value or ora.address attribute", "add value attribute to ora.address.constant operation");
            return;
        }

        const int_value = c.mlirIntegerAttrGetValueSInt(value_attr);
        const value_str = try std.fmt.allocPrint(ctx.allocator, "0x{x:0>40}", .{int_value});
        defer ctx.allocator.free(value_str);

        // Register as constant for inlining
        try ctx.setConstantValue(result, value_str);
    }
}

fn processOraBinaryConstant(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    const value_attr = getStringAttribute(op, "value") orelse {
        try ctx.addError(.MalformedAst, "ora.binary.constant missing value attribute", "add value attribute to ora.binary.constant operation");
        return;
    };

    const result = c.mlirOperationGetResult(op, 0);
    const var_name = try ctx.getValueName(result);

    // Convert binary to hex for Yul
    const hex_value = try convertBinaryToHex(ctx.allocator, value_attr);
    defer ctx.allocator.free(hex_value);

    try ctx.writeIndented("let ");
    try ctx.write(var_name);
    try ctx.write(" := ");
    try ctx.write(hex_value);
    try ctx.writeln("");
}

fn processMemrefAlloca(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    // memref.alloca allocates a local variable
    // In Yul, local variables are just SSA values, so we just register the memref
    const result = c.mlirOperationGetResult(op, 0);
    const var_name = try ctx.getValueName(result);

    // Initialize to 0 (Yul variables must be initialized)
    try ctx.writeIndented("let ");
    try ctx.write(var_name);
    try ctx.writeln(" := 0");
}

fn processMemrefStore(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    // memref.store stores a value to a memref
    // operand 0: value to store
    // operand 1: memref destination
    const value = c.mlirOperationGetOperand(op, 0);
    const memref = c.mlirOperationGetOperand(op, 1);

    const value_name = try ctx.getValueName(value);
    const memref_name = try ctx.getValueName(memref);

    // In Yul, this is just an assignment
    try ctx.writeIndented("");
    try ctx.write(memref_name);
    try ctx.write(" := ");
    try ctx.write(value_name);
    try ctx.writeln("");
}

fn processMemrefLoad(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    // memref.load loads a value from a memref
    // operand 0: memref source
    const memref = c.mlirOperationGetOperand(op, 0);
    const result = c.mlirOperationGetResult(op, 0);

    const memref_name = try ctx.getValueName(memref);
    const result_name = try ctx.getValueName(result);

    // In Yul, just assign the memref value to the result
    try ctx.writeIndented("let ");
    try ctx.write(result_name);
    try ctx.write(" := ");
    try ctx.write(memref_name);
    try ctx.writeln("");
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

fn processArithSubi(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    const lhs = c.mlirOperationGetOperand(op, 0);
    const rhs = c.mlirOperationGetOperand(op, 1);
    const result = c.mlirOperationGetResult(op, 0);

    const lhs_name = try ctx.getValueName(lhs);
    const rhs_name = try ctx.getValueName(rhs);
    const result_name = try ctx.getValueName(result);

    try ctx.writeIndented("let ");
    try ctx.write(result_name);
    try ctx.write(" := sub(");
    try ctx.write(lhs_name);
    try ctx.write(", ");
    try ctx.write(rhs_name);
    try ctx.writeln(")");
}

fn processArithMuli(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    const lhs = c.mlirOperationGetOperand(op, 0);
    const rhs = c.mlirOperationGetOperand(op, 1);
    const result = c.mlirOperationGetResult(op, 0);

    const lhs_name = try ctx.getValueName(lhs);
    const rhs_name = try ctx.getValueName(rhs);
    const result_name = try ctx.getValueName(result);

    try ctx.writeIndented("let ");
    try ctx.write(result_name);
    try ctx.write(" := mul(");
    try ctx.write(lhs_name);
    try ctx.write(", ");
    try ctx.write(rhs_name);
    try ctx.writeln(")");
}

fn processArithDivi(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    const lhs = c.mlirOperationGetOperand(op, 0);
    const rhs = c.mlirOperationGetOperand(op, 1);
    const result = c.mlirOperationGetResult(op, 0);

    const lhs_name = try ctx.getValueName(lhs);
    const rhs_name = try ctx.getValueName(rhs);
    const result_name = try ctx.getValueName(result);

    try ctx.writeIndented("let ");
    try ctx.write(result_name);
    try ctx.write(" := div(");
    try ctx.write(lhs_name);
    try ctx.write(", ");
    try ctx.write(rhs_name);
    try ctx.writeln(")");
}

fn processArithRemi(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    const lhs = c.mlirOperationGetOperand(op, 0);
    const rhs = c.mlirOperationGetOperand(op, 1);
    const result = c.mlirOperationGetResult(op, 0);

    const lhs_name = try ctx.getValueName(lhs);
    const rhs_name = try ctx.getValueName(rhs);
    const result_name = try ctx.getValueName(result);

    try ctx.writeIndented("let ");
    try ctx.write(result_name);
    try ctx.write(" := mod(");
    try ctx.write(lhs_name);
    try ctx.write(", ");
    try ctx.write(rhs_name);
    try ctx.writeln(")");
}

fn processArithAndi(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    const lhs = c.mlirOperationGetOperand(op, 0);
    const rhs = c.mlirOperationGetOperand(op, 1);
    const result = c.mlirOperationGetResult(op, 0);

    const lhs_name = try ctx.getValueName(lhs);
    const rhs_name = try ctx.getValueName(rhs);
    const result_name = try ctx.getValueName(result);

    try ctx.writeIndented("let ");
    try ctx.write(result_name);
    try ctx.write(" := and(");
    try ctx.write(lhs_name);
    try ctx.write(", ");
    try ctx.write(rhs_name);
    try ctx.writeln(")");
}

fn processArithOri(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    const lhs = c.mlirOperationGetOperand(op, 0);
    const rhs = c.mlirOperationGetOperand(op, 1);
    const result = c.mlirOperationGetResult(op, 0);

    const lhs_name = try ctx.getValueName(lhs);
    const rhs_name = try ctx.getValueName(rhs);
    const result_name = try ctx.getValueName(result);

    try ctx.writeIndented("let ");
    try ctx.write(result_name);
    try ctx.write(" := or(");
    try ctx.write(lhs_name);
    try ctx.write(", ");
    try ctx.write(rhs_name);
    try ctx.writeln(")");
}

fn processArithXori(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    const lhs = c.mlirOperationGetOperand(op, 0);
    const rhs = c.mlirOperationGetOperand(op, 1);
    const result = c.mlirOperationGetResult(op, 0);

    const lhs_name = try ctx.getValueName(lhs);
    const rhs_name = try ctx.getValueName(rhs);
    const result_name = try ctx.getValueName(result);

    try ctx.writeIndented("let ");
    try ctx.write(result_name);
    try ctx.write(" := xor(");
    try ctx.write(lhs_name);
    try ctx.write(", ");
    try ctx.write(rhs_name);
    try ctx.writeln(")");
}

fn processArithShli(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    const lhs = c.mlirOperationGetOperand(op, 0);
    const rhs = c.mlirOperationGetOperand(op, 1);
    const result = c.mlirOperationGetResult(op, 0);

    const lhs_name = try ctx.getValueName(lhs);
    const rhs_name = try ctx.getValueName(rhs);
    const result_name = try ctx.getValueName(result);

    try ctx.writeIndented("let ");
    try ctx.write(result_name);
    try ctx.write(" := shl(");
    try ctx.write(lhs_name);
    try ctx.write(", ");
    try ctx.write(rhs_name);
    try ctx.writeln(")");
}

fn processArithShri(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    const lhs = c.mlirOperationGetOperand(op, 0);
    const rhs = c.mlirOperationGetOperand(op, 1);
    const result = c.mlirOperationGetResult(op, 0);

    const lhs_name = try ctx.getValueName(lhs);
    const rhs_name = try ctx.getValueName(rhs);
    const result_name = try ctx.getValueName(result);

    try ctx.writeIndented("let ");
    try ctx.write(result_name);
    try ctx.write(" := shr(");
    try ctx.write(lhs_name);
    try ctx.write(", ");
    try ctx.write(rhs_name);
    try ctx.writeln(")");
}

fn processArithCmpi(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    const predicate = getIntegerAttribute(op, "predicate") orelse {
        try ctx.addError(.MalformedAst, "arith.cmpi missing predicate attribute", "add predicate attribute to arith.cmpi operation");
        return;
    };

    const lhs = c.mlirOperationGetOperand(op, 0);
    const rhs = c.mlirOperationGetOperand(op, 1);
    const result = c.mlirOperationGetResult(op, 0);

    const lhs_name = try ctx.getValueName(lhs);
    const rhs_name = try ctx.getValueName(rhs);
    const result_name = try ctx.getValueName(result);

    // MLIR arith.CmpIPredicate values:
    // 0=eq, 1=ne, 2=slt, 3=sle, 4=sgt, 5=sge, 6=ult, 7=ule, 8=ugt, 9=uge
    // YUL only has: eq, lt, gt, slt, sgt
    // For others we use: ne=iszero(eq), sle=iszero(sgt), sge=iszero(slt), le=iszero(gt), ge=iszero(lt)

    try ctx.writeIndented("let ");
    try ctx.write(result_name);
    try ctx.write(" := ");

    switch (predicate) {
        0 => { // eq
            try ctx.write("eq(");
            try ctx.write(lhs_name);
            try ctx.write(", ");
            try ctx.write(rhs_name);
            try ctx.write(")");
        },
        1 => { // ne = iszero(eq)
            try ctx.write("iszero(eq(");
            try ctx.write(lhs_name);
            try ctx.write(", ");
            try ctx.write(rhs_name);
            try ctx.write("))");
        },
        2 => { // slt
            try ctx.write("slt(");
            try ctx.write(lhs_name);
            try ctx.write(", ");
            try ctx.write(rhs_name);
            try ctx.write(")");
        },
        3 => { // sle = iszero(sgt)
            try ctx.write("iszero(sgt(");
            try ctx.write(lhs_name);
            try ctx.write(", ");
            try ctx.write(rhs_name);
            try ctx.write("))");
        },
        4 => { // sgt
            try ctx.write("sgt(");
            try ctx.write(lhs_name);
            try ctx.write(", ");
            try ctx.write(rhs_name);
            try ctx.write(")");
        },
        5 => { // sge = iszero(slt)
            try ctx.write("iszero(slt(");
            try ctx.write(lhs_name);
            try ctx.write(", ");
            try ctx.write(rhs_name);
            try ctx.write("))");
        },
        6 => { // ult (unsigned lt)
            try ctx.write("lt(");
            try ctx.write(lhs_name);
            try ctx.write(", ");
            try ctx.write(rhs_name);
            try ctx.write(")");
        },
        7 => { // ule = iszero(gt)
            try ctx.write("iszero(gt(");
            try ctx.write(lhs_name);
            try ctx.write(", ");
            try ctx.write(rhs_name);
            try ctx.write("))");
        },
        8 => { // ugt (unsigned gt)
            try ctx.write("gt(");
            try ctx.write(lhs_name);
            try ctx.write(", ");
            try ctx.write(rhs_name);
            try ctx.write(")");
        },
        9 => { // uge = iszero(lt)
            try ctx.write("iszero(lt(");
            try ctx.write(lhs_name);
            try ctx.write(", ");
            try ctx.write(rhs_name);
            try ctx.write("))");
        },
        else => {
            try ctx.addError(.UnsupportedFeature, "Unsupported comparison predicate", "use eq, ne, slt, sle, sgt, sge, ult, ule, ugt, or uge");
            return;
        },
    }

    try ctx.writeln("");
}

fn processArithTrunci(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    // arith.trunci converts i256 to i1 (for boolean storage loads)
    // In Yul, we use iszero(iszero(value)) to normalize to 0 or 1
    const operand = c.mlirOperationGetOperand(op, 0);
    const result = c.mlirOperationGetResult(op, 0);

    const operand_name = try ctx.getValueName(operand);
    const result_name = try ctx.getValueName(result);

    try ctx.writeIndented("let ");
    try ctx.write(result_name);
    try ctx.write(" := iszero(iszero(");
    try ctx.write(operand_name);
    try ctx.writeln("))");
}

fn processArithExtui(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    // arith.extui extends i1 to i256 (for boolean storage stores)
    // In Yul, the value is already 0 or 1, so we just assign it
    const operand = c.mlirOperationGetOperand(op, 0);
    const result = c.mlirOperationGetResult(op, 0);

    const operand_name = try ctx.getValueName(operand);
    const result_name = try ctx.getValueName(result);

    try ctx.writeIndented("let ");
    try ctx.write(result_name);
    try ctx.write(" := ");
    try ctx.write(operand_name);
    try ctx.writeln("");
}

fn processScfIf(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    const condition = c.mlirOperationGetOperand(op, 0);
    const condition_name = try ctx.getValueName(condition);

    // Check if this scf.if has a result (for conditional expressions)
    const has_result = c.mlirOperationGetNumResults(op) > 0;
    var result_name: []const u8 = "";
    var then_yielded = false;
    var else_yielded = false;

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

    // Check if then region contains a func.return (early return)
    const then_region = c.mlirOperationGetRegion(op, 0);
    var then_block = c.mlirRegionGetFirstBlock(then_region);
    var then_has_return = false;
    while (!c.mlirBlockIsNull(then_block)) {
        var current_op = c.mlirBlockGetFirstOperation(then_block);
        while (!c.mlirOperationIsNull(current_op)) {
            const op_name = getOperationName(current_op);
            if (std.mem.eql(u8, op_name, "func.return")) {
                then_has_return = true;
                break;
            }
            current_op = c.mlirOperationGetNextInBlock(current_op);
        }
        if (then_has_return) break;
        then_block = c.mlirBlockGetNextInRegion(then_block);
    }

    // Process then region (region 0)
    then_block = c.mlirRegionGetFirstBlock(then_region);
    while (!c.mlirBlockIsNull(then_block)) {
        var current_op = c.mlirBlockGetFirstOperation(then_block);
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
                    then_yielded = true;

                    // If the yielded value comes from an error call, mark the result as well
                    const yield_value_ptr = @intFromPtr(yield_value.ptr);
                    if (ctx.error_call_results.get(yield_value_ptr)) |err_name| {
                        const result_ptr = @intFromPtr(c.mlirOperationGetResult(op, 0).ptr);
                        const error_name_copy = try ctx.allocator.dupe(u8, err_name);
                        try ctx.error_call_results.put(result_ptr, error_name_copy);
                    }
                }
            } else if (std.mem.eql(u8, op_name, "func.return")) {
                // Early return in then branch - process it directly
                try processFuncReturn(current_op, ctx);
                then_yielded = true;
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
                            else_yielded = true;

                            // If the yielded value comes from an error call, mark the result as well
                            const yield_value_ptr = @intFromPtr(yield_value.ptr);
                            if (ctx.error_call_results.get(yield_value_ptr)) |err_name| {
                                const result_ptr = @intFromPtr(c.mlirOperationGetResult(op, 0).ptr);
                                const error_name_copy = try ctx.allocator.dupe(u8, err_name);
                                try ctx.error_call_results.put(result_ptr, error_name_copy);
                            }
                        }
                    } else if (std.mem.eql(u8, op_name, "func.return")) {
                        // Early return in else branch - process it directly
                        try processFuncReturn(current_op, ctx);
                        else_yielded = true;
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

    // If result is expected but not yielded in either branch, assign default value
    if (has_result and (!then_yielded or !else_yielded)) {
        if (!then_yielded) {
            try ctx.writeIndented("");
            try ctx.write(result_name);
            try ctx.writeln(" := 0");
        }
        if (!else_yielded and num_regions > 1) {
            try ctx.writeIndented("");
            try ctx.write(result_name);
            try ctx.writeln(" := 0");
        }
    }

    // If the result came from an error call and we're in an error union function, check and return early
    if (has_result) {
        const result_ptr = @intFromPtr(c.mlirOperationGetResult(op, 0).ptr);
        if (ctx.error_call_results.get(result_ptr)) |err_name| {
            // This scf.if result is an error - we need to return early
            // Check if we're in an error union function
            const has_error_union = if (ctx.current_function) |func_name|
                ctx.functions_with_error_unions.get(func_name) orelse false
            else
                false;

            if (has_error_union) {
                // Get error code for this error
                const error_code = if (ctx.error_codes.get(err_name)) |code|
                    code
                else blk: {
                    const code = ctx.next_error_code;
                    ctx.next_error_code += 1;
                    const error_name_copy = try ctx.allocator.dupe(u8, err_name);
                    try ctx.error_codes.put(error_name_copy, code);
                    break :blk code;
                };

                // Return early with error if result is non-zero (error occurred)
                // In Yul, we can't return early, so we set the return values and the function will end
                // But we need to prevent subsequent code from overwriting these values
                try ctx.writeIndented("if ");
                try ctx.write(result_name);
                try ctx.writeln(" {");
                ctx.indent_level += 1;
                try ctx.writeIndented("value := 0\n");
                try ctx.writeIndented("error_code := ");
                const code_str = try std.fmt.allocPrint(ctx.allocator, "{d}", .{error_code});
                defer ctx.allocator.free(code_str);
                try ctx.writeln(code_str);
                // In Yul, we can't return early, but we can use leave to exit
                // However, leave only works in inline assembly. For now, we'll rely on the fact
                // that subsequent code should check error_code before overwriting
                ctx.indent_level -= 1;
                try ctx.writelnIndented("}");
            }
        }
    }
}

fn processScfWhile(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    // scf.while has:
    // - operand 0: condition (before loop)
    // - region 0: before region (condition check)
    // - region 1: after region (loop body)

    const condition = c.mlirOperationGetOperand(op, 0);
    const condition_name = try ctx.getValueName(condition);

    try ctx.writeIndented("for {} ");
    try ctx.write(condition_name);
    try ctx.writeln(" {");

    ctx.indent_level += 1;

    // Process the loop body (region 1 - after region)
    const body_region = c.mlirOperationGetRegion(op, 1);
    var body_block = c.mlirRegionGetFirstBlock(body_region);
    while (!c.mlirBlockIsNull(body_block)) {
        var current_op = c.mlirBlockGetFirstOperation(body_block);
        while (!c.mlirOperationIsNull(current_op)) {
            const op_name = getOperationName(current_op);
            if (!std.mem.eql(u8, op_name, "scf.yield")) {
                processOperation(current_op, ctx);
            }
            current_op = c.mlirOperationGetNextInBlock(current_op);
        }
        body_block = c.mlirBlockGetNextInRegion(body_block);
    }

    ctx.indent_level -= 1;
    try ctx.writelnIndented("}");
}

fn processScfFor(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    // scf.for has:
    // - operands: lower bound, upper bound, step, and loop induction variable
    // - region 0: loop body

    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands < 3) {
        ctx.addError(.MlirOperationFailed, "scf.for requires at least 3 operands", "check for loop operands") catch {};
        return;
    }

    const lower_bound = c.mlirOperationGetOperand(op, 0);
    const upper_bound = c.mlirOperationGetOperand(op, 1);
    const step = if (c.mlirOperationGetNumOperands(op) > 2) c.mlirOperationGetOperand(op, 2) else null;

    const lower_name = try ctx.getValueName(lower_bound);
    const upper_name = try ctx.getValueName(upper_bound);

    // Get the loop induction variable (result of the for operation)
    const induction_var = c.mlirOperationGetResult(op, 0);
    const induction_name = try ctx.getValueName(induction_var);

    // Generate Yul for loop: for { let i := lower } lt(i, upper) { i := add(i, step) } { body }
    try ctx.writeIndented("for { let ");
    try ctx.write(induction_name);
    try ctx.write(" := ");
    try ctx.write(lower_name);
    try ctx.write(" } lt(");
    try ctx.write(induction_name);
    try ctx.write(", ");
    try ctx.write(upper_name);
    try ctx.write(") { ");
    try ctx.write(induction_name);
    try ctx.write(" := add(");
    try ctx.write(induction_name);
    try ctx.write(", ");
    if (step) |step_val| {
        const step_name = try ctx.getValueName(step_val);
        try ctx.write(step_name);
    } else {
        try ctx.write("1");
    }
    try ctx.write(") } {");
    try ctx.writeln("");

    ctx.indent_level += 1;

    // Process the loop body (region 0)
    const body_region = c.mlirOperationGetRegion(op, 0);
    var body_block = c.mlirRegionGetFirstBlock(body_region);
    while (!c.mlirBlockIsNull(body_block)) {
        // Register block arguments (loop variables like recipient, i) as local variables
        // For indexed for loops, we need to ensure block arguments are accessible
        const num_args = c.mlirBlockGetNumArguments(body_block);
        if (num_args > 0) {
            // For indexed for loops, block arguments are the loop variables
            // They need to be defined in Yul before use
            for (0..@intCast(num_args)) |arg_idx| {
                const block_arg = c.mlirBlockGetArgument(body_block, @intCast(arg_idx));
                const arg_name = try ctx.getValueName(block_arg);
                // Define the variable in Yul (it will be assigned by the loop)
                // For now, initialize to 0 as a placeholder
                try ctx.writeIndented("let ");
                try ctx.write(arg_name);
                try ctx.writeln(" := 0 // Loop variable placeholder");
            }
        }

        var current_op = c.mlirBlockGetFirstOperation(body_block);
        while (!c.mlirOperationIsNull(current_op)) {
            const op_name = getOperationName(current_op);
            if (!std.mem.eql(u8, op_name, "scf.yield")) {
                processOperation(current_op, ctx);
            }
            current_op = c.mlirOperationGetNextInBlock(current_op);
        }
        body_block = c.mlirBlockGetNextInRegion(body_block);
    }

    ctx.indent_level -= 1;
    try ctx.writelnIndented("}");
}

fn processScfYield(_: c.MlirOperation, _: *YulLoweringContext) !void {
    // scf.yield is handled by the parent scf.if operation
    // No direct Yul generation needed here
}

fn processFuncCall(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    // func.call has:
    // - attribute "callee": function name (FlatSymbolRef)
    // - operands 0..n: arguments
    // - result 0: return value (if any)

    const num_operands = c.mlirOperationGetNumOperands(op);
    const num_results = c.mlirOperationGetNumResults(op);

    // Get the callee function name from the "callee" attribute
    const callee_attr = c.mlirOperationGetAttributeByName(op, c.MlirStringRef{
        .data = "callee".ptr,
        .length = "callee".len,
    });

    if (c.mlirAttributeIsNull(callee_attr)) {
        ctx.addError(.MalformedAst, "func.call missing callee attribute", "add callee attribute to func.call operation") catch {};
        return;
    }

    // Extract function name from FlatSymbolRef attribute
    const callee_ref = c.mlirFlatSymbolRefAttrGetValue(callee_attr);
    const callee_name = callee_ref.data[0..callee_ref.length];

    // Check if this is a call to an error name (error calls like Unauthorized(...))
    const error_names = [_][]const u8{ "Unauthorized", "InvalidAmount", "InsufficientBalance", "InvalidAddress" };
    var is_error_call = false;
    var error_name: []const u8 = "";
    for (error_names) |err_name| {
        if (std.mem.eql(u8, callee_name, err_name)) {
            is_error_call = true;
            error_name = err_name;
            break;
        }
    }

    // Check if this function has an error union return type
    const has_error_union = ctx.functions_with_error_unions.get(callee_name) orelse false;

    // Handle return value if present
    if (num_results > 0) {
        const result = c.mlirOperationGetResult(op, 0);
        const result_name = try ctx.getValueName(result);

        // If this is an error call, track the result value
        if (is_error_call) {
            const result_ptr = @intFromPtr(result.ptr);
            const error_name_copy = try ctx.allocator.dupe(u8, error_name);
            try ctx.error_call_results.put(result_ptr, error_name_copy);
        }

        if (has_error_union) {
            // Error union returns: (value, error_code)
            // Create temp variables for both values
            const value_name = try std.fmt.allocPrint(ctx.allocator, "{s}_value", .{result_name});
            defer ctx.allocator.free(value_name);
            const error_code_name = try std.fmt.allocPrint(ctx.allocator, "{s}_error_code", .{result_name});
            defer ctx.allocator.free(error_code_name);

            try ctx.writeIndented("let ");
            try ctx.write(value_name);
            try ctx.write(", ");
            try ctx.write(error_code_name);
            try ctx.write(" := ");

            // Store the value in the result mapping (for success case)
            const value_ptr = @intFromPtr(result.ptr);
            const value_name_copy = try ctx.allocator.dupe(u8, value_name);
            try ctx.value_map.put(value_ptr, value_name_copy);
        } else {
            // Normal return: result
            try ctx.writeIndented("let ");
            try ctx.write(result_name);
            try ctx.write(" := ");
        }
    }

    // Generate function call
    try ctx.write(callee_name);
    try ctx.write("(");

    // Add arguments (all operands are arguments, not the callee)
    // Ensure all argument values are processed before use
    var first_arg = true;
    for (0..@intCast(num_operands)) |i| {
        if (!first_arg) try ctx.write(", ");
        first_arg = false;

        const arg = c.mlirOperationGetOperand(op, @intCast(i));
        const arg_name = try ctx.getValueName(arg);

        // Check if this argument value is from an unprocessed operation
        // If it's a temp variable that was created by the fallback handler,
        // we need to trace back to find the actual source
        const value_ptr = @intFromPtr(arg.ptr);
        const is_placeholder = std.mem.startsWith(u8, arg_name, "temp_") and
            !ctx.constant_values.contains(value_ptr);

        if (is_placeholder) {
            // Try to find if this value corresponds to a function parameter
            // For now, if we can't find it, use arg0, arg1, etc. as fallback
            // This is a heuristic - ideally we'd trace back to the actual source
            const param_index = i;
            const param_name = try std.fmt.allocPrint(ctx.allocator, "arg{d}", .{param_index});
            defer ctx.allocator.free(param_name);
            try ctx.write(param_name);
        } else {
            try ctx.write(arg_name);
        }
    }

    try ctx.write(")");
    if (num_results == 0) {
        try ctx.writeln("");
    } else {
        try ctx.writeln("");
    }
}

fn processFuncReturn(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    const num_operands = c.mlirOperationGetNumOperands(op);

    // Check if current function has error union return type
    const has_error_union = if (ctx.current_function) |func_name|
        ctx.functions_with_error_unions.get(func_name) orelse false
    else
        false;

    if (has_error_union) {
        // Error union function: encode returns as (value, error_code)
        // Check if this is an error return (has "ora.error" attribute)
        const error_attr = c.mlirOperationGetAttributeByName(op, c.MlirStringRef{
            .data = "ora.error".ptr,
            .length = "ora.error".len,
        });

        const is_error_return = !c.mlirAttributeIsNull(error_attr);

        if (is_error_return) {
            // Error return: encode as (0, error_code)
            const error_name_attr = c.mlirStringAttrGetValue(error_attr);
            const error_name = error_name_attr.data[0..error_name_attr.length];

            // Get or assign error code
            const error_code = if (ctx.error_codes.get(error_name)) |code|
                code
            else blk: {
                const code = ctx.next_error_code;
                ctx.next_error_code += 1;
                const error_name_copy = try ctx.allocator.dupe(u8, error_name);
                try ctx.error_codes.put(error_name_copy, code);
                break :blk code;
            };

            try ctx.writeIndented("value := 0\n");
            try ctx.writeIndented("error_code := ");
            const code_str = try std.fmt.allocPrint(ctx.allocator, "{d}", .{error_code});
            defer ctx.allocator.free(code_str);
            try ctx.writeln(code_str);
        } else if (num_operands > 0) {
            // Check if return value comes from an error call
            const return_value = c.mlirOperationGetOperand(op, 0);
            const return_value_ptr = @intFromPtr(return_value.ptr);

            // Check if this value is from an error call
            if (ctx.error_call_results.get(return_value_ptr)) |err_name| {
                // This return value came from an error call - encode as error return
                // Get or assign error code for this error
                const error_code = if (ctx.error_codes.get(err_name)) |code|
                    code
                else blk: {
                    const code = ctx.next_error_code;
                    ctx.next_error_code += 1;
                    const error_name_copy = try ctx.allocator.dupe(u8, err_name);
                    try ctx.error_codes.put(error_name_copy, code);
                    break :blk code;
                };

                try ctx.writeIndented("value := 0\n");
                try ctx.writeIndented("error_code := ");
                const code_str = try std.fmt.allocPrint(ctx.allocator, "{d}", .{error_code});
                defer ctx.allocator.free(code_str);
                try ctx.writeln(code_str);
            } else {
                // Success return: encode as (value, 0)
                // But first check if error_code was already set (early return happened)
                try ctx.writeIndented("if iszero(error_code) {\n");
                ctx.indent_level += 1;
                const return_value_name = try ctx.getValueName(return_value);
                try ctx.writeIndented("value := ");
                try ctx.writeln(return_value_name);
                try ctx.writeIndented("error_code := 0\n");
                ctx.indent_level -= 1;
                try ctx.writelnIndented("}");
            }
        } else {
            // Void return (shouldn't happen in error union context, but handle gracefully)
            try ctx.writeIndented("value := 0\n");
            try ctx.writeIndented("error_code := 0\n");
        }
    } else {
        // Normal function: just set result
        if (num_operands > 0) {
            const return_value = c.mlirOperationGetOperand(op, 0);
            const return_value_name = try ctx.getValueName(return_value);
            try ctx.writeIndented("result := ");
            try ctx.writeln(return_value_name);
        }
    }
    // Note: In Yul functions, we don't need explicit return statements
    // The result variables are automatically returned
}

fn processOraMove(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    // ora.move has:
    // - operand 0: amount
    // - operand 1: source
    // - operand 2: destination

    const num_operands = c.mlirOperationGetNumOperands(op);
    if (num_operands < 3) {
        try ctx.addError(.MlirOperationFailed, "ora.move requires 3 operands", "check move operation operands");
        return;
    }

    const amount = c.mlirOperationGetOperand(op, 0);
    const source = c.mlirOperationGetOperand(op, 1);
    const destination = c.mlirOperationGetOperand(op, 2);

    const amount_name = try ctx.getValueName(amount);
    const source_name = try ctx.getValueName(source);
    const destination_name = try ctx.getValueName(destination);

    // Move operation: subtract from source, add to destination
    // This is a simplified implementation - in a full implementation we'd need
    // proper balance tracking and validation
    try ctx.writeIndented("// Move ");
    try ctx.write(amount_name);
    try ctx.write(" from ");
    try ctx.write(source_name);
    try ctx.write(" to ");
    try ctx.write(destination_name);
    try ctx.writeln("");

    // Subtract from source
    try ctx.writeIndented("sstore(");
    try ctx.write(source_name);
    try ctx.write(", sub(sload(");
    try ctx.write(source_name);
    try ctx.write("), ");
    try ctx.write(amount_name);
    try ctx.writeln("))");

    // Add to destination
    try ctx.writeIndented("sstore(");
    try ctx.write(destination_name);
    try ctx.write(", add(sload(");
    try ctx.write(destination_name);
    try ctx.write("), ");
    try ctx.write(amount_name);
    try ctx.writeln("))");
}

fn processOraStructInstantiate(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    // ora.struct_instantiate has:
    // - attribute: struct_name
    // - operands: field values
    // - result: struct instance (storage pointer)

    _ = getStringAttribute(op, "struct_name") orelse {
        try ctx.addError(.MalformedAst, "ora.struct_instantiate missing struct_name attribute", "add struct_name attribute");
        return;
    };

    const result = c.mlirOperationGetResult(op, 0);
    const result_name = try ctx.getValueName(result);

    // Get number of operands (field values)
    const num_operands = c.mlirOperationGetNumOperands(op);

    // Allocate struct in memory (use free memory pointer pattern)
    // Base pointer will be at free memory location
    try ctx.writeIndented("let ");
    try ctx.write(result_name);
    try ctx.writeln(" := mload(0x40) // Allocate struct in memory");

    // Initialize each field with corresponding operand value
    // Each field is stored at offset: base_ptr + (field_index * 32)
    var i: u32 = 0;
    while (i < num_operands) : (i += 1) {
        const operand = c.mlirOperationGetOperand(op, i);
        const value_name = try ctx.getValueName(operand);

        // Calculate memory offset for this field (32 bytes per slot)
        const offset = i * 32;
        const offset_str = try std.fmt.allocPrint(ctx.allocator, "{d}", .{offset});
        defer ctx.allocator.free(offset_str);

        try ctx.writeIndented("mstore(add(");
        try ctx.write(result_name);
        try ctx.write(", ");
        try ctx.write(offset_str);
        try ctx.write("), ");
        try ctx.write(value_name);
        try ctx.writeln(")");
    }

    // Update free memory pointer to point after this struct
    const total_size = num_operands * 32;
    const size_str = try std.fmt.allocPrint(ctx.allocator, "{d}", .{total_size});
    defer ctx.allocator.free(size_str);

    try ctx.writeIndented("mstore(0x40, add(");
    try ctx.write(result_name);
    try ctx.write(", ");
    try ctx.write(size_str);
    try ctx.write(")) // Update free memory pointer");
    try ctx.writeln("");
}

fn processOraStructFieldStore(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    // ora.struct_field_store has:
    // - operand 0: struct instance (storage/memory pointer)
    // - operand 1: field value
    // - attribute: field_name
    // - attribute: field_offset (required for correct storage layout)

    const field_name = getStringAttribute(op, "field_name") orelse {
        try ctx.addError(.MalformedAst, "ora.struct_field_store missing field_name attribute", "add struct_name attribute");
        return;
    };

    const struct_instance = c.mlirOperationGetOperand(op, 0);
    const field_value = c.mlirOperationGetOperand(op, 1);

    const struct_ptr = try ctx.getValueName(struct_instance);
    const value_name = try ctx.getValueName(field_value);

    // Get field offset - this is calculated during symbol table registration
    // and represents the byte offset for this field in the struct layout
    const field_offset = getIntegerAttribute(op, "field_offset") orelse 0;

    // Determine storage type from context
    // If struct_ptr contains "storage" or is a known storage variable, use sstore
    // Otherwise use mstore for memory-based structs
    const use_storage = std.mem.indexOf(u8, struct_ptr, "storage") != null or
        ctx.storage_vars.contains(struct_ptr);

    if (use_storage) {
        // Storage-based struct: use sstore
        // Field is at: base_storage_slot + (field_offset / 32)
        const slot_offset = @divFloor(field_offset, 32);
        try ctx.writeIndented("sstore(add(");
        try ctx.write(struct_ptr);
        try ctx.write(", ");
        const offset_str = try std.fmt.allocPrint(ctx.allocator, "{d}", .{slot_offset});
        defer ctx.allocator.free(offset_str);
        try ctx.write(offset_str);
        try ctx.write("), ");
        try ctx.write(value_name);
        try ctx.write(") // field: ");
        try ctx.write(field_name);
        try ctx.write(" at slot offset ");
        try ctx.writeln(offset_str);
    } else {
        // Memory-based struct: use mstore
        // Field is at: base_memory_ptr + field_offset
        try ctx.writeIndented("mstore(add(");
        try ctx.write(struct_ptr);
        try ctx.write(", ");
        const offset_str = try std.fmt.allocPrint(ctx.allocator, "{d}", .{field_offset});
        defer ctx.allocator.free(offset_str);
        try ctx.write(offset_str);
        try ctx.write("), ");
        try ctx.write(value_name);
        try ctx.write(") // field: ");
        try ctx.write(field_name);
        try ctx.write(" at byte offset ");
        try ctx.writeln(offset_str);
    }
}

fn processOraMapGet(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    // ora.map_get has:
    // - operand 0: map (storage slot)
    // - operand 1: key
    // - result: value

    const map = c.mlirOperationGetOperand(op, 0);
    const key = c.mlirOperationGetOperand(op, 1);
    const result = c.mlirOperationGetResult(op, 0);

    const map_name = try ctx.getValueName(map);
    const key_name = try ctx.getValueName(key);
    const result_name = try ctx.getValueName(result);

    // Use proper Solidity map storage layout: keccak256(key . slot)
    // This matches how Solidity stores mappings
    try ctx.writeIndented("let ");
    try ctx.write(result_name);
    try ctx.writeln(" := 0");

    // Store key at position 0
    try ctx.writeIndented("mstore(0, ");
    try ctx.write(key_name);
    try ctx.writeln(")");

    // Store slot at position 32 (0x20)
    try ctx.writeIndented("mstore(32, ");
    try ctx.write(map_name);
    try ctx.writeln(")");

    // Calculate storage slot: keccak256(key . map_slot)
    try ctx.writeIndented(result_name);
    try ctx.writeln(" := sload(keccak256(0, 64))");
}

fn processOraMapStore(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    // ora.map_store has:
    // - operand 0: map (storage slot)
    // - operand 1: key
    // - operand 2: value

    const map = c.mlirOperationGetOperand(op, 0);
    const key = c.mlirOperationGetOperand(op, 1);
    const value = c.mlirOperationGetOperand(op, 2);

    const map_name = try ctx.getValueName(map);
    const key_name = try ctx.getValueName(key);
    const value_name = try ctx.getValueName(value);

    // Use proper Solidity map storage layout: keccak256(key . slot)
    // Store key at position 0
    try ctx.writeIndented("mstore(0, ");
    try ctx.write(key_name);
    try ctx.writeln(")");

    // Store slot at position 32 (0x20)
    try ctx.writeIndented("mstore(32, ");
    try ctx.write(map_name);
    try ctx.writeln(")");

    // Calculate storage slot and store: sstore(keccak256(key . map_slot), value)
    try ctx.writeIndented("sstore(keccak256(0, 64), ");
    try ctx.write(value_name);
    try ctx.writeln(")");
}

/// Process EVM builtin operations (e.g., ora.evm.timestamp, ora.evm.caller, etc.)
/// Maps MLIR operations to Yul EVM opcodes
fn processOraEvmBuiltin(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    const op_name = getOperationName(op);

    // Extract the opcode name from "ora.evm.<opcode>"
    const opcode_name = if (std.mem.startsWith(u8, op_name, "ora.evm."))
        op_name["ora.evm.".len..]
    else {
        try ctx.addError(.MalformedAst, "Invalid ora.evm operation name", "operation should start with ora.evm.");
        return;
    };

    // Get the result value (all EVM builtins return a value)
    const result = c.mlirOperationGetResult(op, 0);
    const var_name = try ctx.getValueName(result);

    // Generate Yul: let <var_name> := <opcode>()
    try ctx.writeIndented("let ");
    try ctx.write(var_name);
    try ctx.write(" := ");
    try ctx.write(opcode_name);
    try ctx.writeln("()");
}

/// Process a block by processing all operations in it
fn processBlock(block: c.MlirBlock, ctx: *YulLoweringContext) void {
    var current_op = c.mlirBlockGetFirstOperation(block);
    while (!c.mlirOperationIsNull(current_op)) {
        processOperation(current_op, ctx);
        current_op = c.mlirOperationGetNextInBlock(current_op);
    }
}

/// Process ora.try operations (try-catch blocks)
/// Handles error union returns by checking error_code and routing to catch block if needed
fn processOraTry(op: c.MlirOperation, ctx: *YulLoweringContext) !void {
    // ora.try has:
    // - region 0: try block
    // - region 1: catch block (optional)
    // - result 0: result value (if any)

    const num_results = c.mlirOperationGetNumResults(op);
    const num_regions = c.mlirOperationGetNumRegions(op);

    // Check if catch block has an error variable (block argument)
    const has_catch = num_regions > 1;
    var catch_has_error_var = false;
    var error_var_name: []const u8 = "";

    if (has_catch) {
        const catch_region = c.mlirOperationGetRegion(op, 1);
        const catch_block = c.mlirRegionGetFirstBlock(catch_region);
        if (!c.mlirBlockIsNull(catch_block)) {
            const num_catch_args = c.mlirBlockGetNumArguments(catch_block);
            if (num_catch_args > 0) {
                catch_has_error_var = true;
                const error_arg = c.mlirBlockGetArgument(catch_block, 0);
                error_var_name = try ctx.getValueName(error_arg);
            }
        }
    }

    // Process try region (region 0)
    // The try block should contain operations that may return error unions
    // We need to track if any function calls return error unions and check their error_code
    if (num_regions > 0) {
        const try_region = c.mlirOperationGetRegion(op, 0);
        var current_block = c.mlirRegionGetFirstBlock(try_region);
        while (!c.mlirBlockIsNull(current_block)) {
            processBlock(current_block, ctx);
            current_block = c.mlirBlockGetNextInRegion(current_block);
        }
    }

    // After processing try block, check for errors
    // If any function calls returned error unions, we need to check their error_code
    // For now, we'll add a check after the try block
    // TODO: Track which operations in try block return error unions and check their error_code

    // Process catch region (region 1) if present
    if (has_catch) {
        const catch_region = c.mlirOperationGetRegion(op, 1);
        const catch_block = c.mlirRegionGetFirstBlock(catch_region);

        if (!c.mlirBlockIsNull(catch_block)) {
            // Check if we should enter catch block (error_code != 0)
            // For now, we'll process catch block unconditionally
            // TODO: Add proper error_code checking to conditionally enter catch block
            if (catch_has_error_var) {
                // The error variable should contain the error_code
                // For now, we'll set it to a placeholder
                try ctx.writeIndented("// Catch block - error variable: ");
                try ctx.writeln(error_var_name);
            }

            var current_block = catch_block;
            while (!c.mlirBlockIsNull(current_block)) {
                processBlock(current_block, ctx);
                current_block = c.mlirBlockGetNextInRegion(current_block);
            }
        }
    }

    // If there's a result, we need to handle it
    // For error unions, the result should be the value (if error_code == 0)
    if (num_results > 0) {
        const result = c.mlirOperationGetResult(op, 0);
        // Try to get a value name - if it doesn't exist, create one
        _ = try ctx.getValueName(result);
    }
}

fn convertBinaryToHex(allocator: std.mem.Allocator, binary_str: []const u8) ![]const u8 {
    // Remove 0b prefix if present
    const binary = if (std.mem.startsWith(u8, binary_str, "0b"))
        binary_str[2..]
    else
        binary_str;

    // Allocate buffer for hex string (each 4 bits = 1 hex char, plus "0x" prefix)
    var hex_buf: [256]u8 = undefined;
    var hex_len: usize = 0;

    // Add "0x" prefix
    hex_buf[hex_len] = '0';
    hex_len += 1;
    hex_buf[hex_len] = 'x';
    hex_len += 1;

    // Process binary string in chunks of 4 bits
    var i: usize = 0;
    while (i < binary.len) {
        var chunk: u4 = 0;
        var bits_in_chunk: u3 = 0;

        // Read up to 4 bits
        while (i < binary.len and bits_in_chunk < 4) {
            const bit = binary[i];
            if (bit == '1') {
                const shift_amount = 3 - bits_in_chunk;
                chunk |= (@as(u4, 1) << @intCast(shift_amount));
            } else if (bit != '0') {
                // Skip non-binary characters
                i += 1;
                continue;
            }
            bits_in_chunk += 1;
            i += 1;
        }

        // Convert chunk to hex
        const hex_char = if (chunk < 10)
            '0' + @as(u8, chunk)
        else
            'a' + @as(u8, chunk - 10);
        hex_buf[hex_len] = hex_char;
        hex_len += 1;
    }

    return allocator.dupe(u8, hex_buf[0..hex_len]);
}
