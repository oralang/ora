const std = @import("std");
pub const ast = @import("../ast.zig");
pub const typer = @import("../typer.zig");
const semantics_errors = @import("semantics_errors.zig");
const semantics_memory_safety = @import("semantics_memory_safety.zig");

// Forward declaration for SemanticAnalyzer
const SemanticAnalyzer = @import("semantics_core.zig").SemanticAnalyzer;

/// Builtin function signature
pub const BuiltinFunctionSignature = struct {
    name: []const u8,
    parameters: []const ParameterInfo,
    return_type: ?typer.OraType,
    category: FunctionCategory,
    description: []const u8,
    validation_rules: []const ValidationRule,

    pub const ParameterInfo = struct {
        name: []const u8,
        param_type: typer.OraType,
        is_optional: bool,
        default_value: ?[]const u8,
    };

    pub const FunctionCategory = enum {
        Math,
        String,
        Array,
        Mapping,
        Crypto,
        Blockchain,
        Conversion,
        Validation,
        Memory,
        System,
    };

    pub const ValidationRule = struct {
        rule_type: RuleType,
        description: []const u8,

        pub const RuleType = enum {
            NonZeroParameter,
            PositiveParameter,
            ValidRange,
            NonEmptyString,
            ValidAddress,
            ValidHash,
            MemoryRegionCompatible,
        };
    };
};

/// Builtin function call validation result
pub const BuiltinCallValidationResult = struct {
    valid: bool,
    function_name: []const u8,
    error_message: ?[]const u8,
    warning_message: ?[]const u8,
    return_type: ?typer.OraType,
    parameter_errors: []ParameterError,

    pub const ParameterError = struct {
        parameter_index: u32,
        parameter_name: []const u8,
        error_message: []const u8,
    };
};

/// Builtin function registry
pub const BuiltinFunctionRegistry = struct {
    allocator: std.mem.Allocator,
    functions: std.HashMap([]const u8, BuiltinFunctionSignature, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),

    pub fn init(allocator: std.mem.Allocator) BuiltinFunctionRegistry {
        var registry = BuiltinFunctionRegistry{
            .allocator = allocator,
            .functions = std.HashMap([]const u8, BuiltinFunctionSignature, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
        };

        // Initialize with builtin functions
        registry.initializeBuiltinFunctions() catch {};
        return registry;
    }

    pub fn deinit(self: *BuiltinFunctionRegistry) void {
        self.functions.deinit();
    }

    /// Initialize builtin function definitions
    fn initializeBuiltinFunctions(self: *BuiltinFunctionRegistry) !void {
        // Math functions
        try self.registerMathFunctions();

        // String functions
        try self.registerStringFunctions();

        // Array functions
        try self.registerArrayFunctions();

        // Mapping functions
        try self.registerMappingFunctions();

        // Crypto functions
        try self.registerCryptoFunctions();

        // Blockchain functions
        try self.registerBlockchainFunctions();

        // Conversion functions
        try self.registerConversionFunctions();

        // Validation functions
        try self.registerValidationFunctions();

        // Memory functions
        try self.registerMemoryFunctions();

        // System functions
        try self.registerSystemFunctions();
    }

    fn registerMathFunctions(self: *BuiltinFunctionRegistry) !void {
        // abs function
        const abs_params = [_]BuiltinFunctionSignature.ParameterInfo{
            .{ .name = "value", .param_type = typer.OraType.I256, .is_optional = false, .default_value = null },
        };
        const abs_signature = BuiltinFunctionSignature{
            .name = "abs",
            .parameters = &abs_params,
            .return_type = typer.OraType.U256,
            .category = .Math,
            .description = "Returns the absolute value of a signed integer",
            .validation_rules = &[_]BuiltinFunctionSignature.ValidationRule{},
        };
        try self.functions.put("abs", abs_signature);

        // max function
        const max_params = [_]BuiltinFunctionSignature.ParameterInfo{
            .{ .name = "a", .param_type = typer.OraType.U256, .is_optional = false, .default_value = null },
            .{ .name = "b", .param_type = typer.OraType.U256, .is_optional = false, .default_value = null },
        };
        const max_signature = BuiltinFunctionSignature{
            .name = "max",
            .parameters = &max_params,
            .return_type = typer.OraType.U256,
            .category = .Math,
            .description = "Returns the maximum of two values",
            .validation_rules = &[_]BuiltinFunctionSignature.ValidationRule{},
        };
        try self.functions.put("max", max_signature);

        // min function
        const min_params = [_]BuiltinFunctionSignature.ParameterInfo{
            .{ .name = "a", .param_type = typer.OraType.U256, .is_optional = false, .default_value = null },
            .{ .name = "b", .param_type = typer.OraType.U256, .is_optional = false, .default_value = null },
        };
        const min_signature = BuiltinFunctionSignature{
            .name = "min",
            .parameters = &min_params,
            .return_type = typer.OraType.U256,
            .category = .Math,
            .description = "Returns the minimum of two values",
            .validation_rules = &[_]BuiltinFunctionSignature.ValidationRule{},
        };
        try self.functions.put("min", min_signature);
    }

    fn registerStringFunctions(self: *BuiltinFunctionRegistry) !void {
        // length function
        const length_params = [_]BuiltinFunctionSignature.ParameterInfo{
            .{ .name = "str", .param_type = typer.OraType.String, .is_optional = false, .default_value = null },
        };
        const length_rules = [_]BuiltinFunctionSignature.ValidationRule{
            .{ .rule_type = .NonEmptyString, .description = "String parameter should not be empty" },
        };
        const length_signature = BuiltinFunctionSignature{
            .name = "length",
            .parameters = &length_params,
            .return_type = typer.OraType.U256,
            .category = .String,
            .description = "Returns the length of a string",
            .validation_rules = &length_rules,
        };
        try self.functions.put("length", length_signature);

        // concat function
        const concat_params = [_]BuiltinFunctionSignature.ParameterInfo{
            .{ .name = "str1", .param_type = typer.OraType.String, .is_optional = false, .default_value = null },
            .{ .name = "str2", .param_type = typer.OraType.String, .is_optional = false, .default_value = null },
        };
        const concat_signature = BuiltinFunctionSignature{
            .name = "concat",
            .parameters = &concat_params,
            .return_type = typer.OraType.String,
            .category = .String,
            .description = "Concatenates two strings",
            .validation_rules = &[_]BuiltinFunctionSignature.ValidationRule{},
        };
        try self.functions.put("concat", concat_signature);
    }

    fn registerArrayFunctions(self: *BuiltinFunctionRegistry) !void {
        // push function (generic for arrays)
        const push_params = [_]BuiltinFunctionSignature.ParameterInfo{
            .{ .name = "array", .param_type = typer.OraType.Generic, .is_optional = false, .default_value = null },
            .{ .name = "element", .param_type = typer.OraType.Generic, .is_optional = false, .default_value = null },
        };
        const push_signature = BuiltinFunctionSignature{
            .name = "push",
            .parameters = &push_params,
            .return_type = typer.OraType.Void,
            .category = .Array,
            .description = "Adds an element to the end of an array",
            .validation_rules = &[_]BuiltinFunctionSignature.ValidationRule{},
        };
        try self.functions.put("push", push_signature);

        // pop function
        const pop_params = [_]BuiltinFunctionSignature.ParameterInfo{
            .{ .name = "array", .param_type = typer.OraType.Generic, .is_optional = false, .default_value = null },
        };
        const pop_signature = BuiltinFunctionSignature{
            .name = "pop",
            .parameters = &pop_params,
            .return_type = typer.OraType.Generic,
            .category = .Array,
            .description = "Removes and returns the last element of an array",
            .validation_rules = &[_]BuiltinFunctionSignature.ValidationRule{},
        };
        try self.functions.put("pop", pop_signature);
    }

    fn registerMappingFunctions(self: *BuiltinFunctionRegistry) !void {
        // contains function
        const contains_params = [_]BuiltinFunctionSignature.ParameterInfo{
            .{ .name = "mapping", .param_type = typer.OraType.Generic, .is_optional = false, .default_value = null },
            .{ .name = "key", .param_type = typer.OraType.Generic, .is_optional = false, .default_value = null },
        };
        const contains_signature = BuiltinFunctionSignature{
            .name = "contains",
            .parameters = &contains_params,
            .return_type = typer.OraType.Bool,
            .category = .Mapping,
            .description = "Checks if a mapping contains a key",
            .validation_rules = &[_]BuiltinFunctionSignature.ValidationRule{},
        };
        try self.functions.put("contains", contains_signature);
    }

    fn registerCryptoFunctions(self: *BuiltinFunctionRegistry) !void {
        // keccak256 function
        const keccak256_params = [_]BuiltinFunctionSignature.ParameterInfo{
            .{ .name = "data", .param_type = typer.OraType.String, .is_optional = false, .default_value = null },
        };
        const keccak256_signature = BuiltinFunctionSignature{
            .name = "keccak256",
            .parameters = &keccak256_params,
            .return_type = typer.OraType.U256,
            .category = .Crypto,
            .description = "Computes the Keccak-256 hash of input data",
            .validation_rules = &[_]BuiltinFunctionSignature.ValidationRule{},
        };
        try self.functions.put("keccak256", keccak256_signature);

        // sha256 function
        const sha256_params = [_]BuiltinFunctionSignature.ParameterInfo{
            .{ .name = "data", .param_type = typer.OraType.String, .is_optional = false, .default_value = null },
        };
        const sha256_signature = BuiltinFunctionSignature{
            .name = "sha256",
            .parameters = &sha256_params,
            .return_type = typer.OraType.U256,
            .category = .Crypto,
            .description = "Computes the SHA-256 hash of input data",
            .validation_rules = &[_]BuiltinFunctionSignature.ValidationRule{},
        };
        try self.functions.put("sha256", sha256_signature);
    }

    fn registerBlockchainFunctions(self: *BuiltinFunctionRegistry) !void {
        // block_number function
        const block_number_signature = BuiltinFunctionSignature{
            .name = "block_number",
            .parameters = &[_]BuiltinFunctionSignature.ParameterInfo{},
            .return_type = typer.OraType.U256,
            .category = .Blockchain,
            .description = "Returns the current block number",
            .validation_rules = &[_]BuiltinFunctionSignature.ValidationRule{},
        };
        try self.functions.put("block_number", block_number_signature);

        // msg_sender function
        const msg_sender_signature = BuiltinFunctionSignature{
            .name = "msg_sender",
            .parameters = &[_]BuiltinFunctionSignature.ParameterInfo{},
            .return_type = typer.OraType.Address,
            .category = .Blockchain,
            .description = "Returns the address of the message sender",
            .validation_rules = &[_]BuiltinFunctionSignature.ValidationRule{},
        };
        try self.functions.put("msg_sender", msg_sender_signature);

        // msg_value function
        const msg_value_signature = BuiltinFunctionSignature{
            .name = "msg_value",
            .parameters = &[_]BuiltinFunctionSignature.ParameterInfo{},
            .return_type = typer.OraType.U256,
            .category = .Blockchain,
            .description = "Returns the value sent with the message",
            .validation_rules = &[_]BuiltinFunctionSignature.ValidationRule{},
        };
        try self.functions.put("msg_value", msg_value_signature);
    }

    fn registerConversionFunctions(self: *BuiltinFunctionRegistry) !void {
        // to_string function
        const to_string_params = [_]BuiltinFunctionSignature.ParameterInfo{
            .{ .name = "value", .param_type = typer.OraType.Generic, .is_optional = false, .default_value = null },
        };
        const to_string_signature = BuiltinFunctionSignature{
            .name = "to_string",
            .parameters = &to_string_params,
            .return_type = typer.OraType.String,
            .category = .Conversion,
            .description = "Converts a value to string representation",
            .validation_rules = &[_]BuiltinFunctionSignature.ValidationRule{},
        };
        try self.functions.put("to_string", to_string_signature);

        // to_address function
        const to_address_params = [_]BuiltinFunctionSignature.ParameterInfo{
            .{ .name = "value", .param_type = typer.OraType.String, .is_optional = false, .default_value = null },
        };
        const to_address_rules = [_]BuiltinFunctionSignature.ValidationRule{
            .{ .rule_type = .ValidAddress, .description = "String must be a valid address format" },
        };
        const to_address_signature = BuiltinFunctionSignature{
            .name = "to_address",
            .parameters = &to_address_params,
            .return_type = typer.OraType.Address,
            .category = .Conversion,
            .description = "Converts a string to an address",
            .validation_rules = &to_address_rules,
        };
        try self.functions.put("to_address", to_address_signature);
    }

    fn registerValidationFunctions(self: *BuiltinFunctionRegistry) !void {
        // is_valid_address function
        const is_valid_address_params = [_]BuiltinFunctionSignature.ParameterInfo{
            .{ .name = "addr", .param_type = typer.OraType.String, .is_optional = false, .default_value = null },
        };
        const is_valid_address_signature = BuiltinFunctionSignature{
            .name = "is_valid_address",
            .parameters = &is_valid_address_params,
            .return_type = typer.OraType.Bool,
            .category = .Validation,
            .description = "Checks if a string is a valid address format",
            .validation_rules = &[_]BuiltinFunctionSignature.ValidationRule{},
        };
        try self.functions.put("is_valid_address", is_valid_address_signature);
    }

    fn registerMemoryFunctions(self: *BuiltinFunctionRegistry) !void {
        // alloc function
        const alloc_params = [_]BuiltinFunctionSignature.ParameterInfo{
            .{ .name = "size", .param_type = typer.OraType.U256, .is_optional = false, .default_value = null },
        };
        const alloc_rules = [_]BuiltinFunctionSignature.ValidationRule{
            .{ .rule_type = .PositiveParameter, .description = "Size must be positive" },
        };
        const alloc_signature = BuiltinFunctionSignature{
            .name = "alloc",
            .parameters = &alloc_params,
            .return_type = typer.OraType.Generic, // Pointer type
            .category = .Memory,
            .description = "Allocates memory of specified size",
            .validation_rules = &alloc_rules,
        };
        try self.functions.put("alloc", alloc_signature);

        // free function
        const free_params = [_]BuiltinFunctionSignature.ParameterInfo{
            .{ .name = "ptr", .param_type = typer.OraType.Generic, .is_optional = false, .default_value = null },
        };
        const free_signature = BuiltinFunctionSignature{
            .name = "free",
            .parameters = &free_params,
            .return_type = typer.OraType.Void,
            .category = .Memory,
            .description = "Frees allocated memory",
            .validation_rules = &[_]BuiltinFunctionSignature.ValidationRule{},
        };
        try self.functions.put("free", free_signature);
    }

    fn registerSystemFunctions(self: *BuiltinFunctionRegistry) !void {
        // require function
        const require_params = [_]BuiltinFunctionSignature.ParameterInfo{
            .{ .name = "condition", .param_type = typer.OraType.Bool, .is_optional = false, .default_value = null },
            .{ .name = "message", .param_type = typer.OraType.String, .is_optional = true, .default_value = "Requirement failed" },
        };
        const require_signature = BuiltinFunctionSignature{
            .name = "require",
            .parameters = &require_params,
            .return_type = typer.OraType.Void,
            .category = .System,
            .description = "Asserts a condition and reverts with message if false",
            .validation_rules = &[_]BuiltinFunctionSignature.ValidationRule{},
        };
        try self.functions.put("require", require_signature);

        // revert function
        const revert_params = [_]BuiltinFunctionSignature.ParameterInfo{
            .{ .name = "message", .param_type = typer.OraType.String, .is_optional = true, .default_value = "Transaction reverted" },
        };
        const revert_signature = BuiltinFunctionSignature{
            .name = "revert",
            .parameters = &revert_params,
            .return_type = typer.OraType.Void,
            .category = .System,
            .description = "Reverts the transaction with an error message",
            .validation_rules = &[_]BuiltinFunctionSignature.ValidationRule{},
        };
        try self.functions.put("revert", revert_signature);
    }

    /// Check if a function is a builtin function
    pub fn isBuiltinFunction(self: *BuiltinFunctionRegistry, function_name: []const u8) bool {
        return self.functions.contains(function_name);
    }

    /// Get builtin function signature
    pub fn getBuiltinFunction(self: *BuiltinFunctionRegistry, function_name: []const u8) ?BuiltinFunctionSignature {
        return self.functions.get(function_name);
    }

    /// Validate builtin function call
    pub fn validateBuiltinCall(self: *BuiltinFunctionRegistry, analyzer: *SemanticAnalyzer, call: *ast.CallNode) !BuiltinCallValidationResult {
        // Extract function name
        const function_name = switch (call.function.*) {
            .Identifier => |*ident| ident.name,
            else => {
                return BuiltinCallValidationResult{
                    .valid = false,
                    .function_name = "unknown",
                    .error_message = "Invalid builtin function call",
                    .warning_message = null,
                    .return_type = null,
                    .parameter_errors = &[_]BuiltinCallValidationResult.ParameterError{},
                };
            },
        };

        // Validate function name
        if (!semantics_memory_safety.isValidString(analyzer, function_name)) {
            try semantics_errors.addErrorStatic(analyzer, "Invalid builtin function name", call.span);
            return BuiltinCallValidationResult{
                .valid = false,
                .function_name = function_name,
                .error_message = "Invalid function name",
                .warning_message = null,
                .return_type = null,
                .parameter_errors = &[_]BuiltinCallValidationResult.ParameterError{},
            };
        }

        // Get function signature
        const signature = self.getBuiltinFunction(function_name) orelse {
            const error_msg = try std.fmt.allocPrint(analyzer.allocator, "Unknown builtin function: {s}", .{function_name});
            try semantics_errors.addError(analyzer, error_msg, call.span);
            return BuiltinCallValidationResult{
                .valid = false,
                .function_name = function_name,
                .error_message = "Unknown builtin function",
                .warning_message = null,
                .return_type = null,
                .parameter_errors = &[_]BuiltinCallValidationResult.ParameterError{},
            };
        };

        // Validate parameter count
        const required_params = self.countRequiredParameters(signature);
        if (call.args.len < required_params or call.args.len > signature.parameters.len) {
            const error_msg = try std.fmt.allocPrint(analyzer.allocator, "Function '{s}' expects {d}-{d} arguments, got {d}", .{ function_name, required_params, signature.parameters.len, call.args.len });
            try semantics_errors.addError(analyzer, error_msg, call.span);
            return BuiltinCallValidationResult{
                .valid = false,
                .function_name = function_name,
                .error_message = "Invalid argument count",
                .warning_message = null,
                .return_type = signature.return_type,
                .parameter_errors = &[_]BuiltinCallValidationResult.ParameterError{},
            };
        }

        // Validate individual parameters
        var parameter_errors = std.ArrayList(BuiltinCallValidationResult.ParameterError).init(analyzer.allocator);
        defer parameter_errors.deinit();

        for (call.args, 0..) |*arg, i| {
            if (i < signature.parameters.len) {
                const param_info = signature.parameters[i];
                const param_valid = try self.validateParameter(analyzer, arg, param_info, @as(u32, @intCast(i)));
                if (!param_valid.valid) {
                    try parameter_errors.append(param_valid);
                }
            }
        }

        // Validate function-specific rules
        try self.validateFunctionRules(analyzer, signature, call);

        const has_errors = parameter_errors.items.len > 0;

        return BuiltinCallValidationResult{
            .valid = !has_errors,
            .function_name = function_name,
            .error_message = if (has_errors) "Parameter validation failed" else null,
            .warning_message = null,
            .return_type = signature.return_type,
            .parameter_errors = try parameter_errors.toOwnedSlice(),
        };
    }

    /// Private helper methods
    fn countRequiredParameters(self: *BuiltinFunctionRegistry, signature: BuiltinFunctionSignature) usize {
        _ = self;
        var count: usize = 0;
        for (signature.parameters) |param| {
            if (!param.is_optional) {
                count += 1;
            }
        }
        return count;
    }

    fn validateParameter(self: *BuiltinFunctionRegistry, analyzer: *SemanticAnalyzer, arg: *ast.ExprNode, param_info: BuiltinFunctionSignature.ParameterInfo, index: u32) !BuiltinCallValidationResult.ParameterError {
        _ = self;
        _ = analyzer;
        _ = arg;

        // TODO: Implement parameter type validation
        // This would involve checking if the argument type matches the expected parameter type

        return BuiltinCallValidationResult.ParameterError{
            .parameter_index = index,
            .parameter_name = param_info.name,
            .error_message = "",
        };
    }

    fn validateFunctionRules(self: *BuiltinFunctionRegistry, analyzer: *SemanticAnalyzer, signature: BuiltinFunctionSignature, call: *ast.CallNode) !void {
        _ = self;
        _ = analyzer;
        _ = signature;
        _ = call;

        // TODO: Implement function-specific validation rules
        // This would check the validation_rules in the signature
    }
};

/// Validate builtin function call
pub fn validateBuiltinFunctionCall(analyzer: *SemanticAnalyzer, call: *ast.CallNode) semantics_errors.SemanticError!bool {
    // Create a temporary registry for this validation
    var registry = BuiltinFunctionRegistry.init(analyzer.allocator);
    defer registry.deinit();

    // Extract function name
    const function_name = switch (call.function.*) {
        .Identifier => |*ident| ident.name,
        else => return true, // Not a simple function call, let other validators handle it
    };

    // Check if it's a builtin function
    if (!registry.isBuiltinFunction(function_name)) {
        return true; // Not a builtin function, validation passes
    }

    // Validate the builtin function call
    const result = registry.validateBuiltinCall(analyzer, call) catch |err| {
        switch (err) {
            error.OutOfMemory => return semantics_errors.SemanticError.OutOfMemory,
            else => {
                try semantics_errors.addErrorStatic(analyzer, "Builtin function validation failed", call.span);
                return false;
            },
        }
    };

    if (!result.valid) {
        if (result.error_message) |msg| {
            try semantics_errors.addErrorStatic(analyzer, msg, call.span);
        }

        // Report parameter errors
        for (result.parameter_errors) |param_error| {
            const error_msg = try std.fmt.allocPrint(analyzer.allocator, "Parameter '{s}': {s}", .{ param_error.parameter_name, param_error.error_message });
            try semantics_errors.addError(analyzer, error_msg, call.span);
        }

        return false;
    }

    return true;
}

/// Check if function is builtin
pub fn isBuiltinFunction(function_name: []const u8) bool {
    // Create a temporary registry to check
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    var registry = BuiltinFunctionRegistry.init(arena.allocator());
    defer registry.deinit();

    return registry.isBuiltinFunction(function_name);
}
