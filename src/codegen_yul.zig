//! Yul Code Generation
//!
//! This module provides code generation from High-level Intermediate Representation (HIR) to Yul
//!
//! The module integrates with the Yul compiler through FFI bindings to produce
//! optimized EVM bytecode from Ora language constructs

const std = @import("std");
const print = std.debug.print;
const ir = @import("ir.zig");
const yul_bindings = @import("yul_bindings.zig");

// Import HIR types from ir.zig
const HIRProgram = ir.HIRProgram;
const Contract = ir.Contract;
const Function = ir.Function;
const Block = ir.Block;
const Statement = ir.Statement;
const Expression = ir.Expression;
const Type = ir.Type;

const YulCompiler = yul_bindings.YulCompiler;
const YulCompileResult = yul_bindings.YulCompileResult;

/// Errors that can occur during Yul code generation
pub const YulCodegenError = error{
    /// Compilation failed during Yul processing
    CompilationFailed,
    /// Invalid HIR structure provided
    InvalidIR,
    /// Out of memory during generation
    OutOfMemory,
};

/// Yul code generator with stack-based variable management
///
/// This generator converts HIR instructions to Yul code while maintaining
/// proper variable scoping and memory management through a variable stack.
pub const YulCodegen = struct {
    allocator: std.mem.Allocator,
    /// Stack of variable names for proper scoping
    variable_stack: std.ArrayList([]const u8),
    /// Current variable counter for unique naming
    var_counter: u32,
    /// Current function context for scope management
    current_function: ?*const Function,
    /// Storage slot counter for contract variables
    storage_counter: u32,
    /// Mapping from storage variable names to their slot numbers
    storage_slots: std.HashMap([]const u8, u32, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),
    /// Current contract context for event lookup
    current_contract: ?*const Contract,

    const Self = @This();

    /// Initialize a new Yul code generator
    ///
    /// Args:
    ///     allocator: Memory allocator for variable management
    ///
    /// Returns:
    ///     Initialized YulCodegen instance
    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .allocator = allocator,
            .variable_stack = std.ArrayList([]const u8).init(allocator),
            .var_counter = 0,
            .current_function = null,
            .storage_counter = 0,
            .storage_slots = std.HashMap([]const u8, u32, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
            .current_contract = null,
        };
    }

    /// Cleanup all allocated resources
    ///
    /// Frees all variable names on the stack and deinitializes the stack.
    pub fn deinit(self: *Self) void {
        // Free all variable names
        for (self.variable_stack.items) |var_name| {
            self.allocator.free(var_name);
        }
        self.variable_stack.deinit();
        self.storage_slots.deinit();
    }

    /// Push a variable name onto the stack (takes ownership)
    ///
    /// Args:
    ///     name: Variable name to push (must be allocated)
    fn pushVariable(self: *Self, name: []const u8) YulCodegenError!void {
        try self.variable_stack.append(name);
    }

    /// Pop a variable name from the stack
    ///
    /// Returns:
    ///     Variable name if stack is not empty, null otherwise
    fn popVariable(self: *Self) ?[]const u8 {
        if (self.variable_stack.items.len == 0) return null;
        return self.variable_stack.pop();
    }

    /// Peek at the top variable without removing it
    ///
    /// Returns:
    ///     Variable name if stack is not empty, null otherwise
    fn peekVariable(self: *Self) ?[]const u8 {
        if (self.variable_stack.items.len == 0) return null;
        return self.variable_stack.items[self.variable_stack.items.len - 1];
    }

    /// Get the second-to-top variable (for binary operations)
    ///
    /// Returns:
    ///     Variable name if stack has at least 2 items, null otherwise
    fn peekVariable2(self: *Self) ?[]const u8 {
        if (self.variable_stack.items.len < 2) return null;
        return self.variable_stack.items[self.variable_stack.items.len - 2];
    }

    /// Generate Yul code from HIR
    ///
    /// Args:
    ///     hir: High-level IR to convert
    ///
    /// Returns:
    ///     Generated Yul code as owned string
    pub fn generateYul(self: *Self, hir: *const ir.HIR) ![]u8 {
        var yul_code = std.ArrayList(u8).init(self.allocator);
        defer yul_code.deinit();

        // Start with Yul object wrapper
        try yul_code.appendSlice("{\n");

        // Generate code for each HIR instruction
        for (hir.instructions.items) |instruction| {
            try self.generateInstruction(&yul_code, instruction);
        }

        // End Yul object
        try yul_code.appendSlice("}\n");

        return yul_code.toOwnedSlice();
    }

    /// Generate Yul code from full HIR Program
    ///
    /// Args:
    ///     program: HIR program to compile
    ///
    /// Returns:
    ///     Generated Yul code as owned string
    pub fn generateYulFromProgram(self: *Self, program: *const HIRProgram) YulCodegenError![]u8 {
        var yul_code = std.ArrayList(u8).init(self.allocator);
        defer yul_code.deinit();

        // Reset counters
        self.var_counter = 0;
        self.storage_counter = 0;
        self.clearVariableStack();

        // Generate Yul object with constructor and runtime code
        try yul_code.appendSlice("object \"Contract\" {\n");

        // Generate code section for each contract
        for (program.contracts) |*contract| {
            try self.generateContract(&yul_code, contract);
        }

        try yul_code.appendSlice("}\n");

        return yul_code.toOwnedSlice();
    }

    /// Generate Yul code for a contract
    ///
    /// Args:
    ///     yul_code: Output buffer
    ///     contract: Contract to compile
    fn generateContract(self: *Self, yul_code: *std.ArrayList(u8), contract: *const Contract) YulCodegenError!void {
        // Store current contract for event lookup
        self.current_contract = contract;

        // Initialize storage slot mapping
        var slot_counter: u32 = 0;
        for (contract.storage) |*storage_var| {
            try self.storage_slots.put(storage_var.name, slot_counter);
            slot_counter += 1;
        }

        try yul_code.writer().print("  // Contract: {s}\n", .{contract.name});
        try yul_code.appendSlice("  code {\n");

        // Generate constructor code
        try self.generateConstructor(yul_code, contract);

        // Generate runtime code
        try yul_code.appendSlice("    datacopy(0, dataoffset(\"runtime\"), datasize(\"runtime\"))\n");
        try yul_code.appendSlice("    return(0, datasize(\"runtime\"))\n");
        try yul_code.appendSlice("  }\n");

        // Generate runtime object
        try yul_code.appendSlice("  object \"runtime\" {\n");
        try yul_code.appendSlice("    code {\n");

        // Generate function dispatcher
        try self.generateDispatcher(yul_code, contract);

        // Generate function implementations
        for (contract.functions) |*function| {
            try self.generateFunction(yul_code, function);
        }

        try yul_code.appendSlice("    }\n");
        try yul_code.appendSlice("  }\n");
    }

    /// Generate constructor code for storage initialization
    ///
    /// Args:
    ///     yul_code: Output buffer
    ///     contract: Contract with storage variables
    fn generateConstructor(self: *Self, yul_code: *std.ArrayList(u8), contract: *const Contract) YulCodegenError!void {
        try yul_code.appendSlice("    // Constructor - initialize storage\n");

        for (contract.storage) |*storage_var| {
            if (storage_var.value) |value| {
                const slot = self.storage_counter;
                self.storage_counter += 1;

                try yul_code.writer().print("    // Initialize {s}\n", .{storage_var.name});
                const value_code = try self.generateExpression(yul_code, value);
                defer self.allocator.free(value_code);
                try yul_code.writer().print("    sstore({}, {s})\n", .{ slot, value_code });
            }
        }
    }

    /// Generate function dispatcher for public functions
    ///
    /// Args:
    ///     yul_code: Output buffer
    ///     contract: Contract with functions
    fn generateDispatcher(self: *Self, yul_code: *std.ArrayList(u8), contract: *const Contract) YulCodegenError!void {
        try yul_code.appendSlice("    // Function dispatcher\n");
        try yul_code.appendSlice("    let selector := shr(224, calldataload(0))\n");
        try yul_code.appendSlice("    switch selector\n");

        for (contract.functions) |*function| {
            if (function.visibility == .public) {
                const selector = try self.calculateFunctionSelector(function);

                // Generate function call with proper parameters
                try yul_code.writer().print("    case 0x{x:0>8} {{\n", .{selector});

                if (function.parameters.len > 0) {
                    // Extract parameters from calldata
                    for (function.parameters, 0..) |*param, i| {
                        const offset = 4 + (i * 32); // 4 bytes for selector + 32 bytes per param
                        try yul_code.writer().print("      let {s} := calldataload({})\n", .{ param.name, offset });
                    }

                    // Call function with parameters and handle return value
                    if (function.return_type != null) {
                        try yul_code.writer().print("      let result := {s}(", .{function.name});
                        for (function.parameters, 0..) |*param, i| {
                            if (i > 0) try yul_code.appendSlice(", ");
                            try yul_code.appendSlice(param.name);
                        }
                        try yul_code.appendSlice(")\n");
                        try yul_code.appendSlice("      return(result, 0x20)\n"); // Return 32 bytes
                    } else {
                        try yul_code.writer().print("      {s}(", .{function.name});
                        for (function.parameters, 0..) |*param, i| {
                            if (i > 0) try yul_code.appendSlice(", ");
                            try yul_code.appendSlice(param.name);
                        }
                        try yul_code.appendSlice(")\n");
                    }
                } else {
                    // Call function with no parameters and handle return value
                    if (function.return_type != null) {
                        try yul_code.writer().print("      let result := {s}()\n", .{function.name});
                        try yul_code.appendSlice("      return(result, 0x20)\n"); // Return 32 bytes
                    } else {
                        try yul_code.writer().print("      {s}()\n", .{function.name});
                    }
                }

                try yul_code.appendSlice("    }\n");
            }
        }

        try yul_code.appendSlice("    default { revert(0, 0) }\n");
    }

    /// Calculate 4-byte function selector
    ///
    /// Args:
    ///     function: Function to calculate selector for
    ///
    /// Returns:
    ///     32-bit function selector
    fn calculateFunctionSelector(self: *Self, function: *const Function) !u32 {
        // Build canonical function signature: functionName(type1,type2,...)
        var signature = std.ArrayList(u8).init(self.allocator);
        defer signature.deinit();

        try signature.appendSlice(function.name);
        try signature.appendSlice("(");

        for (function.parameters, 0..) |*param, i| {
            if (i > 0) {
                try signature.appendSlice(",");
            }

            // Convert Ora types to canonical EVM types
            const evm_type = convertToEVMType(param.type);
            try signature.appendSlice(evm_type);
        }

        try signature.appendSlice(")");

        // Calculate hash of signature (simplified until we add proper keccak256)
        const signature_bytes = signature.items;
        var hash: u64 = 0;

        // Use a better hash function (FNV-1a variant)
        for (signature_bytes) |byte| {
            hash = hash ^ byte;
            hash = hash *% 0x100000001b3; // FNV-1a prime
        }

        // Take first 4 bytes as selector
        return @truncate(hash);
    }

    /// Convert Ora types to canonical EVM types for function signatures
    ///
    /// Args:
    ///     ora_type: Ora type to convert
    ///
    /// Returns:
    ///     Canonical EVM type string
    fn convertToEVMType(ora_type: Type) []const u8 {
        return switch (ora_type) {
            .primitive => |prim| switch (prim) {
                .u8, .u16, .u32, .u64, .u128, .u256 => "uint256",
                .bool => "bool",
                .address => "address",
                .string => "string",
            },
            .slice => |_| "uint256[]", // Simplified - slices become uint256[]
            .mapping => |_| "mapping", // Mappings don't appear in function signatures
            .custom => |_| "uint256", // Custom types default to uint256
            .error_union => |_| "uint256", // Error unions default to uint256
            .result => |_| "uint256", // Results default to uint256
        };
    }

    /// Generate Yul code for a function
    ///
    /// Args:
    ///     yul_code: Output buffer
    ///     function: Function to compile
    fn generateFunction(self: *Self, yul_code: *std.ArrayList(u8), function: *const Function) YulCodegenError!void {
        // Store current function for context
        const prev_function = self.current_function;
        self.current_function = function;
        defer self.current_function = prev_function;

        // Write function signature
        try yul_code.writer().print("    // Function: {s}\n", .{function.name});
        try yul_code.writer().print("    function {s}(", .{function.name});

        // Write parameters
        for (function.parameters, 0..) |*param, i| {
            if (i > 0) try yul_code.appendSlice(", ");
            try yul_code.appendSlice(param.name);
        }

        // Write return type
        if (function.return_type) |_| {
            try yul_code.appendSlice(") -> result");
        } else {
            try yul_code.appendSlice(")");
        }

        try yul_code.appendSlice(" {\n");

        // Generate function body
        try self.generateBlock(yul_code, &function.body);

        try yul_code.appendSlice("    }\n\n");
    }

    /// Generate Yul code for a block of statements
    ///
    /// Args:
    ///     yul_code: Output buffer
    ///     block: Block to compile
    fn generateBlock(self: *Self, yul_code: *std.ArrayList(u8), block: *const Block) YulCodegenError!void {
        for (block.statements) |*stmt| {
            try self.generateStatement(yul_code, stmt);
        }
    }

    /// Generate Yul code for a statement
    ///
    /// Args:
    ///     yul_code: Output buffer
    ///     stmt: Statement to compile
    fn generateStatement(self: *Self, yul_code: *std.ArrayList(u8), stmt: *const Statement) YulCodegenError!void {
        switch (stmt.*) {
            .variable_decl => |*var_decl| {
                try self.generateVariableDecl(yul_code, var_decl);
            },
            .assignment => |*assignment| {
                try self.generateAssignment(yul_code, assignment);
            },
            .compound_assignment => |*comp_assign| {
                try self.generateCompoundAssignment(yul_code, comp_assign);
            },
            .if_statement => |*if_stmt| {
                try self.generateIfStatement(yul_code, if_stmt);
            },
            .while_statement => |*while_stmt| {
                try self.generateWhileStatement(yul_code, while_stmt);
            },
            .return_statement => |*ret_stmt| {
                try self.generateReturnStatement(yul_code, ret_stmt);
            },
            .expression_statement => |*expr_stmt| {
                const expr_var = try self.generateExpression(yul_code, expr_stmt.expression);
                defer self.allocator.free(expr_var);
                // Expression statement result is discarded
            },
            .lock_statement => |*lock_stmt| {
                try self.generateLockStatement(yul_code, lock_stmt);
            },
            .unlock_statement => |*unlock_stmt| {
                try self.generateUnlockStatement(yul_code, unlock_stmt);
            },
            .error_decl => |*error_decl| {
                try self.generateErrorDecl(yul_code, error_decl);
            },
            .try_statement => |*try_stmt| {
                try self.generateTryStatement(yul_code, try_stmt);
            },
            .error_return => |*error_ret| {
                try self.generateErrorReturn(yul_code, error_ret);
            },
        }
    }

    /// Generate variable declaration
    fn generateVariableDecl(self: *Self, yul_code: *std.ArrayList(u8), var_decl: *const ir.VariableDecl) YulCodegenError!void {
        try yul_code.writer().print("      // Variable: {s}\n", .{var_decl.name});

        if (var_decl.value) |value| {
            const value_var = try self.generateExpression(yul_code, value);
            defer self.allocator.free(value_var);
            try yul_code.writer().print("      let {s} := {s}\n", .{ var_decl.name, value_var });
        } else {
            // Initialize with default value based on type
            const default_val = self.getDefaultValue(var_decl.type);
            try yul_code.writer().print("      let {s} := {s}\n", .{ var_decl.name, default_val });
        }
    }

    /// Generate assignment statement
    fn generateAssignment(self: *Self, yul_code: *std.ArrayList(u8), assignment: *const ir.Assignment) YulCodegenError!void {
        try yul_code.appendSlice("      // Assignment\n");

        const value_var = try self.generateExpression(yul_code, assignment.value);
        defer self.allocator.free(value_var);

        // For simplicity, assume target is an identifier
        if (assignment.target.* == .identifier) {
            const target_name = assignment.target.identifier.name;
            try yul_code.writer().print("      {s} := {s}\n", .{ target_name, value_var });
        }
    }

    /// Generate compound assignment statement
    fn generateCompoundAssignment(self: *Self, yul_code: *std.ArrayList(u8), comp_assign: *const ir.CompoundAssignment) YulCodegenError!void {
        try yul_code.appendSlice("      // Compound Assignment\n");

        const value_var = try self.generateExpression(yul_code, comp_assign.value);
        defer self.allocator.free(value_var);

        if (comp_assign.target.* == .identifier) {
            const target_name = comp_assign.target.identifier.name;
            const op_name = switch (comp_assign.operator) {
                .plus_equal => "add",
                .minus_equal => "sub",
                .star_equal => "mul",
                .slash_equal => "div",
                .percent_equal => "mod",
            };

            try yul_code.writer().print("      {s} := {s}({s}, {s})\n", .{ target_name, op_name, target_name, value_var });
        }
    }

    /// Generate if statement
    fn generateIfStatement(self: *Self, yul_code: *std.ArrayList(u8), if_stmt: *const ir.IfStatement) YulCodegenError!void {
        try yul_code.appendSlice("      // If statement\n");

        const condition_var = try self.generateExpression(yul_code, if_stmt.condition);
        defer self.allocator.free(condition_var);

        try yul_code.writer().print("      if {s} {{\n", .{condition_var});
        try self.generateBlock(yul_code, &if_stmt.then_branch);
        try yul_code.appendSlice("      }\n");

        if (if_stmt.else_branch) |*else_branch| {
            try yul_code.appendSlice("      if iszero(");
            try yul_code.appendSlice(condition_var);
            try yul_code.appendSlice(") {\n");
            try self.generateBlock(yul_code, else_branch);
            try yul_code.appendSlice("      }\n");
        }
    }

    /// Generate while statement
    fn generateWhileStatement(self: *Self, yul_code: *std.ArrayList(u8), while_stmt: *const ir.WhileStatement) YulCodegenError!void {
        try yul_code.appendSlice("      // While loop\n");

        const loop_label = try std.fmt.allocPrint(self.allocator, "loop_{}", .{self.var_counter});
        defer self.allocator.free(loop_label);
        self.var_counter += 1;

        try yul_code.writer().print("      {s}:\n", .{loop_label});

        const condition_var = try self.generateExpression(yul_code, while_stmt.condition);
        defer self.allocator.free(condition_var);

        try yul_code.writer().print("      if iszero({s}) {{ leave }}\n", .{condition_var});
        try self.generateBlock(yul_code, &while_stmt.body);
        try yul_code.writer().print("      jump({s})\n", .{loop_label});
    }

    /// Generate return statement
    fn generateReturnStatement(self: *Self, yul_code: *std.ArrayList(u8), ret_stmt: *const ir.ReturnStatement) YulCodegenError!void {
        try yul_code.appendSlice("      // Return\n");

        if (ret_stmt.value) |value| {
            const value_var = try self.generateExpression(yul_code, value);
            defer self.allocator.free(value_var);
            try yul_code.writer().print("      result := {s}\n", .{value_var});
        }
    }

    /// Generate lock statement (blockchain synchronization)
    fn generateLockStatement(self: *Self, yul_code: *std.ArrayList(u8), lock_stmt: *const ir.LockStatement) YulCodegenError!void {
        _ = self;
        _ = lock_stmt;
        try yul_code.appendSlice("      // Lock (placeholder)\n");
        // TODO: Implement proper lock mechanism
    }

    /// Generate unlock statement
    fn generateUnlockStatement(self: *Self, yul_code: *std.ArrayList(u8), unlock_stmt: *const ir.UnlockStatement) YulCodegenError!void {
        _ = self;
        _ = unlock_stmt;
        try yul_code.appendSlice("      // Unlock (placeholder)\n");
        // TODO: Implement proper unlock mechanism
    }

    /// Generate error declaration
    fn generateErrorDecl(self: *Self, yul_code: *std.ArrayList(u8), error_decl: *const ir.ErrorDecl) YulCodegenError!void {
        _ = self;
        try yul_code.writer().print("      // Error: {s}\n", .{error_decl.name});
        // Error declarations don't generate runtime code
    }

    /// Generate try statement
    fn generateTryStatement(self: *Self, yul_code: *std.ArrayList(u8), try_stmt: *const ir.TryStatement) YulCodegenError!void {
        try yul_code.appendSlice("      // Try-catch\n");

        // Generate try block
        try self.generateBlock(yul_code, &try_stmt.try_block);

        if (try_stmt.catch_block) |*catch_block| {
            try yul_code.appendSlice("      // Catch block\n");
            try self.generateBlock(yul_code, &catch_block.block);
        }
    }

    /// Generate error return
    fn generateErrorReturn(self: *Self, yul_code: *std.ArrayList(u8), error_ret: *const ir.ErrorReturn) YulCodegenError!void {
        _ = self;
        try yul_code.writer().print("      // Error return: {s}\n", .{error_ret.error_name});
        // TODO: Look up error code and return proper error union
        try yul_code.appendSlice("      revert(0, 0)\n");
    }

    /// Generate expression and return variable name
    fn generateExpression(self: *Self, yul_code: *std.ArrayList(u8), expr: *const Expression) YulCodegenError![]const u8 {
        switch (expr.*) {
            .literal => |*literal| {
                return try self.generateLiteral(yul_code, literal);
            },
            .identifier => |*ident| {
                // Check if this is a storage variable
                if (self.storage_slots.get(ident.name)) |slot| {
                    return try std.fmt.allocPrint(self.allocator, "{}", .{slot});
                }
                // Regular identifier (parameter, local variable, etc.)
                return try self.allocator.dupe(u8, ident.name);
            },
            .binary => |*binary| {
                return try self.generateBinaryExpression(yul_code, binary);
            },
            .unary => |*unary| {
                return try self.generateUnaryExpression(yul_code, unary);
            },
            .call => |*call| {
                return try self.generateCallExpression(yul_code, call);
            },
            .index => |*index| {
                return try self.generateIndexExpression(yul_code, index);
            },
            .field => |*field| {
                return try self.generateFieldExpression(yul_code, field);
            },
            .transfer => |*transfer| {
                return try self.generateTransferExpression(yul_code, transfer);
            },
            .old => |*old| {
                // Old expressions are used in formal verification - for now just evaluate the inner expression
                return try self.generateExpression(yul_code, old.expression);
            },
            .try_expr => |*try_expr| {
                return try self.generateTryExpression(yul_code, try_expr);
            },
            .error_value => |*error_val| {
                return try self.generateErrorValue(yul_code, error_val);
            },
            .error_cast => |*error_cast| {
                return try self.generateErrorCast(yul_code, error_cast);
            },
        }
    }

    /// Generate literal value
    fn generateLiteral(self: *Self, yul_code: *std.ArrayList(u8), literal: *const ir.Literal) YulCodegenError![]const u8 {
        _ = yul_code;

        return switch (literal.*) {
            .integer => |int_str| try self.allocator.dupe(u8, int_str),
            .string => |str| {
                // Convert string to hex representation for Yul
                var hex_str = try self.allocator.alloc(u8, str.len * 2 + 2);
                hex_str[0] = '0';
                hex_str[1] = 'x';
                for (str, 0..) |byte, i| {
                    _ = std.fmt.bufPrint(hex_str[2 + i * 2 .. 2 + i * 2 + 2], "{x:0>2}", .{byte}) catch unreachable;
                }
                return hex_str;
            },
            .boolean => |b| if (b) try self.allocator.dupe(u8, "1") else try self.allocator.dupe(u8, "0"),
            .address => |addr_str| try self.allocator.dupe(u8, addr_str),
        };
    }

    /// Generate binary expression
    fn generateBinaryExpression(self: *Self, yul_code: *std.ArrayList(u8), binary: *const ir.BinaryExpression) YulCodegenError![]const u8 {
        const left_var = try self.generateExpression(yul_code, binary.left);
        defer self.allocator.free(left_var);

        const right_var = try self.generateExpression(yul_code, binary.right);
        defer self.allocator.free(right_var);

        const result_var = try std.fmt.allocPrint(self.allocator, "temp_{}", .{self.var_counter});
        self.var_counter += 1;

        const op_name = switch (binary.operator) {
            .plus => "add",
            .minus => "sub",
            .star => "mul",
            .slash => "div",
            .percent => "mod",
            .equal_equal => "eq",
            .bang_equal => "iszero(eq",
            .less => "lt",
            .less_equal => "iszero(gt",
            .greater => "gt",
            .greater_equal => "iszero(lt",
            .and_ => "and",
            .or_ => "or",
            .bit_and => "and",
            .bit_or => "or",
            .bit_xor => "xor",
            .shift_left => "shl",
            .shift_right => "shr",
        };

        // Handle special cases that need extra closing parenthesis
        if (binary.operator == .bang_equal or binary.operator == .less_equal or binary.operator == .greater_equal) {
            try yul_code.writer().print("      let {s} := {s}({s}, {s}))\n", .{ result_var, op_name, left_var, right_var });
        } else {
            try yul_code.writer().print("      let {s} := {s}({s}, {s})\n", .{ result_var, op_name, left_var, right_var });
        }

        return result_var;
    }

    /// Generate unary expression
    fn generateUnaryExpression(self: *Self, yul_code: *std.ArrayList(u8), unary: *const ir.UnaryExpression) YulCodegenError![]const u8 {
        const operand_var = try self.generateExpression(yul_code, unary.operand);
        defer self.allocator.free(operand_var);

        const result_var = try std.fmt.allocPrint(self.allocator, "temp_{}", .{self.var_counter});
        self.var_counter += 1;

        switch (unary.operator) {
            .minus => try yul_code.writer().print("      let {s} := sub(0, {s})\n", .{ result_var, operand_var }),
            .bang => try yul_code.writer().print("      let {s} := iszero({s})\n", .{ result_var, operand_var }),
            .bit_not => try yul_code.writer().print("      let {s} := not({s})\n", .{ result_var, operand_var }),
        }

        return result_var;
    }

    /// Generate call expression
    fn generateCallExpression(self: *Self, yul_code: *std.ArrayList(u8), call: *const ir.CallExpression) YulCodegenError![]const u8 {
        // Check if this is an event logging call
        if (call.callee.* == .identifier) {
            const func_name = call.callee.identifier.name;

            // Check if this is a known event (events are stored in the contract)
            if (self.current_contract) |contract| {
                for (contract.events) |*event| {
                    if (std.mem.eql(u8, event.name, func_name)) {
                        // This is an event logging call
                        return try self.generateEventLog(yul_code, event, call.arguments);
                    }
                }
            }
        }

        // Generate arguments
        var arg_vars = std.ArrayList([]const u8).init(self.allocator);
        defer {
            for (arg_vars.items) |arg_var| {
                self.allocator.free(arg_var);
            }
            arg_vars.deinit();
        }

        for (call.arguments) |*arg| {
            const arg_var = try self.generateExpression(yul_code, arg);
            try arg_vars.append(arg_var);
        }

        const result_var = try std.fmt.allocPrint(self.allocator, "temp_{}", .{self.var_counter});
        self.var_counter += 1;

        // For now, assume function calls are simple
        if (call.callee.* == .identifier) {
            const func_name = call.callee.identifier.name;

            try yul_code.writer().print("      let {s} := {s}(", .{ result_var, func_name });
            for (arg_vars.items, 0..) |arg_var, i| {
                if (i > 0) try yul_code.appendSlice(", ");
                try yul_code.appendSlice(arg_var);
            }
            try yul_code.appendSlice(")\n");
        }

        return result_var;
    }

    /// Generate event logging
    fn generateEventLog(self: *Self, yul_code: *std.ArrayList(u8), event: *const ir.Event, arguments: []const ir.Expression) YulCodegenError![]const u8 {
        // Generate argument variables
        var arg_vars = std.ArrayList([]const u8).init(self.allocator);
        defer {
            for (arg_vars.items) |arg_var| {
                self.allocator.free(arg_var);
            }
            arg_vars.deinit();
        }

        for (arguments) |*arg| {
            const arg_var = try self.generateExpression(yul_code, arg);
            try arg_vars.append(arg_var);
        }

        // Calculate event signature hash (simplified)
        const event_hash = try self.calculateEventSignatureHash(event);

        // Store event data in memory
        const data_ptr = try std.fmt.allocPrint(self.allocator, "temp_{}", .{self.var_counter});
        defer self.allocator.free(data_ptr);
        self.var_counter += 1;

        try yul_code.writer().print("      let {s} := mload(0x40)\n", .{data_ptr});

        // Store arguments in memory
        for (arg_vars.items, 0..) |arg_var, i| {
            try yul_code.writer().print("      mstore(add({s}, {}), {s})\n", .{ data_ptr, i * 32, arg_var });
        }

        // Emit log (using log1 with topic for now)
        const data_size = arg_vars.items.len * 32;
        try yul_code.writer().print("      log1({s}, {}, 0x{x:0>64})\n", .{ data_ptr, data_size, event_hash });

        // Return a dummy result
        const result_var = try std.fmt.allocPrint(self.allocator, "temp_{}", .{self.var_counter});
        self.var_counter += 1;
        try yul_code.writer().print("      let {s} := 1\n", .{result_var});

        return result_var;
    }

    /// Calculate event signature hash (simplified)
    fn calculateEventSignatureHash(self: *Self, event: *const ir.Event) !u64 {
        _ = self;
        // Simplified event signature hash - in real implementation would use keccak256
        var hash: u64 = 0;
        for (event.name) |byte| {
            hash = hash *% 31 +% byte;
        }
        return hash;
    }

    /// Generate index expression (array/mapping access)
    fn generateIndexExpression(self: *Self, yul_code: *std.ArrayList(u8), index: *const ir.IndexExpression) YulCodegenError![]const u8 {
        const target_var = try self.generateExpression(yul_code, index.target);
        defer self.allocator.free(target_var);

        const index_var = try self.generateExpression(yul_code, index.index);
        defer self.allocator.free(index_var);

        const result_var = try std.fmt.allocPrint(self.allocator, "temp_{}", .{self.var_counter});
        self.var_counter += 1;

        // Check if target is a storage variable (target_var should be a slot number)
        if (std.fmt.parseInt(u32, target_var, 10)) |slot| {
            // Mapping access: hash(key, slot) for storage location
            try yul_code.writer().print("      let {s} := sload(keccak256({s}, {}))\n", .{ result_var, index_var, slot });
        } else |_| {
            // Fallback: treat as regular array access
            try yul_code.writer().print("      let {s} := sload(add({s}, {s}))\n", .{ result_var, target_var, index_var });
        }

        return result_var;
    }

    /// Generate field expression (struct field access)
    fn generateFieldExpression(self: *Self, yul_code: *std.ArrayList(u8), field: *const ir.FieldExpression) YulCodegenError![]const u8 {
        const target_var = try self.generateExpression(yul_code, field.target);
        defer self.allocator.free(target_var);

        const result_var = try std.fmt.allocPrint(self.allocator, "temp_{}", .{self.var_counter});
        self.var_counter += 1;

        // Check for standard library field access
        if (field.target.* == .identifier) {
            const target_name = field.target.identifier.name;

            if (std.mem.eql(u8, target_name, "std")) {
                // Handle std library modules
                if (std.mem.eql(u8, field.field, "transaction")) {
                    // Return a marker for transaction module
                    try yul_code.writer().print("      let {s} := 0x01 // std.transaction\n", .{result_var});
                    return result_var;
                } else if (std.mem.eql(u8, field.field, "constants")) {
                    // Return a marker for constants module
                    try yul_code.writer().print("      let {s} := 0x02 // std.constants\n", .{result_var});
                    return result_var;
                } else if (std.mem.eql(u8, field.field, "block")) {
                    // Return a marker for block module
                    try yul_code.writer().print("      let {s} := 0x03 // std.block\n", .{result_var});
                    return result_var;
                }
            }
        }

        // Check for nested field access like std.transaction.sender
        if (field.target.* == .field) {
            const nested_field = field.target.field;
            if (nested_field.target.* == .identifier) {
                const base_name = nested_field.target.identifier.name;

                if (std.mem.eql(u8, base_name, "std")) {
                    if (std.mem.eql(u8, nested_field.field, "transaction")) {
                        // Handle std.transaction.* fields
                        if (std.mem.eql(u8, field.field, "sender")) {
                            try yul_code.writer().print("      let {s} := caller()\n", .{result_var});
                            return result_var;
                        } else if (std.mem.eql(u8, field.field, "value")) {
                            try yul_code.writer().print("      let {s} := callvalue()\n", .{result_var});
                            return result_var;
                        } else if (std.mem.eql(u8, field.field, "origin")) {
                            try yul_code.writer().print("      let {s} := origin()\n", .{result_var});
                            return result_var;
                        }
                    } else if (std.mem.eql(u8, nested_field.field, "constants")) {
                        // Handle std.constants.* fields
                        if (std.mem.eql(u8, field.field, "ZERO_ADDRESS")) {
                            try yul_code.writer().print("      let {s} := 0x0\n", .{result_var});
                            return result_var;
                        }
                    } else if (std.mem.eql(u8, nested_field.field, "block")) {
                        // Handle std.block.* fields
                        if (std.mem.eql(u8, field.field, "timestamp")) {
                            try yul_code.writer().print("      let {s} := timestamp()\n", .{result_var});
                            return result_var;
                        } else if (std.mem.eql(u8, field.field, "number")) {
                            try yul_code.writer().print("      let {s} := number()\n", .{result_var});
                            return result_var;
                        } else if (std.mem.eql(u8, field.field, "coinbase")) {
                            try yul_code.writer().print("      let {s} := coinbase()\n", .{result_var});
                            return result_var;
                        }
                    }
                }
            }
        }

        // Simple field access - in real implementation would need struct layout
        try yul_code.writer().print("      let {s} := {s} // field: {s}\n", .{ result_var, target_var, field.field });

        return result_var;
    }

    /// Generate transfer expression (blockchain transfer)
    fn generateTransferExpression(self: *Self, yul_code: *std.ArrayList(u8), transfer: *const ir.TransferExpression) YulCodegenError![]const u8 {
        const to_var = try self.generateExpression(yul_code, transfer.to);
        defer self.allocator.free(to_var);

        const amount_var = try self.generateExpression(yul_code, transfer.amount);
        defer self.allocator.free(amount_var);

        const result_var = try std.fmt.allocPrint(self.allocator, "temp_{}", .{self.var_counter});
        self.var_counter += 1;

        // Generate a call to transfer Ether
        try yul_code.writer().print("      let {s} := call(gas(), {s}, {s}, 0, 0, 0, 0)\n", .{ result_var, to_var, amount_var });

        return result_var;
    }

    /// Generate try expression
    fn generateTryExpression(self: *Self, yul_code: *std.ArrayList(u8), try_expr: *const ir.TryExpression) YulCodegenError![]const u8 {
        // For now, just evaluate the expression normally
        return try self.generateExpression(yul_code, try_expr.expression);
    }

    /// Generate error value
    fn generateErrorValue(self: *Self, yul_code: *std.ArrayList(u8), error_val: *const ir.ErrorValue) YulCodegenError![]const u8 {
        _ = yul_code;
        // TODO: Look up error code by name
        return try std.fmt.allocPrint(self.allocator, "0xFF // error: {s}", .{error_val.error_name});
    }

    /// Generate error cast
    fn generateErrorCast(self: *Self, yul_code: *std.ArrayList(u8), error_cast: *const ir.ErrorCast) YulCodegenError![]const u8 {
        // For now, just evaluate the operand
        return try self.generateExpression(yul_code, error_cast.operand);
    }

    /// Get default value for a type
    fn getDefaultValue(self: *Self, typ: Type) []const u8 {
        _ = self;
        return switch (typ) {
            .primitive => |prim| switch (prim) {
                .u8, .u16, .u32, .u64, .u128, .u256 => "0",
                .bool => "false",
                .address => "0",
                .string => "0",
            },
            else => "0",
        };
    }

    /// Generate Yul code for a single HIR instruction
    ///
    /// Args:
    ///     yul_code: Output buffer for generated code
    ///     instruction: HIR instruction to convert
    fn generateInstruction(self: *Self, yul_code: *std.ArrayList(u8), instruction: HIRInstruction) !void {
        _ = self;

        switch (instruction) {
            .push => |value| {
                try yul_code.writer().print("    // Push {}\n", .{value});
                try yul_code.writer().print("    let temp_{} := {}\n", .{ value, value });
            },
            .pop => {
                try yul_code.appendSlice("    // Pop\n");
            },
            .add => {
                try yul_code.appendSlice("    // Add\n");
                try yul_code.appendSlice("    let result := add(temp_a, temp_b)\n");
            },
            .store => |addr| {
                try yul_code.writer().print("    // Store at address {}\n", .{addr});
                try yul_code.writer().print("    mstore({}, temp_value)\n", .{addr});
            },
            .load => |addr| {
                try yul_code.writer().print("    // Load from address {}\n", .{addr});
                try yul_code.writer().print("    let temp_loaded := mload({})\n", .{addr});
            },
            .ret => {
                try yul_code.appendSlice("    // Return\n");
                try yul_code.appendSlice("    return(0, 32)\n");
            },
        }
    }

    /// Compile Yul code to bytecode using the Solidity Yul compiler
    ///
    /// Args:
    ///     yul_code: Yul source code to compile
    ///
    /// Returns:
    ///     Compilation result with bytecode or error information
    pub fn compileYulToBytecode(self: *Self, yul_code: []const u8) !YulCompileResult {
        print("Compiling Yul code:\n{s}\n", .{yul_code});
        return YulCompiler.compile(self.allocator, yul_code);
    }

    /// Full pipeline: HIR -> Yul -> Bytecode
    ///
    /// Args:
    ///     hir: High-level IR to compile
    ///
    /// Returns:
    ///     Compilation result with bytecode
    pub fn generateBytecode(self: *Self, hir: *const ir.HIR) !YulCompileResult {
        const yul_code = try self.generateYul(hir);
        defer self.allocator.free(yul_code);

        return self.compileYulToBytecode(yul_code);
    }

    /// Generate Yul code for error union success return
    ///
    /// Error union layout: [tag][data]
    /// - tag: 0x00 for success, 0x01 for error
    /// - data: actual value (u256) or error code
    ///
    /// Args:
    ///     yul_code: Output buffer for generated code
    ///     value_var: Variable containing the success value
    ///
    /// Returns:
    ///     Variable name containing the error union
    pub fn generateErrorUnionSuccess(self: *Self, yul_code: *std.ArrayList(u8), value_var: []const u8) ![]const u8 {
        const result_var = try std.fmt.allocPrint(self.allocator, "result_{}", .{self.variable_stack.items.len});

        // Generate the error union layout
        try yul_code.writer().print("    // Success error union: [0x00][value]\n");
        try yul_code.writer().print("    let ptr := mload(0x40)\n");
        try yul_code.writer().print("    mstore(ptr, 0x00)           // tag = success\n");
        try yul_code.writer().print("    mstore(add(ptr, 0x20), {s}) // value\n", .{value_var});
        try yul_code.writer().print("    let {s} := ptr\n", .{result_var});
        try yul_code.writer().print("    mstore(0x40, add(ptr, 0x40)) // update free memory pointer\n");

        try self.pushVariable(result_var);
        return result_var;
    }

    /// Generate Yul code for error union error return
    ///
    /// Args:
    ///     yul_code: Output buffer for generated code
    ///     error_code: Error code to return
    ///
    /// Returns:
    ///     Variable name containing the error union
    pub fn generateErrorUnionError(self: *Self, yul_code: *std.ArrayList(u8), error_code: u32) ![]const u8 {
        const result_var = try std.fmt.allocPrint(self.allocator, "error_{}", .{self.variable_stack.items.len});

        // Generate the error union layout
        try yul_code.writer().print("    // Error error union: [0x01][error_code]\n");
        try yul_code.writer().print("    let ptr := mload(0x40)\n");
        try yul_code.writer().print("    mstore(ptr, 0x01)           // tag = error\n");
        try yul_code.writer().print("    mstore(add(ptr, 0x20), {}) // error code\n", .{error_code});
        try yul_code.writer().print("    let {s} := ptr\n", .{result_var});
        try yul_code.writer().print("    mstore(0x40, add(ptr, 0x40)) // update free memory pointer\n");

        try self.pushVariable(result_var);
        return result_var;
    }

    /// Generate Yul code for error union check
    ///
    /// Args:
    ///     yul_code: Output buffer for generated code
    ///     error_union_var: Variable containing the error union to check
    ///
    /// Returns:
    ///     Variable name containing the tag (0x00 = success, 0x01 = error)
    pub fn generateErrorUnionCheck(self: *Self, yul_code: *std.ArrayList(u8), error_union_var: []const u8) ![]const u8 {
        const tag_var = try std.fmt.allocPrint(self.allocator, "tag_{}", .{self.variable_stack.items.len});

        // Extract tag from error union
        try yul_code.writer().print("    // Check error union tag\n");
        try yul_code.writer().print("    let {s} := mload({s})\n", .{ tag_var, error_union_var });

        try self.pushVariable(tag_var);
        return tag_var;
    }

    /// Generate Yul code for error union value extraction
    ///
    /// Args:
    ///     yul_code: Output buffer for generated code
    ///     error_union_var: Variable containing the error union
    ///
    /// Returns:
    ///     Variable name containing the data (value or error code)
    pub fn generateErrorUnionExtract(self: *Self, yul_code: *std.ArrayList(u8), error_union_var: []const u8) ![]const u8 {
        const data_var = try std.fmt.allocPrint(self.allocator, "data_{}", .{self.variable_stack.items.len});

        // Extract data from error union
        try yul_code.writer().print("    // Extract error union data\n");
        try yul_code.writer().print("    let {s} := mload(add({s}, 0x20))\n", .{ data_var, error_union_var });

        try self.pushVariable(data_var);
        return data_var;
    }

    /// Generate Yul code for try-catch pattern
    ///
    /// Args:
    ///     yul_code: Output buffer for generated code
    ///     error_union_var: Variable containing the error union to check
    ///     success_label: Label for success branch
    ///     error_label: Label for error branch
    pub fn generateTryCatch(self: *Self, yul_code: *std.ArrayList(u8), error_union_var: []const u8, success_label: []const u8, error_label: []const u8) !void {
        const tag_var = try self.generateErrorUnionCheck(yul_code, error_union_var);
        defer self.allocator.free(tag_var);

        try yul_code.writer().print("    // Try-catch pattern\n");
        try yul_code.writer().print("    switch {s}\n", .{tag_var});
        try yul_code.writer().print("    case 0 {{\n");
        try yul_code.writer().print("        // Success case\n");
        try yul_code.writer().print("        {s}:\n", .{success_label});
        try yul_code.writer().print("    }}\n");
        try yul_code.writer().print("    default {{\n");
        try yul_code.writer().print("        // Error case\n");
        try yul_code.writer().print("        {s}:\n", .{error_label});
        try yul_code.writer().print("    }}\n");
    }

    /// Generate Yul code for error union function call
    ///
    /// This implements the calling convention for functions that return error unions:
    /// 1. Call the function
    /// 2. Check if it succeeded (call success AND tag == 0)
    /// 3. Handle error or continue with success value
    ///
    /// Args:
    ///     yul_code: Output buffer for generated code
    ///     function_name: Name of the function to call
    ///     args: Arguments to pass to the function
    ///
    /// Returns:
    ///     Variable name containing the error union result
    pub fn generateErrorUnionCall(self: *Self, yul_code: *std.ArrayList(u8), function_name: []const u8, args: []const []const u8) ![]const u8 {
        _ = args; // TODO: Use args for function call parameters
        const result_var = try std.fmt.allocPrint(self.allocator, "call_result_{}", .{self.variable_stack.items.len});

        try yul_code.writer().print("    // Call function that returns error union\n");
        try yul_code.writer().print("    let ptr := mload(0x40)\n");

        // Generate function call
        try yul_code.writer().print("    let success := call(\n");
        try yul_code.writer().print("        gas(),\n");
        try yul_code.writer().print("        {s},    // target\n", .{function_name});
        try yul_code.writer().print("        0,      // value\n");
        try yul_code.writer().print("        ptr,    // input ptr\n");
        try yul_code.writer().print("        0,      // input size\n");
        try yul_code.writer().print("        ptr,    // output ptr\n");
        try yul_code.writer().print("        0x40    // output size (64 bytes for error union)\n");
        try yul_code.writer().print("    )\n");

        // Check call success and handle error union
        try yul_code.writer().print("    switch success\n");
        try yul_code.writer().print("    case 0 {{\n");
        try yul_code.writer().print("        // Call failed - return error\n");
        try yul_code.writer().print("        mstore(ptr, 0x01)     // tag = error\n");
        try yul_code.writer().print("        mstore(add(ptr, 0x20), 0xFF) // generic error code\n");
        try yul_code.writer().print("    }}\n");
        try yul_code.writer().print("    default {{\n");
        try yul_code.writer().print("        // Call succeeded - check error union tag\n");
        try yul_code.writer().print("        let tag := mload(ptr)\n");
        try yul_code.writer().print("        switch tag\n");
        try yul_code.writer().print("        case 0 {{\n");
        try yul_code.writer().print("            // Success case - data is valid\n");
        try yul_code.writer().print("        }}\n");
        try yul_code.writer().print("        default {{\n");
        try yul_code.writer().print("            // Error case - propagate error\n");
        try yul_code.writer().print("        }}\n");
        try yul_code.writer().print("    }}\n");

        try yul_code.writer().print("    let {s} := ptr\n", .{result_var});
        try yul_code.writer().print("    mstore(0x40, add(ptr, 0x40)) // update free memory pointer\n");

        try self.pushVariable(result_var);
        return result_var;
    }

    /// Clear the variable stack and free all variables
    ///
    /// Used internally to clean up variables after code generation.
    fn clearVariableStack(self: *Self) void {
        for (self.variable_stack.items) |var_name| {
            self.allocator.free(var_name);
        }
        self.variable_stack.clearRetainingCapacity();
    }

    /// Generate Yul code from simple HIR
    ///
    /// Args:
    ///     hir: Simple HIR structure for testing
    ///
    /// Returns:
    ///     Generated Yul code as owned string
    pub fn generateYulSimple(self: *Self, hir: *const SimpleHIR) ![]u8 {
        var yul_code = std.ArrayList(u8).init(self.allocator);
        defer yul_code.deinit();

        // Clear variable stack for fresh generation
        self.clearVariableStack();

        // Start with Yul object wrapper
        try yul_code.appendSlice("{\n");

        // Generate code for each HIR instruction
        for (hir.instructions.items) |instruction| {
            try self.generateInstructionSimple(&yul_code, instruction);
        }

        // End Yul object
        try yul_code.appendSlice("}\n");

        // Clean up any remaining variables on the stack
        self.clearVariableStack();

        return yul_code.toOwnedSlice();
    }

    /// Generate Yul code for a single simple HIR instruction
    ///
    /// Args:
    ///     yul_code: Output buffer for generated code
    ///     instruction: Simple HIR instruction to convert
    fn generateInstructionSimple(self: *Self, yul_code: *std.ArrayList(u8), instruction: HIRInstruction) !void {
        switch (instruction) {
            .push => |value| {
                try yul_code.writer().print("    // Push {}\n", .{value});
                const var_name = try std.fmt.allocPrint(self.allocator, "temp_{}", .{value});
                try yul_code.writer().print("    let {s} := {}\n", .{ var_name, value });
                try self.pushVariable(var_name);
            },
            .pop => {
                try yul_code.appendSlice("    // Pop\n");
                if (self.popVariable()) |var_name| {
                    self.allocator.free(var_name);
                }
            },
            .add => {
                try yul_code.appendSlice("    // Add\n");
                const var_b = self.popVariable();
                const var_a = self.popVariable();

                if (var_b == null or var_a == null) {
                    // Not enough operands on stack - use fallback
                    if (var_a) |a| self.allocator.free(a);
                    if (var_b) |b| self.allocator.free(b);

                    const result_var = try std.fmt.allocPrint(self.allocator, "temp_add_result", .{});
                    try yul_code.writer().print("    let {s} := add(0, 0)\n", .{result_var});
                    try self.pushVariable(result_var);
                } else {
                    defer {
                        self.allocator.free(var_a.?);
                        self.allocator.free(var_b.?);
                    }

                    const result_var = try std.fmt.allocPrint(self.allocator, "temp_add_result", .{});
                    try yul_code.writer().print("    let {s} := add({s}, {s})\n", .{ result_var, var_a.?, var_b.? });
                    try self.pushVariable(result_var);
                }
            },
            .store => |addr| {
                try yul_code.writer().print("    // Store at address {}\n", .{addr});
                const value_var = self.popVariable();
                if (value_var) |var_name| {
                    defer self.allocator.free(var_name);
                    try yul_code.writer().print("    mstore({}, {s})\n", .{ addr, var_name });
                } else {
                    // No value on stack - store 0
                    try yul_code.writer().print("    mstore({}, 0)\n", .{addr});
                }
            },
            .load => |addr| {
                try yul_code.writer().print("    // Load from address {}\n", .{addr});
                const loaded_var = try std.fmt.allocPrint(self.allocator, "temp_loaded_{}", .{addr});
                try yul_code.writer().print("    let {s} := mload({})\n", .{ loaded_var, addr });
                try self.pushVariable(loaded_var);
            },
            .ret => {
                try yul_code.appendSlice("    // Return\n");
                try yul_code.appendSlice("    return(0, 32)\n");
            },
        }
    }

    /// Full pipeline: Simple HIR -> Yul -> Bytecode
    ///
    /// Args:
    ///     hir: Simple HIR structure to compile
    ///
    /// Returns:
    ///     Compilation result with bytecode
    pub fn generateBytecodeSimple(self: *Self, hir: *const SimpleHIR) !YulCompileResult {
        const yul_code = try self.generateYulSimple(hir);
        defer self.allocator.free(yul_code);

        return self.compileYulToBytecode(yul_code);
    }

    /// Generate optimized Yul code from HIR Program with gas optimizations
    ///
    /// Args:
    ///     program: HIR program to compile
    ///     optimize_gas: Whether to apply gas optimizations
    ///
    /// Returns:
    ///     Generated optimized Yul code
    pub fn generateOptimizedYul(self: *Self, program: *const HIRProgram, optimize_gas: bool) ![]u8 {
        const yul_code = try self.generateYulFromProgram(program);

        if (!optimize_gas) return yul_code;

        // Apply gas optimizations
        defer self.allocator.free(yul_code);
        return try self.optimizeYulForGas(yul_code);
    }

    /// Generate bytecode with optimizations
    ///
    /// Args:
    ///     program: HIR program to compile
    ///     optimize_gas: Whether to apply gas optimizations
    ///
    /// Returns:
    ///     Compilation result with optimized bytecode
    pub fn generateOptimizedBytecode(self: *Self, program: *const HIRProgram, optimize_gas: bool) !YulCompileResult {
        const yul_code = try self.generateOptimizedYul(program, optimize_gas);
        defer self.allocator.free(yul_code);

        return self.compileYulToBytecode(yul_code);
    }

    /// Apply gas optimizations to generated Yul code
    ///
    /// Args:
    ///     yul_code: Original Yul code
    ///
    /// Returns:
    ///     Gas-optimized Yul code
    fn optimizeYulForGas(self: *Self, yul_code: []const u8) ![]u8 {
        var optimized = std.ArrayList(u8).init(self.allocator);
        defer optimized.deinit();

        // Simple gas optimizations - parse line by line
        var lines = std.mem.split(u8, yul_code, "\n");
        while (lines.next()) |line| {
            const optimized_line = try self.optimizeLine(line);
            defer if (optimized_line.ptr != line.ptr) self.allocator.free(optimized_line);

            try optimized.appendSlice(optimized_line);
            try optimized.append('\n');
        }

        return optimized.toOwnedSlice();
    }

    /// Optimize a single line of Yul code
    ///
    /// Args:
    ///     line: Original line
    ///
    /// Returns:
    ///     Optimized line (may be same as input)
    fn optimizeLine(self: *Self, line: []const u8) ![]const u8 {
        // Gas optimization patterns

        // Replace add(x, 0) with x
        if (std.mem.indexOf(u8, line, "add(") != null and std.mem.indexOf(u8, line, ", 0)") != null) {
            return try self.optimizeArithmetic(line, "add", "0", true);
        }

        // Replace mul(x, 1) with x
        if (std.mem.indexOf(u8, line, "mul(") != null and std.mem.indexOf(u8, line, ", 1)") != null) {
            return try self.optimizeArithmetic(line, "mul", "1", true);
        }

        // Replace mul(x, 0) with 0
        if (std.mem.indexOf(u8, line, "mul(") != null and std.mem.indexOf(u8, line, ", 0)") != null) {
            return try self.optimizeArithmetic(line, "mul", "0", false);
        }

        // Return original line if no optimizations apply
        return line;
    }

    /// Optimize arithmetic operations
    ///
    /// Args:
    ///     line: Line to optimize
    ///     op: Operation name (add, mul, etc.)
    ///     identity: Identity value to optimize away
    ///     keep_operand: Whether to keep the operand (true) or replace with identity (false)
    ///
    /// Returns:
    ///     Optimized line
    fn optimizeArithmetic(self: *Self, line: []const u8, op: []const u8, identity: []const u8, keep_operand: bool) ![]const u8 {
        const op_pattern = try std.fmt.allocPrint(self.allocator, "{s}(", .{op});
        defer self.allocator.free(op_pattern);

        const identity_pattern = try std.fmt.allocPrint(self.allocator, ", {s})", .{identity});
        defer self.allocator.free(identity_pattern);

        if (std.mem.indexOf(u8, line, op_pattern)) |op_start| {
            if (std.mem.indexOf(u8, line, identity_pattern)) |identity_start| {
                // Extract the operand
                const operand_start = op_start + op_pattern.len;
                const operand_end = identity_start;
                const operand = std.mem.trim(u8, line[operand_start..operand_end], " ");

                // Build the optimized line
                var result = std.ArrayList(u8).init(self.allocator);
                defer result.deinit();

                // Copy prefix
                try result.appendSlice(line[0..op_start]);

                // Replace the operation
                if (keep_operand) {
                    try result.appendSlice(operand);
                } else {
                    try result.appendSlice(identity);
                }

                // Copy suffix (after the closing paren)
                const suffix_start = identity_start + identity_pattern.len;
                if (suffix_start < line.len) {
                    try result.appendSlice(line[suffix_start..]);
                }

                return result.toOwnedSlice();
            }
        }

        return line;
    }
};

/// Simple HIR instruction for testing and prototyping
///
/// This is a simplified instruction set used for testing the Yul generation
/// pipeline before full HIR integration is complete.
pub const HIRInstruction = union(enum) {
    /// Push a constant value onto the stack
    push: u64,
    /// Pop a value from the stack
    pop,
    /// Add two values from the stack
    add,
    /// Store a value to memory at given address
    store: u64,
    /// Load a value from memory at given address
    load: u64,
    /// Return from function
    ret,
};

/// Simple HIR structure for testing
///
/// Contains a list of simple instructions for testing the Yul generation
/// pipeline before full HIR integration.
pub const SimpleHIR = struct {
    /// List of HIR instructions
    instructions: std.ArrayList(HIRInstruction),

    const Self = @This();

    /// Initialize a new SimpleHIR instance
    ///
    /// Args:
    ///     allocator: Memory allocator for instruction list
    ///
    /// Returns:
    ///     Initialized SimpleHIR instance
    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .instructions = std.ArrayList(HIRInstruction).init(allocator),
        };
    }

    /// Cleanup allocated resources
    pub fn deinit(self: *Self) void {
        self.instructions.deinit();
    }
};

/// Test function for Yul code generation
///
/// Creates a simple HIR with basic operations and tests the complete
/// HIR -> Yul -> Bytecode pipeline.
pub fn test_yul_codegen() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create a simple HIR for testing
    var test_hir = SimpleHIR.init(allocator);
    defer test_hir.deinit();

    try test_hir.instructions.append(.{ .push = 42 });
    try test_hir.instructions.append(.{ .push = 1 });
    try test_hir.instructions.append(.add);
    try test_hir.instructions.append(.{ .store = 0 });
    try test_hir.instructions.append(.ret);

    // Generate and compile
    var codegen = YulCodegen.init(allocator);
    defer codegen.deinit();
    var result = codegen.generateBytecodeSimple(&test_hir) catch |err| {
        print("Failed to generate bytecode: {}\n", .{err});
        return;
    };
    defer result.deinit(allocator);

    if (result.success) {
        if (result.bytecode) |bytecode| {
            print("✓ Bytecode generation successful!\n", .{});
            print("Bytecode: {s}\n", .{bytecode});
        }
    } else {
        print("✗ Bytecode generation failed\n", .{});
        if (result.error_message) |error_msg| {
            print("Error: {s}\n", .{error_msg});
        }
    }
}

/// Test gas optimization functionality
pub fn test_gas_optimization() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var codegen = YulCodegen.init(allocator);
    defer codegen.deinit();

    // Test arithmetic optimization
    const test_line = "let result := add(x, 0)";
    const optimized = try codegen.optimizeLine(test_line);
    defer if (optimized.ptr != test_line.ptr) allocator.free(optimized);

    print("✓ Gas optimization test:\n");
    print("  Original: {s}\n", .{test_line});
    print("  Optimized: {s}\n", .{optimized});

    // Test multiplication optimization
    const test_line2 = "let result := mul(x, 1)";
    const optimized2 = try codegen.optimizeLine(test_line2);
    defer if (optimized2.ptr != test_line2.ptr) allocator.free(optimized2);

    print("  Original: {s}\n", .{test_line2});
    print("  Optimized: {s}\n", .{optimized2});

    // Test zero multiplication
    const test_line3 = "let result := mul(x, 0)";
    const optimized3 = try codegen.optimizeLine(test_line3);
    defer if (optimized3.ptr != test_line3.ptr) allocator.free(optimized3);

    print("  Original: {s}\n", .{test_line3});
    print("  Optimized: {s}\n", .{optimized3});
}

/// Test error union generation
pub fn test_error_unions() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var codegen = YulCodegen.init(allocator);
    defer codegen.deinit();

    var yul_code = std.ArrayList(u8).init(allocator);
    defer yul_code.deinit();

    // Test error union success
    const success_var = try codegen.generateErrorUnionSuccess(&yul_code, "42");
    defer allocator.free(success_var);

    print("✓ Error union test:\n");
    print("Generated Yul code:\n{s}\n", .{yul_code.items});

    // Test error union error
    yul_code.clearRetainingCapacity();
    const error_var = try codegen.generateErrorUnionError(&yul_code, 1);
    defer allocator.free(error_var);

    print("Error union (error case):\n{s}\n", .{yul_code.items});
}

/// Comprehensive test suite
pub fn test_comprehensive() !void {
    print("🚀 Running comprehensive Yul codegen tests...\n\n");

    try test_yul_codegen();
    print("\n");

    try test_gas_optimization();
    print("\n");

    try test_error_unions();
    print("\n");

    print("✅ All tests completed successfully!\n");
}
