// ============================================================================
// Ora Language - Built-in Standard Library Registry
// ============================================================================
//
// This module provides the registry for Ora's built-in standard library
// functions and constants. These are compiler-provided and require no imports.
//
// Design:
// - `std` is a built-in namespace (no file, no import)
// - Users write: std.block.timestamp(), std.msg.sender(), etc.
// - Compiler validates against this registry at semantic analysis
// - MLIR lowers to ora.evm.* operations
// - Yul maps to EVM opcodes
//
// Scope (Phase 1 - Bare Minimum):
// - Block data (timestamp, number, chainid, etc.)
// - Transaction/message data (sender, caller, value, etc.)
// - Constants (ZERO_ADDRESS, U256_MAX, etc.)
//
// ============================================================================

const std = @import("std");
const ast = @import("../ast.zig");
const lib = @import("../ast/type_info.zig");

/// Information about a built-in function or constant
pub const BuiltinInfo = struct {
    /// Full qualified path (e.g., "std.block.timestamp")
    full_path: []const u8,

    /// Return type of the builtin
    return_type: lib.OraType,

    /// Parameter types (empty for constants and no-arg functions)
    param_types: []const lib.OraType,

    /// EVM opcode name (e.g., "TIMESTAMP", "ORIGIN", "CALLER")
    evm_opcode: []const u8,

    /// Whether this is a function call (true) or constant (false)
    is_call: bool,

    /// Gas cost (if known, for future optimization)
    gas_cost: ?u64,
};

/// Registry of all built-in functions and constants
pub const BuiltinRegistry = struct {
    builtins: std.StringHashMap(BuiltinInfo),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !BuiltinRegistry {
        var registry = BuiltinRegistry{
            .builtins = std.StringHashMap(BuiltinInfo).init(allocator),
            .allocator = allocator,
        };

        try registry.registerAllBuiltins();
        return registry;
    }

    pub fn deinit(self: *BuiltinRegistry) void {
        self.builtins.deinit();
    }

    /// Register all built-in functions and constants
    fn registerAllBuiltins(self: *BuiltinRegistry) !void {
        // ===================================================================
        // block DATA
        // ===================================================================

        try self.register(.{
            .full_path = "std.block.timestamp",
            .return_type = .u256,
            .param_types = &.{},
            .evm_opcode = "timestamp",
            .is_call = true,
            .gas_cost = 2,
        });

        try self.register(.{
            .full_path = "std.block.number",
            .return_type = .u256,
            .param_types = &.{},
            .evm_opcode = "number",
            .is_call = true,
            .gas_cost = 2,
        });

        try self.register(.{
            .full_path = "std.block.coinbase",
            .return_type = .address,
            .param_types = &.{},
            .evm_opcode = "coinbase",
            .is_call = true,
            .gas_cost = 2,
        });

        try self.register(.{
            .full_path = "std.block.difficulty",
            .return_type = .u256,
            .param_types = &.{},
            .evm_opcode = "prevrandao", // Post-merge: DIFFICULTY is PREVRANDAO
            .is_call = true,
            .gas_cost = 2,
        });

        try self.register(.{
            .full_path = "std.block.gaslimit",
            .return_type = .u256,
            .param_types = &.{},
            .evm_opcode = "gaslimit",
            .is_call = true,
            .gas_cost = 2,
        });

        try self.register(.{
            .full_path = "std.block.chainid",
            .return_type = .u256,
            .param_types = &.{},
            .evm_opcode = "chainid",
            .is_call = true,
            .gas_cost = 2,
        });

        try self.register(.{
            .full_path = "std.block.basefee",
            .return_type = .u256,
            .param_types = &.{},
            .evm_opcode = "basefee",
            .is_call = true,
            .gas_cost = 2,
        });

        // ===================================================================
        // transaction DATA (tx.*)
        // ===================================================================

        try self.register(.{
            .full_path = "std.transaction.sender",
            .return_type = .non_zero_address,
            .param_types = &.{},
            .evm_opcode = "origin",
            .is_call = true,
            .gas_cost = 2,
        });

        try self.register(.{
            .full_path = "std.transaction.gasprice",
            .return_type = .u256,
            .param_types = &.{},
            .evm_opcode = "gasprice",
            .is_call = true,
            .gas_cost = 2,
        });

        // ===================================================================
        // message DATA (msg.*)
        // ===================================================================

        try self.register(.{
            .full_path = "std.msg.sender",
            .return_type = .non_zero_address,
            .param_types = &.{},
            .evm_opcode = "caller",
            .is_call = true,
            .gas_cost = 2,
        });

        try self.register(.{
            .full_path = "std.msg.value",
            .return_type = .u256,
            .param_types = &.{},
            .evm_opcode = "callvalue",
            .is_call = true,
            .gas_cost = 2,
        });

        try self.register(.{
            .full_path = "std.msg.gas",
            .return_type = .u256,
            .param_types = &.{},
            .evm_opcode = "gas",
            .is_call = true,
            .gas_cost = 2,
        });

        // ===================================================================
        // constants
        // ===================================================================

        try self.register(.{
            .full_path = "std.constants.ZERO_ADDRESS",
            .return_type = .address,
            .param_types = &.{},
            .evm_opcode = "ZERO_ADDR", // Special: constant 0
            .is_call = false,
            .gas_cost = null,
        });

        try self.register(.{
            .full_path = "std.constants.U256_MAX",
            .return_type = .u256,
            .param_types = &.{},
            .evm_opcode = "MAX_U256", // Special: constant 2^256-1
            .is_call = false,
            .gas_cost = null,
        });

        try self.register(.{
            .full_path = "std.constants.U128_MAX",
            .return_type = .u128,
            .param_types = &.{},
            .evm_opcode = "MAX_U128",
            .is_call = false,
            .gas_cost = null,
        });

        try self.register(.{
            .full_path = "std.constants.U64_MAX",
            .return_type = .u64,
            .param_types = &.{},
            .evm_opcode = "MAX_U64",
            .is_call = false,
            .gas_cost = null,
        });

        try self.register(.{
            .full_path = "std.constants.U32_MAX",
            .return_type = .u32,
            .param_types = &.{},
            .evm_opcode = "MAX_U32",
            .is_call = false,
            .gas_cost = null,
        });
    }

    /// Register a single builtin
    fn register(self: *BuiltinRegistry, info: BuiltinInfo) !void {
        try self.builtins.put(info.full_path, info);
    }

    /// Look up a builtin by full path
    pub fn lookup(self: *const BuiltinRegistry, path: []const u8) ?BuiltinInfo {
        return self.builtins.get(path);
    }

    /// Check if a path is a known builtin
    pub fn isBuiltin(self: *const BuiltinRegistry, path: []const u8) bool {
        return self.builtins.contains(path);
    }

    /// Check if a path starts with "std."
    pub fn isStdNamespace(path: []const u8) bool {
        return std.mem.startsWith(u8, path, "std.");
    }
};

/// Helper to build a full path from member access chain
/// e.g., buildPath(allocator, "std", &.{"block", "timestamp"}) -> "std.block.timestamp"
pub fn buildPath(allocator: std.mem.Allocator, base: []const u8, members: []const []const u8) ![]const u8 {
    var result = std.ArrayList(u8){};
    try result.appendSlice(allocator, base);

    for (members) |member| {
        try result.append(allocator, '.');
        try result.appendSlice(allocator, member);
    }

    return result.toOwnedSlice(allocator);
}

/// Extract the full path from a member access expression
/// e.g., std.block.timestamp() -> "std.block.timestamp"
pub fn getMemberAccessPath(allocator: std.mem.Allocator, expr: *const ast.Expressions.ExprNode) ![]const u8 {
    var parts = std.ArrayList([]const u8){};
    defer parts.deinit(allocator);

    var current = expr;
    while (true) {
        switch (current.*) {
            .FieldAccess => |fa| {
                try parts.insert(allocator, 0, fa.field);
                current = fa.target;
            },
            .Identifier => |id| {
                try parts.insert(allocator, 0, id.name);
                break;
            },
            else => break,
        }
    }

    // join with dots
    return std.mem.join(allocator, ".", parts.items);
}

/// Check if an expression is a member access chain
pub fn isMemberAccessChain(expr: *const ast.Expressions.ExprNode) bool {
    return switch (expr.*) {
        .FieldAccess => true,
        else => false,
    };
}
