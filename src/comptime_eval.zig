const std = @import("std");
const ast = @import("ast.zig");
const Allocator = std.mem.Allocator;

/// Errors that can occur during compile-time evaluation
pub const ComptimeError = error{
    NotCompileTimeEvaluable,
    DivisionByZero,
    IntegerOverflow,
    IntegerUnderflow,
    TypeMismatch,
    UndefinedVariable,
    InvalidOperation,
    InvalidLiteral,
    OutOfMemory,
    ConstantTooLarge,
    UnsupportedOperation,
};

/// Values that can be computed at compile time
pub const ComptimeValue = union(enum) {
    // Primitive values
    bool: bool,
    u8: u8,
    u16: u16,
    u32: u32,
    u64: u64,
    u128: u128,
    u256: [32]u8, // Store u256 as byte array
    string: []const u8,
    address: [20]u8, // Ethereum address

    // Special values
    undefined_value: void,

    /// Convert to string representation for debugging
    pub fn toString(self: ComptimeValue, allocator: Allocator) ![]const u8 {
        return switch (self) {
            .bool => |b| if (b) "true" else "false",
            .u8 => |v| try std.fmt.allocPrint(allocator, "{}", .{v}),
            .u16 => |v| try std.fmt.allocPrint(allocator, "{}", .{v}),
            .u32 => |v| try std.fmt.allocPrint(allocator, "{}", .{v}),
            .u64 => |v| try std.fmt.allocPrint(allocator, "{}", .{v}),
            .u128 => |v| try std.fmt.allocPrint(allocator, "{}", .{v}),
            .u256 => |bytes| try std.fmt.allocPrint(allocator, "0x{}", .{std.fmt.fmtSliceHexUpper(&bytes)}),
            .string => |s| try allocator.dupe(u8, s),
            .address => |addr| try std.fmt.allocPrint(allocator, "0x{}", .{std.fmt.fmtSliceHexUpper(&addr)}),
            .undefined_value => "undefined",
        };
    }

    /// Check if two comptime values are equal
    pub fn equals(self: ComptimeValue, other: ComptimeValue) bool {
        return switch (self) {
            .bool => |a| switch (other) {
                .bool => |b| a == b,
                else => false,
            },
            .u8 => |a| switch (other) {
                .u8 => |b| a == b,
                else => false,
            },
            .u16 => |a| switch (other) {
                .u16 => |b| a == b,
                else => false,
            },
            .u32 => |a| switch (other) {
                .u32 => |b| a == b,
                else => false,
            },
            .u64 => |a| switch (other) {
                .u64 => |b| a == b,
                else => false,
            },
            .u128 => |a| switch (other) {
                .u128 => |b| a == b,
                else => false,
            },
            .u256 => |a| switch (other) {
                .u256 => |b| std.mem.eql(u8, &a, &b),
                else => false,
            },
            .string => |a| switch (other) {
                .string => |b| std.mem.eql(u8, a, b),
                else => false,
            },
            .address => |a| switch (other) {
                .address => |b| std.mem.eql(u8, &a, &b),
                else => false,
            },
            .undefined_value => switch (other) {
                .undefined_value => true,
                else => false,
            },
        };
    }
};

/// Symbol table for compile-time constants
const ComptimeSymbolTable = struct {
    symbols: std.HashMap([]const u8, ComptimeValue, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),
    allocator: Allocator,

    pub fn init(allocator: Allocator) ComptimeSymbolTable {
        return ComptimeSymbolTable{
            .symbols = std.HashMap([]const u8, ComptimeValue, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ComptimeSymbolTable) void {
        self.symbols.deinit();
    }

    pub fn define(self: *ComptimeSymbolTable, name: []const u8, value: ComptimeValue) !void {
        try self.symbols.put(name, value);
    }

    pub fn lookup(self: *ComptimeSymbolTable, name: []const u8) ?ComptimeValue {
        return self.symbols.get(name);
    }
};

/// Compile-time evaluator for Ora expressions
pub const ComptimeEvaluator = struct {
    allocator: Allocator,
    symbol_table: ComptimeSymbolTable,

    pub fn init(allocator: Allocator) ComptimeEvaluator {
        return ComptimeEvaluator{
            .allocator = allocator,
            .symbol_table = ComptimeSymbolTable.init(allocator),
        };
    }

    pub fn deinit(self: *ComptimeEvaluator) void {
        self.symbol_table.deinit();
    }

    /// Evaluate an expression at compile time
    pub fn evaluate(self: *ComptimeEvaluator, expr: *ast.ExprNode) ComptimeError!ComptimeValue {
        return switch (expr.*) {
            .Literal => |*lit| self.evaluateLiteral(lit),
            .Identifier => |*ident| self.evaluateIdentifier(ident),
            .Binary => |*bin| self.evaluateBinary(bin),
            .Unary => |*unary| self.evaluateUnary(unary),
            .Comptime => |*comp| self.evaluateComptimeBlock(comp),
            else => ComptimeError.NotCompileTimeEvaluable,
        };
    }

    /// Evaluate a literal value
    fn evaluateLiteral(self: *ComptimeEvaluator, literal: *ast.LiteralNode) ComptimeError!ComptimeValue {
        return switch (literal.*) {
            .Integer => |*int| self.parseIntegerLiteral(int.value),
            .String => |*str| ComptimeValue{ .string = str.value },
            .Bool => |*b| ComptimeValue{ .bool = b.value },
            .Address => |*addr| self.parseAddressLiteral(addr.value),
            .Hex => |*hex| self.parseHexLiteral(hex.value),
        };
    }

    /// Parse integer literal to appropriate type
    fn parseIntegerLiteral(self: *ComptimeEvaluator, value_str: []const u8) ComptimeError!ComptimeValue {
        _ = self;

        // Try parsing as different integer types, starting with smallest
        if (std.fmt.parseInt(u8, value_str, 10)) |val| {
            return ComptimeValue{ .u8 = val };
        } else |_| {}

        if (std.fmt.parseInt(u16, value_str, 10)) |val| {
            return ComptimeValue{ .u16 = val };
        } else |_| {}

        if (std.fmt.parseInt(u32, value_str, 10)) |val| {
            return ComptimeValue{ .u32 = val };
        } else |_| {}

        if (std.fmt.parseInt(u64, value_str, 10)) |val| {
            return ComptimeValue{ .u64 = val };
        } else |_| {}

        if (std.fmt.parseInt(u128, value_str, 10)) |val| {
            return ComptimeValue{ .u128 = val };
        } else |_| {}

        // Default to u256 for very large numbers
        // For now, just use the first 32 bytes of a hash of the string
        // In a real implementation, you'd want proper big integer parsing
        var bytes: [32]u8 = [_]u8{0} ** 32;
        const hash = std.hash.CityHash32.hash(value_str);
        std.mem.writeInt(u32, bytes[28..32], hash, .big);
        return ComptimeValue{ .u256 = bytes };
    }

    /// Parse address literal
    fn parseAddressLiteral(self: *ComptimeEvaluator, value_str: []const u8) ComptimeError!ComptimeValue {
        _ = self;

        var addr: [20]u8 = [_]u8{0} ** 20;

        // Expect "0x" prefix for addresses
        if (!std.mem.startsWith(u8, value_str, "0x")) {
            return ComptimeError.InvalidOperation;
        }

        const hex_part = value_str[2..];
        if (hex_part.len != 40) { // 20 bytes = 40 hex chars
            return ComptimeError.InvalidOperation;
        }

        // Parse hex string to bytes
        for (0..20) |i| {
            const hex_byte = hex_part[i * 2 .. i * 2 + 2];
            addr[i] = std.fmt.parseInt(u8, hex_byte, 16) catch return ComptimeError.InvalidOperation;
        }

        return ComptimeValue{ .address = addr };
    }

    /// Parse hex literal
    fn parseHexLiteral(self: *ComptimeEvaluator, value_str: []const u8) ComptimeError!ComptimeValue {
        _ = self;

        // For now, treat hex literals as u256
        var bytes: [32]u8 = [_]u8{0} ** 32;

        if (!std.mem.startsWith(u8, value_str, "0x")) {
            return ComptimeError.InvalidOperation;
        }

        const hex_part = value_str[2..];
        const max_chars = @min(hex_part.len, 64); // Max 32 bytes = 64 hex chars

        // Parse from right to left to handle variable length hex strings
        var byte_idx: i32 = 31;
        var char_idx: i32 = @intCast(max_chars);

        while (char_idx >= 2 and byte_idx >= 0) {
            const hex_byte = hex_part[@intCast(char_idx - 2)..@intCast(char_idx)];
            bytes[@intCast(byte_idx)] = std.fmt.parseInt(u8, hex_byte, 16) catch return ComptimeError.InvalidOperation;
            char_idx -= 2;
            byte_idx -= 1;
        }

        return ComptimeValue{ .u256 = bytes };
    }

    /// Evaluate identifier (const variable lookup)
    fn evaluateIdentifier(self: *ComptimeEvaluator, ident: *ast.IdentifierExpr) ComptimeError!ComptimeValue {
        if (self.symbol_table.lookup(ident.name)) |value| {
            return value;
        }
        return ComptimeError.UndefinedVariable;
    }

    /// Evaluate binary operation
    fn evaluateBinary(self: *ComptimeEvaluator, binary: *ast.BinaryExpr) ComptimeError!ComptimeValue {
        const left = try self.evaluate(binary.lhs);
        const right = try self.evaluate(binary.rhs);

        return switch (binary.operator) {
            .Plus => self.add(left, right),
            .Minus => self.subtract(left, right),
            .Star => self.multiply(left, right),
            .Slash => self.divide(left, right),
            .Percent => self.modulo(left, right),
            .EqualEqual => ComptimeValue{ .bool = left.equals(right) },
            .BangEqual => ComptimeValue{ .bool = !left.equals(right) },
            .Less => self.compare(left, right, .less),
            .LessEqual => self.compare(left, right, .less_equal),
            .Greater => self.compare(left, right, .greater),
            .GreaterEqual => self.compare(left, right, .greater_equal),
            .And => self.logicalAnd(left, right),
            .Or => self.logicalOr(left, right),
            .BitAnd => self.bitwiseAnd(left, right),
            .BitOr => self.bitwiseOr(left, right),
            .BitXor => self.bitwiseXor(left, right),
            .ShiftLeft => self.shiftLeft(left, right),
            .ShiftRight => self.shiftRight(left, right),
        };
    }

    /// Evaluate unary operation
    fn evaluateUnary(self: *ComptimeEvaluator, unary: *ast.UnaryExpr) ComptimeError!ComptimeValue {
        const operand = try self.evaluate(unary.operand);

        return switch (unary.operator) {
            .Minus => self.negate(operand),
            .Bang => self.logicalNot(operand),
            .BitNot => self.bitwiseNot(operand),
        };
    }

    /// Evaluate comptime block
    fn evaluateComptimeBlock(self: *ComptimeEvaluator, comptime_block: *ast.ComptimeExpr) ComptimeError!ComptimeValue {
        // For now, just evaluate the last expression in the block
        // In a full implementation, you'd execute all statements sequentially
        if (comptime_block.block.statements.len == 0) {
            return ComptimeValue{ .undefined_value = {} };
        }

        // Find the last expression statement
        for (comptime_block.block.statements) |*stmt| {
            switch (stmt.*) {
                .Expr => |*expr| {
                    return self.evaluate(expr);
                },
                else => continue,
            }
        }

        return ComptimeValue{ .undefined_value = {} };
    }

    /// Define a compile-time constant
    pub fn defineConstant(self: *ComptimeEvaluator, name: []const u8, value: ComptimeValue) !void {
        try self.symbol_table.define(name, value);
    }

    /// Evaluate with optimization (constant folding, dead code elimination)
    pub fn evaluateOptimized(self: *ComptimeEvaluator, expr: *ast.ExprNode) ComptimeError!ComptimeValue {
        // First try simple evaluation
        if (self.evaluate(expr)) |result| {
            return result;
        } else |err| {
            // If evaluation fails, try optimization strategies
            return switch (err) {
                ComptimeError.NotCompileTimeEvaluable => self.tryConstantFolding(expr),
                else => err,
            };
        }
    }

    /// Try constant folding optimizations
    fn tryConstantFolding(self: *ComptimeEvaluator, expr: *ast.ExprNode) ComptimeError!ComptimeValue {
        return switch (expr.*) {
            .Binary => |*binary| {
                // Try to evaluate both sides
                const left_result = self.evaluate(binary.lhs) catch return ComptimeError.NotCompileTimeEvaluable;
                const right_result = self.evaluate(binary.rhs) catch return ComptimeError.NotCompileTimeEvaluable;

                // Apply constant folding optimizations
                return switch (binary.operator) {
                    .Plus => {
                        // 0 + x = x, x + 0 = x
                        if (self.isZero(left_result)) return right_result;
                        if (self.isZero(right_result)) return left_result;
                        return self.add(left_result, right_result);
                    },
                    .Star => {
                        // 0 * x = 0, x * 0 = 0
                        if (self.isZero(left_result) or self.isZero(right_result)) {
                            return self.getZeroOfType(left_result);
                        }
                        // 1 * x = x, x * 1 = x
                        if (self.isOne(left_result)) return right_result;
                        if (self.isOne(right_result)) return left_result;
                        return self.multiply(left_result, right_result);
                    },
                    .And => {
                        // false && x = false, x && false = false
                        if (left_result == .bool and !left_result.bool) return left_result;
                        if (right_result == .bool and !right_result.bool) return right_result;
                        // true && x = x, x && true = x
                        if (left_result == .bool and left_result.bool) return right_result;
                        if (right_result == .bool and right_result.bool) return left_result;
                        return self.logicalAnd(left_result, right_result);
                    },
                    .Or => {
                        // true || x = true, x || true = true
                        if (left_result == .bool and left_result.bool) return left_result;
                        if (right_result == .bool and right_result.bool) return right_result;
                        // false || x = x, x || false = x
                        if (left_result == .bool and !left_result.bool) return right_result;
                        if (right_result == .bool and !right_result.bool) return left_result;
                        return self.logicalOr(left_result, right_result);
                    },
                    else => return self.evaluateBinary(binary),
                };
            },
            else => ComptimeError.NotCompileTimeEvaluable,
        };
    }

    /// Check if a value is zero
    fn isZero(self: *ComptimeEvaluator, value: ComptimeValue) bool {
        return switch (value) {
            .u8 => |v| v == 0,
            .u16 => |v| v == 0,
            .u32 => |v| v == 0,
            .u64 => |v| v == 0,
            .u128 => |v| v == 0,
            .u256 => |v| self.isZeroU256(v),
            else => false,
        };
    }

    /// Check if a value is one
    fn isOne(self: *ComptimeEvaluator, value: ComptimeValue) bool {
        _ = self;
        return switch (value) {
            .u8 => |v| v == 1,
            .u16 => |v| v == 1,
            .u32 => |v| v == 1,
            .u64 => |v| v == 1,
            .u128 => |v| v == 1,
            .u256 => |v| {
                // Check if u256 equals 1
                for (v[0..31]) |byte| {
                    if (byte != 0) return false;
                }
                return v[31] == 1;
            },
            else => false,
        };
    }

    /// Get zero value of the same type
    fn getZeroOfType(self: *ComptimeEvaluator, value: ComptimeValue) ComptimeValue {
        _ = self;
        return switch (value) {
            .u8 => ComptimeValue{ .u8 = 0 },
            .u16 => ComptimeValue{ .u16 = 0 },
            .u32 => ComptimeValue{ .u32 = 0 },
            .u64 => ComptimeValue{ .u64 = 0 },
            .u128 => ComptimeValue{ .u128 = 0 },
            .u256 => ComptimeValue{ .u256 = [_]u8{0} ** 32 },
            .bool => ComptimeValue{ .bool = false },
            else => value, // Return original for non-numeric types
        };
    }

    /// Arithmetic operations with type promotion
    fn add(self: *ComptimeEvaluator, left: ComptimeValue, right: ComptimeValue) ComptimeError!ComptimeValue {
        // Try type promotion for mixed integer operations
        const promoted = try self.promoteTypes(left, right);

        return switch (promoted.left) {
            .u8 => |a| switch (promoted.right) {
                .u8 => |b| ComptimeValue{ .u8 = a +% b }, // Wrapping add to avoid overflow
                else => ComptimeError.TypeMismatch,
            },
            .u16 => |a| switch (promoted.right) {
                .u16 => |b| ComptimeValue{ .u16 = a +% b },
                else => ComptimeError.TypeMismatch,
            },
            .u32 => |a| switch (promoted.right) {
                .u32 => |b| ComptimeValue{ .u32 = a +% b },
                else => ComptimeError.TypeMismatch,
            },
            .u64 => |a| switch (promoted.right) {
                .u64 => |b| ComptimeValue{ .u64 = a +% b },
                else => ComptimeError.TypeMismatch,
            },
            .u128 => |a| switch (promoted.right) {
                .u128 => |b| ComptimeValue{ .u128 = a +% b },
                else => ComptimeError.TypeMismatch,
            },
            .u256 => |a| switch (promoted.right) {
                .u256 => |b| ComptimeValue{ .u256 = try self.addU256(a, b) },
                else => ComptimeError.TypeMismatch,
            },
            else => ComptimeError.InvalidOperation,
        };
    }

    fn subtract(self: *ComptimeEvaluator, left: ComptimeValue, right: ComptimeValue) ComptimeError!ComptimeValue {
        const promoted = try self.promoteTypes(left, right);
        return switch (promoted.left) {
            .u8 => |a| switch (promoted.right) {
                .u8 => |b| ComptimeValue{ .u8 = a -% b },
                else => ComptimeError.TypeMismatch,
            },
            .u16 => |a| switch (promoted.right) {
                .u16 => |b| ComptimeValue{ .u16 = a -% b },
                else => ComptimeError.TypeMismatch,
            },
            .u32 => |a| switch (promoted.right) {
                .u32 => |b| ComptimeValue{ .u32 = a -% b },
                else => ComptimeError.TypeMismatch,
            },
            .u64 => |a| switch (promoted.right) {
                .u64 => |b| ComptimeValue{ .u64 = a -% b },
                else => ComptimeError.TypeMismatch,
            },
            .u128 => |a| switch (promoted.right) {
                .u128 => |b| ComptimeValue{ .u128 = a -% b },
                else => ComptimeError.TypeMismatch,
            },
            .u256 => |a| switch (promoted.right) {
                .u256 => |b| ComptimeValue{ .u256 = try self.subtractU256(a, b) },
                else => ComptimeError.TypeMismatch,
            },
            else => ComptimeError.InvalidOperation,
        };
    }

    fn multiply(self: *ComptimeEvaluator, left: ComptimeValue, right: ComptimeValue) ComptimeError!ComptimeValue {
        const promoted = try self.promoteTypes(left, right);
        return switch (promoted.left) {
            .u8 => |a| switch (promoted.right) {
                .u8 => |b| ComptimeValue{ .u8 = a *% b },
                else => ComptimeError.TypeMismatch,
            },
            .u16 => |a| switch (promoted.right) {
                .u16 => |b| ComptimeValue{ .u16 = a *% b },
                else => ComptimeError.TypeMismatch,
            },
            .u32 => |a| switch (promoted.right) {
                .u32 => |b| ComptimeValue{ .u32 = a *% b },
                else => ComptimeError.TypeMismatch,
            },
            .u64 => |a| switch (promoted.right) {
                .u64 => |b| ComptimeValue{ .u64 = a *% b },
                else => ComptimeError.TypeMismatch,
            },
            .u128 => |a| switch (promoted.right) {
                .u128 => |b| ComptimeValue{ .u128 = a *% b },
                else => ComptimeError.TypeMismatch,
            },
            .u256 => |a| switch (promoted.right) {
                .u256 => |b| ComptimeValue{ .u256 = try self.multiplyU256(a, b) },
                else => ComptimeError.TypeMismatch,
            },
            else => ComptimeError.InvalidOperation,
        };
    }

    fn divide(self: *ComptimeEvaluator, left: ComptimeValue, right: ComptimeValue) ComptimeError!ComptimeValue {
        _ = self;
        return switch (left) {
            .u8 => |a| switch (right) {
                .u8 => |b| if (b == 0) ComptimeError.DivisionByZero else ComptimeValue{ .u8 = a / b },
                else => ComptimeError.TypeMismatch,
            },
            .u16 => |a| switch (right) {
                .u16 => |b| if (b == 0) ComptimeError.DivisionByZero else ComptimeValue{ .u16 = a / b },
                else => ComptimeError.TypeMismatch,
            },
            .u32 => |a| switch (right) {
                .u32 => |b| if (b == 0) ComptimeError.DivisionByZero else ComptimeValue{ .u32 = a / b },
                else => ComptimeError.TypeMismatch,
            },
            .u64 => |a| switch (right) {
                .u64 => |b| if (b == 0) ComptimeError.DivisionByZero else ComptimeValue{ .u64 = a / b },
                else => ComptimeError.TypeMismatch,
            },
            .u128 => |a| switch (right) {
                .u128 => |b| if (b == 0) ComptimeError.DivisionByZero else ComptimeValue{ .u128 = a / b },
                else => ComptimeError.TypeMismatch,
            },
            else => ComptimeError.InvalidOperation,
        };
    }

    fn modulo(self: *ComptimeEvaluator, left: ComptimeValue, right: ComptimeValue) ComptimeError!ComptimeValue {
        _ = self;
        return switch (left) {
            .u8 => |a| switch (right) {
                .u8 => |b| if (b == 0) ComptimeError.DivisionByZero else ComptimeValue{ .u8 = a % b },
                else => ComptimeError.TypeMismatch,
            },
            .u16 => |a| switch (right) {
                .u16 => |b| if (b == 0) ComptimeError.DivisionByZero else ComptimeValue{ .u16 = a % b },
                else => ComptimeError.TypeMismatch,
            },
            .u32 => |a| switch (right) {
                .u32 => |b| if (b == 0) ComptimeError.DivisionByZero else ComptimeValue{ .u32 = a % b },
                else => ComptimeError.TypeMismatch,
            },
            .u64 => |a| switch (right) {
                .u64 => |b| if (b == 0) ComptimeError.DivisionByZero else ComptimeValue{ .u64 = a % b },
                else => ComptimeError.TypeMismatch,
            },
            .u128 => |a| switch (right) {
                .u128 => |b| if (b == 0) ComptimeError.DivisionByZero else ComptimeValue{ .u128 = a % b },
                else => ComptimeError.TypeMismatch,
            },
            else => ComptimeError.InvalidOperation,
        };
    }

    /// Comparison operations
    fn compare(self: *ComptimeEvaluator, left: ComptimeValue, right: ComptimeValue, op: enum { less, less_equal, greater, greater_equal }) ComptimeError!ComptimeValue {
        _ = self;
        const result = switch (left) {
            .u8 => |a| switch (right) {
                .u8 => |b| switch (op) {
                    .less => a < b,
                    .less_equal => a <= b,
                    .greater => a > b,
                    .greater_equal => a >= b,
                },
                else => return ComptimeError.TypeMismatch,
            },
            .u16 => |a| switch (right) {
                .u16 => |b| switch (op) {
                    .less => a < b,
                    .less_equal => a <= b,
                    .greater => a > b,
                    .greater_equal => a >= b,
                },
                else => return ComptimeError.TypeMismatch,
            },
            .u32 => |a| switch (right) {
                .u32 => |b| switch (op) {
                    .less => a < b,
                    .less_equal => a <= b,
                    .greater => a > b,
                    .greater_equal => a >= b,
                },
                else => return ComptimeError.TypeMismatch,
            },
            .u64 => |a| switch (right) {
                .u64 => |b| switch (op) {
                    .less => a < b,
                    .less_equal => a <= b,
                    .greater => a > b,
                    .greater_equal => a >= b,
                },
                else => return ComptimeError.TypeMismatch,
            },
            .u128 => |a| switch (right) {
                .u128 => |b| switch (op) {
                    .less => a < b,
                    .less_equal => a <= b,
                    .greater => a > b,
                    .greater_equal => a >= b,
                },
                else => return ComptimeError.TypeMismatch,
            },
            else => return ComptimeError.InvalidOperation,
        };
        return ComptimeValue{ .bool = result };
    }

    /// Logical operations
    fn logicalAnd(self: *ComptimeEvaluator, left: ComptimeValue, right: ComptimeValue) ComptimeError!ComptimeValue {
        _ = self;
        return switch (left) {
            .bool => |a| switch (right) {
                .bool => |b| ComptimeValue{ .bool = a and b },
                else => ComptimeError.TypeMismatch,
            },
            else => ComptimeError.InvalidOperation,
        };
    }

    fn logicalOr(self: *ComptimeEvaluator, left: ComptimeValue, right: ComptimeValue) ComptimeError!ComptimeValue {
        _ = self;
        return switch (left) {
            .bool => |a| switch (right) {
                .bool => |b| ComptimeValue{ .bool = a or b },
                else => ComptimeError.TypeMismatch,
            },
            else => ComptimeError.InvalidOperation,
        };
    }

    fn logicalNot(self: *ComptimeEvaluator, operand: ComptimeValue) ComptimeError!ComptimeValue {
        _ = self;
        return switch (operand) {
            .bool => |a| ComptimeValue{ .bool = !a },
            else => ComptimeError.InvalidOperation,
        };
    }

    /// Bitwise operations
    fn bitwiseAnd(self: *ComptimeEvaluator, left: ComptimeValue, right: ComptimeValue) ComptimeError!ComptimeValue {
        _ = self;
        return switch (left) {
            .u8 => |a| switch (right) {
                .u8 => |b| ComptimeValue{ .u8 = a & b },
                else => ComptimeError.TypeMismatch,
            },
            .u16 => |a| switch (right) {
                .u16 => |b| ComptimeValue{ .u16 = a & b },
                else => ComptimeError.TypeMismatch,
            },
            .u32 => |a| switch (right) {
                .u32 => |b| ComptimeValue{ .u32 = a & b },
                else => ComptimeError.TypeMismatch,
            },
            .u64 => |a| switch (right) {
                .u64 => |b| ComptimeValue{ .u64 = a & b },
                else => ComptimeError.TypeMismatch,
            },
            .u128 => |a| switch (right) {
                .u128 => |b| ComptimeValue{ .u128 = a & b },
                else => ComptimeError.TypeMismatch,
            },
            else => ComptimeError.InvalidOperation,
        };
    }

    fn bitwiseOr(self: *ComptimeEvaluator, left: ComptimeValue, right: ComptimeValue) ComptimeError!ComptimeValue {
        _ = self;
        return switch (left) {
            .u8 => |a| switch (right) {
                .u8 => |b| ComptimeValue{ .u8 = a | b },
                else => ComptimeError.TypeMismatch,
            },
            .u16 => |a| switch (right) {
                .u16 => |b| ComptimeValue{ .u16 = a | b },
                else => ComptimeError.TypeMismatch,
            },
            .u32 => |a| switch (right) {
                .u32 => |b| ComptimeValue{ .u32 = a | b },
                else => ComptimeError.TypeMismatch,
            },
            .u64 => |a| switch (right) {
                .u64 => |b| ComptimeValue{ .u64 = a | b },
                else => ComptimeError.TypeMismatch,
            },
            .u128 => |a| switch (right) {
                .u128 => |b| ComptimeValue{ .u128 = a | b },
                else => ComptimeError.TypeMismatch,
            },
            else => ComptimeError.InvalidOperation,
        };
    }

    fn bitwiseXor(self: *ComptimeEvaluator, left: ComptimeValue, right: ComptimeValue) ComptimeError!ComptimeValue {
        _ = self;
        return switch (left) {
            .u8 => |a| switch (right) {
                .u8 => |b| ComptimeValue{ .u8 = a ^ b },
                else => ComptimeError.TypeMismatch,
            },
            .u16 => |a| switch (right) {
                .u16 => |b| ComptimeValue{ .u16 = a ^ b },
                else => ComptimeError.TypeMismatch,
            },
            .u32 => |a| switch (right) {
                .u32 => |b| ComptimeValue{ .u32 = a ^ b },
                else => ComptimeError.TypeMismatch,
            },
            .u64 => |a| switch (right) {
                .u64 => |b| ComptimeValue{ .u64 = a ^ b },
                else => ComptimeError.TypeMismatch,
            },
            .u128 => |a| switch (right) {
                .u128 => |b| ComptimeValue{ .u128 = a ^ b },
                else => ComptimeError.TypeMismatch,
            },
            else => ComptimeError.InvalidOperation,
        };
    }

    fn bitwiseNot(self: *ComptimeEvaluator, operand: ComptimeValue) ComptimeError!ComptimeValue {
        _ = self;
        return switch (operand) {
            .u8 => |a| ComptimeValue{ .u8 = ~a },
            .u16 => |a| ComptimeValue{ .u16 = ~a },
            .u32 => |a| ComptimeValue{ .u32 = ~a },
            .u64 => |a| ComptimeValue{ .u64 = ~a },
            .u128 => |a| ComptimeValue{ .u128 = ~a },
            else => ComptimeError.InvalidOperation,
        };
    }

    /// Shift operations
    fn shiftLeft(self: *ComptimeEvaluator, left: ComptimeValue, right: ComptimeValue) ComptimeError!ComptimeValue {
        _ = self;
        return switch (left) {
            .u8 => |a| switch (right) {
                .u8 => |b| ComptimeValue{ .u8 = a << @intCast(b) },
                else => ComptimeError.TypeMismatch,
            },
            .u16 => |a| switch (right) {
                .u16 => |b| ComptimeValue{ .u16 = a << @intCast(b) },
                else => ComptimeError.TypeMismatch,
            },
            .u32 => |a| switch (right) {
                .u32 => |b| ComptimeValue{ .u32 = a << @intCast(b) },
                else => ComptimeError.TypeMismatch,
            },
            .u64 => |a| switch (right) {
                .u64 => |b| ComptimeValue{ .u64 = a << @intCast(b) },
                else => ComptimeError.TypeMismatch,
            },
            .u128 => |a| switch (right) {
                .u128 => |b| ComptimeValue{ .u128 = a << @intCast(b) },
                else => ComptimeError.TypeMismatch,
            },
            else => ComptimeError.InvalidOperation,
        };
    }

    fn shiftRight(self: *ComptimeEvaluator, left: ComptimeValue, right: ComptimeValue) ComptimeError!ComptimeValue {
        _ = self;
        return switch (left) {
            .u8 => |a| switch (right) {
                .u8 => |b| ComptimeValue{ .u8 = a >> @intCast(b) },
                else => ComptimeError.TypeMismatch,
            },
            .u16 => |a| switch (right) {
                .u16 => |b| ComptimeValue{ .u16 = a >> @intCast(b) },
                else => ComptimeError.TypeMismatch,
            },
            .u32 => |a| switch (right) {
                .u32 => |b| ComptimeValue{ .u32 = a >> @intCast(b) },
                else => ComptimeError.TypeMismatch,
            },
            .u64 => |a| switch (right) {
                .u64 => |b| ComptimeValue{ .u64 = a >> @intCast(b) },
                else => ComptimeError.TypeMismatch,
            },
            .u128 => |a| switch (right) {
                .u128 => |b| ComptimeValue{ .u128 = a >> @intCast(b) },
                else => ComptimeError.TypeMismatch,
            },
            else => ComptimeError.InvalidOperation,
        };
    }

    fn negate(self: *ComptimeEvaluator, operand: ComptimeValue) ComptimeError!ComptimeValue {
        _ = self;
        return switch (operand) {
            .u8 => |a| ComptimeValue{ .u8 = -%a },
            .u16 => |a| ComptimeValue{ .u16 = -%a },
            .u32 => |a| ComptimeValue{ .u32 = -%a },
            .u64 => |a| ComptimeValue{ .u64 = -%a },
            .u128 => |a| ComptimeValue{ .u128 = -%a },
            else => ComptimeError.InvalidOperation,
        };
    }

    /// U256 arithmetic operations (simplified big integer operations)
    fn addU256(self: *ComptimeEvaluator, a: [32]u8, b: [32]u8) ComptimeError![32]u8 {
        _ = self;
        var result: [32]u8 = [_]u8{0} ** 32;
        var carry: u16 = 0;

        // Add from least significant byte
        var i: usize = 31;
        while (i < 32) {
            const sum = @as(u16, a[i]) + @as(u16, b[i]) + carry;
            result[i] = @truncate(sum);
            carry = sum >> 8;
            if (i == 0) break;
            i -= 1;
        }

        // Check for overflow
        if (carry != 0) {
            return ComptimeError.IntegerOverflow;
        }

        return result;
    }

    fn subtractU256(self: *ComptimeEvaluator, a: [32]u8, b: [32]u8) ComptimeError![32]u8 {
        _ = self;
        var result: [32]u8 = [_]u8{0} ** 32;
        var borrow: i16 = 0;

        // Subtract from least significant byte
        var i: usize = 31;
        while (i < 32) {
            const diff = @as(i16, a[i]) - @as(i16, b[i]) - borrow;
            if (diff < 0) {
                result[i] = @intCast(diff + 256);
                borrow = 1;
            } else {
                result[i] = @intCast(diff);
                borrow = 0;
            }
            if (i == 0) break;
            i -= 1;
        }

        // Check for underflow
        if (borrow != 0) {
            return ComptimeError.IntegerOverflow;
        }

        return result;
    }

    fn multiplyU256(self: *ComptimeEvaluator, a: [32]u8, b: [32]u8) ComptimeError![32]u8 {
        _ = self;
        // Simplified multiplication for compile-time evaluation
        // In practice, you'd want a more sophisticated algorithm
        var result: [32]u8 = [_]u8{0} ** 32;

        // Simple byte-by-byte multiplication (may overflow)
        var carry: u32 = 0;
        for (0..32) |i| {
            const prod = @as(u32, a[31 - i]) * @as(u32, b[31 - i]) + carry;
            result[31 - i] = @truncate(prod);
            carry = prod >> 8;
        }

        if (carry != 0) {
            return ComptimeError.IntegerOverflow;
        }

        return result;
    }

    fn isZeroU256(self: *ComptimeEvaluator, value: [32]u8) bool {
        _ = self;
        for (value) |byte| {
            if (byte != 0) return false;
        }
        return true;
    }

    /// Type promotion for mixed arithmetic operations
    fn promoteTypes(self: *ComptimeEvaluator, left: ComptimeValue, right: ComptimeValue) ComptimeError!struct { left: ComptimeValue, right: ComptimeValue } {
        // If types are the same, no promotion needed
        if (std.meta.activeTag(left) == std.meta.activeTag(right)) {
            return .{ .left = left, .right = right };
        }

        // Promote to larger integer type
        return switch (left) {
            .u8 => |a| switch (right) {
                .u16 => |_| .{ .left = ComptimeValue{ .u16 = a }, .right = right },
                .u32 => |_| .{ .left = ComptimeValue{ .u32 = a }, .right = right },
                .u64 => |_| .{ .left = ComptimeValue{ .u64 = a }, .right = right },
                .u128 => |_| .{ .left = ComptimeValue{ .u128 = a }, .right = right },
                .u256 => |_| .{ .left = self.promoteToU256(ComptimeValue{ .u8 = a }), .right = right },
                else => ComptimeError.TypeMismatch,
            },
            .u16 => |a| switch (right) {
                .u8 => |b| .{ .left = left, .right = ComptimeValue{ .u16 = b } },
                .u32 => |_| .{ .left = ComptimeValue{ .u32 = a }, .right = right },
                .u64 => |_| .{ .left = ComptimeValue{ .u64 = a }, .right = right },
                .u128 => |_| .{ .left = ComptimeValue{ .u128 = a }, .right = right },
                .u256 => |_| .{ .left = self.promoteToU256(ComptimeValue{ .u16 = a }), .right = right },
                else => ComptimeError.TypeMismatch,
            },
            .u32 => |a| switch (right) {
                .u8 => |b| .{ .left = left, .right = ComptimeValue{ .u32 = b } },
                .u16 => |b| .{ .left = left, .right = ComptimeValue{ .u32 = b } },
                .u64 => |_| .{ .left = ComptimeValue{ .u64 = a }, .right = right },
                .u128 => |_| .{ .left = ComptimeValue{ .u128 = a }, .right = right },
                .u256 => |_| .{ .left = self.promoteToU256(ComptimeValue{ .u32 = a }), .right = right },
                else => ComptimeError.TypeMismatch,
            },
            .u64 => |a| switch (right) {
                .u8 => |b| .{ .left = left, .right = ComptimeValue{ .u64 = b } },
                .u16 => |b| .{ .left = left, .right = ComptimeValue{ .u64 = b } },
                .u32 => |b| .{ .left = left, .right = ComptimeValue{ .u64 = b } },
                .u128 => |_| .{ .left = ComptimeValue{ .u128 = a }, .right = right },
                .u256 => |_| .{ .left = self.promoteToU256(ComptimeValue{ .u64 = a }), .right = right },
                else => ComptimeError.TypeMismatch,
            },
            .u128 => |a| switch (right) {
                .u8 => |b| .{ .left = left, .right = ComptimeValue{ .u128 = b } },
                .u16 => |b| .{ .left = left, .right = ComptimeValue{ .u128 = b } },
                .u32 => |b| .{ .left = left, .right = ComptimeValue{ .u128 = b } },
                .u64 => |b| .{ .left = left, .right = ComptimeValue{ .u128 = b } },
                .u256 => |_| .{ .left = self.promoteToU256(ComptimeValue{ .u128 = a }), .right = right },
                else => ComptimeError.TypeMismatch,
            },
            .u256 => |_| switch (right) {
                .u8, .u16, .u32, .u64, .u128 => .{ .left = left, .right = self.promoteToU256(right) },
                else => ComptimeError.TypeMismatch,
            },
            else => ComptimeError.TypeMismatch,
        };
    }

    /// Promote any integer value to U256
    fn promoteToU256(self: *ComptimeEvaluator, value: ComptimeValue) ComptimeValue {
        _ = self;
        var bytes: [32]u8 = [_]u8{0} ** 32;

        switch (value) {
            .u8 => |v| bytes[31] = v,
            .u16 => |v| std.mem.writeInt(u16, bytes[30..32], v, .big),
            .u32 => |v| std.mem.writeInt(u32, bytes[28..32], v, .big),
            .u64 => |v| std.mem.writeInt(u64, bytes[24..32], v, .big),
            .u128 => |v| std.mem.writeInt(u128, bytes[16..32], v, .big),
            .u256 => |v| return ComptimeValue{ .u256 = v },
            else => {}, // Should not happen for integer types
        }

        return ComptimeValue{ .u256 = bytes };
    }

    /// Validate compile-time constant definitions
    pub fn validateConstant(self: *ComptimeEvaluator, name: []const u8, value: ComptimeValue) ComptimeError!void {
        _ = self;

        // Validate constant name
        if (name.len == 0) {
            return ComptimeError.InvalidOperation;
        }

        // Validate value constraints
        switch (value) {
            .u256 => |bytes| {
                // Check if u256 value is too large for practical use
                var non_zero_count: u32 = 0;
                for (bytes) |byte| {
                    if (byte != 0) non_zero_count += 1;
                }
                if (non_zero_count > 16) { // More than half the bytes are non-zero
                    return ComptimeError.ConstantTooLarge;
                }
            },
            .string => |str| {
                if (str.len > 1024) { // Reasonable string length limit
                    return ComptimeError.ConstantTooLarge;
                }
            },
            else => {}, // Other types are generally safe
        }
    }

    /// Get detailed error information for diagnostics
    pub fn getErrorMessage(self: *ComptimeEvaluator, err: ComptimeError) []const u8 {
        _ = self;
        return switch (err) {
            ComptimeError.NotCompileTimeEvaluable => "Expression cannot be evaluated at compile time",
            ComptimeError.DivisionByZero => "Division by zero in compile-time expression",
            ComptimeError.IntegerOverflow => "Integer overflow in compile-time calculation",
            ComptimeError.IntegerUnderflow => "Integer underflow in compile-time calculation",
            ComptimeError.TypeMismatch => "Type mismatch in compile-time operation",
            ComptimeError.UndefinedVariable => "Undefined variable in compile-time expression",
            ComptimeError.InvalidOperation => "Invalid operation for compile-time evaluation",
            ComptimeError.InvalidLiteral => "Invalid literal format in compile-time expression",
            ComptimeError.OutOfMemory => "Out of memory during compile-time evaluation",
            ComptimeError.ConstantTooLarge => "Compile-time constant value is too large",
            ComptimeError.UnsupportedOperation => "Operation not supported in compile-time context",
        };
    }

    /// Check if an expression is potentially compile-time evaluable
    pub fn isComptimeEvaluable(self: *ComptimeEvaluator, expr: *ast.ExprNode) bool {
        return switch (expr.*) {
            .Literal => true,
            .Identifier => |*ident| self.symbol_table.lookup(ident.name) != null,
            .Binary => |*binary| {
                return self.isComptimeEvaluable(binary.lhs) and
                    self.isComptimeEvaluable(binary.rhs) and
                    self.isBinaryOpComptimeEvaluable(binary.operator);
            },
            .Unary => |*unary| {
                return self.isComptimeEvaluable(unary.operand) and
                    self.isUnaryOpComptimeEvaluable(unary.operator);
            },
            .Comptime => true, // Comptime blocks are always evaluable
            else => false,
        };
    }

    fn isBinaryOpComptimeEvaluable(self: *ComptimeEvaluator, op: ast.BinaryOp) bool {
        _ = self;
        return switch (op) {
            .Plus, .Minus, .Star, .Slash, .Percent => true,
            .EqualEqual, .BangEqual, .Less, .LessEqual, .Greater, .GreaterEqual => true,
            .And, .Or => true,
            .BitAnd, .BitOr, .BitXor, .ShiftLeft, .ShiftRight => true,
        };
    }

    fn isUnaryOpComptimeEvaluable(self: *ComptimeEvaluator, op: ast.UnaryOp) bool {
        _ = self;
        return switch (op) {
            .Minus, .Bang, .BitNot => true,
        };
    }
};

// Tests
const testing = std.testing;

test "comptime evaluator basic literals" {
    var evaluator = ComptimeEvaluator.init(testing.allocator);
    defer evaluator.deinit();

    // Test integer literal
    var int_literal = ast.LiteralNode{ .Integer = ast.IntegerLiteral{ .value = "42", .span = ast.SourceSpan{ .line = 1, .column = 1, .length = 2 } } };
    const int_result = try evaluator.evaluateLiteral(&int_literal);
    try testing.expect(int_result == .u8);
    try testing.expectEqual(@as(u8, 42), int_result.u8);

    // Test boolean literal
    var bool_literal = ast.LiteralNode{ .Bool = ast.BoolLiteral{ .value = true, .span = ast.SourceSpan{ .line = 1, .column = 1, .length = 4 } } };
    const bool_result = try evaluator.evaluateLiteral(&bool_literal);
    try testing.expect(bool_result == .bool);
    try testing.expectEqual(true, bool_result.bool);

    // Test string literal
    var str_literal = ast.LiteralNode{ .String = ast.StringLiteral{ .value = "hello", .span = ast.SourceSpan{ .line = 1, .column = 1, .length = 7 } } };
    const str_result = try evaluator.evaluateLiteral(&str_literal);
    try testing.expect(str_result == .string);
    try testing.expectEqualStrings("hello", str_result.string);
}

test "comptime evaluator arithmetic" {
    var evaluator = ComptimeEvaluator.init(testing.allocator);
    defer evaluator.deinit();

    const a = ComptimeValue{ .u32 = 10 };
    const b = ComptimeValue{ .u32 = 5 };

    // Test addition
    const add_result = try evaluator.add(a, b);
    try testing.expectEqual(@as(u32, 15), add_result.u32);

    // Test subtraction
    const sub_result = try evaluator.subtract(a, b);
    try testing.expectEqual(@as(u32, 5), sub_result.u32);

    // Test multiplication
    const mul_result = try evaluator.multiply(a, b);
    try testing.expectEqual(@as(u32, 50), mul_result.u32);

    // Test division
    const div_result = try evaluator.divide(a, b);
    try testing.expectEqual(@as(u32, 2), div_result.u32);

    // Test modulo
    const mod_result = try evaluator.modulo(a, b);
    try testing.expectEqual(@as(u32, 0), mod_result.u32);
}

test "comptime evaluator comparison" {
    var evaluator = ComptimeEvaluator.init(testing.allocator);
    defer evaluator.deinit();

    const a = ComptimeValue{ .u32 = 10 };
    const b = ComptimeValue{ .u32 = 5 };

    // Test comparisons
    const less_result = try evaluator.compare(b, a, .less);
    try testing.expectEqual(true, less_result.bool);

    const greater_result = try evaluator.compare(a, b, .greater);
    try testing.expectEqual(true, greater_result.bool);

    const equal_result = a.equals(a);
    try testing.expectEqual(true, equal_result);
}

test "comptime evaluator constants" {
    var evaluator = ComptimeEvaluator.init(testing.allocator);
    defer evaluator.deinit();

    // Define a constant
    try evaluator.defineConstant("MAX_SUPPLY", ComptimeValue{ .u64 = 1000000 });

    // Look it up
    const value = evaluator.symbol_table.lookup("MAX_SUPPLY");
    try testing.expect(value != null);
    try testing.expectEqual(@as(u64, 1000000), value.?.u64);
}

test "comptime evaluator type promotion" {
    var evaluator = ComptimeEvaluator.init(testing.allocator);
    defer evaluator.deinit();

    const a = ComptimeValue{ .u8 = 10 };
    const b = ComptimeValue{ .u32 = 5 };

    // Test type promotion in addition
    const result = try evaluator.add(a, b);
    try testing.expect(result == .u32);
    try testing.expectEqual(@as(u32, 15), result.u32);
}

test "comptime evaluator optimization" {
    var evaluator = ComptimeEvaluator.init(testing.allocator);
    defer evaluator.deinit();

    // Test zero optimization
    try testing.expect(evaluator.isZero(ComptimeValue{ .u32 = 0 }));
    try testing.expect(!evaluator.isZero(ComptimeValue{ .u32 = 1 }));

    // Test one optimization
    try testing.expect(evaluator.isOne(ComptimeValue{ .u32 = 1 }));
    try testing.expect(!evaluator.isOne(ComptimeValue{ .u32 = 0 }));
}

test "comptime evaluator validation" {
    var evaluator = ComptimeEvaluator.init(testing.allocator);
    defer evaluator.deinit();

    // Test valid constant
    try evaluator.validateConstant("VALID_CONST", ComptimeValue{ .u64 = 1000 });

    // Test invalid constant name
    try testing.expectError(ComptimeError.InvalidOperation, evaluator.validateConstant("", ComptimeValue{ .u64 = 1000 }));

    // Test string too large
    const large_string = "a" ** 2000;
    try testing.expectError(ComptimeError.ConstantTooLarge, evaluator.validateConstant("LARGE", ComptimeValue{ .string = large_string }));
}
