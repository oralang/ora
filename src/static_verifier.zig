const std = @import("std");
const ast = @import("ast.zig");
const comptime_eval = @import("comptime_eval.zig");
const Allocator = std.mem.Allocator;

/// Errors that can occur during static verification
pub const VerificationError = error{
    ContractViolation,
    UnsatisfiableCondition,
    InvalidOldExpression,
    UnknownSymbol,
    TypeMismatch,
    IncompatibleConstraints,
    OutOfMemory,
};

/// Represents a verification condition
pub const VerificationCondition = struct {
    condition: *ast.ExprNode,
    kind: Kind,
    context: Context,
    span: ast.SourceSpan,

    pub const Kind = enum {
        Precondition, // requires clause
        Postcondition, // ensures clause
        Invariant, // loop invariant
        Assertion, // assert statement
    };

    pub const Context = struct {
        function_name: ?[]const u8,
        old_state: ?*OldStateContext,
    };
};

/// Represents the "old" state for postconditions
pub const OldStateContext = struct {
    variables: std.HashMap([]const u8, comptime_eval.ComptimeValue, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),
    allocator: Allocator,

    pub fn init(allocator: Allocator) OldStateContext {
        return OldStateContext{
            .variables = std.HashMap([]const u8, comptime_eval.ComptimeValue, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *OldStateContext) void {
        self.variables.deinit();
    }

    pub fn captureVariable(self: *OldStateContext, name: []const u8, value: comptime_eval.ComptimeValue) !void {
        try self.variables.put(name, value);
    }

    pub fn getOldValue(self: *OldStateContext, name: []const u8) ?comptime_eval.ComptimeValue {
        return self.variables.get(name);
    }
};

/// Static verification result
pub const VerificationResult = struct {
    verified: bool,
    violations: []VerificationViolation,
    warnings: []VerificationWarning,

    pub const VerificationViolation = struct {
        message: []const u8,
        span: ast.SourceSpan,
        condition: *VerificationCondition,
    };

    pub const VerificationWarning = struct {
        message: []const u8,
        span: ast.SourceSpan,
        condition: *VerificationCondition,
    };
};

/// The main static verifier
pub const StaticVerifier = struct {
    allocator: Allocator,
    comptime_evaluator: comptime_eval.ComptimeEvaluator,
    conditions: std.ArrayList(VerificationCondition),
    violations: std.ArrayList(VerificationResult.VerificationViolation),
    warnings: std.ArrayList(VerificationResult.VerificationWarning),

    pub fn init(allocator: Allocator) StaticVerifier {
        return StaticVerifier{
            .allocator = allocator,
            .comptime_evaluator = comptime_eval.ComptimeEvaluator.init(allocator),
            .conditions = std.ArrayList(VerificationCondition).init(allocator),
            .violations = std.ArrayList(VerificationResult.VerificationViolation).init(allocator),
            .warnings = std.ArrayList(VerificationResult.VerificationWarning).init(allocator),
        };
    }

    pub fn deinit(self: *StaticVerifier) void {
        self.comptime_evaluator.deinit();
        self.conditions.deinit();
        self.violations.deinit();
        self.warnings.deinit();
    }

    /// Add a verification condition
    pub fn addCondition(self: *StaticVerifier, condition: VerificationCondition) !void {
        try self.conditions.append(condition);
    }

    /// Verify all conditions
    pub fn verifyAll(self: *StaticVerifier) !VerificationResult {
        // Clear previous results
        self.violations.clearRetainingCapacity();
        self.warnings.clearRetainingCapacity();

        // Verify each condition
        for (self.conditions.items) |*condition| {
            try self.verifyCondition(condition);
        }

        return VerificationResult{
            .verified = self.violations.items.len == 0,
            .violations = try self.violations.toOwnedSlice(),
            .warnings = try self.warnings.toOwnedSlice(),
        };
    }

    /// Verify a single condition
    fn verifyCondition(self: *StaticVerifier, condition: *VerificationCondition) !void {
        switch (condition.kind) {
            .Precondition => try self.verifyPrecondition(condition),
            .Postcondition => try self.verifyPostcondition(condition),
            .Invariant => try self.verifyInvariant(condition),
            .Assertion => try self.verifyAssertion(condition),
        }
    }

    /// Verify a precondition (requires clause)
    fn verifyPrecondition(self: *StaticVerifier, condition: *VerificationCondition) !void {
        // Try to evaluate the condition at compile time
        if (self.comptime_evaluator.evaluate(condition.condition)) |result| {
            switch (result) {
                .bool => |b| {
                    if (!b) {
                        try self.addViolation("Precondition always false", condition);
                    }
                    // If true, precondition is satisfied
                },
                else => {
                    try self.addWarning("Precondition is not boolean", condition);
                },
            }
        } else |err| {
            switch (err) {
                comptime_eval.ComptimeError.NotCompileTimeEvaluable => {
                    // Cannot evaluate at compile time - this is expected for many conditions
                    // Perform static analysis instead
                    try self.analyzeConditionStatically(condition);
                },
                else => {
                    try self.addWarning("Error evaluating precondition", condition);
                },
            }
        }
    }

    /// Verify a postcondition (ensures clause)
    fn verifyPostcondition(self: *StaticVerifier, condition: *VerificationCondition) !void {
        // First, extract and validate old() expressions
        try self.validateOldExpressions(condition);

        // Try to evaluate the condition at compile time
        if (self.comptime_evaluator.evaluate(condition.condition)) |result| {
            switch (result) {
                .bool => |b| {
                    if (!b) {
                        try self.addViolation("Postcondition always false", condition);
                    }
                },
                else => {
                    try self.addWarning("Postcondition is not boolean", condition);
                },
            }
        } else |err| {
            switch (err) {
                comptime_eval.ComptimeError.NotCompileTimeEvaluable => {
                    // Handle old() expressions and perform static analysis
                    try self.analyzePostconditionStatically(condition);
                },
                else => {
                    try self.addWarning("Error evaluating postcondition", condition);
                },
            }
        }
    }

    /// Verify an invariant
    fn verifyInvariant(self: *StaticVerifier, condition: *VerificationCondition) !void {
        // Invariants must hold at the beginning and end of each loop iteration
        try self.verifyPrecondition(condition); // Must hold initially

        // TODO: Verify it's maintained through the loop body
        // This requires more sophisticated analysis
    }

    /// Verify an assertion
    fn verifyAssertion(self: *StaticVerifier, condition: *VerificationCondition) !void {
        // Similar to precondition but different error message
        if (self.comptime_evaluator.evaluate(condition.condition)) |result| {
            switch (result) {
                .bool => |b| {
                    if (!b) {
                        try self.addViolation("Assertion always false", condition);
                    }
                },
                else => {
                    try self.addWarning("Assertion is not boolean", condition);
                },
            }
        } else |err| {
            switch (err) {
                comptime_eval.ComptimeError.NotCompileTimeEvaluable => {
                    try self.analyzeConditionStatically(condition);
                },
                else => {
                    try self.addWarning("Error evaluating assertion", condition);
                },
            }
        }
    }

    /// Validate old() expressions in postconditions
    fn validateOldExpressions(self: *StaticVerifier, condition: *VerificationCondition) !void {
        if (condition.kind != .Postcondition) {
            // Check that old() is not used in non-postconditions
            if (self.containsOldExpression(condition.condition)) {
                try self.addViolation("old() expressions only allowed in postconditions", condition);
            }
        }

        // TODO: Validate that old() expressions reference valid variables
    }

    /// Check if expression contains old() calls
    fn containsOldExpression(self: *StaticVerifier, expr: *ast.ExprNode) bool {
        return switch (expr.*) {
            .Old => true,
            .Binary => |*binary| {
                return self.containsOldExpression(binary.lhs) or self.containsOldExpression(binary.rhs);
            },
            .Unary => |*unary| {
                return self.containsOldExpression(unary.operand);
            },
            .Call => |*call| {
                if (self.containsOldExpression(call.callee)) return true;
                for (call.arguments) |*arg| {
                    if (self.containsOldExpression(arg)) return true;
                }
                return false;
            },
            .FieldAccess => |*field| {
                return self.containsOldExpression(field.target);
            },
            .Index => |*index| {
                return self.containsOldExpression(index.target) or self.containsOldExpression(index.index);
            },
            else => false,
        };
    }

    /// Perform static analysis on a condition that can't be evaluated at compile time
    fn analyzeConditionStatically(self: *StaticVerifier, condition: *VerificationCondition) !void {
        // Basic static analysis patterns
        try self.checkForCommonPatterns(condition);
        try self.checkForContradictions(condition);
        try self.checkForTautologies(condition);
    }

    /// Analyze postcondition with old() expressions
    fn analyzePostconditionStatically(self: *StaticVerifier, condition: *VerificationCondition) !void {
        // Extract old() expressions and check their validity
        try self.extractOldExpressions(condition);

        // Perform general static analysis
        try self.analyzeConditionStatically(condition);
    }

    /// Extract and validate old() expressions
    fn extractOldExpressions(self: *StaticVerifier, condition: *VerificationCondition) !void {
        try self.extractOldExpressionsFromExpr(condition.condition, condition);
    }

    /// Recursively extract old() expressions from an expression
    fn extractOldExpressionsFromExpr(self: *StaticVerifier, expr: *ast.ExprNode, condition: *VerificationCondition) !void {
        switch (expr.*) {
            .Old => |*old_expr| {
                // Validate the old() expression
                try self.validateOldExpression(old_expr, condition);
            },
            .Binary => |*binary| {
                try self.extractOldExpressionsFromExpr(binary.lhs, condition);
                try self.extractOldExpressionsFromExpr(binary.rhs, condition);
            },
            .Unary => |*unary| {
                try self.extractOldExpressionsFromExpr(unary.operand, condition);
            },
            .Call => |*call| {
                try self.extractOldExpressionsFromExpr(call.callee, condition);
                for (call.arguments) |*arg| {
                    try self.extractOldExpressionsFromExpr(arg, condition);
                }
            },
            .FieldAccess => |*field| {
                try self.extractOldExpressionsFromExpr(field.target, condition);
            },
            .Index => |*index| {
                try self.extractOldExpressionsFromExpr(index.target, condition);
                try self.extractOldExpressionsFromExpr(index.index, condition);
            },
            else => {
                // Other expressions don't contain old() calls
            },
        }
    }

    /// Validate a specific old() expression
    fn validateOldExpression(self: *StaticVerifier, old_expr: *ast.OldExpr, condition: *VerificationCondition) !void {
        // Check if the old expression references a valid variable
        switch (old_expr.expr.*) {
            .Identifier => |*ident| {
                // This is a simple old(variable) case
                if (condition.context.old_state) |old_state| {
                    if (old_state.getOldValue(ident.name) == null) {
                        try self.addWarning("old() references unknown variable", condition);
                    }
                } else {
                    try self.addWarning("old() expression without old state context", condition);
                }
            },
            .FieldAccess => |*field| {
                // This is old(obj.field) case
                // TODO: Validate field access in old state
                _ = field;
            },
            .Index => |*index| {
                // This is old(array[index]) case
                // TODO: Validate array access in old state
                _ = index;
            },
            else => {
                try self.addWarning("Complex old() expression may not be supported", condition);
            },
        }
    }

    /// Check for common verification patterns
    fn checkForCommonPatterns(self: *StaticVerifier, condition: *VerificationCondition) !void {
        try self.checkForNullPointerPattern(condition);
        try self.checkForOverflowPattern(condition);
        try self.checkForDivisionByZeroPattern(condition);
        try self.checkForArrayBoundsPattern(condition);
    }

    /// Check for null pointer/zero address patterns
    fn checkForNullPointerPattern(self: *StaticVerifier, condition: *VerificationCondition) !void {
        // Look for patterns like "addr != 0x0" or "addr != address(0)"
        if (self.isNullAddressCheck(condition.condition)) {
            // This is a good pattern, no warning needed
            return;
        }

        // Check if we're accessing something that might be null
        if (self.mightAccessNullPointer(condition.condition)) {
            try self.addWarning("Potential null pointer access", condition);
        }
    }

    /// Check for integer overflow patterns
    fn checkForOverflowPattern(self: *StaticVerifier, condition: *VerificationCondition) !void {
        // Look for patterns like "a + b >= a" (overflow check)
        if (self.isOverflowCheck(condition.condition)) {
            // This is a good pattern
            return;
        }

        // Check for potential overflow in arithmetic
        if (self.mightOverflow(condition.condition)) {
            try self.addWarning("Potential integer overflow", condition);
        }
    }

    /// Check for division by zero patterns
    fn checkForDivisionByZeroPattern(self: *StaticVerifier, condition: *VerificationCondition) !void {
        // Look for patterns like "denominator != 0"
        if (self.isDivisionByZeroCheck(condition.condition)) {
            return;
        }

        // Check for potential division by zero
        if (self.mightDivideByZero(condition.condition)) {
            try self.addWarning("Potential division by zero", condition);
        }
    }

    /// Check for array bounds patterns
    fn checkForArrayBoundsPattern(self: *StaticVerifier, condition: *VerificationCondition) !void {
        // Look for patterns like "index < array.length"
        if (self.isArrayBoundsCheck(condition.condition)) {
            return;
        }

        // Check for potential out-of-bounds access
        if (self.mightAccessOutOfBounds(condition.condition)) {
            try self.addWarning("Potential array out-of-bounds access", condition);
        }
    }

    /// Check for logical contradictions
    fn checkForContradictions(self: *StaticVerifier, condition: *VerificationCondition) !void {
        // Look for patterns like "x > 5 && x < 3"
        if (self.isContradiction(condition.condition)) {
            try self.addViolation("Logical contradiction in condition", condition);
        }
    }

    /// Check for tautologies (always true conditions)
    fn checkForTautologies(self: *StaticVerifier, condition: *VerificationCondition) !void {
        // Look for patterns like "x == x" or "true"
        if (self.isTautology(condition.condition)) {
            try self.addWarning("Condition is always true (tautology)", condition);
        }
    }

    // Helper methods for pattern recognition
    fn isNullAddressCheck(self: *StaticVerifier, expr: *ast.ExprNode) bool {
        return switch (expr.*) {
            .Binary => |*binary| {
                return binary.operator == .BangEqual and self.isZeroAddress(binary.rhs);
            },
            else => false,
        };
    }

    fn isZeroAddress(self: *StaticVerifier, expr: *ast.ExprNode) bool {
        _ = self;
        return switch (expr.*) {
            .Literal => |*lit| {
                return switch (lit.*) {
                    .Address => |*addr| std.mem.eql(u8, addr.value, "0x0000000000000000000000000000000000000000"),
                    else => false,
                };
            },
            else => false,
        };
    }

    fn isZeroValue(self: *StaticVerifier, expr: *ast.ExprNode) bool {
        _ = self;
        return switch (expr.*) {
            .Literal => |*lit| {
                return switch (lit.*) {
                    .Integer => |*int| std.mem.eql(u8, int.value, "0"),
                    .Bool => |*b| !b.value, // false is "zero" for booleans
                    .Address => |*addr| std.mem.eql(u8, addr.value, "0x0") or
                        std.mem.eql(u8, addr.value, "0x0000000000000000000000000000000000000000"),
                    .Hex => |*hex| std.mem.eql(u8, hex.value, "0x0") or std.mem.eql(u8, hex.value, "0x00"),
                    else => false,
                };
            },
            else => false,
        };
    }

    fn isLargeLiteral(self: *StaticVerifier, expr: *ast.ExprNode) bool {
        _ = self;
        return switch (expr.*) {
            .Literal => |*lit| {
                return switch (lit.*) {
                    .Integer => |*int| {
                        // Consider numbers > 2^128 as "large" for overflow detection
                        const large_threshold = "340282366920938463463374607431768211456"; // 2^128
                        return int.value.len > large_threshold.len or
                            (int.value.len == large_threshold.len and
                            std.mem.order(u8, int.value, large_threshold) == .gt);
                    },
                    .Hex => |*hex| {
                        // Consider hex values with many bytes as large
                        return hex.value.len > 34; // "0x" + 32 hex chars (16 bytes)
                    },
                    else => false,
                };
            },
            else => false,
        };
    }

    fn mightAccessNullPointer(self: *StaticVerifier, expr: *ast.ExprNode) bool {
        _ = self;
        _ = expr;
        // TODO: Implement null pointer analysis
        return false;
    }

    fn isOverflowCheck(self: *StaticVerifier, expr: *ast.ExprNode) bool {
        return switch (expr.*) {
            .Binary => |*binary| {
                // Look for patterns like "a + b >= a" or "a + b >= b" (overflow checks)
                if (binary.operator == .GreaterEqual) {
                    if (binary.lhs.* == .Binary and binary.lhs.Binary.operator == .Plus) {
                        const add_expr = &binary.lhs.Binary;
                        // Check if rhs is one of the addends
                        return self.expressionsEqual(binary.rhs, add_expr.lhs) or
                            self.expressionsEqual(binary.rhs, add_expr.rhs);
                    }
                }
                return false;
            },
            else => false,
        };
    }

    fn mightOverflow(self: *StaticVerifier, expr: *ast.ExprNode) bool {
        return switch (expr.*) {
            .Binary => |*binary| {
                return switch (binary.operator) {
                    .Plus, .Minus, .Star =>
                    // Arithmetic operations can overflow
                    // If both operands are large literals, might overflow
                    self.isLargeLiteral(binary.lhs) or self.isLargeLiteral(binary.rhs),
                    else => false,
                };
            },
            else => false,
        };
    }

    fn isDivisionByZeroCheck(self: *StaticVerifier, expr: *ast.ExprNode) bool {
        return switch (expr.*) {
            .Binary => |*binary| {
                // Look for patterns like "denominator != 0"
                if (binary.operator == .BangEqual) {
                    return self.isZeroValue(binary.rhs);
                }
                return false;
            },
            else => false,
        };
    }

    fn mightDivideByZero(self: *StaticVerifier, expr: *ast.ExprNode) bool {
        _ = self;
        _ = expr;
        // TODO: Implement division by zero analysis
        return false;
    }

    fn isArrayBoundsCheck(self: *StaticVerifier, expr: *ast.ExprNode) bool {
        _ = self;
        _ = expr;
        // TODO: Implement array bounds check pattern recognition
        return false;
    }

    fn mightAccessOutOfBounds(self: *StaticVerifier, expr: *ast.ExprNode) bool {
        _ = self;
        _ = expr;
        // TODO: Implement out-of-bounds analysis
        return false;
    }

    fn isContradiction(self: *StaticVerifier, expr: *ast.ExprNode) bool {
        return switch (expr.*) {
            .Binary => |*binary| {
                return switch (binary.operator) {
                    .And => {
                        // Look for patterns like "x > 5 && x < 3"
                        if (self.isRangeContradiction(binary.lhs, binary.rhs)) {
                            return true;
                        }
                        // Check if either operand is false
                        return self.isFalseLiteral(binary.lhs) or self.isFalseLiteral(binary.rhs);
                    },
                    .Or =>
                    // "false || false" is not a contradiction, just false
                    false,
                    .EqualEqual =>
                    // "x == x" is always true, not a contradiction
                    // "5 == 3" would be a contradiction
                    self.isLiteralComparison(binary) and !self.literalsEqualDirect(binary.lhs, binary.rhs),
                    else => false,
                };
            },
            else => false,
        };
    }

    fn isRangeContradiction(self: *StaticVerifier, lhs: *ast.ExprNode, rhs: *ast.ExprNode) bool {
        // Simple case: "x > 5 && x < 3" - same variable with contradictory bounds
        if (lhs.* == .Binary and rhs.* == .Binary) {
            const lhs_bin = &lhs.Binary;
            const rhs_bin = &rhs.Binary;

            // Check if both compare the same variable
            if (self.expressionsEqual(lhs_bin.lhs, rhs_bin.lhs)) {
                return self.areContradictoryComparisons(lhs_bin.operator, lhs_bin.rhs, rhs_bin.operator, rhs_bin.rhs);
            }
        }
        return false;
    }

    fn areContradictoryComparisons(self: *StaticVerifier, op1: ast.BinaryOp, val1: *ast.ExprNode, op2: ast.BinaryOp, val2: *ast.ExprNode) bool {
        // Check for patterns like "x > 5 && x < 3"
        if (val1.* == .Literal and val2.* == .Literal) {
            const lit1 = &val1.Literal;
            const lit2 = &val2.Literal;
            if (lit1.* == .Integer and lit2.* == .Integer) {
                const num1 = lit1.Integer.value;
                const num2 = lit2.Integer.value;

                return switch (op1) {
                    .Greater => switch (op2) {
                        .Less => std.mem.order(u8, num1, num2) != .lt, // x > 5 && x < 3 (5 >= 3)
                        .LessEqual => std.mem.order(u8, num1, num2) != .lt, // x > 5 && x <= 3
                        else => false,
                    },
                    .GreaterEqual => switch (op2) {
                        .Less => std.mem.order(u8, num1, num2) != .lt, // x >= 5 && x < 3
                        else => false,
                    },
                    else => false,
                };
            }
        }
        _ = self;
        return false;
    }

    fn isFalseLiteral(self: *StaticVerifier, expr: *ast.ExprNode) bool {
        _ = self;
        return switch (expr.*) {
            .Literal => |*lit| {
                return switch (lit.*) {
                    .Bool => |*b| !b.value,
                    else => false,
                };
            },
            else => false,
        };
    }

    fn isLiteralComparison(self: *StaticVerifier, binary: *ast.BinaryExpr) bool {
        _ = self;
        return binary.lhs.* == .Literal and binary.rhs.* == .Literal;
    }

    fn literalsEqualDirect(self: *StaticVerifier, lhs: *ast.ExprNode, rhs: *ast.ExprNode) bool {
        if (lhs.* == .Literal and rhs.* == .Literal) {
            return literalsEqual(&lhs.Literal, &rhs.Literal);
        }
        _ = self;
        return false;
    }

    fn isTautology(self: *StaticVerifier, expr: *ast.ExprNode) bool {
        _ = self;
        return switch (expr.*) {
            .Literal => |*lit| {
                return switch (lit.*) {
                    .Bool => |*b| b.value == true,
                    else => false,
                };
            },
            else => false,
        };
    }

    /// Add a violation
    fn addViolation(self: *StaticVerifier, message: []const u8, condition: *VerificationCondition) !void {
        try self.violations.append(VerificationResult.VerificationViolation{
            .message = message,
            .span = condition.span,
            .condition = condition,
        });
    }

    /// Add a warning
    fn addWarning(self: *StaticVerifier, message: []const u8, condition: *VerificationCondition) !void {
        try self.warnings.append(VerificationResult.VerificationWarning{
            .message = message,
            .span = condition.span,
            .condition = condition,
        });
    }

    /// Define a constant for verification
    pub fn defineConstant(self: *StaticVerifier, name: []const u8, value: comptime_eval.ComptimeValue) !void {
        try self.comptime_evaluator.defineConstant(name, value);
    }

    /// Create old state context
    pub fn createOldStateContext(self: *StaticVerifier) OldStateContext {
        return OldStateContext.init(self.allocator);
    }

    /// Check if two expressions are structurally equal
    fn expressionsEqual(self: *StaticVerifier, lhs: *ast.ExprNode, rhs: *ast.ExprNode) bool {
        return switch (lhs.*) {
            .Identifier => |*lhs_ident| {
                return switch (rhs.*) {
                    .Identifier => |*rhs_ident| std.mem.eql(u8, lhs_ident.name, rhs_ident.name),
                    else => false,
                };
            },
            .Literal => |*lhs_lit| {
                return switch (rhs.*) {
                    .Literal => |*rhs_lit| literalsEqual(lhs_lit, rhs_lit),
                    else => false,
                };
            },
            .Binary => |*lhs_bin| {
                return switch (rhs.*) {
                    .Binary => |*rhs_bin| {
                        return lhs_bin.operator == rhs_bin.operator and
                            self.expressionsEqual(lhs_bin.lhs, rhs_bin.lhs) and
                            self.expressionsEqual(lhs_bin.rhs, rhs_bin.rhs);
                    },
                    else => false,
                };
            },
            else => false,
        };
    }

    /// Check if two literals are equal
    fn literalsEqual(lhs: *ast.LiteralNode, rhs: *ast.LiteralNode) bool {
        return switch (lhs.*) {
            .Integer => |*lhs_int| {
                return switch (rhs.*) {
                    .Integer => |*rhs_int| std.mem.eql(u8, lhs_int.value, rhs_int.value),
                    else => false,
                };
            },
            .Bool => |*lhs_bool| {
                return switch (rhs.*) {
                    .Bool => |*rhs_bool| lhs_bool.value == rhs_bool.value,
                    else => false,
                };
            },
            .String => |*lhs_str| {
                return switch (rhs.*) {
                    .String => |*rhs_str| std.mem.eql(u8, lhs_str.value, rhs_str.value),
                    else => false,
                };
            },
            .Address => |*lhs_addr| {
                return switch (rhs.*) {
                    .Address => |*rhs_addr| std.mem.eql(u8, lhs_addr.value, rhs_addr.value),
                    else => false,
                };
            },
            .Hex => |*lhs_hex| {
                return switch (rhs.*) {
                    .Hex => |*rhs_hex| std.mem.eql(u8, lhs_hex.value, rhs_hex.value),
                    else => false,
                };
            },
        };
    }

    /// Verify a function with pre/postconditions
    pub fn verifyFunction(self: *StaticVerifier, func: *ast.FunctionNode) !VerificationResult {
        // Create old state context for postconditions
        var old_state = self.createOldStateContext();
        defer old_state.deinit();

        // Capture initial state for old() expressions
        // TODO: This would need to be integrated with the semantic analyzer
        // to capture actual variable values

        // Add preconditions
        for (func.requires_clauses) |*req| {
            try self.addCondition(VerificationCondition{
                .condition = req,
                .kind = .Precondition,
                .context = VerificationCondition.Context{
                    .function_name = func.name,
                    .old_state = null,
                },
                .span = ast.SourceSpan{ .line = 0, .column = 0, .length = 0 }, // TODO: Get actual span
            });
        }

        // Add postconditions
        for (func.ensures_clauses) |*ens| {
            try self.addCondition(VerificationCondition{
                .condition = ens,
                .kind = .Postcondition,
                .context = VerificationCondition.Context{
                    .function_name = func.name,
                    .old_state = &old_state,
                },
                .span = ast.SourceSpan{ .line = 0, .column = 0, .length = 0 }, // TODO: Get actual span
            });
        }

        return self.verifyAll();
    }
};

// Tests
const testing = std.testing;

test "static verifier basic functionality" {
    var verifier = StaticVerifier.init(testing.allocator);
    defer verifier.deinit();

    // Test that verifier initializes correctly
    try testing.expect(verifier.conditions.items.len == 0);
    try testing.expect(verifier.violations.items.len == 0);
    try testing.expect(verifier.warnings.items.len == 0);
}

test "static verifier tautology detection" {
    var verifier = StaticVerifier.init(testing.allocator);
    defer verifier.deinit();

    // Create a tautology condition
    var true_literal = ast.LiteralNode{ .Bool = ast.BoolLiteral{ .value = true, .span = ast.SourceSpan{ .line = 1, .column = 1, .length = 4 } } };
    var true_expr = ast.ExprNode{ .Literal = &true_literal };

    try verifier.addCondition(VerificationCondition{
        .condition = &true_expr,
        .kind = .Precondition,
        .context = VerificationCondition.Context{ .function_name = "test", .old_state = null },
        .span = ast.SourceSpan{ .line = 1, .column = 1, .length = 4 },
    });

    const result = try verifier.verifyAll();
    try testing.expect(result.verified); // Should be verified (no violations)
    try testing.expect(result.warnings.len > 0); // Should have warning about tautology
}

test "static verifier false precondition" {
    var verifier = StaticVerifier.init(testing.allocator);
    defer verifier.deinit();

    // Create a false precondition
    var false_literal = ast.LiteralNode{ .Bool = ast.BoolLiteral{ .value = false, .span = ast.SourceSpan{ .line = 1, .column = 1, .length = 5 } } };
    var false_expr = ast.ExprNode{ .Literal = &false_literal };

    try verifier.addCondition(VerificationCondition{
        .condition = &false_expr,
        .kind = .Precondition,
        .context = VerificationCondition.Context{ .function_name = "test", .old_state = null },
        .span = ast.SourceSpan{ .line = 1, .column = 1, .length = 5 },
    });

    const result = try verifier.verifyAll();
    try testing.expect(!result.verified); // Should not be verified
    try testing.expect(result.violations.len > 0); // Should have violation
}

test "static verifier old state context" {
    var verifier = StaticVerifier.init(testing.allocator);
    defer verifier.deinit();

    var old_state = verifier.createOldStateContext();
    defer old_state.deinit();

    // Capture an old value
    try old_state.captureVariable("balance", comptime_eval.ComptimeValue{ .u256 = [_]u8{0} ** 31 ++ [_]u8{100} });

    // Retrieve the old value
    const old_value = old_state.getOldValue("balance");
    try testing.expect(old_value != null);
    try testing.expect(old_value.? == .u256);
}

test "static verifier constants integration" {
    var verifier = StaticVerifier.init(testing.allocator);
    defer verifier.deinit();

    // Define a constant
    try verifier.defineConstant("MAX_VALUE", comptime_eval.ComptimeValue{ .u64 = 1000 });

    // Verify the constant was stored
    const value = verifier.comptime_evaluator.symbol_table.lookup("MAX_VALUE");
    try testing.expect(value != null);
    try testing.expect(value.? == .u64);
    try testing.expectEqual(@as(u64, 1000), value.?.u64);
}
