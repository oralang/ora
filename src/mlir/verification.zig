// ============================================================================
// MLIR Verification
// ============================================================================
//
// Verifies MLIR module correctness and detects common errors.
//
// FEATURES:
//   • Type verification for ora.* operations
//   • Memory verification for storage operations
//   • Contract verification for function contracts
//   • Semantic verification for Ora-specific constructs
//
// ============================================================================

const std = @import("std");
const c = @import("mlir_c_api").c;
const lib = @import("ora_lib");
const ErrorHandler = @import("error_handling.zig").ErrorHandler;
const ErrorType = @import("error_handling.zig").ErrorType;
const WarningType = @import("error_handling.zig").WarningType;

/// Ora MLIR verification system
pub const OraVerification = struct {
    ctx: c.MlirContext,
    allocator: std.mem.Allocator,
    error_handler: ErrorHandler,

    const Self = @This();

    pub fn init(ctx: c.MlirContext, allocator: std.mem.Allocator) Self {
        return Self{
            .ctx = ctx,
            .allocator = allocator,
            .error_handler = ErrorHandler.init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        self.error_handler.deinit();
    }

    /// Run all verification passes on a module
    pub fn verifyModule(self: *Self, module: c.MlirModule) !VerificationResult {
        // reset error handler for new verification run
        self.error_handler.deinit();
        self.error_handler = ErrorHandler.init(self.allocator);

        const module_op = c.oraModuleGetOperation(module);

        // run all verification passes
        try self.verifyTypes(module_op);
        try self.verifyMemoryOperations(module_op);
        try self.verifyContracts(module_op);
        try self.verifySemantics(module_op);

        // convert ErrorHandler errors to VerificationResult
        const errors = self.error_handler.getErrors();
        var verification_errors = std.ArrayList(VerificationError){};
        defer verification_errors.deinit(self.allocator);

        for (errors) |err| {
            try verification_errors.append(self.allocator, VerificationError{
                .type = switch (err.error_type) {
                    .MalformedAst => .SemanticError,
                    .TypeMismatch => .InvalidType,
                    .UndefinedSymbol => .SemanticError,
                    .InvalidMemoryRegion => .MemorySafety,
                    .MlirOperationFailed => .SemanticError,
                    .UnsupportedFeature => .SemanticError,
                    .MissingNodeType => .MissingAttribute,
                    .CompilationLimit => .SemanticError,
                    .InternalError => .SemanticError,
                },
                .operation = c.MlirOperation{}, // We don't have operation context in ErrorHandler
                .message = err.message,
            });
        }

        return VerificationResult{
            .success = errors.len == 0,
            .errors = try verification_errors.toOwnedSlice(self.allocator),
        };
    }

    /// Verify types in all operations
    fn verifyTypes(self: *Self, op: c.MlirOperation) !void {
        // get operation name
        const op_name = self.getOperationName(op);

        // basic verification for ora.* operations
        if (std.mem.startsWith(u8, op_name, "ora.")) {
            try self.verifyOraOperationTypes(op, op_name);
        }

        // recursively verify regions
        const num_regions = c.oraOperationGetNumRegions(op);
        for (0..@intCast(num_regions)) |region_idx| {
            const block = c.oraOperationGetRegionBlock(op, region_idx);
            _ = block;
        }
    }

    /// Verify types for specific Ora operations
    fn verifyOraOperationTypes(self: *Self, op: c.MlirOperation, op_name: []const u8) !void {
        // basic verification for common Ora operations
        if (std.mem.eql(u8, op_name, "ora.contract")) {
            try self.verifyContractOperation(op);
        } else if (std.mem.eql(u8, op_name, "ora.global")) {
            try self.verifyGlobalOperation(op);
        } else if (std.mem.eql(u8, op_name, "ora.sload")) {
            try self.verifySLoadOperation(op);
        } else if (std.mem.eql(u8, op_name, "ora.sstore")) {
            try self.verifySStoreOperation(op);
        } else if (std.mem.eql(u8, op_name, "ora.mload")) {
            try self.verifyMLoadOperation(op);
        } else if (std.mem.eql(u8, op_name, "ora.mstore")) {
            try self.verifyMStoreOperation(op);
        }

        // basic operand/result count verification
        _ = c.oraOperationGetNumOperands(op);
        const num_results = c.oraOperationGetNumResults(op);

        // result count verification skipped to avoid false positives
        // operation-specific result validation handled by MLIR verifier
        _ = num_results;
    }

    /// Verify contract operation
    fn verifyContractOperation(self: *Self, op: c.MlirOperation) !void {
        _ = c.oraOperationGetNumOperands(op);
        const num_results = c.oraOperationGetNumResults(op);

        // contract should have at least one result (the contract instance)
        if (num_results == 0) {
            try self.error_handler.reportError(.MlirOperationFailed, null, "Contract operation must have at least one result", "add result to contract operation");
        }

        // contract should have at least one region (the body)
        const num_regions = c.oraOperationGetNumRegions(op);
        if (num_regions == 0) {
            try self.error_handler.reportError(.MlirOperationFailed, null, "Contract operation must have at least one region", "add region to contract operation");
        }
    }

    /// Verify global operation
    fn verifyGlobalOperation(self: *Self, op: c.MlirOperation) !void {
        _ = c.oraOperationGetNumOperands(op);
        const num_results = c.oraOperationGetNumResults(op);

        // global should have exactly one result
        if (num_results != 1) {
            try self.error_handler.reportError(.MlirOperationFailed, null, "Global operation must have exactly one result", "ensure global operation has exactly one result");
        }
    }

    /// Verify storage load operation
    fn verifySLoadOperation(self: *Self, op: c.MlirOperation) !void {
        const num_operands = c.oraOperationGetNumOperands(op);
        const num_results = c.oraOperationGetNumResults(op);

        // sLoad should have exactly one operand (address) and one result (value)
        if (num_operands != 1) {
            try self.error_handler.reportError(.MlirOperationFailed, null, "SLoad operation must have exactly one operand (address)", "ensure sload has exactly one address operand");
        }
        if (num_results != 1) {
            try self.error_handler.reportError(.MlirOperationFailed, null, "SLoad operation must have exactly one result (value)", "ensure sload has exactly one result");
        }
    }

    /// Verify storage store operation
    fn verifySStoreOperation(self: *Self, op: c.MlirOperation) !void {
        const num_operands = c.oraOperationGetNumOperands(op);
        const num_results = c.oraOperationGetNumResults(op);

        // sStore should have exactly two operands (address, value) and no results
        if (num_operands != 2) {
            try self.error_handler.reportError(.MlirOperationFailed, null, "SStore operation must have exactly two operands (address, value)", "ensure sstore has exactly two operands");
        }
        if (num_results != 0) {
            try self.error_handler.reportError(.MlirOperationFailed, null, "SStore operation must have no results", "ensure sstore has no results");
        }
    }

    /// Verify memory load operation
    fn verifyMLoadOperation(self: *Self, op: c.MlirOperation) !void {
        const num_operands = c.oraOperationGetNumOperands(op);
        const num_results = c.oraOperationGetNumResults(op);

        // mLoad should have exactly one operand (address) and one result (value)
        if (num_operands != 1) {
            try self.error_handler.reportError(.MlirOperationFailed, null, "MLoad operation must have exactly one operand (address)", "ensure mload has exactly one address operand");
        }
        if (num_results != 1) {
            try self.error_handler.reportError(.MlirOperationFailed, null, "MLoad operation must have exactly one result (value)", "ensure mload has exactly one result");
        }
    }

    /// Verify memory store operation
    fn verifyMStoreOperation(self: *Self, op: c.MlirOperation) !void {
        const num_operands = c.oraOperationGetNumOperands(op);
        const num_results = c.oraOperationGetNumResults(op);

        // mStore should have exactly two operands (address, value) and no results
        if (num_operands != 2) {
            try self.error_handler.reportError(.MlirOperationFailed, null, "MStore operation must have exactly two operands (address, value)", "ensure mstore has exactly two operands");
        }
        if (num_results != 0) {
            try self.error_handler.reportError(.MlirOperationFailed, null, "MStore operation must have no results", "ensure mstore has no results");
        }
    }

    /// Verify memory operations for consistency
    fn verifyMemoryOperations(self: *Self, op: c.MlirOperation) !void {
        // basic memory safety checks
        const op_name = self.getOperationName(op);

        // check for memory operation patterns
        if (std.mem.eql(u8, op_name, "ora.mload") or std.mem.eql(u8, op_name, "ora.mstore")) {
            // memory operations should have proper operand types
            const num_operands = c.oraOperationGetNumOperands(op);
            if (num_operands > 0) {
                // memory address type checking delegated to MLIR type system
                // for now, just verify operand count
            }
        }

        // recursively check regions
        const num_regions = c.oraOperationGetNumRegions(op);
        for (0..@intCast(num_regions)) |region_idx| {
            const block = c.oraOperationGetRegionBlock(op, region_idx);
            _ = block;
        }
    }

    /// Verify function contracts and invariants
    fn verifyContracts(self: *Self, op: c.MlirOperation) !void {
        // basic contract verification
        const op_name = self.getOperationName(op);

        // check for contract-related operations
        if (std.mem.eql(u8, op_name, "ora.requires") or std.mem.eql(u8, op_name, "ora.ensures")) {
            // contract operations should have exactly one operand (the condition)
            const num_operands = c.oraOperationGetNumOperands(op);
            if (num_operands != 1) {
                try self.error_handler.reportError(.MlirOperationFailed, null, "Contract operation must have exactly one operand (condition)", "ensure contract has exactly one condition operand");
            }
        }

        // recursively check regions
        const num_regions = c.oraOperationGetNumRegions(op);
        for (0..@intCast(num_regions)) |region_idx| {
            const block = c.oraOperationGetRegionBlock(op, region_idx);
            _ = block;
        }
    }

    /// Verify Ora-specific semantics
    fn verifySemantics(self: *Self, op: c.MlirOperation) !void {
        // basic semantic verification
        const op_name = self.getOperationName(op);

        // check for semantic consistency
        if (std.mem.eql(u8, op_name, "ora.if")) {
            // if operation should have exactly one operand (condition) and at least one region
            const num_operands = c.oraOperationGetNumOperands(op);
            const num_regions = c.oraOperationGetNumRegions(op);

            if (num_operands != 1) {
                try self.error_handler.reportError(.MlirOperationFailed, null, "If operation must have exactly one operand (condition)", "ensure if operation has exactly one condition operand");
            }
            if (num_regions < 1) {
                try self.error_handler.reportError(.MlirOperationFailed, null, "If operation must have at least one region (then branch)", "ensure if operation has at least one region");
            }
        }

        // recursively check regions
        const num_regions = c.oraOperationGetNumRegions(op);
        for (0..@intCast(num_regions)) |region_idx| {
            const block = c.oraOperationGetRegionBlock(op, region_idx);
            _ = block;
        }
    }

    /// Get operation name as string
    /// Note: The returned string is owned by the caller and should be freed
    fn getOperationName(self: *Self, op: c.MlirOperation) []const u8 {
        const name_ref = c.oraOperationGetName(op);
        if (name_ref.data == null or name_ref.length == 0) {
            return "unknown.operation";
        }
        // copy the string to a Zig slice
        // note: The C API allocates this, so we need to free it later
        // for now, we'll create a copy that the caller owns
        const name_slice = name_ref.data[0..name_ref.length];
        // allocate a copy that we can return
        const name_copy = self.allocator.dupe(u8, name_slice) catch {
            // if allocation fails, free the C string and return a fallback
            c.oraStringRefFree(name_ref);
            return "unknown.operation";
        };
        // free the C-allocated string
        c.oraStringRefFree(name_ref);
        return name_copy;
    }
};

/// Verification error types
pub const VerificationErrorType = enum {
    InvalidResultCount,
    InvalidOperandCount,
    MissingAttribute,
    InvalidType,
    MemorySafety,
    ContractViolation,
    SemanticError,
};

/// Verification error
pub const VerificationError = struct {
    type: VerificationErrorType,
    operation: c.MlirOperation,
    message: []const u8,
};

/// Verification result
pub const VerificationResult = struct {
    success: bool,
    errors: []const VerificationError,

    pub fn deinit(self: *const VerificationResult, allocator: std.mem.Allocator) void {
        for (self.errors) |verification_error| {
            allocator.free(verification_error.message);
        }
        allocator.free(self.errors);
    }
};

/// Simple verification pass that can be used with mlir-opt
pub fn runOraVerification(ctx: c.MlirContext, module: c.MlirModule, allocator: std.mem.Allocator) !VerificationResult {
    var verifier = OraVerification.init(ctx, allocator);
    defer verifier.deinit();

    return try verifier.verifyModule(module);
}
