const std = @import("std");
pub const ast = @import("../ast.zig");
const semantics_errors = @import("semantics_errors.zig");
const semantics_memory_safety = @import("semantics_memory_safety.zig");

// Forward declaration for SemanticAnalyzer
const SemanticAnalyzer = @import("semantics_core.zig").SemanticAnalyzer;

/// Immutable variable information
pub const ImmutableVarInfo = struct {
    name: []const u8,
    declared_span: ast.SourceSpan,
    initialized: bool,
    init_span: ?ast.SourceSpan,
    init_value: ?*ast.ExprNode,
    variable_type: ?ast.TypeNode,
    is_storage: bool,
    contract_name: ?[]const u8,

    pub fn init(name: []const u8, declared_span: ast.SourceSpan) ImmutableVarInfo {
        return ImmutableVarInfo{
            .name = name,
            .declared_span = declared_span,
            .initialized = false,
            .init_span = null,
            .init_value = null,
            .variable_type = null,
            .is_storage = false,
            .contract_name = null,
        };
    }

    pub fn markInitialized(self: *ImmutableVarInfo, init_span: ast.SourceSpan, init_value: *ast.ExprNode) void {
        self.initialized = true;
        self.init_span = init_span;
        self.init_value = init_value;
    }

    pub fn setType(self: *ImmutableVarInfo, var_type: ast.TypeNode) void {
        self.variable_type = var_type;
    }

    pub fn setStorageContext(self: *ImmutableVarInfo, contract_name: []const u8) void {
        self.is_storage = true;
        self.contract_name = contract_name;
    }
};

/// Immutable variable assignment attempt
pub const ImmutableAssignmentAttempt = struct {
    variable_name: []const u8,
    assignment_span: ast.SourceSpan,
    assignment_context: AssignmentContext,
    is_valid: bool,
    error_message: ?[]const u8,

    pub const AssignmentContext = enum {
        Declaration,
        Constructor,
        RegularFunction,
        GlobalScope,
    };
};

/// Immutable variable validation result
pub const ImmutableValidationResult = struct {
    valid: bool,
    variable_name: []const u8,
    error_message: ?[]const u8,
    warning_message: ?[]const u8,
    uninitialized_variables: [][]const u8,
    invalid_assignments: []ImmutableAssignmentAttempt,
};

/// Immutable variable tracker for comprehensive immutable semantics
pub const ImmutableTracker = struct {
    allocator: std.mem.Allocator,
    immutable_variables: std.HashMap([]const u8, ImmutableVarInfo, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),
    assignment_attempts: std.ArrayList(ImmutableAssignmentAttempt),
    current_contract: ?[]const u8,
    in_constructor: bool,
    constructor_name: ?[]const u8,

    pub fn init(allocator: std.mem.Allocator) ImmutableTracker {
        return ImmutableTracker{
            .allocator = allocator,
            .immutable_variables = std.HashMap([]const u8, ImmutableVarInfo, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
            .assignment_attempts = std.ArrayList(ImmutableAssignmentAttempt).init(allocator),
            .current_contract = null,
            .in_constructor = false,
            .constructor_name = null,
        };
    }

    pub fn deinit(self: *ImmutableTracker) void {
        self.immutable_variables.deinit();
        self.assignment_attempts.deinit();
    }

    /// Set contract context
    pub fn setContractContext(self: *ImmutableTracker, contract_name: []const u8) void {
        self.current_contract = contract_name;
    }

    /// Set constructor context
    pub fn setConstructorContext(self: *ImmutableTracker, constructor_name: []const u8) void {
        self.in_constructor = true;
        self.constructor_name = constructor_name;
    }

    /// Clear constructor context
    pub fn clearConstructorContext(self: *ImmutableTracker) void {
        self.in_constructor = false;
        self.constructor_name = null;
    }

    /// Clear contract context
    pub fn clearContractContext(self: *ImmutableTracker) void {
        self.current_contract = null;
        self.clearConstructorContext();
        self.immutable_variables.clearRetainingCapacity();
        self.assignment_attempts.clearRetainingCapacity();
    }

    /// Register immutable variable declaration
    pub fn registerImmutableVariable(self: *ImmutableTracker, analyzer: *SemanticAnalyzer, var_decl: *ast.VariableDeclNode) !void {
        // Validate variable name
        if (!semantics_memory_safety.isValidString(analyzer, var_decl.name)) {
            try semantics_errors.addErrorStatic(analyzer, "Invalid immutable variable name", var_decl.span);
            return;
        }

        // Check for duplicate declarations
        if (self.immutable_variables.contains(var_decl.name)) {
            const error_msg = try std.fmt.allocPrint(analyzer.allocator, "Duplicate immutable variable declaration: {s}", .{var_decl.name});
            try semantics_errors.addError(analyzer, error_msg, var_decl.span);
            return;
        }

        // Create immutable variable info
        var var_info = ImmutableVarInfo.init(var_decl.name, var_decl.span);
        var_info.setType(var_decl.typ);

        // Set storage context if in contract
        if (self.current_contract) |contract_name| {
            var_info.setStorageContext(contract_name);
        }

        // Check if initialized at declaration
        if (var_decl.value) |init_value| {
            var_info.markInitialized(var_decl.span, init_value);
        }

        // Register the variable
        try self.immutable_variables.put(var_decl.name, var_info);
    }

    /// Track immutable variable assignment attempt
    pub fn trackAssignmentAttempt(self: *ImmutableTracker, analyzer: *SemanticAnalyzer, variable_name: []const u8, assignment_span: ast.SourceSpan) !bool {
        // Validate variable name
        if (!semantics_memory_safety.isValidString(analyzer, variable_name)) {
            try semantics_errors.addErrorStatic(analyzer, "Invalid variable name in assignment", assignment_span);
            return false;
        }

        // Check if variable is registered as immutable
        var var_info = self.immutable_variables.getPtr(variable_name) orelse {
            // Not an immutable variable, assignment is allowed
            return true;
        };

        // Determine assignment context
        const context = self.getAssignmentContext();

        // Validate assignment based on context
        const is_valid = self.validateAssignmentContext(context, var_info);

        // Create assignment attempt record
        const attempt = ImmutableAssignmentAttempt{
            .variable_name = variable_name,
            .assignment_span = assignment_span,
            .assignment_context = context,
            .is_valid = is_valid,
            .error_message = if (!is_valid) self.getAssignmentErrorMessage(context, var_info) else null,
        };

        try self.assignment_attempts.append(attempt);

        // Report error if invalid
        if (!is_valid) {
            if (attempt.error_message) |msg| {
                try semantics_errors.addErrorStatic(analyzer, msg, assignment_span);
            } else {
                const error_msg = try std.fmt.allocPrint(analyzer.allocator, "Invalid assignment to immutable variable '{s}'", .{variable_name});
                try semantics_errors.addError(analyzer, error_msg, assignment_span);
            }
            return false;
        }

        // Mark variable as initialized if this is a valid initialization
        if (context == .Constructor or context == .Declaration) {
            if (!var_info.initialized) {
                var_info.initialized = true;
                var_info.init_span = assignment_span;
            }
        }

        return true;
    }

    /// Validate all immutable variables are properly initialized
    pub fn validateAllInitialized(self: *ImmutableTracker, analyzer: *SemanticAnalyzer) !ImmutableValidationResult {
        var uninitialized = std.ArrayList([]const u8).init(self.allocator);
        defer uninitialized.deinit();

        var invalid_assignments = std.ArrayList(ImmutableAssignmentAttempt).init(self.allocator);
        defer invalid_assignments.deinit();

        // Check for uninitialized immutable variables
        var iterator = self.immutable_variables.iterator();
        while (iterator.next()) |entry| {
            const var_info = entry.value_ptr;
            if (!var_info.initialized) {
                try uninitialized.append(var_info.name);

                const error_msg = try std.fmt.allocPrint(analyzer.allocator, "Immutable variable '{s}' is not initialized", .{var_info.name});
                try semantics_errors.addError(analyzer, error_msg, var_info.declared_span);
            }
        }

        // Collect invalid assignment attempts
        for (self.assignment_attempts.items) |attempt| {
            if (!attempt.is_valid) {
                try invalid_assignments.append(attempt);
            }
        }

        const has_errors = uninitialized.items.len > 0 or invalid_assignments.items.len > 0;

        return ImmutableValidationResult{
            .valid = !has_errors,
            .variable_name = if (self.current_contract) |name| name else "global",
            .error_message = if (has_errors) "Immutable variable validation failed" else null,
            .warning_message = null,
            .uninitialized_variables = try uninitialized.toOwnedSlice(),
            .invalid_assignments = try invalid_assignments.toOwnedSlice(),
        };
    }

    /// Check if variable is immutable
    pub fn isVariableImmutable(self: *ImmutableTracker, variable_name: []const u8) bool {
        return self.immutable_variables.contains(variable_name);
    }

    /// Get immutable variable info
    pub fn getVariableInfo(self: *ImmutableTracker, variable_name: []const u8) ?ImmutableVarInfo {
        return self.immutable_variables.get(variable_name);
    }

    /// Validate immutable variable access
    pub fn validateVariableAccess(self: *ImmutableTracker, analyzer: *SemanticAnalyzer, variable_name: []const u8, access_span: ast.SourceSpan, is_write_access: bool) !bool {
        const var_info = self.immutable_variables.get(variable_name) orelse {
            // Not an immutable variable, access is allowed
            return true;
        };

        // Read access is always allowed for initialized variables
        if (!is_write_access) {
            if (!var_info.initialized) {
                const error_msg = try std.fmt.allocPrint(analyzer.allocator, "Reading uninitialized immutable variable '{s}'", .{variable_name});
                try semantics_errors.addError(analyzer, error_msg, access_span);
                return false;
            }
            return true;
        }

        // Write access validation
        return self.trackAssignmentAttempt(analyzer, variable_name, access_span);
    }

    /// Private helper methods
    fn getAssignmentContext(self: *ImmutableTracker) ImmutableAssignmentAttempt.AssignmentContext {
        if (self.in_constructor) {
            return .Constructor;
        } else if (self.current_contract != null) {
            return .RegularFunction;
        } else {
            return .GlobalScope;
        }
    }

    fn validateAssignmentContext(self: *ImmutableTracker, context: ImmutableAssignmentAttempt.AssignmentContext, var_info: *ImmutableVarInfo) bool {
        _ = self;

        switch (context) {
            .Declaration => return true, // Always allowed at declaration
            .Constructor => {
                // Allowed in constructor if not already initialized
                return !var_info.initialized;
            },
            .RegularFunction => {
                // Not allowed in regular functions
                return false;
            },
            .GlobalScope => {
                // Only allowed if not in storage context
                return !var_info.is_storage;
            },
        }
    }

    fn getAssignmentErrorMessage(self: *ImmutableTracker, context: ImmutableAssignmentAttempt.AssignmentContext, var_info: *ImmutableVarInfo) []const u8 {
        _ = self;

        switch (context) {
            .Declaration => return "Invalid assignment at declaration", // Should not happen
            .Constructor => {
                if (var_info.initialized) {
                    return "Immutable variable already initialized";
                } else {
                    return "Invalid constructor assignment";
                }
            },
            .RegularFunction => return "Immutable variables can only be assigned in constructor",
            .GlobalScope => {
                if (var_info.is_storage) {
                    return "Storage immutable variables can only be assigned in constructor";
                } else {
                    return "Invalid global assignment to immutable variable";
                }
            },
        }
    }
};

/// Track immutable variable declaration
pub fn trackImmutableDeclaration(analyzer: *SemanticAnalyzer, var_decl: *ast.VariableDeclNode) semantics_errors.SemanticError!void {
    // Only track if variable is immutable or storage const
    if (var_decl.region != .Immutable and !(var_decl.region == .Storage and !var_decl.mutable)) {
        return;
    }

    // Create a temporary tracker for this validation
    var tracker = ImmutableTracker.init(analyzer.allocator);
    defer tracker.deinit();

    // Set up context
    if (analyzer.current_contract) |contract| {
        tracker.setContractContext(contract.name);
    }
    if (analyzer.in_constructor) {
        tracker.setConstructorContext("init");
    }

    // Register the variable
    tracker.registerImmutableVariable(analyzer, var_decl) catch |err| {
        switch (err) {
            error.OutOfMemory => return semantics_errors.SemanticError.OutOfMemory,
            else => {
                try semantics_errors.addErrorStatic(analyzer, "Failed to track immutable variable", var_decl.span);
                return;
            },
        }
    };
}

/// Validate immutable variable assignment
pub fn validateImmutableAssignment(analyzer: *SemanticAnalyzer, variable_name: []const u8, assignment_span: ast.SourceSpan) semantics_errors.SemanticError!bool {
    // Check if this is an immutable variable in the analyzer's tracking
    const var_info = analyzer.immutable_variables.get(variable_name) orelse {
        // Not tracked as immutable, assignment is allowed
        return true;
    };

    // Check assignment context
    if (!analyzer.in_constructor) {
        const error_msg = try std.fmt.allocPrint(analyzer.allocator, "Cannot assign to immutable variable '{s}' outside of constructor", .{variable_name});
        try semantics_errors.addError(analyzer, error_msg, assignment_span);
        return false;
    }

    // Check if already initialized
    if (var_info.initialized) {
        const error_msg = try std.fmt.allocPrint(analyzer.allocator, "Immutable variable '{s}' is already initialized", .{variable_name});
        try semantics_errors.addError(analyzer, error_msg, assignment_span);
        return false;
    }

    return true;
}

/// Validate all immutable variables are initialized
pub fn validateImmutableInitialization(analyzer: *SemanticAnalyzer) semantics_errors.SemanticError!void {
    var iterator = analyzer.immutable_variables.iterator();
    while (iterator.next()) |entry| {
        const var_info = entry.value_ptr;
        if (!var_info.initialized) {
            const error_msg = try std.fmt.allocPrint(analyzer.allocator, "Immutable variable '{s}' is not initialized", .{var_info.name});
            try semantics_errors.addError(analyzer, error_msg, var_info.declared_span);
        }
    }
}
