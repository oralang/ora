const std = @import("std");
pub const ast = @import("../ast.zig");
const semantics_errors = @import("semantics_errors.zig");
const semantics_memory_safety = @import("semantics_memory_safety.zig");

// Forward declaration for SemanticAnalyzer
const SemanticAnalyzer = @import("semantics_core.zig").SemanticAnalyzer;

/// Memory region transition rules
pub const MemoryRegionTransition = struct {
    from: ast.MemoryRegion,
    to: ast.MemoryRegion,
    allowed: bool,
    requires_explicit_cast: bool,
    warning_message: ?[]const u8,
};

/// Memory region validation context
pub const MemoryRegionContext = struct {
    current_function: ?[]const u8,
    in_contract: bool,
    in_constructor: bool,
    contract_name: ?[]const u8,
    allowed_regions: std.EnumSet(ast.MemoryRegion),

    pub fn init() MemoryRegionContext {
        return MemoryRegionContext{
            .current_function = null,
            .in_contract = false,
            .in_constructor = false,
            .contract_name = null,
            .allowed_regions = std.EnumSet(ast.MemoryRegion).initFull(),
        };
    }

    pub fn setContractContext(self: *MemoryRegionContext, contract_name: []const u8, in_constructor: bool) void {
        self.in_contract = true;
        self.contract_name = contract_name;
        self.in_constructor = in_constructor;
    }

    pub fn setFunctionContext(self: *MemoryRegionContext, function_name: []const u8) void {
        self.current_function = function_name;
    }

    pub fn isRegionAllowed(self: *MemoryRegionContext, region: ast.MemoryRegion) bool {
        return self.allowed_regions.contains(region);
    }

    pub fn restrictRegion(self: *MemoryRegionContext, region: ast.MemoryRegion) void {
        self.allowed_regions.remove(region);
    }

    pub fn allowRegion(self: *MemoryRegionContext, region: ast.MemoryRegion) void {
        self.allowed_regions.insert(region);
    }
};

/// Memory region validation result
pub const MemoryRegionValidationResult = struct {
    valid: bool,
    region: ast.MemoryRegion,
    error_message: ?[]const u8,
    warning_message: ?[]const u8,
    suggested_region: ?ast.MemoryRegion,
};

/// Memory region validator for comprehensive memory semantics validation
pub const MemoryRegionValidator = struct {
    allocator: std.mem.Allocator,
    transition_rules: []const MemoryRegionTransition,
    context_stack: std.ArrayList(MemoryRegionContext),

    pub fn init(allocator: std.mem.Allocator) MemoryRegionValidator {
        var validator = MemoryRegionValidator{
            .allocator = allocator,
            .transition_rules = &default_transition_rules,
            .context_stack = std.ArrayList(MemoryRegionContext).init(allocator),
        };

        // Initialize with default context
        validator.context_stack.append(MemoryRegionContext.init()) catch {};
        return validator;
    }

    pub fn deinit(self: *MemoryRegionValidator) void {
        self.context_stack.deinit();
    }

    /// Push a new memory region context
    pub fn pushContext(self: *MemoryRegionValidator, context: MemoryRegionContext) !void {
        try self.context_stack.append(context);
    }

    /// Pop the current memory region context
    pub fn popContext(self: *MemoryRegionValidator) void {
        if (self.context_stack.items.len > 1) {
            _ = self.context_stack.pop();
        }
    }

    /// Get the current memory region context
    pub fn getCurrentContext(self: *MemoryRegionValidator) *MemoryRegionContext {
        return &self.context_stack.items[self.context_stack.items.len - 1];
    }

    /// Validate memory region for variable declaration
    pub fn validateVariableRegion(self: *MemoryRegionValidator, analyzer: *SemanticAnalyzer, var_decl: *ast.VariableDeclNode) !MemoryRegionValidationResult {
        const context = self.getCurrentContext();
        const region = var_decl.region;

        // Validate region name
        if (!semantics_memory_safety.isValidString(analyzer, var_decl.name)) {
            try semantics_errors.addErrorStatic(analyzer, "Invalid variable name", var_decl.span);
            return MemoryRegionValidationResult{
                .valid = false,
                .region = region,
                .error_message = "Invalid variable name",
                .warning_message = null,
                .suggested_region = null,
            };
        }

        // Check if region is allowed in current context
        if (!context.isRegionAllowed(region)) {
            const error_msg = try std.fmt.allocPrint(analyzer.allocator, "Memory region '{s}' not allowed in current context", .{@tagName(region)});
            try semantics_errors.addError(analyzer, error_msg, var_decl.span);
            return MemoryRegionValidationResult{
                .valid = false,
                .region = region,
                .error_message = "Region not allowed in context",
                .warning_message = null,
                .suggested_region = self.suggestAlternativeRegion(context, region),
            };
        }

        // Validate region-specific rules
        switch (region) {
            .Storage => {
                if (!context.in_contract) {
                    try semantics_errors.addErrorStatic(analyzer, "Storage variables can only be declared in contracts", var_decl.span);
                    return MemoryRegionValidationResult{
                        .valid = false,
                        .region = region,
                        .error_message = "Storage variables only allowed in contracts",
                        .warning_message = null,
                        .suggested_region = .Stack,
                    };
                }
            },
            .Immutable => {
                if (!context.in_contract) {
                    try semantics_errors.addErrorStatic(analyzer, "Immutable variables can only be declared in contracts", var_decl.span);
                    return MemoryRegionValidationResult{
                        .valid = false,
                        .region = region,
                        .error_message = "Immutable variables only allowed in contracts",
                        .warning_message = null,
                        .suggested_region = .Stack,
                    };
                }

                // Immutable variables must be initialized
                if (var_decl.value == null and !context.in_constructor) {
                    try semantics_errors.addErrorStatic(analyzer, "Immutable variables must be initialized", var_decl.span);
                    return MemoryRegionValidationResult{
                        .valid = false,
                        .region = region,
                        .error_message = "Immutable variables must be initialized",
                        .warning_message = null,
                        .suggested_region = null,
                    };
                }
            },
            .Stack => {
                // Stack variables are generally allowed everywhere
                // But warn if used in contract storage context where Storage might be intended
                if (context.in_contract and context.current_function == null) {
                    const warning_msg = try std.fmt.allocPrint(analyzer.allocator, "Stack variable in contract scope - consider using Storage region");
                    try semantics_errors.addWarning(analyzer, warning_msg, var_decl.span);
                    return MemoryRegionValidationResult{
                        .valid = true,
                        .region = region,
                        .error_message = null,
                        .warning_message = "Consider using Storage region",
                        .suggested_region = .Storage,
                    };
                }
            },
            .Heap => {
                // Heap variables require careful memory management
                const warning_msg = try std.fmt.allocPrint(analyzer.allocator, "Heap allocation requires explicit memory management");
                try semantics_errors.addWarning(analyzer, warning_msg, var_decl.span);
            },
        }

        return MemoryRegionValidationResult{
            .valid = true,
            .region = region,
            .error_message = null,
            .warning_message = null,
            .suggested_region = null,
        };
    }

    /// Validate memory region transition
    pub fn validateRegionTransition(self: *MemoryRegionValidator, analyzer: *SemanticAnalyzer, from_region: ast.MemoryRegion, to_region: ast.MemoryRegion, span: ast.SourceSpan) !bool {
        // Find applicable transition rule
        for (self.transition_rules) |rule| {
            if (rule.from == from_region and rule.to == to_region) {
                if (!rule.allowed) {
                    const error_msg = try std.fmt.allocPrint(analyzer.allocator, "Invalid memory region transition from {s} to {s}", .{ @tagName(from_region), @tagName(to_region) });
                    try semantics_errors.addError(analyzer, error_msg, span);
                    return false;
                }

                if (rule.requires_explicit_cast) {
                    const warning_msg = try std.fmt.allocPrint(analyzer.allocator, "Memory region transition from {s} to {s} requires explicit cast", .{ @tagName(from_region), @tagName(to_region) });
                    try semantics_errors.addWarning(analyzer, warning_msg, span);
                }

                if (rule.warning_message) |msg| {
                    try semantics_errors.addWarningStatic(analyzer, msg, span);
                }

                return true;
            }
        }

        // No specific rule found - default to allowing with warning
        const warning_msg = try std.fmt.allocPrint(analyzer.allocator, "Unspecified memory region transition from {s} to {s}", .{ @tagName(from_region), @tagName(to_region) });
        try semantics_errors.addWarning(analyzer, warning_msg, span);
        return true;
    }

    /// Validate assignment between different memory regions
    pub fn validateRegionAssignment(self: *MemoryRegionValidator, analyzer: *SemanticAnalyzer, target_region: ast.MemoryRegion, source_region: ast.MemoryRegion, span: ast.SourceSpan) !bool {
        // Storage assignments have special rules
        if (target_region == .Storage) {
            const context = self.getCurrentContext();

            // Storage can only be modified in contract functions
            if (!context.in_contract) {
                try semantics_errors.addErrorStatic(analyzer, "Storage can only be modified within contract functions", span);
                return false;
            }
        }

        // Immutable assignments have special rules
        if (target_region == .Immutable) {
            const context = self.getCurrentContext();

            // Immutable can only be assigned in constructor
            if (!context.in_constructor) {
                try semantics_errors.addErrorStatic(analyzer, "Immutable variables can only be assigned in constructor", span);
                return false;
            }
        }

        // Validate the transition
        return self.validateRegionTransition(analyzer, source_region, target_region, span);
    }

    /// Suggest alternative memory region
    fn suggestAlternativeRegion(self: *MemoryRegionValidator, context: *MemoryRegionContext, requested_region: ast.MemoryRegion) ?ast.MemoryRegion {
        _ = self;

        switch (requested_region) {
            .Storage => {
                if (!context.in_contract) {
                    return .Stack; // Suggest stack for non-contract contexts
                }
            },
            .Immutable => {
                if (!context.in_contract) {
                    return .Stack; // Suggest stack for non-contract contexts
                }
            },
            .Stack => {
                if (context.in_contract and context.current_function == null) {
                    return .Storage; // Suggest storage for contract-level variables
                }
            },
            .Heap => {
                return .Stack; // Generally suggest stack over heap for safety
            },
        }

        return null;
    }

    /// Validate memory region access
    pub fn validateRegionAccess(self: *MemoryRegionValidator, analyzer: *SemanticAnalyzer, region: ast.MemoryRegion, access_type: RegionAccessType, span: ast.SourceSpan) !bool {
        const context = self.getCurrentContext();

        switch (region) {
            .Storage => {
                if (!context.in_contract) {
                    try semantics_errors.addErrorStatic(analyzer, "Storage access only allowed within contracts", span);
                    return false;
                }

                if (access_type == .Write and context.current_function != null) {
                    // Storage writes in functions should be validated for state changes
                    const warning_msg = try std.fmt.allocPrint(analyzer.allocator, "Storage modification in function - ensure proper state management");
                    try semantics_errors.addWarning(analyzer, warning_msg, span);
                }
            },
            .Immutable => {
                if (access_type == .Write and !context.in_constructor) {
                    try semantics_errors.addErrorStatic(analyzer, "Immutable variables can only be written in constructor", span);
                    return false;
                }
            },
            .Stack => {
                // Stack access is generally safe
            },
            .Heap => {
                // Heap access requires careful validation
                const warning_msg = try std.fmt.allocPrint(analyzer.allocator, "Heap access - ensure proper memory management");
                try semantics_errors.addWarning(analyzer, warning_msg, span);
            },
        }

        return true;
    }
};

/// Memory region access type
pub const RegionAccessType = enum {
    Read,
    Write,
};

/// Default memory region transition rules
const default_transition_rules = [_]MemoryRegionTransition{
    // Stack transitions
    .{ .from = .Stack, .to = .Stack, .allowed = true, .requires_explicit_cast = false, .warning_message = null },
    .{ .from = .Stack, .to = .Heap, .allowed = true, .requires_explicit_cast = true, .warning_message = "Stack to heap transition may cause memory leaks" },
    .{ .from = .Stack, .to = .Storage, .allowed = false, .requires_explicit_cast = false, .warning_message = null },
    .{ .from = .Stack, .to = .Immutable, .allowed = false, .requires_explicit_cast = false, .warning_message = null },

    // Heap transitions
    .{ .from = .Heap, .to = .Stack, .allowed = true, .requires_explicit_cast = true, .warning_message = "Heap to stack transition - ensure memory is properly freed" },
    .{ .from = .Heap, .to = .Heap, .allowed = true, .requires_explicit_cast = false, .warning_message = null },
    .{ .from = .Heap, .to = .Storage, .allowed = false, .requires_explicit_cast = false, .warning_message = null },
    .{ .from = .Heap, .to = .Immutable, .allowed = false, .requires_explicit_cast = false, .warning_message = null },

    // Storage transitions
    .{ .from = .Storage, .to = .Stack, .allowed = true, .requires_explicit_cast = false, .warning_message = "Storage to stack - value will be copied" },
    .{ .from = .Storage, .to = .Heap, .allowed = true, .requires_explicit_cast = true, .warning_message = "Storage to heap transition requires careful memory management" },
    .{ .from = .Storage, .to = .Storage, .allowed = true, .requires_explicit_cast = false, .warning_message = null },
    .{ .from = .Storage, .to = .Immutable, .allowed = false, .requires_explicit_cast = false, .warning_message = null },

    // Immutable transitions
    .{ .from = .Immutable, .to = .Stack, .allowed = true, .requires_explicit_cast = false, .warning_message = "Immutable to stack - value will be copied" },
    .{ .from = .Immutable, .to = .Heap, .allowed = true, .requires_explicit_cast = true, .warning_message = "Immutable to heap transition requires careful memory management" },
    .{ .from = .Immutable, .to = .Storage, .allowed = false, .requires_explicit_cast = false, .warning_message = null },
    .{ .from = .Immutable, .to = .Immutable, .allowed = true, .requires_explicit_cast = false, .warning_message = null },
};

/// Validate memory region for variable declaration
pub fn validateVariableRegion(analyzer: *SemanticAnalyzer, var_decl: *ast.VariableDeclNode) semantics_errors.SemanticError!void {
    // Create a temporary validator for this validation
    var validator = MemoryRegionValidator.init(analyzer.allocator);
    defer validator.deinit();

    // Set up context based on analyzer state
    var context = MemoryRegionContext.init();
    if (analyzer.current_contract) |contract| {
        context.setContractContext(contract.name, analyzer.in_constructor);
    }
    if (analyzer.current_function) |function| {
        context.setFunctionContext(function);
    }

    try validator.pushContext(context);
    defer validator.popContext();

    const result = validator.validateVariableRegion(analyzer, var_decl) catch |err| {
        switch (err) {
            error.OutOfMemory => return semantics_errors.SemanticError.OutOfMemory,
            else => {
                try semantics_errors.addErrorStatic(analyzer, "Memory region validation failed", var_decl.span);
                return;
            },
        }
    };

    if (!result.valid) {
        if (result.error_message) |msg| {
            try semantics_errors.addErrorStatic(analyzer, msg, var_decl.span);
        }
    }
}

/// Validate memory region assignment
pub fn validateRegionAssignment(analyzer: *SemanticAnalyzer, target_region: ast.MemoryRegion, source_region: ast.MemoryRegion, span: ast.SourceSpan) semantics_errors.SemanticError!void {
    // Create a temporary validator for this validation
    var validator = MemoryRegionValidator.init(analyzer.allocator);
    defer validator.deinit();

    // Set up context based on analyzer state
    var context = MemoryRegionContext.init();
    if (analyzer.current_contract) |contract| {
        context.setContractContext(contract.name, analyzer.in_constructor);
    }
    if (analyzer.current_function) |function| {
        context.setFunctionContext(function);
    }

    try validator.pushContext(context);
    defer validator.popContext();

    const valid = validator.validateRegionAssignment(analyzer, target_region, source_region, span) catch |err| {
        switch (err) {
            error.OutOfMemory => return semantics_errors.SemanticError.OutOfMemory,
            else => {
                try semantics_errors.addErrorStatic(analyzer, "Memory region assignment validation failed", span);
                return;
            },
        }
    };

    if (!valid) {
        try semantics_errors.addErrorStatic(analyzer, "Invalid memory region assignment", span);
    }
}
