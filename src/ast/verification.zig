const std = @import("std");
const SourceSpan = @import("../ast.zig").SourceSpan;
const TypeInfo = @import("type_info.zig").TypeInfo;

/// Verification attribute types for formal verification constructs
pub const VerificationAttributeType = enum {
    /// Quantified expression attribute (forall/exists)
    Quantified,
    /// Assertion attribute
    Assertion,
    /// Invariant attribute
    Invariant,
    /// Precondition attribute (requires)
    Precondition,
    /// Postcondition attribute (ensures)
    Postcondition,
    /// Loop invariant attribute
    LoopInvariant,
    /// Custom verification attribute
    Custom,
};

/// Verification attribute with metadata
pub const VerificationAttribute = struct {
    /// Type of verification attribute
    attr_type: VerificationAttributeType,
    /// Name of the attribute (for custom attributes)
    name: ?[]const u8,
    /// Value of the attribute (string representation)
    value: ?[]const u8,
    /// Source span for error reporting
    span: SourceSpan,

    pub fn init(attr_type: VerificationAttributeType, span: SourceSpan) VerificationAttribute {
        return VerificationAttribute{
            .attr_type = attr_type,
            .name = null,
            .value = null,
            .span = span,
        };
    }

    pub fn initCustom(name: []const u8, value: ?[]const u8, span: SourceSpan) VerificationAttribute {
        return VerificationAttribute{
            .attr_type = .Custom,
            .name = name,
            .value = value,
            .span = span,
        };
    }

    pub fn deinit(self: *VerificationAttribute, allocator: std.mem.Allocator) void {
        // Only free strings if they were allocated (not string literals)
        // For now, we'll skip freeing to avoid crashes with string literals
        // In a real implementation, we'd track whether strings were allocated
        _ = allocator;
        _ = self;
    }
};

/// Verification metadata for quantified expressions
pub const QuantifiedMetadata = struct {
    /// Quantifier type (forall/exists)
    quantifier_type: QuantifierType,
    /// Bound variable name
    variable_name: []const u8,
    /// Type of bound variable
    variable_type: TypeInfo,
    /// Optional condition (where clause)
    has_condition: bool,
    /// Verification domain (e.g., "arithmetic", "array", "custom")
    domain: ?[]const u8,
    /// Additional verification attributes
    attributes: []VerificationAttribute,
    /// Source span
    span: SourceSpan,

    pub fn init(quantifier_type: QuantifierType, variable_name: []const u8, variable_type: TypeInfo, span: SourceSpan) QuantifiedMetadata {
        return QuantifiedMetadata{
            .quantifier_type = quantifier_type,
            .variable_name = variable_name,
            .variable_type = variable_type,
            .has_condition = false,
            .domain = null,
            .attributes = &[_]VerificationAttribute{},
            .span = span,
        };
    }

    pub fn deinit(self: *QuantifiedMetadata, allocator: std.mem.Allocator) void {
        // Only free strings if they were allocated (not string literals)
        // For now, we'll skip freeing to avoid crashes with string literals
        // In a real implementation, we'd track whether strings were allocated
        _ = allocator;
        _ = self;
    }
};

// Use the existing QuantifierType from expressions
const QuantifierType = @import("expressions.zig").QuantifierType;

/// Verification context for tracking verification constructs
pub const VerificationContext = struct {
    /// Current verification mode
    mode: VerificationMode,
    /// Stack of verification scopes
    scope_stack: std.ArrayList(VerificationScope),
    /// Current verification attributes
    current_attributes: std.ArrayList(VerificationAttribute),
    /// Allocator
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) VerificationContext {
        return VerificationContext{
            .mode = .None,
            .scope_stack = std.ArrayList(VerificationScope).init(allocator),
            .current_attributes = std.ArrayList(VerificationAttribute).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *VerificationContext) void {
        // Only free strings if they were allocated (not string literals)
        // For now, we'll skip freeing to avoid crashes with string literals
        // In a real implementation, we'd track whether strings were allocated
        self.scope_stack.deinit();
        self.current_attributes.deinit();
    }

    pub fn pushScope(self: *VerificationContext, scope: VerificationScope) !void {
        try self.scope_stack.append(scope);
    }

    pub fn popScope(self: *VerificationContext) ?VerificationScope {
        return self.scope_stack.popOrNull();
    }

    pub fn addAttribute(self: *VerificationContext, attr: VerificationAttribute) !void {
        try self.current_attributes.append(attr);
    }

    pub fn clearAttributes(self: *VerificationContext) void {
        for (self.current_attributes.items) |*attr| {
            attr.deinit(self.allocator);
        }
        self.current_attributes.clearRetainingCapacity();
    }
};

/// Verification mode for different contexts
pub const VerificationMode = enum {
    None, // No verification
    Precondition, // Precondition verification
    Postcondition, // Postcondition verification
    Invariant, // Invariant verification
    Quantified, // Quantified expression verification
    Assertion, // Assertion verification
};

/// Verification scope for tracking verification constructs
pub const VerificationScope = struct {
    /// Scope type
    scope_type: VerificationScopeType,
    /// Scope name/identifier
    name: ?[]const u8,
    /// Verification attributes in this scope
    attributes: []VerificationAttribute,
    /// Source span
    span: SourceSpan,

    pub fn init(scope_type: VerificationScopeType, span: SourceSpan) VerificationScope {
        return VerificationScope{
            .scope_type = scope_type,
            .name = null,
            .attributes = &[_]VerificationAttribute{},
            .span = span,
        };
    }

    pub fn deinit(self: *VerificationScope, allocator: std.mem.Allocator) void {
        if (self.name) |name| {
            allocator.free(name);
        }
        for (self.attributes) |*attr| {
            attr.deinit(allocator);
        }
        allocator.free(self.attributes);
    }
};

/// Verification scope types
pub const VerificationScopeType = enum {
    Function, // Function scope
    Contract, // Contract scope
    Loop, // Loop scope
    Quantified, // Quantified expression scope
    Block, // Block scope
};

/// Verification result for verification operations
pub const VerificationResult = union(enum) {
    Success: struct {
        message: ?[]const u8,
    },
    Warning: struct {
        message: []const u8,
        span: SourceSpan,
    },
    Error: struct {
        message: []const u8,
        span: SourceSpan,
    },
};

/// Verification error types
pub const VerificationError = error{
    InvalidQuantifiedExpression,
    InvalidVerificationAttribute,
    UnsupportedVerificationConstruct,
    VerificationContextMismatch,
    InvalidVerificationScope,
    VerificationMetadataError,
};
