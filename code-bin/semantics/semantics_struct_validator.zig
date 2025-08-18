const std = @import("std");
pub const ast = @import("../ast.zig");
pub const typer = @import("../typer.zig");
const semantics_errors = @import("semantics_errors.zig");
const semantics_memory_safety = @import("semantics_memory_safety.zig");

// Forward declaration for SemanticAnalyzer
const SemanticAnalyzer = @import("semantics_core.zig").SemanticAnalyzer;

/// Struct validation context
pub const StructValidationContext = struct {
    struct_name: []const u8,
    fields: std.ArrayList(FieldInfo),
    methods: std.ArrayList(MethodInfo),
    is_packed: bool,
    alignment: ?u32,
    allocator: std.mem.Allocator,

    pub const FieldInfo = struct {
        name: []const u8,
        field_type: typer.OraType,
        is_public: bool,
        is_mutable: bool,
        default_value: ?*ast.ExprNode,
        span: ast.SourceSpan,
    };

    pub const MethodInfo = struct {
        name: []const u8,
        is_public: bool,
        is_static: bool,
        parameters: []typer.OraType,
        return_type: ?typer.OraType,
        span: ast.SourceSpan,
    };

    pub fn init(allocator: std.mem.Allocator, name: []const u8) StructValidationContext {
        return StructValidationContext{
            .struct_name = name,
            .fields = std.ArrayList(FieldInfo).init(allocator),
            .methods = std.ArrayList(MethodInfo).init(allocator),
            .is_packed = false,
            .alignment = null,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *StructValidationContext) void {
        self.fields.deinit();
        self.methods.deinit();
    }

    pub fn addField(self: *StructValidationContext, field: FieldInfo) !void {
        try self.fields.append(field);
    }

    pub fn addMethod(self: *StructValidationContext, method: MethodInfo) !void {
        try self.methods.append(method);
    }

    pub fn hasField(self: *StructValidationContext, name: []const u8) bool {
        for (self.fields.items) |field| {
            if (std.mem.eql(u8, field.name, name)) {
                return true;
            }
        }
        return false;
    }

    pub fn getField(self: *StructValidationContext, name: []const u8) ?FieldInfo {
        for (self.fields.items) |field| {
            if (std.mem.eql(u8, field.name, name)) {
                return field;
            }
        }
        return null;
    }

    pub fn hasMethod(self: *StructValidationContext, name: []const u8) bool {
        for (self.methods.items) |method| {
            if (std.mem.eql(u8, method.name, name)) {
                return true;
            }
        }
        return false;
    }
};

/// Struct field validation result
pub const FieldValidationResult = struct {
    valid: bool,
    field_name: []const u8,
    error_message: ?[]const u8,
    warnings: [][]const u8,
};

/// Struct method validation result
pub const MethodValidationResult = struct {
    valid: bool,
    method_name: []const u8,
    error_message: ?[]const u8,
    warnings: [][]const u8,
};

/// Struct validation result
pub const StructValidationResult = struct {
    valid: bool,
    struct_name: []const u8,
    field_results: []FieldValidationResult,
    method_results: []MethodValidationResult,
    error_message: ?[]const u8,
    warnings: [][]const u8,
};

/// Struct validator for comprehensive struct validation
pub const StructValidator = struct {
    allocator: std.mem.Allocator,
    validation_contexts: std.HashMap([]const u8, StructValidationContext, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),

    pub fn init(allocator: std.mem.Allocator) StructValidator {
        return StructValidator{
            .allocator = allocator,
            .validation_contexts = std.HashMap([]const u8, StructValidationContext, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
        };
    }

    pub fn deinit(self: *StructValidator) void {
        var iterator = self.validation_contexts.iterator();
        while (iterator.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.validation_contexts.deinit();
    }

    /// Validate struct declaration
    pub fn validateStruct(self: *StructValidator, analyzer: *SemanticAnalyzer, struct_decl: *ast.StructNode) !StructValidationResult {
        // Validate struct name
        if (!semantics_memory_safety.isValidString(analyzer, struct_decl.name)) {
            try semantics_errors.addErrorStatic(analyzer, "Invalid struct name", struct_decl.span);
            return StructValidationResult{
                .valid = false,
                .struct_name = struct_decl.name,
                .field_results = &[_]FieldValidationResult{},
                .method_results = &[_]MethodValidationResult{},
                .error_message = "Invalid struct name",
                .warnings = &[_][]const u8{},
            };
        }

        // Create validation context
        var context = StructValidationContext.init(self.allocator, struct_decl.name);
        defer context.deinit();

        // Validate struct fields
        var field_results = std.ArrayList(FieldValidationResult).init(self.allocator);
        defer field_results.deinit();

        for (struct_decl.fields) |*field| {
            const field_result = try self.validateStructField(analyzer, &context, field);
            try field_results.append(field_result);
        }

        // Validate struct methods (if any)
        var method_results = std.ArrayList(MethodValidationResult).init(self.allocator);
        defer method_results.deinit();

        // Check for duplicate field names
        try self.validateNoDuplicateFields(analyzer, &context);

        // Check for circular dependencies in field types
        try self.validateNoCircularFieldDependencies(analyzer, &context);

        // Validate struct layout and alignment
        try self.validateStructLayout(analyzer, &context);

        // Store validation context for future reference
        try self.validation_contexts.put(struct_decl.name, context);

        return StructValidationResult{
            .valid = true,
            .struct_name = struct_decl.name,
            .field_results = try field_results.toOwnedSlice(),
            .method_results = try method_results.toOwnedSlice(),
            .error_message = null,
            .warnings = &[_][]const u8{},
        };
    }

    /// Validate struct field
    fn validateStructField(self: *StructValidator, analyzer: *SemanticAnalyzer, context: *StructValidationContext, field: *ast.StructFieldNode) !FieldValidationResult {
        _ = self;

        // Validate field name
        if (!semantics_memory_safety.isValidString(analyzer, field.name)) {
            try semantics_errors.addErrorStatic(analyzer, "Invalid field name", field.span);
            return FieldValidationResult{
                .valid = false,
                .field_name = field.name,
                .error_message = "Invalid field name",
                .warnings = &[_][]const u8{},
            };
        }

        // Check for duplicate field names
        if (context.hasField(field.name)) {
            const error_msg = try std.fmt.allocPrint(analyzer.allocator, "Duplicate field name: {s}", .{field.name});
            try semantics_errors.addError(analyzer, error_msg, field.span);
            return FieldValidationResult{
                .valid = false,
                .field_name = field.name,
                .error_message = "Duplicate field name",
                .warnings = &[_][]const u8{},
            };
        }

        // Convert AST type to Ora type for validation
        const field_type = analyzer.type_checker.convertAstTypeToOraType(&field.field_type) catch |err| {
            const error_msg = switch (err) {
                typer.TyperError.TypeMismatch => "Invalid field type",
                typer.TyperError.OutOfMemory => return semantics_errors.SemanticError.OutOfMemory,
                else => "Unknown type error",
            };
            try semantics_errors.addErrorStatic(analyzer, error_msg, field.span);
            return FieldValidationResult{
                .valid = false,
                .field_name = field.name,
                .error_message = error_msg,
                .warnings = &[_][]const u8{},
            };
        };

        // Validate field type is valid for struct fields
        if (!isValidStructFieldType(field_type)) {
            try semantics_errors.addErrorStatic(analyzer, "Invalid type for struct field", field.span);
            return FieldValidationResult{
                .valid = false,
                .field_name = field.name,
                .error_message = "Invalid type for struct field",
                .warnings = &[_][]const u8{},
            };
        }

        // Add field to context
        const field_info = StructValidationContext.FieldInfo{
            .name = field.name,
            .field_type = field_type,
            .is_public = field.pub_,
            .is_mutable = field.mutable,
            .default_value = field.default_value,
            .span = field.span,
        };
        try context.addField(field_info);

        return FieldValidationResult{
            .valid = true,
            .field_name = field.name,
            .error_message = null,
            .warnings = &[_][]const u8{},
        };
    }

    /// Validate no duplicate fields
    fn validateNoDuplicateFields(self: *StructValidator, analyzer: *SemanticAnalyzer, context: *StructValidationContext) !void {
        _ = self;
        var seen_fields = std.HashMap([]const u8, bool, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(analyzer.allocator);
        defer seen_fields.deinit();

        for (context.fields.items) |field| {
            if (seen_fields.contains(field.name)) {
                const error_msg = try std.fmt.allocPrint(analyzer.allocator, "Duplicate field '{s}' in struct '{s}'", .{ field.name, context.struct_name });
                try semantics_errors.addError(analyzer, error_msg, field.span);
            } else {
                try seen_fields.put(field.name, true);
            }
        }
    }

    /// Validate no circular field dependencies
    fn validateNoCircularFieldDependencies(self: *StructValidator, analyzer: *SemanticAnalyzer, context: *StructValidationContext) !void {
        _ = self;
        _ = analyzer;
        _ = context;
        // TODO: Implement circular dependency detection for struct fields
        // This would involve checking if any field type references the struct itself
        // either directly or through a chain of other structs
    }

    /// Validate struct layout and alignment
    fn validateStructLayout(self: *StructValidator, analyzer: *SemanticAnalyzer, context: *StructValidationContext) !void {
        _ = self;
        _ = analyzer;
        _ = context;
        // TODO: Implement struct layout validation
        // This would check for proper field alignment, padding, and memory layout
    }

    /// Validate struct instantiation
    pub fn validateStructInstantiation(self: *StructValidator, analyzer: *SemanticAnalyzer, struct_inst: *ast.StructInstantiationNode) !bool {
        // Get struct validation context
        const context = self.validation_contexts.get(struct_inst.struct_name) orelse {
            try semantics_errors.addErrorStatic(analyzer, "Unknown struct type", struct_inst.span);
            return false;
        };

        // Validate all required fields are provided
        var provided_fields = std.HashMap([]const u8, bool, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(analyzer.allocator);
        defer provided_fields.deinit();

        for (struct_inst.fields) |field| {
            try provided_fields.put(field.name, true);

            // Check if field exists in struct
            if (!context.hasField(field.name)) {
                const error_msg = try std.fmt.allocPrint(analyzer.allocator, "Field '{s}' does not exist in struct '{s}'", .{ field.name, struct_inst.struct_name });
                try semantics_errors.addError(analyzer, error_msg, struct_inst.span);
                return false;
            }

            // Validate field value type matches field type
            // This would be done by the type checker, but we can add semantic validation here
        }

        // Check for missing required fields (fields without default values)
        for (context.fields.items) |field| {
            if (!provided_fields.contains(field.name) and field.default_value == null) {
                const error_msg = try std.fmt.allocPrint(analyzer.allocator, "Missing required field '{s}' in struct instantiation", .{field.name});
                try semantics_errors.addError(analyzer, error_msg, struct_inst.span);
                return false;
            }
        }

        return true;
    }

    /// Validate struct field access
    pub fn validateStructFieldAccess(self: *StructValidator, analyzer: *SemanticAnalyzer, field_access: *ast.FieldAccessNode, struct_type_name: []const u8) !bool {
        // Get struct validation context
        const context = self.validation_contexts.get(struct_type_name) orelse {
            try semantics_errors.addErrorStatic(analyzer, "Unknown struct type for field access", field_access.span);
            return false;
        };

        // Check if field exists
        const field_info = context.getField(field_access.field) orelse {
            const error_msg = try std.fmt.allocPrint(analyzer.allocator, "Field '{s}' does not exist in struct '{s}'", .{ field_access.field, struct_type_name });
            try semantics_errors.addError(analyzer, error_msg, field_access.span);
            return false;
        };

        // Check field visibility (if accessing from outside the struct)
        if (!field_info.is_public) {
            // TODO: Add context checking to determine if access is from within the same module/struct
            const warning_msg = try std.fmt.allocPrint(analyzer.allocator, "Accessing private field '{s}' of struct '{s}'", .{ field_access.field, struct_type_name });
            try semantics_errors.addWarning(analyzer, warning_msg, field_access.span);
        }

        return true;
    }
};

/// Check if a type is valid for struct fields
fn isValidStructFieldType(field_type: typer.OraType) bool {
    return switch (field_type) {
        .Unknown => false,
        .Void => false, // Void fields don't make sense
        .U8, .U16, .U32, .U64, .U128, .U256 => true,
        .I8, .I16, .I32, .I64, .I128, .I256 => true,
        .Bool => true,
        .Address => true,
        .String => true,
        .Array => true,
        .Mapping => true,
        .Struct => true,
        .Enum => true,
        .Function => false, // Function fields need special handling
        .ErrorUnion => true,
        .Generic => true,
    };
}

/// Validate struct declaration
pub fn validateStruct(analyzer: *SemanticAnalyzer, struct_decl: *ast.StructNode) semantics_errors.SemanticError!void {
    // Create a temporary validator for this validation
    var validator = StructValidator.init(analyzer.allocator);
    defer validator.deinit();

    const result = validator.validateStruct(analyzer, struct_decl) catch |err| {
        switch (err) {
            error.OutOfMemory => return semantics_errors.SemanticError.OutOfMemory,
            else => {
                try semantics_errors.addErrorStatic(analyzer, "Struct validation failed", struct_decl.span);
                return;
            },
        }
    };

    if (!result.valid) {
        if (result.error_message) |msg| {
            try semantics_errors.addErrorStatic(analyzer, msg, struct_decl.span);
        }
    }
}

/// Validate struct instantiation
pub fn validateStructInstantiation(analyzer: *SemanticAnalyzer, struct_inst: *ast.StructInstantiationNode) semantics_errors.SemanticError!void {
    // Create a temporary validator for this validation
    var validator = StructValidator.init(analyzer.allocator);
    defer validator.deinit();

    const valid = validator.validateStructInstantiation(analyzer, struct_inst) catch |err| {
        switch (err) {
            error.OutOfMemory => return semantics_errors.SemanticError.OutOfMemory,
            else => {
                try semantics_errors.addErrorStatic(analyzer, "Struct instantiation validation failed", struct_inst.span);
                return;
            },
        }
    };

    if (!valid) {
        try semantics_errors.addErrorStatic(analyzer, "Invalid struct instantiation", struct_inst.span);
    }
}
