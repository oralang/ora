const std = @import("std");
const ast = @import("ast.zig");
const typer = @import("typer.zig");
const comptime_eval = @import("comptime_eval.zig");
const static_verifier = @import("static_verifier.zig");
const formal_verifier = @import("formal_verifier.zig");
const optimizer = @import("optimizer.zig");

/// Semantic analysis errors
pub const SemanticError = error{
    // Contract-level errors
    MissingInitFunction,
    InvalidContractStructure,
    DuplicateInitFunction,
    InitFunctionNotPublic,

    // Memory semantics errors
    InvalidStorageAccess,
    ImmutableViolation,
    InvalidMemoryTransition,
    StorageInNonPersistentContext,

    // Function semantics errors
    MissingReturnStatement,
    UnreachableCode,
    InvalidReturnType,
    VoidReturnInNonVoidFunction,

    // Formal verification errors
    InvalidRequiresClause,
    InvalidEnsuresClause,
    OldExpressionInRequires,
    InvalidInvariant,

    // Error union semantic errors
    DuplicateErrorDeclaration,
    UndeclaredError,
    InvalidErrorType,
    InvalidErrorUnionCast,
    InvalidErrorUnionTarget,

    // General semantic errors
    UndeclaredIdentifier,
    TypeMismatch,
    InvalidOperation,
    CircularDependency,
    OutOfMemory,

    // Memory safety errors
    PointerValidationFailed,
    MessageValidationFailed,
    AnalysisStateCorrupted,
    RecoveryFailed,
};

/// Semantic diagnostic with location and severity
pub const Diagnostic = struct {
    message: []const u8,
    span: ast.SourceSpan,
    severity: Severity,
    context: ?DiagnosticContext, // NEW: Additional error context

    pub const Severity = enum {
        Error,
        Warning,
        Info,
    };

    pub fn format(self: Diagnostic, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;

        // Safe message handling with validation
        const safe_message = self.validateMessage();
        const safe_span = self.validateSpan();

        // Enhanced formatting with context
        try writer.print("{s} at line {}, column {}: {s}", .{ @tagName(self.severity), safe_span.line, safe_span.column, safe_message });

        // Add context information if available
        if (self.context) |ctx| {
            try writer.print(" [Node: {s}, Phase: {s}]", .{ @tagName(ctx.node_type), @tagName(ctx.analysis_phase) });
        }
    }

    /// Validate and sanitize diagnostic message
    fn validateMessage(self: *const Diagnostic) []const u8 {
        if (self.message.len == 0) {
            return "<empty message>";
        }

        // Check for reasonable message length
        if (self.message.len > 1024) {
            return "<message too long>";
        }

        // Basic UTF-8 validation - check if all bytes are printable ASCII or valid UTF-8 start bytes
        for (self.message) |byte| {
            if (byte < 32 or byte == 127) { // Non-printable ASCII
                if (byte != '\t' and byte != '\n' and byte != '\r') {
                    return "<corrupted message>";
                }
            }
        }

        return self.message;
    }

    /// Validate and sanitize source span
    fn validateSpan(self: *const Diagnostic) ast.SourceSpan {
        return ast.SourceSpan{
            .line = if (self.span.line > 1000000) 0 else self.span.line,
            .column = if (self.span.column > 10000) 0 else self.span.column,
            .length = if (self.span.length > 10000) 0 else self.span.length,
        };
    }
};

/// Analysis state tracking for memory safety
pub const AnalysisState = struct {
    phase: AnalysisPhase,
    current_node_type: ?std.meta.Tag(ast.AstNode),
    error_count: u32,
    warning_count: u32,
    safety_checks_enabled: bool,

    pub const AnalysisPhase = enum {
        PreInitialization,
        TypeChecking,
        SemanticAnalysis,
        Validation,
    };

    pub fn init() AnalysisState {
        return AnalysisState{
            .phase = .PreInitialization,
            .current_node_type = null,
            .error_count = 0,
            .warning_count = 0,
            .safety_checks_enabled = true,
        };
    }
};

/// Validation coverage tracking
pub const ValidationCoverage = struct {
    visited_node_types: std.EnumSet(std.meta.Tag(ast.AstNode)),
    missing_implementations: std.ArrayList(std.meta.Tag(ast.AstNode)),
    validation_stats: ValidationStats,

    pub const ValidationStats = struct {
        nodes_analyzed: u32,
        errors_found: u32,
        warnings_generated: u32,
        validations_skipped: u32,
        recovery_attempts: u32,

        pub fn init() ValidationStats {
            return ValidationStats{
                .nodes_analyzed = 0,
                .errors_found = 0,
                .warnings_generated = 0,
                .validations_skipped = 0,
                .recovery_attempts = 0,
            };
        }
    };

    pub fn init(allocator: std.mem.Allocator) ValidationCoverage {
        return ValidationCoverage{
            .visited_node_types = std.EnumSet(std.meta.Tag(ast.AstNode)).initEmpty(),
            .missing_implementations = std.ArrayList(std.meta.Tag(ast.AstNode)).init(allocator),
            .validation_stats = ValidationStats.init(),
        };
    }

    pub fn deinit(self: *ValidationCoverage) void {
        self.missing_implementations.deinit();
    }
};

/// Enhanced diagnostic with context for better error reporting
pub const DiagnosticContext = struct {
    node_type: std.meta.Tag(ast.AstNode),
    analysis_phase: AnalysisState.AnalysisPhase,
    recovery_attempted: bool,
    additional_info: ?[]const u8,
};

/// Contract analysis context
pub const ContractContext = struct {
    name: []const u8,
    has_init: bool,
    init_is_public: bool,
    storage_variables: std.ArrayList([]const u8),
    functions: std.ArrayList([]const u8),
    events: std.ArrayList([]const u8),

    pub fn init(allocator: std.mem.Allocator, name: []const u8) ContractContext {
        return ContractContext{
            .name = name,
            .has_init = false,
            .init_is_public = false,
            .storage_variables = std.ArrayList([]const u8).init(allocator),
            .functions = std.ArrayList([]const u8).init(allocator),
            .events = std.ArrayList([]const u8).init(allocator),
        };
    }

    pub fn deinit(self: *ContractContext) void {
        self.storage_variables.deinit();
        self.functions.deinit();
        self.events.deinit();
    }
};

/// Semantic analyzer for Ora
pub const SemanticAnalyzer = struct {
    allocator: std.mem.Allocator,
    type_checker: typer.Typer,
    comptime_evaluator: comptime_eval.ComptimeEvaluator,
    static_verifier: static_verifier.StaticVerifier,
    formal_verifier: formal_verifier.FormalVerifier,
    optimizer: optimizer.Optimizer,
    diagnostics: std.ArrayList(Diagnostic),
    current_function: ?[]const u8,
    in_loop: bool,
    in_assignment_target: bool, // Flag to indicate if we are currently analyzing an assignment target

    // Error propagation tracking
    in_error_propagation_context: bool, // Track if we're in a context where errors can propagate
    current_function_returns_error_union: bool, // Track if current function returns error union

    // Immutable variable tracking
    current_contract: ?*ast.ContractNode,
    immutable_variables: std.HashMap([]const u8, ImmutableVarInfo, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),
    in_constructor: bool, // Flag to indicate if we're in a constructor (init function)

    // Memory safety and error recovery
    error_recovery_mode: bool, // Flag to indicate if we're in error recovery mode
    analysis_state: AnalysisState, // Track current analysis state
    validation_coverage: ValidationCoverage, // Track validation completeness

    /// Information about an immutable variable
    const ImmutableVarInfo = struct {
        name: []const u8,
        declared_span: ast.SourceSpan,
        initialized: bool,
        init_span: ?ast.SourceSpan,
    };

    pub fn init(allocator: std.mem.Allocator) SemanticAnalyzer {
        return SemanticAnalyzer{
            .allocator = allocator,
            .type_checker = typer.Typer.init(allocator),
            .comptime_evaluator = comptime_eval.ComptimeEvaluator.init(allocator),
            .static_verifier = static_verifier.StaticVerifier.init(allocator),
            .formal_verifier = formal_verifier.FormalVerifier.init(allocator),
            .optimizer = optimizer.Optimizer.init(allocator),
            .diagnostics = std.ArrayList(Diagnostic).init(allocator),
            .current_function = null,
            .in_loop = false,
            .in_assignment_target = false,
            .in_error_propagation_context = false,
            .current_function_returns_error_union = false,
            .current_contract = null,
            .immutable_variables = std.HashMap([]const u8, ImmutableVarInfo, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
            .in_constructor = false,
            .error_recovery_mode = false,
            .analysis_state = AnalysisState.init(),
            .validation_coverage = ValidationCoverage.init(allocator),
        };
    }

    /// Initialize self-references after the struct is in its final location
    pub fn initSelfReferences(self: *SemanticAnalyzer) void {
        self.type_checker.fixSelfReferences();
    }

    // MEMORY SAFETY UTILITIES - Critical for preventing segfaults

    /// Validate that an AST node pointer is safe to dereference
    fn isValidNodePointer(self: *SemanticAnalyzer, node: ?*ast.AstNode) bool {
        _ = self;
        if (node == null) return false;

        // Basic pointer validation - check if it's in a reasonable memory range
        const ptr_value = @intFromPtr(node.?);

        // Check for null and obviously invalid pointers
        if (ptr_value == 0) return false;

        // Platform-independent validation using standard page size
        const page_size: usize = 4096; // Use standard page size (4KB)
        if (ptr_value < page_size) return false; // Likely null pointer dereference

        // Check alignment for typical AST node structures (should be word-aligned)
        if (ptr_value % @alignOf(*ast.AstNode) != 0) return false;

        return true;
    }

    /// Validate that a string is safe to use
    fn isValidString(self: *SemanticAnalyzer, str: []const u8) bool {
        _ = self;
        // Check for reasonable string length (prevent reading garbage memory)
        if (str.len > 1024 * 1024) return false; // 1MB max string length

        // Check if the pointer looks valid
        const ptr_value = @intFromPtr(str.ptr);
        if (ptr_value == 0) return false;
        if (ptr_value < 0x1000) return false;

        return true;
    }

    /// Validate that a SourceSpan is safe to use
    fn validateSpan(self: *SemanticAnalyzer, span: ast.SourceSpan) ast.SourceSpan {
        _ = self;
        // Ensure span values are reasonable
        const safe_span = ast.SourceSpan{
            .line = if (span.line > 1000000) 0 else span.line,
            .column = if (span.column > 10000) 0 else span.column,
            .length = if (span.length > 10000) 0 else span.length,
        };
        return safe_span;
    }

    /// Get a default safe span for error reporting
    fn getDefaultSpan(self: *SemanticAnalyzer) ast.SourceSpan {
        _ = self;
        return ast.SourceSpan{ .line = 0, .column = 0, .length = 0 };
    }

    /// Extract span from any expression type
    fn getExpressionSpan(self: *SemanticAnalyzer, expr: *ast.ExprNode) ast.SourceSpan {
        _ = self;
        return switch (expr.*) {
            .Identifier => |*ident| ident.span,
            .Literal => |*lit| switch (lit.*) {
                .Integer => |*int| int.span,
                .String => |*str| str.span,
                .Bool => |*b| b.span,
                .Address => |*addr| addr.span,
                .Hex => |*hex| hex.span,
            },
            .Binary => |*bin| bin.span,
            .Unary => |*un| un.span,
            .Assignment => |*assign| assign.span,
            .CompoundAssignment => |*comp| comp.span,
            .Call => |*call| call.span,
            .Index => |*index| index.span,
            .FieldAccess => |*field| field.span,
            .Cast => |*cast| cast.span,
            .Comptime => |*comp| comp.span,
            .Old => |*old| old.span,
            .Tuple => |*tuple| tuple.span,
            .Try => |*try_expr| try_expr.span,
            .ErrorReturn => |*error_ret| error_ret.span,
            .ErrorCast => |*error_cast| error_cast.span,
            .Shift => |*shift| shift.span,
            .StructInstantiation => |*struct_inst| struct_inst.span,
            .EnumLiteral => |*enum_lit| enum_lit.span,
        };
    }

    /// Safely analyze a node with error recovery
    fn safeAnalyzeNode(self: *SemanticAnalyzer, node: *ast.AstNode) SemanticError!void {
        // Validate node pointer before proceeding
        if (!self.isValidNodePointer(node)) {
            try self.addError("Invalid node pointer detected", self.getDefaultSpan());
            self.validation_coverage.validation_stats.recovery_attempts += 1;
            return SemanticError.PointerValidationFailed;
        }

        // Set error recovery mode
        const prev_recovery = self.error_recovery_mode;
        self.error_recovery_mode = true;
        defer self.error_recovery_mode = prev_recovery;

        // Update analysis state
        self.analysis_state.current_node_type = @as(std.meta.Tag(ast.AstNode), node.*);
        self.validation_coverage.visited_node_types.insert(@as(std.meta.Tag(ast.AstNode), node.*));
        self.validation_coverage.validation_stats.nodes_analyzed += 1;

        // Attempt analysis with safety checks
        self.analyzeNode(node) catch |err| {
            self.validation_coverage.validation_stats.recovery_attempts += 1;
            try self.handleAnalysisError(err, node);
        };
    }

    /// Handle analysis errors gracefully
    fn handleAnalysisError(self: *SemanticAnalyzer, err: SemanticError, node: *ast.AstNode) SemanticError!void {
        const node_type_name = @tagName(@as(std.meta.Tag(ast.AstNode), node.*));

        switch (err) {
            SemanticError.PointerValidationFailed => {
                const message = std.fmt.allocPrint(self.allocator, "Pointer validation failed for {s} node", .{node_type_name}) catch "Pointer validation failed";
                defer if (!std.mem.eql(u8, message, "Pointer validation failed")) self.allocator.free(message);
                try self.addError(message, self.getDefaultSpan());
            },
            SemanticError.OutOfMemory => {
                try self.addError("Out of memory during analysis", self.getDefaultSpan());
                return err; // Don't recover from OOM
            },
            else => {
                const message = std.fmt.allocPrint(self.allocator, "Analysis error in {s} node: {s}", .{ node_type_name, @errorName(err) }) catch "Analysis error occurred";
                defer if (!std.mem.eql(u8, message, "Analysis error occurred")) self.allocator.free(message);
                try self.addWarning(message, self.getDefaultSpan());
            },
        }

        self.validation_coverage.validation_stats.errors_found += 1;
    }

    pub fn deinit(self: *SemanticAnalyzer) void {
        self.type_checker.deinit();
        self.comptime_evaluator.deinit();
        self.static_verifier.deinit();
        self.formal_verifier.deinit();
        self.optimizer.deinit();
        self.diagnostics.deinit();
        self.immutable_variables.deinit();
        self.validation_coverage.deinit();
    }

    /// Perform complete semantic analysis on AST nodes
    pub fn analyze(self: *SemanticAnalyzer, nodes: []ast.AstNode) SemanticError![]Diagnostic {
        // Phase 0: Pre-initialize type checker with contract symbols
        // This must happen before type checking so symbols are available
        for (nodes) |*node| {
            if (node.* == .Contract) {
                try self.preInitializeContractSymbols(&node.Contract);
            }
        }

        // Phase 1: Type checking
        self.type_checker.typeCheck(nodes) catch |err| {
            switch (err) {
                typer.TyperError.UndeclaredVariable => {
                    try self.addError("Undeclared variable", ast.SourceSpan{ .line = 0, .column = 0, .length = 0 });
                },
                typer.TyperError.TypeMismatch => {
                    try self.addError("Type mismatch", ast.SourceSpan{ .line = 0, .column = 0, .length = 0 });
                },
                else => return SemanticError.TypeMismatch,
            }
        };

        // Phase 2: Semantic analysis with safety checks
        self.analysis_state.phase = .SemanticAnalysis;
        for (nodes) |*node| {
            try self.safeAnalyzeNode(node);
        }

        // Phase 3: Contract-level validation is now handled within analyzeContract

        return try self.diagnostics.toOwnedSlice();
    }

    /// Pre-initialize contract symbols in type checker before type checking runs
    fn preInitializeContractSymbols(self: *SemanticAnalyzer, contract: *ast.ContractNode) SemanticError!void {
        // Initialize standard library in the current scope before processing contract
        self.type_checker.initStandardLibrary() catch |err| {
            switch (err) {
                typer.TyperError.OutOfMemory => return SemanticError.OutOfMemory,
                else => {
                    // Log the error but continue - standard library initialization is not critical
                    try self.addWarning("Failed to initialize standard library", ast.SourceSpan{ .line = 0, .column = 0, .length = 0 });
                },
            }
        };

        // Add storage variables and events to type checker symbol table
        for (contract.body) |*member| {
            switch (member.*) {
                .VariableDecl => |*var_decl| {
                    if (var_decl.region == .Storage or var_decl.region == .Immutable) {
                        // Add to type checker's global scope
                        const ora_type = self.type_checker.convertAstTypeToOraType(&var_decl.typ) catch |err| {
                            switch (err) {
                                typer.TyperError.TypeMismatch => return SemanticError.TypeMismatch,
                                typer.TyperError.OutOfMemory => return SemanticError.OutOfMemory,
                                else => return SemanticError.TypeMismatch,
                            }
                        };

                        const symbol = typer.Symbol{
                            .name = var_decl.name,
                            .typ = ora_type,
                            .region = var_decl.region,
                            .mutable = var_decl.mutable,
                            .span = var_decl.span,
                        };
                        try self.type_checker.current_scope.declare(symbol);
                    }
                },
                .LogDecl => |*log_decl| {
                    // Add event to symbol table
                    const event_symbol = typer.Symbol{
                        .name = log_decl.name,
                        .typ = typer.OraType.Unknown, // Event type
                        .region = ast.MemoryRegion.Stack,
                        .mutable = false,
                        .span = log_decl.span,
                    };
                    try self.type_checker.current_scope.declare(event_symbol);
                },
                else => {},
            }
        }
    }

    /// Analyze a single AST node
    fn analyzeNode(self: *SemanticAnalyzer, node: *ast.AstNode) SemanticError!void {
        switch (node.*) {
            .Contract => |*contract| {
                try self.analyzeContract(contract);
            },
            .Function => |*function| {
                try self.analyzeFunction(function);
            },
            .VariableDecl => |*var_decl| {
                try self.analyzeVariableDecl(var_decl);
            },
            .LogDecl => |*log_decl| {
                try self.analyzeLogDecl(log_decl);
            },
            .EnumDecl => |*enum_decl| {
                try self.analyzeEnumDecl(enum_decl);
            },
            .StructDecl => |*struct_decl| {
                try self.analyzeStructDecl(struct_decl);
            },
            .Import => |*import| {
                try self.analyzeImport(import);
            },
            .ErrorDecl => |*error_decl| {
                try self.analyzeErrorDecl(error_decl);
            },
            .Block => |*block| {
                try self.analyzeBlock(block);
            },
            .Expression => |*expr| {
                try self.analyzeExpression(expr);
            },
            .Statement => |*stmt| {
                try self.analyzeStatement(stmt);
            },
            .TryBlock => |*try_block| {
                try self.analyzeTryBlock(try_block);
            },
        }
    }

    /// Analyze contract declaration
    fn analyzeContract(self: *SemanticAnalyzer, contract: *ast.ContractNode) SemanticError!void {
        // Set current contract for immutable variable tracking
        self.current_contract = contract;
        defer self.current_contract = null;

        // Clear immutable variables from previous contract
        self.immutable_variables.clearRetainingCapacity();

        // Initialize contract context
        var contract_ctx = ContractContext.init(self.allocator, contract.name);
        // Ensure cleanup on error - this guarantees memory is freed even if analysis fails
        defer contract_ctx.deinit();

        // First pass: collect contract member names and immutable variables
        for (contract.body) |*member| {
            switch (member.*) {
                .VariableDecl => |*var_decl| {
                    if (var_decl.region == .Storage or var_decl.region == .Immutable) {
                        try contract_ctx.storage_variables.append(var_decl.name);
                    }

                    // Track immutable variables (including storage const)
                    if (var_decl.region == .Immutable or (var_decl.region == .Storage and !var_decl.mutable)) {
                        try self.immutable_variables.put(var_decl.name, ImmutableVarInfo{
                            .name = var_decl.name,
                            .declared_span = var_decl.span,
                            .initialized = var_decl.value != null,
                            .init_span = if (var_decl.value != null) var_decl.span else null,
                        });
                    }
                },
                .LogDecl => |*log_decl| {
                    try contract_ctx.events.append(log_decl.name);
                },
                else => {},
            }
        }

        // Second pass: analyze contract members
        for (contract.body) |*member| {
            switch (member.*) {
                .Function => |*function| {
                    // Track init function
                    if (std.mem.eql(u8, function.name, "init")) {
                        if (contract_ctx.has_init) {
                            try self.addError("Duplicate init function", function.span);
                            return SemanticError.DuplicateInitFunction;
                        }
                        contract_ctx.has_init = true;
                        contract_ctx.init_is_public = function.pub_;

                        // Set constructor flag for immutable variable validation
                        self.in_constructor = true;
                        defer self.in_constructor = false;
                    }

                    try contract_ctx.functions.append(function.name);
                    try self.analyzeFunction(function);
                },
                .VariableDecl => |*var_decl| {
                    try self.analyzeVariableDecl(var_decl);
                },
                .LogDecl => |*log_decl| {
                    try self.analyzeLogDecl(log_decl);
                },
                else => {
                    try self.analyzeNode(member);
                },
            }
        }

        // Validate all immutable variables are initialized
        try self.validateImmutableInitialization();

        // Validate contract after all members are analyzed
        try self.validateContract(&contract_ctx);
    }

    /// Analyze function declaration
    fn analyzeFunction(self: *SemanticAnalyzer, function: *ast.FunctionNode) SemanticError!void {
        const prev_function = self.current_function;
        self.current_function = function.name;
        defer self.current_function = prev_function;

        // Check if function returns error union
        const prev_returns_error_union = self.current_function_returns_error_union;
        self.current_function_returns_error_union = self.functionReturnsErrorUnion(function);
        defer self.current_function_returns_error_union = prev_returns_error_union;

        // Note: The type checker has already validated the function body in Phase 1
        // The semantic analyzer should not create new scopes or interfere with type checker scopes

        // Validate init function requirements
        if (std.mem.eql(u8, function.name, "init")) {
            try self.validateInitFunction(function);
        }

        // Analyze requires clauses
        for (function.requires_clauses) |*clause| {
            try self.analyzeRequiresClause(clause, function.span);
        }

        // Analyze ensures clauses
        for (function.ensures_clauses) |*clause| {
            try self.analyzeEnsuresClause(clause, function.span);
        }

        // Perform static verification on requires/ensures clauses
        try self.performStaticVerification(function);

        // Analyze function body with proper scope context
        try self.analyzeBlock(&function.body);

        // Validate return statements if function has return type
        if (function.return_type != null) {
            try self.validateReturnStatements(&function.body, function.span);
        }
    }

    /// Validate init function requirements
    fn validateInitFunction(self: *SemanticAnalyzer, function: *ast.FunctionNode) SemanticError!void {
        // Init functions should be public
        if (!function.pub_) {
            try self.addWarning("Init function should be public", function.span);
        }

        // Init functions should not have return type
        if (function.return_type != null) {
            try self.addError("Init function cannot have return type", function.span);
        }

        // Init functions should not have requires/ensures (for now)
        if (function.requires_clauses.len > 0) {
            try self.addWarning("Init function with requires clauses - verify carefully", function.span);
        }
    }

    /// Analyze variable declaration
    fn analyzeVariableDecl(self: *SemanticAnalyzer, var_decl: *ast.VariableDeclNode) SemanticError!void {
        // Validate memory region semantics
        try self.validateMemoryRegionSemantics(var_decl);

        // Validate immutable constraints
        if (!var_decl.mutable) {
            try self.validateImmutableSemantics(var_decl);
        }

        // Analyze initializer
        if (var_decl.value) |*init_expr| {
            try self.analyzeExpression(init_expr);

            // If this is a const variable, try to evaluate it at compile time
            if (var_decl.region == .Const) {
                if (self.comptime_evaluator.evaluate(init_expr)) |comptime_value| {
                    // Successfully evaluated at compile time - store the constant
                    self.comptime_evaluator.defineConstant(var_decl.name, comptime_value) catch |err| {
                        switch (err) {
                            error.OutOfMemory => return SemanticError.OutOfMemory,
                        }
                    };

                    // Add info diagnostic about successful comptime evaluation
                    const value_str = comptime_value.toString(self.allocator) catch "unknown";
                    defer self.allocator.free(value_str);

                    const message = std.fmt.allocPrint(self.allocator, "Const '{s}' evaluated at compile time: {s}", .{ var_decl.name, value_str }) catch "Const evaluated at compile time";
                    defer if (!std.mem.eql(u8, message, "Const evaluated at compile time")) self.allocator.free(message);

                    try self.addInfo(message, var_decl.span);
                } else |err| {
                    // Failed to evaluate at compile time
                    const error_msg = switch (err) {
                        comptime_eval.ComptimeError.NotCompileTimeEvaluable => "Expression is not evaluable at compile time",
                        comptime_eval.ComptimeError.DivisionByZero => "Division by zero in const expression",
                        comptime_eval.ComptimeError.TypeMismatch => "Type mismatch in const expression",
                        comptime_eval.ComptimeError.UndefinedVariable => "Undefined variable in const expression",
                        comptime_eval.ComptimeError.InvalidOperation => "Invalid operation in const expression",
                        comptime_eval.ComptimeError.IntegerOverflow => "Integer overflow in const expression",
                        comptime_eval.ComptimeError.IntegerUnderflow => "Integer underflow in const expression",
                        comptime_eval.ComptimeError.InvalidLiteral => "Invalid literal format in const expression",
                        comptime_eval.ComptimeError.ConstantTooLarge => "Constant value too large in const expression",
                        comptime_eval.ComptimeError.UnsupportedOperation => "Unsupported operation in const expression",
                        comptime_eval.ComptimeError.OutOfMemory => return SemanticError.OutOfMemory,
                    };
                    try self.addWarning(error_msg, var_decl.span);
                }
            }
        }

        // Note: Variable is added to symbol table by type checker during type checking phase
    }

    /// Analyze log declaration
    fn analyzeLogDecl(self: *SemanticAnalyzer, log_decl: *ast.LogDeclNode) SemanticError!void {
        // Validate log field types
        for (log_decl.fields) |*field| {
            try self.validateLogFieldType(&field.typ, field.span);
        }
    }

    /// Analyze enum declaration
    fn analyzeEnumDecl(self: *SemanticAnalyzer, enum_decl: *ast.EnumDeclNode) SemanticError!void {
        // Determine base type (defaults to u32)
        const base_type = if (enum_decl.base_type) |*base_type_ref|
            self.type_checker.convertAstTypeToOraType(base_type_ref) catch typer.OraType.U32
        else
            typer.OraType.U32;

        // Calculate discriminant values for each variant
        var current_discriminant: u64 = 0;
        for (enum_decl.variants) |*variant| {
            if (variant.value) |*value_expr| {
                // Analyze explicit value expression
                try self.analyzeExpression(value_expr);

                // Try to evaluate the expression as a compile-time constant
                const comptime_value = self.comptime_evaluator.evaluate(value_expr) catch |err| {
                    switch (err) {
                        comptime_eval.ComptimeError.NotCompileTimeEvaluable => {
                            try self.addError("Enum variant value must be a compile-time constant", variant.span);
                            return SemanticError.InvalidOperation;
                        },
                        else => {
                            try self.addError("Invalid enum variant value", variant.span);
                            return SemanticError.InvalidOperation;
                        },
                    }
                };

                // Extract discriminant value from compile-time value
                current_discriminant = switch (comptime_value) {
                    .u8 => |val| val,
                    .u16 => |val| val,
                    .u32 => |val| val,
                    .u64 => |val| val,
                    .i8 => |val| @as(u64, @as(u8, @intCast(val))),
                    .i16 => |val| @as(u64, @as(u16, @intCast(val))),
                    .i32 => |val| @as(u64, @as(u32, @intCast(val))),
                    .i64 => |val| @bitCast(val),
                    else => {
                        try self.addError("Enum variant value must be an integer", variant.span);
                        return SemanticError.InvalidOperation;
                    },
                };
            }

            // Set the computed discriminant value
            variant.discriminant = current_discriminant;
            current_discriminant += 1;
        }

        // Register the enum type in the type checker
        var enum_type = typer.EnumType.init(self.allocator, enum_decl.name, base_type);

        // Add variants to the enum type
        for (enum_decl.variants) |*variant| {
            const enum_variant = typer.EnumType.EnumVariant{
                .name = variant.name,
                .discriminant = variant.discriminant.?,
                .span = variant.span,
            };
            try enum_type.addVariant(enum_variant);
        }

        // Register the enum type in the symbol table
        const enum_type_ptr = try self.allocator.create(typer.EnumType);
        enum_type_ptr.* = enum_type;

        const enum_symbol = typer.Symbol{
            .name = enum_decl.name,
            .typ = typer.OraType{ .Enum = enum_type_ptr },
            .region = ast.MemoryRegion.Stack,
            .mutable = false,
            .span = enum_decl.span,
        };

        try self.type_checker.current_scope.declare(enum_symbol);

        try self.addInfo("Enum type registered", enum_decl.span);
    }

    /// Analyze struct declaration with comprehensive validation
    fn analyzeStructDecl(self: *SemanticAnalyzer, struct_decl: *ast.StructDeclNode) SemanticError!void {
        // 1. Validate struct name
        if (struct_decl.name.len == 0) {
            try self.addError("Struct name cannot be empty", struct_decl.span);
            return SemanticError.InvalidOperation;
        }

        // 2. Verify struct was registered by type checker (it should be from the first pass)
        if (self.type_checker.getStructType(struct_decl.name) == null) {
            try self.addError("Struct was not properly registered during type checking", struct_decl.span);
            return SemanticError.InvalidOperation;
        }

        // 3. Validate minimum field requirement
        if (struct_decl.fields.len == 0) {
            try self.addWarning("Empty struct - consider using a unit type instead", struct_decl.span);
        }

        // 4. Validate and analyze each field
        var field_names = std.HashMap([]const u8, ast.SourceSpan, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(self.allocator);
        defer field_names.deinit();

        var total_size: u32 = 0;
        var has_complex_types = false;

        for (struct_decl.fields) |*field| {
            // Check for duplicate field names
            if (field_names.get(field.name)) |existing_span| {
                const message = std.fmt.allocPrint(self.allocator, "Duplicate field name '{s}' in struct '{s}' (first defined at line {})", .{ field.name, struct_decl.name, existing_span.line }) catch "Duplicate field name";
                defer if (!std.mem.eql(u8, message, "Duplicate field name")) self.allocator.free(message);
                try self.addError(message, field.span);
                continue;
            }
            try field_names.put(field.name, field.span);

            // Validate field name
            if (field.name.len == 0) {
                try self.addError("Field name cannot be empty", field.span);
                continue;
            }

            // Validate field type
            const field_type = self.type_checker.convertAstTypeToOraType(&field.typ) catch {
                const message = std.fmt.allocPrint(self.allocator, "Invalid field type for '{s}' in struct '{s}'", .{ field.name, struct_decl.name }) catch "Invalid field type";
                defer if (!std.mem.eql(u8, message, "Invalid field type")) self.allocator.free(message);
                try self.addError(message, field.span);
                continue;
            };

            // Analyze field type implications
            switch (field_type) {
                .Mapping, .DoubleMap => {
                    has_complex_types = true;
                    try self.addWarning("Mapping fields require careful gas management", field.span);
                },
                .String, .Bytes, .Slice => {
                    has_complex_types = true;
                    try self.addInfo("Dynamic-size field detected - consider gas implications", field.span);
                },
                .Enum => {
                    has_complex_types = true;
                    // TODO: Check for circular dependencies when we have proper struct type tracking
                },
                else => {
                    // Calculate size for primitive types
                    total_size += self.getTypeSize(field_type);
                },
            }

            // Validate field name doesn't conflict with built-in methods
            if (self.isReservedFieldName(field.name)) {
                const message = std.fmt.allocPrint(self.allocator, "Field name '{s}' conflicts with built-in struct methods", .{field.name}) catch "Reserved field name";
                defer if (!std.mem.eql(u8, message, "Reserved field name")) self.allocator.free(message);
                try self.addWarning(message, field.span);
            }
        }

        // 5. Struct type already registered by type checker - just validate it exists
        // (Registration happens in type checker's first pass)

        // 6. Memory layout analysis and optimization suggestions
        try self.analyzeStructMemoryLayout(struct_decl, total_size, has_complex_types);

        // 7. Symbol already registered by type checker during first pass

        // 8. Success reporting
        const message = std.fmt.allocPrint(self.allocator, "Struct '{s}' registered with {} fields", .{ struct_decl.name, struct_decl.fields.len }) catch "Struct registered";
        defer if (!std.mem.eql(u8, message, "Struct registered")) self.allocator.free(message);
        try self.addInfo(message, struct_decl.span);
    }

    /// Check for circular dependencies in struct fields
    fn checkStructCircularDependency(self: *SemanticAnalyzer, struct_name: []const u8, field_type: typer.OraType, span: ast.SourceSpan) SemanticError!void {
        _ = field_type; // TODO: Implement proper circular dependency checking

        // For now, just add a warning about potential circular dependencies
        const message = std.fmt.allocPrint(self.allocator, "Struct '{s}' contains nested struct field - verify no circular dependencies", .{struct_name}) catch "Potential circular dependency";
        defer if (!std.mem.eql(u8, message, "Potential circular dependency")) self.allocator.free(message);
        try self.addWarning(message, span);
    }

    /// Analyze struct memory layout and provide optimization suggestions
    fn analyzeStructMemoryLayout(self: *SemanticAnalyzer, struct_decl: *ast.StructDeclNode, total_size: u32, has_complex_types: bool) SemanticError!void {
        // Calculate storage slots for EVM storage
        const storage_slots = (total_size + 31) / 32; // Round up to 32-byte slots

        if (storage_slots > 1) {
            const message = std.fmt.allocPrint(self.allocator, "Struct '{s}' uses {} storage slots ({} bytes) - consider field ordering for gas optimization", .{ struct_decl.name, storage_slots, total_size }) catch "Storage layout info";
            defer if (!std.mem.eql(u8, message, "Storage layout info")) self.allocator.free(message);
            try self.addInfo(message, struct_decl.span);
        }

        if (has_complex_types) {
            const message = std.fmt.allocPrint(self.allocator, "Struct '{s}' contains complex types - actual gas costs will depend on usage patterns", .{struct_decl.name}) catch "Complex types warning";
            defer if (!std.mem.eql(u8, message, "Complex types warning")) self.allocator.free(message);
            try self.addWarning(message, struct_decl.span);
        }

        // Field ordering suggestions for gas optimization
        if (struct_decl.fields.len > 2 and !has_complex_types) {
            try self.analyzeFieldOrdering(struct_decl);
        }
    }

    /// Get the size in bytes for primitive types
    fn getTypeSize(self: *SemanticAnalyzer, typ: typer.OraType) u32 {
        _ = self;
        return switch (typ) {
            .Bool => 1,
            .Address => 20,
            .U8, .I8 => 1,
            .U16, .I16 => 2,
            .U32, .I32 => 4,
            .U64, .I64 => 8,
            .U128, .I128 => 16,
            .U256, .I256 => 32,
            .String, .Bytes => 32, // Dynamic, but metadata is 32 bytes
            .Slice => 64, // Pointer + length
            .Mapping, .DoubleMap => 32, // Storage key hash
            .Function => 32, // Function selector (simplified)
            .Tuple => 64, // Estimate for tuple metadata
            .Struct => 64, // Estimate, actual size depends on fields
            .Enum => 4, // Default enum size
            .Unknown, .Void => 0,
            .Error => 4, // Error code
        };
    }

    /// Check if a field name conflicts with built-in struct methods
    fn isReservedFieldName(self: *SemanticAnalyzer, name: []const u8) bool {
        _ = self;
        const reserved_names = [_][]const u8{
            "init",      "deinit",      "clone", "copy",   "move", "drop",
            "serialize", "deserialize", "hash",  "equals", "size", "length",
            "capacity",  "isEmpty",
        };

        for (reserved_names) |reserved| {
            if (std.mem.eql(u8, name, reserved)) {
                return true;
            }
        }
        return false;
    }

    /// Analyze field ordering for optimal gas usage
    fn analyzeFieldOrdering(self: *SemanticAnalyzer, struct_decl: *ast.StructDeclNode) SemanticError!void {
        var field_sizes = std.ArrayList(struct { name: []const u8, size: u32, span: ast.SourceSpan }).init(self.allocator);
        defer field_sizes.deinit();

        // Collect field sizes
        for (struct_decl.fields) |*field| {
            const field_type = self.type_checker.convertAstTypeToOraType(&field.typ) catch continue;
            const size = self.getTypeSize(field_type);
            try field_sizes.append(.{ .name = field.name, .size = size, .span = field.span });
        }

        // Check if current ordering is optimal (larger fields first for better packing)
        var is_optimal = true;
        for (field_sizes.items[0 .. field_sizes.items.len - 1], 1..) |field, i| {
            if (field.size < field_sizes.items[i].size) {
                is_optimal = false;
                break;
            }
        }

        if (!is_optimal) {
            try self.addInfo("Consider reordering fields (largest first) for better storage packing", struct_decl.span);
        }
    }

    /// Analyze import statement with comprehensive validation
    fn analyzeImport(self: *SemanticAnalyzer, import: *ast.ImportNode) SemanticError!void {
        // 1. Validate import path
        if (import.path.len == 0) {
            try self.addError("Empty import path", import.span);
            return SemanticError.InvalidOperation;
        }

        // 2. Validate path format
        try self.validateImportPath(import.path, import.span);

        // 3. Check for duplicate imports
        try self.checkDuplicateImport(import);

        // 4. Validate import name
        if (import.name.len == 0) {
            try self.addWarning("Import without alias - using module name", import.span);
        } else {
            if (!self.isValidIdentifier(import.name)) {
                const message = std.fmt.allocPrint(self.allocator, "Invalid import alias '{s}'", .{import.name}) catch "Invalid import alias";
                defer if (!std.mem.eql(u8, message, "Invalid import alias")) self.allocator.free(message);
                try self.addError(message, import.span);
                return SemanticError.InvalidOperation;
            }
        }

        // 5. Module resolution (simulate - actual implementation would resolve files)
        try self.resolveImportModule(import);

        // 6. Register import in symbol table
        try self.registerImport(import);

        // 7. Success reporting
        const message = std.fmt.allocPrint(self.allocator, "Import '{s}' registered", .{import.path}) catch "Import registered";
        defer if (!std.mem.eql(u8, message, "Import registered")) self.allocator.free(message);
        try self.addInfo(message, import.span);
    }

    /// Validate import path format
    fn validateImportPath(self: *SemanticAnalyzer, path: []const u8, span: ast.SourceSpan) SemanticError!void {
        // Check for valid characters
        for (path) |char| {
            switch (char) {
                'a'...'z', 'A'...'Z', '0'...'9', '/', '.', '_', '-' => {}, // Valid
                else => {
                    const message = std.fmt.allocPrint(self.allocator, "Invalid character in import path: '{c}'", .{char}) catch "Invalid character in import path";
                    defer if (!std.mem.eql(u8, message, "Invalid character in import path")) self.allocator.free(message);
                    try self.addError(message, span);
                    return;
                },
            }
        }

        // Check for path traversal attempts
        if (std.mem.indexOf(u8, path, "..") != null) {
            try self.addError("Path traversal not allowed in imports", span);
            return;
        }

        // Check for absolute paths (security)
        if (path.len > 0 and path[0] == '/') {
            try self.addWarning("Absolute paths not recommended for imports", span);
        }

        // Check for common file extensions
        if (std.mem.endsWith(u8, path, ".ora")) {
            try self.addInfo("Standard .ora module import", span);
        } else if (std.mem.endsWith(u8, path, ".lib")) {
            try self.addInfo("Library import detected", span);
        } else {
            try self.addWarning("Import path without standard extension", span);
        }
    }

    /// Check for duplicate imports
    fn checkDuplicateImport(self: *SemanticAnalyzer, import: *ast.ImportNode) SemanticError!void {
        // This would check against a registry of already imported modules
        // For now, we'll just add a placeholder implementation

        // Check if already imported by path
        if (std.mem.eql(u8, import.path, "std/transaction") or
            std.mem.eql(u8, import.path, "std/block") or
            std.mem.eql(u8, import.path, "std/constants"))
        {
            // Standard library modules - allow multiple imports
            return;
        }

        // TODO: Implement actual duplicate checking with import registry
        const message = std.fmt.allocPrint(self.allocator, "Import duplicate checking not implemented for '{s}'", .{import.path}) catch "Duplicate checking placeholder";
        defer if (!std.mem.eql(u8, message, "Duplicate checking placeholder")) self.allocator.free(message);
        try self.addInfo(message, import.span);
    }

    /// Resolve import module (simulation of file system resolution)
    fn resolveImportModule(self: *SemanticAnalyzer, import: *ast.ImportNode) SemanticError!void {
        // Standard library imports
        if (std.mem.startsWith(u8, import.path, "std/")) {
            try self.resolveStandardLibraryModule(import);
            return;
        }

        // Third-party library imports
        if (std.mem.startsWith(u8, import.path, "lib/")) {
            try self.resolveLibraryModule(import);
            return;
        }

        // Local module imports
        if (std.mem.startsWith(u8, import.path, "./") or std.mem.startsWith(u8, import.path, "../")) {
            try self.resolveLocalModule(import);
            return;
        }

        // Default: treat as local module
        try self.resolveLocalModule(import);
    }

    /// Resolve standard library module
    fn resolveStandardLibraryModule(self: *SemanticAnalyzer, import: *ast.ImportNode) SemanticError!void {
        const module_name = import.path[4..]; // Remove "std/" prefix

        const valid_std_modules = [_][]const u8{
            "transaction", "block", "constants", "math", "string", "array", "crypto",
        };

        var is_valid = false;
        for (valid_std_modules) |valid_module| {
            if (std.mem.eql(u8, module_name, valid_module)) {
                is_valid = true;
                break;
            }
        }

        if (!is_valid) {
            const message = std.fmt.allocPrint(self.allocator, "Unknown standard library module: '{s}'", .{module_name}) catch "Unknown std module";
            defer if (!std.mem.eql(u8, message, "Unknown std module")) self.allocator.free(message);
            try self.addError(message, import.span);
            return;
        }

        // Register standard library symbols
        try self.registerStandardLibrarySymbols(module_name, import);
    }

    /// Resolve library module
    fn resolveLibraryModule(self: *SemanticAnalyzer, import: *ast.ImportNode) SemanticError!void {
        // This would resolve third-party libraries
        // For now, just add a placeholder
        const message = std.fmt.allocPrint(self.allocator, "Library module '{s}' resolution not implemented", .{import.path}) catch "Library module resolution";
        defer if (!std.mem.eql(u8, message, "Library module resolution")) self.allocator.free(message);
        try self.addWarning(message, import.span);
    }

    /// Resolve local module
    fn resolveLocalModule(self: *SemanticAnalyzer, import: *ast.ImportNode) SemanticError!void {
        // This would resolve local .ora files
        // For now, just add a placeholder
        const message = std.fmt.allocPrint(self.allocator, "Local module '{s}' resolution not implemented - ensure file exists", .{import.path}) catch "Local module resolution";
        defer if (!std.mem.eql(u8, message, "Local module resolution")) self.allocator.free(message);
        try self.addWarning(message, import.span);
    }

    /// Register standard library symbols
    fn registerStandardLibrarySymbols(self: *SemanticAnalyzer, module_name: []const u8, import: *ast.ImportNode) SemanticError!void {
        if (std.mem.eql(u8, module_name, "transaction")) {
            // Register transaction module symbols
            try self.registerTransactionSymbols(import);
        } else if (std.mem.eql(u8, module_name, "block")) {
            // Register block module symbols
            try self.registerBlockSymbols(import);
        } else if (std.mem.eql(u8, module_name, "constants")) {
            // Register constants module symbols
            try self.registerConstantsSymbols(import);
        } else if (std.mem.eql(u8, module_name, "math")) {
            // Register math module symbols
            try self.registerMathSymbols(import);
        }
        // ... other standard library modules
    }

    /// Register import symbol
    fn registerImport(self: *SemanticAnalyzer, import: *ast.ImportNode) SemanticError!void {
        const import_name = if (import.name.len > 0) import.name else self.extractModuleName(import.path);

        const import_symbol = typer.Symbol{
            .name = import_name,
            .typ = typer.OraType.Unknown, // Module type
            .region = ast.MemoryRegion.Const,
            .mutable = false,
            .span = import.span,
        };

        try self.type_checker.current_scope.declare(import_symbol);
    }

    /// Check if a string is a valid Ora identifier
    fn isValidIdentifier(self: *SemanticAnalyzer, name: []const u8) bool {
        _ = self;
        if (name.len == 0) return false;

        // First character must be letter or underscore
        switch (name[0]) {
            'a'...'z', 'A'...'Z', '_' => {},
            else => return false,
        }

        // Remaining characters can be letters, digits, or underscores
        for (name[1..]) |char| {
            switch (char) {
                'a'...'z', 'A'...'Z', '0'...'9', '_' => {},
                else => return false,
            }
        }

        return true;
    }

    /// Extract module name from import path
    fn extractModuleName(self: *SemanticAnalyzer, path: []const u8) []const u8 {
        _ = self;

        // Find last slash or use whole path
        var start: usize = 0;
        var end: usize = path.len;

        // Find last '/' to get module name
        if (std.mem.lastIndexOf(u8, path, "/")) |last_slash| {
            start = last_slash + 1;
        }

        // Remove file extension if present
        if (std.mem.lastIndexOf(u8, path[start..], ".")) |last_dot| {
            end = start + last_dot;
        }

        return path[start..end];
    }

    /// Register transaction module symbols
    fn registerTransactionSymbols(self: *SemanticAnalyzer, import: *ast.ImportNode) SemanticError!void {
        const symbols = [_][]const u8{ "sender", "value", "origin", "gasprice" };

        for (symbols) |symbol_name| {
            const transaction_symbol = typer.Symbol{
                .name = symbol_name,
                .typ = if (std.mem.eql(u8, symbol_name, "sender") or std.mem.eql(u8, symbol_name, "origin"))
                    typer.OraType.Address
                else
                    typer.OraType.U256,
                .region = ast.MemoryRegion.Const,
                .mutable = false,
                .span = import.span,
            };
            try self.type_checker.current_scope.declare(transaction_symbol);
        }
    }

    /// Register block module symbols
    fn registerBlockSymbols(self: *SemanticAnalyzer, import: *ast.ImportNode) SemanticError!void {
        const symbols = [_][]const u8{ "timestamp", "number", "coinbase", "difficulty", "gaslimit" };

        for (symbols) |symbol_name| {
            const block_symbol = typer.Symbol{
                .name = symbol_name,
                .typ = if (std.mem.eql(u8, symbol_name, "coinbase"))
                    typer.OraType.Address
                else
                    typer.OraType.U256,
                .region = ast.MemoryRegion.Const,
                .mutable = false,
                .span = import.span,
            };
            try self.type_checker.current_scope.declare(block_symbol);
        }
    }

    /// Register constants module symbols
    fn registerConstantsSymbols(self: *SemanticAnalyzer, import: *ast.ImportNode) SemanticError!void {
        const constants = [_]struct { name: []const u8, typ: typer.OraType }{
            .{ .name = "ZERO_ADDRESS", .typ = typer.OraType.Address },
            .{ .name = "MAX_UINT256", .typ = typer.OraType.U256 },
            .{ .name = "MIN_UINT256", .typ = typer.OraType.U256 },
        };

        for (constants) |constant| {
            const constant_symbol = typer.Symbol{
                .name = constant.name,
                .typ = constant.typ,
                .region = ast.MemoryRegion.Const,
                .mutable = false,
                .span = import.span,
            };
            try self.type_checker.current_scope.declare(constant_symbol);
        }
    }

    /// Register math module symbols
    fn registerMathSymbols(self: *SemanticAnalyzer, import: *ast.ImportNode) SemanticError!void {
        const functions = [_][]const u8{ "max", "min", "abs", "sqrt", "pow", "log", "exp" };

        for (functions) |func_name| {
            const math_symbol = typer.Symbol{
                .name = func_name,
                .typ = typer.OraType{
                    .Function = .{
                        .params = &[_]typer.OraType{}, // Empty parameters for now
                        .return_type = null, // No return type specified for now
                    },
                },
                .region = ast.MemoryRegion.Const,
                .mutable = false,
                .span = import.span,
            };
            try self.type_checker.current_scope.declare(math_symbol);
        }
    }

    /// Analyze error declaration
    fn analyzeErrorDecl(self: *SemanticAnalyzer, error_decl: *ast.ErrorDeclNode) SemanticError!void {
        // Register error in symbol table
        const error_symbol = typer.Symbol{
            .name = error_decl.name,
            .typ = typer.OraType.Error,
            .region = ast.MemoryRegion.Const, // Errors are compile-time constants
            .mutable = false,
            .span = error_decl.span,
        };

        // Check if error already exists
        if (self.type_checker.current_scope.lookup(error_decl.name)) |existing| {
            if (existing.typ == typer.OraType.Error) {
                const message = try std.fmt.allocPrint(self.allocator, "Error '{s}' already declared", .{error_decl.name});
                try self.addError(message, error_decl.span);
                return;
            }
        }

        // Register the error
        try self.type_checker.current_scope.declare(error_symbol);

        // Add info diagnostic
        const message = try std.fmt.allocPrint(self.allocator, "Error '{s}' registered", .{error_decl.name});
        try self.addInfo(message, error_decl.span);
    }

    /// Analyze block of statements
    fn analyzeBlock(self: *SemanticAnalyzer, block: *ast.BlockNode) SemanticError!void {
        for (block.statements) |*stmt| {
            try self.analyzeStatement(stmt);
        }
    }

    /// Analyze statement
    fn analyzeStatement(self: *SemanticAnalyzer, stmt: *ast.StmtNode) SemanticError!void {
        switch (stmt.*) {
            .Expr => |*expr| {
                try self.analyzeExpression(expr);
            },
            .VariableDecl => |*var_decl| {
                try self.analyzeVariableDecl(var_decl);
            },
            .Return => |*ret| {
                if (ret.value) |*value| {
                    try self.analyzeExpression(value);
                }
                try self.validateReturnContext(ret.span);
            },
            .Log => |*log| {
                try self.analyzeLogStatement(log);
            },
            .Lock => |*lock| {
                try self.analyzeExpression(&lock.path);
            },
            .Break, .Continue => |span| {
                if (!self.in_loop) {
                    try self.addError("Break/continue outside loop", span);
                }
            },
            .While => |*while_stmt| {
                try self.analyzeWhileStatement(while_stmt);
            },
            .If => |*if_stmt| {
                try self.analyzeIfStatement(if_stmt);
            },
            .Invariant => |*inv| {
                try self.analyzeInvariant(inv);
            },
            .Requires => |*req| {
                try self.analyzeRequiresClause(&req.condition, req.span);
            },
            .Ensures => |*ens| {
                try self.analyzeEnsuresClause(&ens.condition, ens.span);
            },
            .ErrorDecl => |*error_decl| {
                try self.analyzeErrorDecl(error_decl);
            },
            .TryBlock => |*try_block| {
                try self.analyzeTryBlock(try_block);
            },
        }
    }

    /// Analyze expression
    fn analyzeExpression(self: *SemanticAnalyzer, expr: *ast.ExprNode) SemanticError!void {
        switch (expr.*) {
            .Assignment => |*assign| {
                try self.analyzeAssignment(assign);
            },
            .CompoundAssignment => |*compound| {
                try self.analyzeCompoundAssignment(compound);
            },
            .Call => |*call| {
                try self.analyzeFunctionCall(call);
            },
            .FieldAccess => |*field| {
                try self.analyzeFieldAccess(field);
            },
            .Index => |*index| {
                try self.analyzeIndexAccess(index);
            },
            .Binary => |*binary| {
                try self.analyzeBinaryExpression(binary);
            },
            .Unary => |*unary| {
                try self.analyzeUnaryExpression(unary);
            },
            .Try => |*try_expr| {
                try self.analyzeExpression(try_expr.expr);

                // Validate that the try expression is applied to an error union
                const expr_type = self.type_checker.typeCheckExpression(try_expr.expr) catch {
                    const message = try std.fmt.allocPrint(self.allocator, "Cannot determine type of try expression", .{});
                    try self.addError(message, try_expr.span);
                    return;
                };

                // Check if the expression can be used with try
                if (!self.canUseTryWithType(expr_type)) {
                    const message = try std.fmt.allocPrint(self.allocator, "try can only be used with error union types", .{});
                    try self.addError(message, try_expr.span);
                }
            },
            .ErrorReturn => |*error_return| {
                // Validate error name exists in symbol table
                if (self.type_checker.current_scope.lookup(error_return.error_name)) |symbol| {
                    if (symbol.typ != typer.OraType.Error) {
                        const message = try std.fmt.allocPrint(self.allocator, "'{s}' is not an error type", .{error_return.error_name});
                        try self.addError(message, error_return.span);
                    }
                } else {
                    const message = try std.fmt.allocPrint(self.allocator, "Undefined error '{s}'", .{error_return.error_name});
                    try self.addError(message, error_return.span);
                }
            },
            .ErrorCast => |*error_cast| {
                try self.analyzeExpression(error_cast.operand);

                // Validate target type is error union type
                const target_type = self.type_checker.convertAstTypeToOraType(&error_cast.target_type) catch {
                    const message = try std.fmt.allocPrint(self.allocator, "Invalid target type in error cast", .{});
                    try self.addError(message, error_cast.span);
                    return;
                };

                // Check if target type is compatible with error union
                if (!self.isErrorUnionCompatible(target_type)) {
                    const message = try std.fmt.allocPrint(self.allocator, "Cannot cast to non-error-union type", .{});
                    try self.addError(message, error_cast.span);
                }
            },
            .Shift => |*shift| {
                try self.analyzeShiftExpression(shift);
            },
            .Identifier => |*ident| {
                // Note: Identifier existence is already validated by the type checker in Phase 1
                // The semantic analyzer focuses on higher-level semantic validation

                // Check for immutable variable assignment attempts (only for storage/immutable variables)
                if (self.type_checker.current_scope.lookup(ident.name)) |symbol| {
                    if ((symbol.region == .Immutable or (symbol.region == .Storage and !symbol.mutable)) and self.in_assignment_target) {
                        if (!self.in_constructor) {
                            const var_type = if (symbol.region == .Immutable) "immutable" else "storage const";
                            const message = std.fmt.allocPrint(self.allocator, "Cannot assign to {s} variable '{s}' outside constructor", .{ var_type, ident.name }) catch "Cannot assign to variable outside constructor";
                            defer if (!std.mem.eql(u8, message, "Cannot assign to variable outside constructor")) self.allocator.free(message);
                            try self.addError(message, ident.span);
                        }
                        // Constructor assignment is handled in analyzeAssignment
                    }
                }
                // Note: Don't error on undeclared identifiers - the type checker handles that
            },
            .Literal => |*literal| {
                // Literals are always valid, but check for potential overflow
                if (literal.* == .Integer) {
                    // TODO: Add integer overflow validation for large constants
                }
            },
            .Cast => |*cast| {
                try self.analyzeExpression(cast.operand);
                // TODO: Validate cast safety (e.g., no truncation warnings)
            },
            .Old => |*old| {
                try self.analyzeExpression(old.expr);
                // Old expressions are validated in requires/ensures context
            },
            .Comptime => |*comptime_expr| {
                try self.analyzeBlock(&comptime_expr.block);
                // Comptime blocks are valid if they parse correctly
                try self.addInfo("Comptime block analyzed", comptime_expr.span);
            },
            .Tuple => |*tuple| {
                // Analyze all elements in the tuple
                for (tuple.elements) |*element| {
                    try self.analyzeExpression(element);
                }
            },
            .StructInstantiation => |*struct_inst| {
                // Analyze the struct name (should be an identifier)
                try self.analyzeExpression(struct_inst.struct_name);

                // Analyze all field initializers
                for (struct_inst.fields) |*field| {
                    try self.analyzeExpression(field.value);
                }

                // TODO: Validate struct exists and all required fields are provided
            },
            .EnumLiteral => |*enum_literal| {
                // Validate enum type exists
                const enum_type_symbol = self.type_checker.current_scope.lookup(enum_literal.enum_name);
                if (enum_type_symbol == null) {
                    try self.addError("Undefined enum type", enum_literal.span);
                    return SemanticError.UndeclaredIdentifier;
                }

                // Validate enum variant exists
                if (enum_type_symbol.?.typ == .Enum) {
                    const enum_type = enum_type_symbol.?.typ.Enum;
                    if (enum_type.findVariant(enum_literal.variant_name) == null) {
                        try self.addError("Undefined enum variant", enum_literal.span);
                        return SemanticError.UndeclaredIdentifier;
                    }
                } else {
                    try self.addError("Not an enum type", enum_literal.span);
                    return SemanticError.TypeMismatch;
                }
            },
        }
    }

    /// Analyze assignment expression
    fn analyzeAssignment(self: *SemanticAnalyzer, assign: *ast.AssignmentExpr) SemanticError!void {
        // Set flag to indicate we're analyzing assignment target
        const prev_in_assignment_target = self.in_assignment_target;
        self.in_assignment_target = true;
        defer self.in_assignment_target = prev_in_assignment_target;

        try self.analyzeExpression(assign.target);

        // Reset flag for analyzing value
        self.in_assignment_target = false;
        try self.analyzeExpression(assign.value);

        // Check if target is an immutable variable
        if (assign.target.* == .Identifier) {
            const ident = assign.target.Identifier;
            if (self.type_checker.current_scope.lookup(ident.name)) |symbol| {
                if (symbol.region == .Immutable or (symbol.region == .Storage and !symbol.mutable)) {
                    if (self.in_constructor) {
                        // Allow assignment in constructor, but track initialization
                        if (self.immutable_variables.getPtr(ident.name)) |info| {
                            if (info.initialized) {
                                const var_type = if (symbol.region == .Immutable) "Immutable" else "Storage const";
                                const message = std.fmt.allocPrint(self.allocator, "{s} variable '{s}' is already initialized", .{ var_type, ident.name }) catch "Variable is already initialized";
                                defer if (!std.mem.eql(u8, message, "Variable is already initialized")) self.allocator.free(message);
                                try self.addError(message, ident.span);
                                return SemanticError.ImmutableViolation;
                            }
                            // Mark as initialized
                            info.initialized = true;
                            info.init_span = ident.span;
                        }
                    } else {
                        const var_type = if (symbol.region == .Immutable) "immutable" else "storage const";
                        const message = std.fmt.allocPrint(self.allocator, "Cannot assign to {s} variable '{s}' outside constructor", .{ var_type, ident.name }) catch "Cannot assign to variable outside constructor";
                        defer if (!std.mem.eql(u8, message, "Cannot assign to variable outside constructor")) self.allocator.free(message);
                        try self.addError(message, ident.span);
                        return SemanticError.ImmutableViolation;
                    }
                }
            }
        }
    }

    /// Analyze compound assignment
    fn analyzeCompoundAssignment(self: *SemanticAnalyzer, compound: *ast.CompoundAssignmentExpr) SemanticError!void {
        try self.analyzeExpression(compound.target);
        try self.analyzeExpression(compound.value);

        // TODO: Validate target is mutable and supports the operation
    }

    /// Analyze binary expression
    fn analyzeBinaryExpression(self: *SemanticAnalyzer, binary: *ast.BinaryExpr) SemanticError!void {
        try self.analyzeExpression(binary.lhs);
        try self.analyzeExpression(binary.rhs);

        // Additional semantic validation for binary operations
        switch (binary.operator) {
            .Slash, .Percent => {
                // Check for potential division by zero in compile-time constants
                if (self.comptime_evaluator.isComptimeEvaluable(binary.rhs)) {
                    if (self.comptime_evaluator.evaluate(binary.rhs)) |value| {
                        if (self.isZeroValue(value)) {
                            try self.addError("Potential division by zero", binary.span);
                        }
                    } else |_| {
                        // Could not evaluate - that's fine, runtime check needed
                    }
                }
            },
            .ShiftLeft, .ShiftRight => {
                // Validate shift amounts are reasonable
                try self.validateShiftAmount(binary.rhs);
            },
            else => {},
        }
    }

    /// Analyze unary expression
    fn analyzeUnaryExpression(self: *SemanticAnalyzer, unary: *ast.UnaryExpr) SemanticError!void {
        try self.analyzeExpression(unary.operand);

        // Additional semantic validation for unary operations
        switch (unary.operator) {
            .Minus => {
                // Check for potential overflow on signed integer minimum values
                if (self.comptime_evaluator.isComptimeEvaluable(unary.operand)) {
                    try self.validateNegationOverflow(unary);
                }
            },
            .Bang => {
                // Logical NOT should only be applied to boolean expressions
                // This is handled by the type checker, but we can add semantic warnings
            },
            .BitNot => {
                // Bitwise NOT should only be applied to integer types
                // This is handled by the type checker
            },
        }
    }

    /// Check if a compile-time value is zero
    fn isZeroValue(self: *SemanticAnalyzer, value: comptime_eval.ComptimeValue) bool {
        _ = self;
        return switch (value) {
            .u8 => |v| v == 0,
            .u16 => |v| v == 0,
            .u32 => |v| v == 0,
            .u64 => |v| v == 0,
            .i8 => |v| v == 0,
            .i16 => |v| v == 0,
            .i32 => |v| v == 0,
            .i64 => |v| v == 0,
            else => false,
        };
    }

    /// Validate shift amount is reasonable
    fn validateShiftAmount(self: *SemanticAnalyzer, shift_expr: *ast.ExprNode) SemanticError!void {
        if (self.comptime_evaluator.isComptimeEvaluable(shift_expr)) {
            if (self.comptime_evaluator.evaluate(shift_expr)) |value| {
                const shift_amount = switch (value) {
                    .u8 => |v| @as(u64, v),
                    .u16 => |v| @as(u64, v),
                    .u32 => |v| @as(u64, v),
                    .u64 => |v| v,
                    else => return, // Non-numeric, let type checker handle
                };

                if (shift_amount > 256) {
                    try self.addWarning("Large shift amount may cause unexpected behavior", self.getExpressionSpan(shift_expr));
                }
            } else |_| {
                // Could not evaluate - that's fine
            }
        }
    }

    /// Validate negation doesn't cause overflow
    fn validateNegationOverflow(self: *SemanticAnalyzer, unary: *ast.UnaryExpr) SemanticError!void {
        if (self.comptime_evaluator.evaluate(unary.operand)) |value| {
            const will_overflow = switch (value) {
                .i8 => |v| v == std.math.minInt(i8),
                .i16 => |v| v == std.math.minInt(i16),
                .i32 => |v| v == std.math.minInt(i32),
                .i64 => |v| v == std.math.minInt(i64),
                else => false,
            };

            if (will_overflow) {
                try self.addWarning("Negation of minimum integer value causes overflow", unary.span);
            }
        } else |_| {
            // Could not evaluate - that's fine
        }
    }

    /// Analyze function call
    fn analyzeFunctionCall(self: *SemanticAnalyzer, call: *ast.CallExpr) SemanticError!void {
        try self.analyzeExpression(call.callee);

        for (call.arguments) |*arg| {
            try self.analyzeExpression(arg);
        }

        // Validate function exists using type checker's symbol table
        if (call.callee.* == .Identifier) {
            const func_name = call.callee.Identifier.name;
            if (self.type_checker.current_scope.lookup(func_name)) |symbol| {
                // Function exists, validate it's actually callable
                if (symbol.typ != .Function and symbol.typ != .Unknown) {
                    try self.addError("Attempt to call non-function", call.span);
                }
            } else if (self.isBuiltinFunction(func_name)) {
                // Built-in function - no additional validation needed
            } else {
                try self.addError("Undeclared function", call.span);
            }
        }
    }

    /// Analyze shift expression (mapping from source -> dest : amount)
    fn analyzeShiftExpression(self: *SemanticAnalyzer, shift: *ast.ShiftExpr) SemanticError!void {
        // Analyze all sub-expressions
        try self.analyzeExpression(shift.mapping);
        try self.analyzeExpression(shift.source);
        try self.analyzeExpression(shift.dest);
        try self.analyzeExpression(shift.amount);

        // TODO: Add semantic validation:
        // - mapping should be a mapping type
        // - source and dest should be address-compatible
        // - amount should be numeric
    }

    /// Analyze field access with proper module-based resolution
    fn analyzeFieldAccess(self: *SemanticAnalyzer, field: *ast.FieldAccessExpr) SemanticError!void {
        // First analyze the target expression
        try self.analyzeExpression(field.target);

        // Get the type of the target expression
        const target_type = self.type_checker.typeCheckExpression(field.target) catch {
            try self.addError("Cannot determine type of field access target", field.span);
            return;
        };

        // Handle different types of field access
        try self.validateFieldAccess(field, target_type);
    }

    /// Validate field access based on target type and context
    fn validateFieldAccess(self: *SemanticAnalyzer, field: *ast.FieldAccessExpr, target_type: typer.OraType) SemanticError!void {
        switch (target_type) {
            // Handle module/namespace access (e.g., std.transaction, imported modules)
            .Unknown => {
                // Unknown type might be a module reference
                try self.validateModuleFieldAccess(field);
            },

            // Handle struct field access
            .Enum => {
                try self.validateStructFieldAccess(field, target_type);
            },

            // Handle other types
            else => {
                const message = std.fmt.allocPrint(self.allocator, "Type '{s}' does not support field access", .{@tagName(target_type)}) catch "Invalid field access";
                defer if (!std.mem.eql(u8, message, "Invalid field access")) self.allocator.free(message);
                try self.addError(message, field.span);
            },
        }
    }

    /// Validate module-based field access (replaces the hardcoded hack)
    fn validateModuleFieldAccess(self: *SemanticAnalyzer, field: *ast.FieldAccessExpr) SemanticError!void {
        // Handle direct identifier access (e.g., module.field)
        if (field.target.* == .Identifier) {
            const module_name = field.target.Identifier.name;
            try self.validateModuleField(module_name, field.field, field.span);
            return;
        }

        // Handle nested field access (e.g., std.transaction.sender)
        if (field.target.* == .FieldAccess) {
            const nested_access = field.target.FieldAccess;
            if (nested_access.target.* == .Identifier) {
                const root_module = nested_access.target.Identifier.name;
                const sub_module = nested_access.field;
                try self.validateNestedModuleField(root_module, sub_module, field.field, field.span);
                return;
            }
        }

        // If we get here, it's an unsupported field access pattern
        try self.addWarning("Unsupported field access pattern - ensure proper import and module structure", field.span);
    }

    /// Validate field access on a specific module
    fn validateModuleField(self: *SemanticAnalyzer, module_name: []const u8, field_name: []const u8, span: ast.SourceSpan) SemanticError!void {
        // Check if the module exists in symbol table
        if (self.type_checker.current_scope.lookup(module_name)) |module_symbol| {
            // Module exists, check if field is valid for this module type
            _ = module_symbol; // Use the symbol to determine valid fields in future

            // For now, provide generic validation
            const message = std.fmt.allocPrint(self.allocator, "Field '{s}' access on module '{s}' - ensure field exists", .{ field_name, module_name }) catch "Module field access";
            defer if (!std.mem.eql(u8, message, "Module field access")) self.allocator.free(message);
            try self.addInfo(message, span);
        } else {
            const message = std.fmt.allocPrint(self.allocator, "Module '{s}' not found - ensure proper import", .{module_name}) catch "Module not found";
            defer if (!std.mem.eql(u8, message, "Module not found")) self.allocator.free(message);
            try self.addError(message, span);
        }
    }

    /// Validate nested module field access (e.g., std.transaction.sender)
    fn validateNestedModuleField(self: *SemanticAnalyzer, root_module: []const u8, sub_module: []const u8, field_name: []const u8, span: ast.SourceSpan) SemanticError!void {
        // Handle the common case of std library access
        if (std.mem.eql(u8, root_module, "std")) {
            try self.validateStandardLibraryAccess(sub_module, field_name, span);
        } else {
            // Handle other imported modules with nested access
            const message = std.fmt.allocPrint(self.allocator, "Nested field access '{s}.{s}.{s}' - ensure proper module structure", .{ root_module, sub_module, field_name }) catch "Nested field access";
            defer if (!std.mem.eql(u8, message, "Nested field access")) self.allocator.free(message);
            try self.addInfo(message, span);
        }
    }

    /// Validate standard library access (proper replacement for hardcoded validation)
    fn validateStandardLibraryAccess(self: *SemanticAnalyzer, module_name: []const u8, field_name: []const u8, span: ast.SourceSpan) SemanticError!void {
        // Define valid fields for each std module
        if (std.mem.eql(u8, module_name, "transaction")) {
            const valid_fields = [_][]const u8{ "sender", "value", "origin", "gasprice" };
            if (self.isFieldValid(field_name, &valid_fields)) {
                try self.addInfo("Valid std.transaction field access", span);
            } else {
                const message = std.fmt.allocPrint(self.allocator, "Invalid field '{s}' for std.transaction module", .{field_name}) catch "Invalid transaction field";
                defer if (!std.mem.eql(u8, message, "Invalid transaction field")) self.allocator.free(message);
                try self.addError(message, span);
            }
        } else if (std.mem.eql(u8, module_name, "block")) {
            const valid_fields = [_][]const u8{ "timestamp", "number", "coinbase", "difficulty", "gaslimit" };
            if (self.isFieldValid(field_name, &valid_fields)) {
                try self.addInfo("Valid std.block field access", span);
            } else {
                const message = std.fmt.allocPrint(self.allocator, "Invalid field '{s}' for std.block module", .{field_name}) catch "Invalid block field";
                defer if (!std.mem.eql(u8, message, "Invalid block field")) self.allocator.free(message);
                try self.addError(message, span);
            }
        } else if (std.mem.eql(u8, module_name, "constants")) {
            const valid_fields = [_][]const u8{ "ZERO_ADDRESS", "MAX_UINT256", "MIN_UINT256" };
            if (self.isFieldValid(field_name, &valid_fields)) {
                try self.addInfo("Valid std.constants field access", span);
            } else {
                const message = std.fmt.allocPrint(self.allocator, "Invalid constant '{s}' for std.constants module", .{field_name}) catch "Invalid constant";
                defer if (!std.mem.eql(u8, message, "Invalid constant")) self.allocator.free(message);
                try self.addError(message, span);
            }
        } else {
            const message = std.fmt.allocPrint(self.allocator, "Unknown std library module: '{s}'", .{module_name}) catch "Unknown std module";
            defer if (!std.mem.eql(u8, message, "Unknown std module")) self.allocator.free(message);
            try self.addError(message, span);
        }
    }

    /// Validate struct field access
    fn validateStructFieldAccess(self: *SemanticAnalyzer, field: *ast.FieldAccessExpr, target_type: typer.OraType) SemanticError!void {
        _ = target_type; // TODO: Use actual struct type information

        // This would validate that the field exists on the struct
        // For now, provide basic validation
        const message = std.fmt.allocPrint(self.allocator, "Struct field access '.{s}' - validation not fully implemented", .{field.field}) catch "Struct field access";
        defer if (!std.mem.eql(u8, message, "Struct field access")) self.allocator.free(message);
        try self.addWarning(message, field.span);
    }

    /// Check if a field name is valid for a given set of valid fields
    fn isFieldValid(self: *SemanticAnalyzer, field_name: []const u8, valid_fields: []const []const u8) bool {
        _ = self;
        for (valid_fields) |valid_field| {
            if (std.mem.eql(u8, field_name, valid_field)) {
                return true;
            }
        }
        return false;
    }

    /// Analyze index access (for mappings, arrays)
    fn analyzeIndexAccess(self: *SemanticAnalyzer, index: *ast.IndexExpr) SemanticError!void {
        try self.analyzeExpression(index.target);
        try self.analyzeExpression(index.index);

        // TODO: Validate target is indexable and index type is compatible
    }

    /// Analyze while statement
    fn analyzeWhileStatement(self: *SemanticAnalyzer, while_stmt: *ast.WhileNode) SemanticError!void {
        try self.analyzeExpression(&while_stmt.condition);

        // Analyze invariants
        for (while_stmt.invariants) |*inv| {
            try self.analyzeExpression(inv);
        }

        // Analyze body with loop context
        const prev_in_loop = self.in_loop;
        self.in_loop = true;
        defer self.in_loop = prev_in_loop;

        try self.analyzeBlock(&while_stmt.body);
    }

    /// Analyze if statement
    fn analyzeIfStatement(self: *SemanticAnalyzer, if_stmt: *ast.IfNode) SemanticError!void {
        try self.analyzeExpression(&if_stmt.condition);
        try self.analyzeBlock(&if_stmt.then_branch);

        if (if_stmt.else_branch) |*else_branch| {
            try self.analyzeBlock(else_branch);
        }
    }

    /// Analyze log statement
    fn analyzeLogStatement(self: *SemanticAnalyzer, log: *ast.LogNode) SemanticError!void {
        for (log.args) |*arg| {
            try self.analyzeExpression(arg);
        }

        // Validate log event exists in symbol table
        if (self.type_checker.current_scope.lookup(log.event_name)) |symbol| {
            // Check if it's actually a log declaration (simplified check)
            if (symbol.region != .Stack) {
                try self.addWarning("Log event may not be properly declared", log.span);
            }
        } else {
            try self.addError("Undeclared log event", log.span);
        }
    }

    /// Analyze requires clause
    fn analyzeRequiresClause(self: *SemanticAnalyzer, clause: *ast.ExprNode, context_span: ast.SourceSpan) SemanticError!void {
        try self.analyzeExpression(clause);

        // Validate no old() expressions in requires
        if (self.containsOldExpression(clause)) {
            try self.addError("old() expressions not allowed in requires clauses", context_span);
            return SemanticError.OldExpressionInRequires;
        }
    }

    /// Analyze ensures clause
    fn analyzeEnsuresClause(self: *SemanticAnalyzer, clause: *ast.ExprNode, context_span: ast.SourceSpan) SemanticError!void {
        _ = context_span;
        try self.analyzeExpression(clause);

        // TODO: Validate old() expressions are used correctly
    }

    /// Analyze invariant
    fn analyzeInvariant(self: *SemanticAnalyzer, inv: *ast.InvariantNode) SemanticError!void {
        if (!self.in_loop) {
            try self.addError("Invariant outside loop context", inv.span);
            return SemanticError.InvalidInvariant;
        }

        try self.analyzeExpression(&inv.condition);
    }

    /// Analyze try-catch block
    fn analyzeTryBlock(self: *SemanticAnalyzer, try_block: *ast.TryBlockNode) SemanticError!void {
        // Track error propagation from try block
        const previous_error_context = self.in_error_propagation_context;
        self.in_error_propagation_context = true;

        // Analyze try block
        try self.analyzeBlock(&try_block.try_block);

        // Restore error context
        self.in_error_propagation_context = previous_error_context;

        // Analyze catch block if present
        if (try_block.catch_block) |*catch_block| {
            try self.analyzeCatchBlock(catch_block);
        } else {
            // No catch block means errors must be handled by caller
            if (!self.current_function_returns_error_union) {
                const message = try std.fmt.allocPrint(self.allocator, "try block without catch requires function to return error union", .{});
                try self.addError(message, try_block.span);
            }
        }
    }

    /// Analyze catch block with proper error variable scope management
    fn analyzeCatchBlock(self: *SemanticAnalyzer, catch_block: *ast.CatchBlock) SemanticError!void {
        // Note: Scope management is handled by the type checker
        // For now, we'll register the error variable in the current scope
        // TODO: Implement proper scope management for catch blocks

        // If error variable is specified, add it to the scope
        if (catch_block.error_variable) |error_var| {
            const error_symbol = typer.Symbol{
                .name = error_var,
                .typ = typer.OraType.Error,
                .region = ast.MemoryRegion.Stack,
                .mutable = false,
                .span = catch_block.span,
            };

            try self.type_checker.current_scope.declare(error_symbol);

            const message = try std.fmt.allocPrint(self.allocator, "Error variable '{s}' available in catch block", .{error_var});
            try self.addInfo(message, catch_block.span);
        }

        // Analyze catch block body
        try self.analyzeBlock(&catch_block.block);
    }

    /// Check if a type is compatible with error union operations
    fn isErrorUnionCompatible(self: *SemanticAnalyzer, ora_type: typer.OraType) bool {
        _ = self;
        return switch (ora_type) {
            // Error unions are obviously compatible
            .Error => true,
            // Other types could potentially be wrapped in error unions
            .Bool, .Address, .U8, .U16, .U32, .U64, .U128, .U256, .I8, .I16, .I32, .I64, .I128, .I256, .String, .Bytes, .Slice, .Mapping, .DoubleMap, .Function, .Tuple, .Struct, .Enum => true,
            // Unknown and Void are not compatible
            .Unknown, .Void => false,
        };
    }

    /// Check if a type can be used with try expressions
    fn canUseTryWithType(self: *SemanticAnalyzer, ora_type: typer.OraType) bool {
        _ = self;
        return switch (ora_type) {
            // Error type can be used with try
            .Error => true,
            // Function calls that return error unions can be used with try
            .Function => |func| func.return_type != null,
            // Other types cannot be used with try directly
            else => false,
        };
    }

    /// Check if a function returns an error union type
    fn functionReturnsErrorUnion(self: *SemanticAnalyzer, function: *ast.FunctionNode) bool {
        _ = self;
        if (function.return_type) |*return_type| {
            return switch (return_type.*) {
                .ErrorUnion => true,
                .Result => true, // Result types are also error-like
                else => false,
            };
        }
        return false;
    }

    /// Check if expression contains old() calls
    fn containsOldExpression(self: *SemanticAnalyzer, expr: *ast.ExprNode) bool {
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
            else => false,
        };
    }

    /// Validate memory region semantics
    fn validateMemoryRegionSemantics(self: *SemanticAnalyzer, var_decl: *ast.VariableDeclNode) SemanticError!void {
        switch (var_decl.region) {
            .Storage => {
                // Storage variables must be at contract level
                if (self.current_function != null) {
                    try self.addError("Storage variables must be declared at contract level", var_decl.span);
                    return SemanticError.StorageInNonPersistentContext;
                }

                // Storage const variables must have initializers
                if (var_decl.region == .Storage and !var_decl.mutable and var_decl.value == null) {
                    try self.addError("Storage const variables must have initializers", var_decl.span);
                }
            },
            .Immutable => {
                // Immutable variables must be at contract level
                if (self.current_function != null) {
                    try self.addError("Immutable variables must be declared at contract level", var_decl.span);
                }
            },
            .Const => {
                // Const can be anywhere but must have initializer
                if (var_decl.value == null) {
                    try self.addError("Const variables must have initializer", var_decl.span);
                }
            },
            else => {
                // Stack, memory, tstore are fine in functions
            },
        }
    }

    /// Validate immutable semantics
    fn validateImmutableSemantics(self: *SemanticAnalyzer, var_decl: *ast.VariableDeclNode) SemanticError!void {
        // Handle true immutable variables and storage const variables
        if (var_decl.region == .Immutable or (var_decl.region == .Storage and !var_decl.mutable)) {
            // These variables must be declared at contract level
            if (self.current_function != null) {
                const var_type = if (var_decl.region == .Immutable) "Immutable" else "Storage const";
                const message = std.fmt.allocPrint(self.allocator, "{s} variables must be declared at contract level", .{var_type}) catch "Variables must be declared at contract level";
                defer if (!std.mem.eql(u8, message, "Variables must be declared at contract level")) self.allocator.free(message);
                try self.addError(message, var_decl.span);
                return SemanticError.ImmutableViolation;
            }

            // These variables can be initialized at declaration OR in constructor
            // They are validated for initialization at the end of contract analysis

            // Add info about immutability
            const info_msg = if (var_decl.region == .Immutable)
                "Immutable variable - can only be initialized once in constructor"
            else
                "Storage const variable - can only be initialized once in constructor";
            try self.addInfo(info_msg, var_decl.span);
        } else {
            // For other non-mutable variables, handle const semantics
            if (var_decl.region == .Const) {
                // Const variables must have initializers
                if (var_decl.value == null) {
                    try self.addError("Const variables must have initializer", var_decl.span);
                }
            } else {
                // Other immutable variables (let declarations)
                if (var_decl.value == null) {
                    try self.addError("Immutable variables must have initializers", var_decl.span);
                }
            }
        }
    }

    /// Validate log field type
    fn validateLogFieldType(self: *SemanticAnalyzer, field_type: *ast.TypeRef, span: ast.SourceSpan) SemanticError!void {
        switch (field_type.*) {
            .Mapping, .DoubleMap => {
                try self.addError("Mappings not allowed in log fields", span);
            },
            .Slice => {
                try self.addWarning("Slice types in logs may have gas implications", span);
            },
            else => {
                // Other types are fine
            },
        }
    }

    /// Validate return context
    fn validateReturnContext(self: *SemanticAnalyzer, span: ast.SourceSpan) SemanticError!void {
        if (self.current_function == null) {
            try self.addError("Return statement outside function", span);
        }
    }

    /// Validate return statements in function
    fn validateReturnStatements(self: *SemanticAnalyzer, block: *ast.BlockNode, function_span: ast.SourceSpan) SemanticError!void {
        var has_return = false;

        for (block.statements) |*stmt| {
            if (stmt.* == .Return) {
                has_return = true;
                break;
            }
        }

        if (!has_return) {
            try self.addError("Function with return type must have return statement", function_span);
            return SemanticError.MissingReturnStatement;
        }
    }

    /// Validate contract requirements
    fn validateContract(self: *SemanticAnalyzer, contract: *ContractContext) SemanticError!void {
        // Check for init function
        if (!contract.has_init) {
            try self.addError("Contract missing init function", ast.SourceSpan{ .line = 0, .column = 0, .length = 0 });
            return SemanticError.MissingInitFunction;
        }

        // Validate init function is public
        if (!contract.init_is_public) {
            try self.addWarning("Init function should be public", ast.SourceSpan{ .line = 0, .column = 0, .length = 0 });
        }
    }

    /// Add error diagnostic (takes ownership of message if it's allocated)
    fn addError(self: *SemanticAnalyzer, message: []const u8, span: ast.SourceSpan) !void {
        self.analysis_state.error_count += 1;
        self.validation_coverage.validation_stats.errors_found += 1;

        // Duplicate the message to ensure it stays alive after the caller frees it
        const owned_message = self.allocator.dupe(u8, message) catch return; // Skip on OOM
        const safe_span = self.validateSpan(span);

        try self.diagnostics.append(Diagnostic{
            .message = owned_message,
            .span = safe_span,
            .severity = .Error,
            .context = if (self.analysis_state.current_node_type) |node_type| DiagnosticContext{
                .node_type = node_type,
                .analysis_phase = self.analysis_state.phase,
                .recovery_attempted = self.error_recovery_mode,
                .additional_info = null,
            } else null,
        });
    }

    /// Add warning diagnostic
    fn addWarning(self: *SemanticAnalyzer, message: []const u8, span: ast.SourceSpan) !void {
        self.analysis_state.warning_count += 1;
        self.validation_coverage.validation_stats.warnings_generated += 1;

        // Duplicate the message to ensure it stays alive after the caller frees it
        const owned_message = self.allocator.dupe(u8, message) catch return; // Skip on OOM

        try self.diagnostics.append(Diagnostic{
            .message = owned_message,
            .span = span,
            .severity = .Warning,
            .context = if (self.analysis_state.current_node_type) |node_type| DiagnosticContext{
                .node_type = node_type,
                .analysis_phase = self.analysis_state.phase,
                .recovery_attempted = self.error_recovery_mode,
                .additional_info = null,
            } else null,
        });
    }

    /// Add info diagnostic
    fn addInfo(self: *SemanticAnalyzer, message: []const u8, span: ast.SourceSpan) !void {
        // Duplicate the message to ensure it stays alive after the caller frees it
        const owned_message = self.allocator.dupe(u8, message) catch return; // Skip on OOM

        try self.diagnostics.append(Diagnostic{
            .message = owned_message,
            .span = span,
            .severity = .Info,
            .context = if (self.analysis_state.current_node_type) |node_type| DiagnosticContext{
                .node_type = node_type,
                .analysis_phase = self.analysis_state.phase,
                .recovery_attempted = self.error_recovery_mode,
                .additional_info = null,
            } else null,
        });
    }

    /// Check if a function is a built-in function
    fn isBuiltinFunction(self: *SemanticAnalyzer, name: []const u8) bool {
        _ = self;
        // Actual built-in functions in Ora language
        return std.mem.eql(u8, name, "requires") or
            std.mem.eql(u8, name, "ensures") or
            std.mem.eql(u8, name, "invariant") or
            std.mem.eql(u8, name, "old") or
            std.mem.eql(u8, name, "log") or
            // Division functions (with @ prefix)
            std.mem.eql(u8, name, "@divmod") or
            std.mem.eql(u8, name, "@divTrunc") or
            std.mem.eql(u8, name, "@divFloor") or
            std.mem.eql(u8, name, "@divCeil") or
            std.mem.eql(u8, name, "@divExact");
    }

    /// Perform static verification on function requires/ensures clauses
    fn performStaticVerification(self: *SemanticAnalyzer, function: *ast.FunctionNode) SemanticError!void {
        // Share constants between comptime evaluator and static verifier
        var constant_iter = self.comptime_evaluator.symbol_table.symbols.iterator();
        while (constant_iter.next()) |entry| {
            self.static_verifier.defineConstant(entry.key_ptr.*, entry.value_ptr.*) catch |err| {
                switch (err) {
                    error.OutOfMemory => return SemanticError.OutOfMemory,
                }
            };
        }

        // Create verification conditions for requires clauses
        for (function.requires_clauses) |*clause| {
            const condition = static_verifier.VerificationCondition{
                .condition = clause,
                .kind = .Precondition,
                .context = static_verifier.VerificationCondition.Context{
                    .function_name = function.name,
                    .old_state = null,
                },
                .span = ast.SourceSpan{ .line = 0, .column = 0, .length = 0 }, // TODO: Get actual span
            };
            self.static_verifier.addCondition(condition) catch |err| {
                switch (err) {
                    error.OutOfMemory => return SemanticError.OutOfMemory,
                }
            };
        }

        // Create verification conditions for ensures clauses
        for (function.ensures_clauses) |*clause| {
            const condition = static_verifier.VerificationCondition{
                .condition = clause,
                .kind = .Postcondition,
                .context = static_verifier.VerificationCondition.Context{
                    .function_name = function.name,
                    .old_state = null, // TODO: Implement old state context
                },
                .span = ast.SourceSpan{ .line = 0, .column = 0, .length = 0 }, // TODO: Get actual span
            };
            self.static_verifier.addCondition(condition) catch |err| {
                switch (err) {
                    error.OutOfMemory => return SemanticError.OutOfMemory,
                }
            };
        }

        // Run static verification
        const result = self.static_verifier.verifyAll() catch |err| {
            switch (err) {
                error.OutOfMemory => return SemanticError.OutOfMemory,
            }
        };

        // Report verification results
        for (result.violations) |violation| {
            try self.addError(violation.message, violation.span);
        }

        for (result.warnings) |warning| {
            try self.addWarning(warning.message, warning.span);
        }
    }

    /// Perform formal verification for complex conditions
    fn performFormalVerification(self: *SemanticAnalyzer, function: *ast.FunctionNode, static_result: static_verifier.VerificationResult) SemanticError!void {
        // Try formal verification for the entire function
        const formal_result = self.formal_verifier.verifyFunction(function) catch |err| {
            switch (err) {
                formal_verifier.FormalVerificationError.ComplexityTooHigh => {
                    try self.addWarning("Function too complex for formal verification", function.span);
                    return;
                },
                formal_verifier.FormalVerificationError.TimeoutError => {
                    try self.addWarning("Formal verification timeout", function.span);
                    return;
                },
                formal_verifier.FormalVerificationError.OutOfMemory => return SemanticError.OutOfMemory,
                else => {
                    try self.addWarning("Formal verification failed", function.span);
                    return;
                },
            }
        };

        // Report formal verification results
        if (formal_result.proven) {
            const message = std.fmt.allocPrint(self.allocator, "Formal verification succeeded for function '{s}' (confidence: {d:.1}%)", .{
                function.name,
                formal_result.confidence_level * 100,
            }) catch "Formal verification succeeded";
            defer if (!std.mem.eql(u8, message, "Formal verification succeeded")) self.allocator.free(message);
            try self.addInfo(message, function.span);

            // Report proof strategy used
            const strategy_message = std.fmt.allocPrint(self.allocator, "Proof strategy: {s}", .{@tagName(formal_result.verification_method)}) catch "Proof strategy used";
            defer if (!std.mem.eql(u8, strategy_message, "Proof strategy used")) self.allocator.free(strategy_message);
            try self.addInfo(strategy_message, function.span);
        } else {
            try self.addWarning("Formal verification could not prove function correctness", function.span);

            // Report counterexample if available
            if (formal_result.counterexample != null) {
                try self.addWarning("Counterexample found - function may have bugs", function.span);
            }
        }

        // Perform formal verification on individual complex conditions
        try self.verifyComplexConditions(function, static_result);
    }

    /// Verify complex conditions individually using formal verification
    fn verifyComplexConditions(self: *SemanticAnalyzer, function: *ast.FunctionNode, static_result: static_verifier.VerificationResult) SemanticError!void {
        // Check for complex requires clauses that need formal verification
        for (function.requires_clauses) |*clause| {
            if (self.isComplexCondition(clause)) {
                try self.verifyComplexCondition(clause, function, .Precondition);
            }
        }

        // Check for complex ensures clauses that need formal verification
        for (function.ensures_clauses) |*clause| {
            if (self.isComplexCondition(clause)) {
                try self.verifyComplexCondition(clause, function, .Postcondition);
            }
        }

        // Check for complex invariants
        for (function.body.statements) |*stmt| {
            if (stmt.* == .Invariant) {
                const invariant_expr = &stmt.Invariant.condition;
                if (self.isComplexCondition(invariant_expr)) {
                    try self.verifyComplexCondition(invariant_expr, function, .Invariant);
                }
            }
        }

        _ = static_result; // May be used in future for enhanced verification
    }

    /// Check if a condition is complex enough to require formal verification
    fn isComplexCondition(self: *SemanticAnalyzer, condition: *ast.ExprNode) bool {
        const complexity = self.calculateConditionComplexity(condition);
        return complexity > 10; // Threshold for complex conditions
    }

    /// Calculate the complexity of a condition
    fn calculateConditionComplexity(self: *SemanticAnalyzer, condition: *ast.ExprNode) u32 {
        return switch (condition.*) {
            .Literal => 1,
            .Identifier => 1,
            .Binary => |*binary| 1 + self.calculateConditionComplexity(binary.lhs) + self.calculateConditionComplexity(binary.rhs),
            .Unary => |*unary| 1 + self.calculateConditionComplexity(unary.operand),
            .Call => |*call| {
                var complexity: u32 = 5; // Function calls are inherently complex
                complexity += self.calculateConditionComplexity(call.callee);
                for (call.arguments) |*arg| {
                    complexity += self.calculateConditionComplexity(arg);
                }
                return complexity;
            },
            .Old => |*old| 3 + self.calculateConditionComplexity(old.expr), // Old expressions add complexity
            .FieldAccess => |*field| 2 + self.calculateConditionComplexity(field.target),
            .Index => |*index| 2 + self.calculateConditionComplexity(index.target) + self.calculateConditionComplexity(index.index),
            else => 2,
        };
    }

    /// Verify a single complex condition using formal verification
    fn verifyComplexCondition(self: *SemanticAnalyzer, condition: *ast.ExprNode, function: *ast.FunctionNode, kind: ConditionKind) SemanticError!void {
        // Create formal condition structure
        var formal_condition = formal_verifier.FormalCondition{
            .expression = condition,
            .domain = formal_verifier.MathDomain.Integer, // Default domain
            .quantifiers = &[_]formal_verifier.FormalCondition.Quantifier{}, // No quantifiers for now
            .axioms = &[_]formal_verifier.FormalCondition.Axiom{}, // No custom axioms for now
            .proof_strategy = self.chooseProofStrategy(condition),
            .complexity_bound = 1000,
            .timeout_ms = 30000, // 30 seconds timeout
        };

        // Perform formal verification
        const result = self.formal_verifier.verify(&formal_condition) catch |err| {
            switch (err) {
                formal_verifier.FormalVerificationError.ComplexityTooHigh => {
                    try self.addWarning("Condition too complex for formal verification", function.span);
                    return;
                },
                formal_verifier.FormalVerificationError.TimeoutError => {
                    try self.addWarning("Formal verification timeout for condition", function.span);
                    return;
                },
                formal_verifier.FormalVerificationError.OutOfMemory => return SemanticError.OutOfMemory,
                else => {
                    try self.addWarning("Formal verification failed for condition", function.span);
                    return;
                },
            }
        };

        // Report results
        const kind_str = switch (kind) {
            .Precondition => "precondition",
            .Postcondition => "postcondition",
            .Invariant => "invariant",
        };

        if (result.proven) {
            const message = std.fmt.allocPrint(self.allocator, "Formal verification proved {s} (confidence: {d:.1}%)", .{
                kind_str,
                result.confidence_level * 100,
            }) catch "Formal verification proved condition";
            defer if (!std.mem.eql(u8, message, "Formal verification proved condition")) self.allocator.free(message);
            try self.addInfo(message, function.span);
        } else {
            const message = std.fmt.allocPrint(self.allocator, "Could not formally verify {s}", .{kind_str}) catch "Could not formally verify condition";
            defer if (!std.mem.eql(u8, message, "Could not formally verify condition")) self.allocator.free(message);
            try self.addWarning(message, function.span);
        }
    }

    /// Choose appropriate proof strategy for a condition
    fn chooseProofStrategy(self: *SemanticAnalyzer, condition: *ast.ExprNode) formal_verifier.ProofStrategy {
        _ = self;
        return switch (condition.*) {
            .Call => formal_verifier.ProofStrategy.SymbolicExecution, // Function calls need symbolic execution
            .Old => formal_verifier.ProofStrategy.StructuralInduction, // Old expressions need structural reasoning
            .Binary => |*binary| switch (binary.operator) {
                .And, .Or => formal_verifier.ProofStrategy.CaseAnalysis, // Logical operators benefit from case analysis
                .EqualEqual, .BangEqual => formal_verifier.ProofStrategy.DirectProof, // Equality can often be proven directly
                .Less, .LessEqual, .Greater, .GreaterEqual => formal_verifier.ProofStrategy.MathematicalInduction, // Comparisons may need induction
                else => formal_verifier.ProofStrategy.DirectProof,
            },
            else => formal_verifier.ProofStrategy.DirectProof,
        };
    }

    /// Condition kinds for formal verification
    const ConditionKind = enum {
        Precondition,
        Postcondition,
        Invariant,
    };

    /// Perform optimization passes on function after static verification
    fn performOptimizationPasses(self: *SemanticAnalyzer, function: *ast.FunctionNode, verification_result: static_verifier.VerificationResult) SemanticError!void {
        // Set up optimizer with verification results
        var results_array = [_]static_verifier.VerificationResult{verification_result};
        self.optimizer.setVerificationResults(&results_array);

        // Share constants between comptime evaluator and optimizer
        var constant_iter = self.comptime_evaluator.symbol_table.symbols.iterator();
        while (constant_iter.next()) |entry| {
            try self.optimizer.addConstant(entry.key_ptr.*, entry.value_ptr.*);
        }

        // Mark proven conditions for optimization
        if (verification_result.verified) {
            // Mark all verified requires clauses as proven
            for (function.requires_clauses) |_| {
                try self.optimizer.addProvenCondition("requires_clause", true);
            }

            // Mark all verified ensures clauses as proven
            for (function.ensures_clauses) |_| {
                try self.optimizer.addProvenCondition("ensures_clause", true);
            }
        }

        // Run optimization passes
        const optimization_result = self.optimizer.optimizeFunction(function) catch |err| {
            switch (err) {
                error.OutOfMemory => return SemanticError.OutOfMemory,
            }
        };

        // Report optimization results
        try self.reportOptimizationResults(function, optimization_result);
    }

    /// Report optimization results as diagnostics
    fn reportOptimizationResults(self: *SemanticAnalyzer, function: *ast.FunctionNode, result: optimizer.OptimizationResult) SemanticError!void {
        // Add info about successful optimizations
        if (result.transformations_applied > 0) {
            const message = std.fmt.allocPrint(self.allocator, "Function '{s}' optimized: {} transformations, {} checks eliminated, {} gas saved", .{
                function.name,
                result.transformations_applied,
                result.checks_eliminated,
                self.optimizer.getTotalGasSavings(),
            }) catch "Function optimized";
            defer if (!std.mem.eql(u8, message, "Function optimized")) self.allocator.free(message);
            try self.addInfo(message, function.span);
        }

        // Report specific optimizations
        for (result.optimizations) |opt| {
            const opt_message = std.fmt.allocPrint(self.allocator, "{s}: {} gas saved", .{
                opt.description,
                opt.savings.gas_saved,
            }) catch opt.description;
            defer if (!std.mem.eql(u8, opt_message, opt.description)) self.allocator.free(opt_message);
            try self.addInfo(opt_message, opt.span);
        }

        // Report significant savings
        const total_gas_saved = self.optimizer.getTotalGasSavings();
        if (total_gas_saved > 1000) {
            const savings_message = std.fmt.allocPrint(self.allocator, "Significant optimization: {} total gas saved in function '{s}'", .{
                total_gas_saved,
                function.name,
            }) catch "Significant optimization achieved";
            defer if (!std.mem.eql(u8, savings_message, "Significant optimization achieved")) self.allocator.free(savings_message);
            try self.addInfo(savings_message, function.span);
        }
    }

    /// Validate all immutable variables are initialized
    fn validateImmutableInitialization(self: *SemanticAnalyzer) SemanticError!void {
        var iter = self.immutable_variables.iterator();
        while (iter.next()) |entry| {
            const info = entry.value_ptr.*;
            if (!info.initialized) {
                const message = std.fmt.allocPrint(self.allocator, "Immutable variable '{s}' is not initialized", .{info.name}) catch "Immutable variable is not initialized";
                defer if (!std.mem.eql(u8, message, "Immutable variable is not initialized")) self.allocator.free(message);
                try self.addError(message, info.declared_span);
                return SemanticError.ImmutableViolation;
            }
        }
    }
};

/// Convenience function for semantic analysis
pub fn analyze(allocator: std.mem.Allocator, nodes: []ast.AstNode) SemanticError![]Diagnostic {
    var analyzer = SemanticAnalyzer.init(allocator);
    defer analyzer.deinit();

    return analyzer.analyze(nodes);
}
