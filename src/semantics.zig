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
};

/// Semantic diagnostic with location and severity
pub const Diagnostic = struct {
    message: []const u8,
    span: ast.SourceSpan,
    severity: Severity,

    pub const Severity = enum {
        Error,
        Warning,
        Info,
    };

    pub fn format(self: Diagnostic, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;

        // Safely handle potentially invalid message pointers
        const safe_message = if (self.message.len == 0)
            "<empty message>"
        else
            self.message;

        try writer.print("{s} at line {}, column {}: {s}", .{ @tagName(self.severity), self.span.line, self.span.column, safe_message });
    }
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
        };
    }

    /// Initialize self-references after the struct is in its final location
    pub fn initSelfReferences(self: *SemanticAnalyzer) void {
        self.type_checker.fixSelfReferences();
    }

    pub fn deinit(self: *SemanticAnalyzer) void {
        self.type_checker.deinit();
        self.comptime_evaluator.deinit();
        self.static_verifier.deinit();
        self.formal_verifier.deinit();
        self.optimizer.deinit();
        self.diagnostics.deinit();
        self.immutable_variables.deinit();
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

        // Phase 2: Semantic analysis
        for (nodes) |*node| {
            try self.analyzeNode(node);
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
            else => {
                // TODO: Add node analysis for: StructDecl, EnumDecl, Import, ErrorDecl (top-level), Block, Expression, Statement, TryBlock
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
                try self.analyzeExpression(binary.lhs);
                try self.analyzeExpression(binary.rhs);
            },
            .Unary => |*unary| {
                try self.analyzeExpression(unary.operand);
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

    /// Analyze field access (for std.transaction.sender, etc.)
    /// TODO: This is a hack to get the compiler to work, we need to fix it, the library is not yet implemented
    fn analyzeFieldAccess(self: *SemanticAnalyzer, field: *ast.FieldAccessExpr) SemanticError!void {
        try self.analyzeExpression(field.target);

        // Special handling for std.transaction.*, std.block.*, etc.
        if (field.target.* == .Identifier) {
            const base_name = field.target.Identifier.name;
            if (std.mem.eql(u8, base_name, "std")) {
                // Validate std library field access
                if (std.mem.eql(u8, field.field, "transaction") or
                    std.mem.eql(u8, field.field, "block") or
                    std.mem.eql(u8, field.field, "msg") or
                    std.mem.eql(u8, field.field, "constants"))
                {
                    // Valid std library modules
                    return;
                } else {
                    try self.addError("Invalid std library module", field.span);
                }
            }
        }

        // Handle nested field access like std.transaction.sender
        if (field.target.* == .FieldAccess) {
            const nested_field = field.target.FieldAccess;
            if (nested_field.target.* == .Identifier and
                std.mem.eql(u8, nested_field.target.Identifier.name, "std"))
            {
                if (std.mem.eql(u8, nested_field.field, "transaction")) {
                    // std.transaction.* fields
                    if (std.mem.eql(u8, field.field, "sender") or
                        std.mem.eql(u8, field.field, "value") or
                        std.mem.eql(u8, field.field, "origin"))
                    {
                        return; // Valid transaction field
                    }
                } else if (std.mem.eql(u8, nested_field.field, "block")) {
                    // std.block.* fields
                    if (std.mem.eql(u8, field.field, "timestamp") or
                        std.mem.eql(u8, field.field, "number") or
                        std.mem.eql(u8, field.field, "coinbase"))
                    {
                        return; // Valid block field
                    }
                } else if (std.mem.eql(u8, nested_field.field, "constants")) {
                    // std.constants.* fields
                    if (std.mem.eql(u8, field.field, "ZERO_ADDRESS")) {
                        return; // Valid constant
                    }
                }
                try self.addError("Invalid std library field", field.span);
                return;
            }
        }

        // TODO: Validate field exists on custom struct types
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
            .Bool, .Address, .U8, .U16, .U32, .U64, .U128, .U256, .I8, .I16, .I32, .I64, .I128, .I256, .String, .Bytes, .Slice, .Mapping, .DoubleMap, .Function, .Tuple => true,
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

    /// Add error diagnostic
    fn addError(self: *SemanticAnalyzer, message: []const u8, span: ast.SourceSpan) !void {
        try self.diagnostics.append(Diagnostic{
            .message = message,
            .span = span,
            .severity = .Error,
        });
    }

    /// Add warning diagnostic
    fn addWarning(self: *SemanticAnalyzer, message: []const u8, span: ast.SourceSpan) !void {
        try self.diagnostics.append(Diagnostic{
            .message = message,
            .span = span,
            .severity = .Warning,
        });
    }

    /// Add info diagnostic
    fn addInfo(self: *SemanticAnalyzer, message: []const u8, span: ast.SourceSpan) !void {
        try self.diagnostics.append(Diagnostic{
            .message = message,
            .span = span,
            .severity = .Info,
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
