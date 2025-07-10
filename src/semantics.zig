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
        try writer.print("{s} at line {}, column {}: {s}", .{ @tagName(self.severity), self.span.line, self.span.column, self.message });
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
    }

    /// Perform complete semantic analysis on AST nodes
    pub fn analyze(self: *SemanticAnalyzer, nodes: []ast.AstNode) SemanticError![]Diagnostic {
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
        // Initialize contract context
        var contract_ctx = ContractContext.init(self.allocator, contract.name);
        // Ensure cleanup on error - this guarantees memory is freed even if analysis fails
        defer contract_ctx.deinit();

        // Analyze contract members
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
                    }

                    try contract_ctx.functions.append(function.name);
                    try self.analyzeFunction(function);
                },
                .VariableDecl => |*var_decl| {
                    // Track storage variables
                    if (var_decl.region == .Storage) {
                        try contract_ctx.storage_variables.append(var_decl.name);
                    }
                    try self.analyzeVariableDecl(var_decl);
                },
                .LogDecl => |*log_decl| {
                    try contract_ctx.events.append(log_decl.name);
                    try self.analyzeLogDecl(log_decl);
                },
                else => {
                    try self.analyzeNode(member);
                },
            }
        }

        // Validate contract after all members are analyzed
        try self.validateContract(&contract_ctx);
    }

    /// Analyze function declaration
    fn analyzeFunction(self: *SemanticAnalyzer, function: *ast.FunctionNode) SemanticError!void {
        const prev_function = self.current_function;
        self.current_function = function.name;
        defer self.current_function = prev_function;

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

        // Analyze function body
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
            },
            .ErrorReturn => |*error_return| {
                // TODO: Validate error name exists in symbol table
                _ = error_return;
            },
            .ErrorCast => |*error_cast| {
                try self.analyzeExpression(error_cast.operand);
                // TODO: Validate target type is error union type
            },
            .Identifier => |*ident| {
                // Validate identifier exists in scope
                if (self.type_checker.current_scope.lookup(ident.name) == null) {
                    try self.addError("Undeclared identifier", ident.span);
                }
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
        }
    }

    /// Analyze assignment for mutability constraints
    fn analyzeAssignment(self: *SemanticAnalyzer, assign: *ast.AssignmentExpr) SemanticError!void {
        try self.analyzeExpression(assign.target);
        try self.analyzeExpression(assign.value);

        // Validate target is mutable using type checker's symbol table
        if (assign.target.* == .Identifier) {
            const identifier = assign.target.Identifier;
            if (self.type_checker.current_scope.lookup(identifier.name)) |symbol| {
                if (!symbol.mutable) {
                    try self.addError("Cannot assign to immutable variable", assign.span);
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
            } else {
                try self.addError("Undeclared function", call.span);
            }
        }
    }

    /// Analyze field access (for std.transaction.sender, etc.)
    fn analyzeFieldAccess(self: *SemanticAnalyzer, field: *ast.FieldAccessExpr) SemanticError!void {
        try self.analyzeExpression(field.target);

        // Special handling for std.transaction.*, std.block.*, etc.
        if (field.target.* == .Identifier) {
            const base_name = field.target.Identifier.name;
            if (std.mem.eql(u8, base_name, "std")) {
                // Validate std library field access
                if (std.mem.eql(u8, field.field, "transaction") or
                    std.mem.eql(u8, field.field, "block") or
                    std.mem.eql(u8, field.field, "msg"))
                {
                    // Valid std library modules
                    return;
                } else {
                    try self.addError("Invalid std library module", field.span);
                }
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
        // TODO: Register error in symbol table
        _ = self;
        _ = error_decl;
        // Error declarations don't need complex analysis
    }

    /// Analyze try-catch block
    fn analyzeTryBlock(self: *SemanticAnalyzer, try_block: *ast.TryBlockNode) SemanticError!void {
        // Analyze try block
        try self.analyzeBlock(&try_block.try_block);

        // Analyze catch block if present
        if (try_block.catch_block) |*catch_block| {
            try self.analyzeBlock(&catch_block.block);
        }
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
        // Immutable variables must have initializers (except for storage variables)
        if (var_decl.region != .Storage and var_decl.value == null) {
            try self.addError("Immutable variables must have initializers", var_decl.span);
        }

        // Add info about immutability
        if (var_decl.region == .Immutable) {
            try self.addInfo("Immutable variable - cannot be reassigned after initialization", var_decl.span);
        }

        // TODO: Track immutable variables and validate they're never reassigned in function bodies
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

    /// Perform static verification on function requires/ensures clauses
    fn performStaticVerification(self: *SemanticAnalyzer, function: *ast.FunctionNode) SemanticError!void {
        // Share constants between comptime evaluator and static verifier
        var constant_iter = self.comptime_evaluator.symbol_table.symbols.iterator();
        while (constant_iter.next()) |entry| {
            try self.static_verifier.defineConstant(entry.key_ptr.*, entry.value_ptr.*);
        }

        // Create verification conditions for requires clauses
        for (function.requires_clauses) |*clause| {
            try self.static_verifier.addCondition(static_verifier.VerificationCondition{
                .condition = clause,
                .kind = .Precondition,
                .context = static_verifier.VerificationCondition.Context{
                    .function_name = function.name,
                    .old_state = null,
                },
                .span = function.span,
            });
        }

        // Create old state context for ensures clauses
        var old_state = self.static_verifier.createOldStateContext();
        defer old_state.deinit();

        // TODO: Capture actual variable states for old() expressions
        // For now, we'll just create the context structure

        // Create verification conditions for ensures clauses
        for (function.ensures_clauses) |*clause| {
            try self.static_verifier.addCondition(static_verifier.VerificationCondition{
                .condition = clause,
                .kind = .Postcondition,
                .context = static_verifier.VerificationCondition.Context{
                    .function_name = function.name,
                    .old_state = &old_state,
                },
                .span = function.span,
            });
        }

        // Run basic static verification first
        const verification_result = self.static_verifier.verifyAll() catch |err| {
            switch (err) {
                error.OutOfMemory => return SemanticError.OutOfMemory,
            }
        };

        // Perform formal verification for complex conditions
        try self.performFormalVerification(function, verification_result);

        // Process verification results
        for (verification_result.violations) |violation| {
            try self.addError(violation.message, violation.span);
        }

        for (verification_result.warnings) |warning| {
            try self.addWarning(warning.message, warning.span);
        }

        // Add info if verification passed
        if (verification_result.verified and verification_result.violations.len == 0) {
            const message = std.fmt.allocPrint(self.allocator, "Static verification passed for function '{s}'", .{function.name}) catch "Static verification passed";
            defer if (!std.mem.eql(u8, message, "Static verification passed")) self.allocator.free(message);
            try self.addInfo(message, function.span);
        }

        // Run optimization passes based on verification results
        try self.performOptimizationPasses(function, verification_result);
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
};

/// Convenience function for semantic analysis
pub fn analyze(allocator: std.mem.Allocator, nodes: []ast.AstNode) SemanticError![]Diagnostic {
    var analyzer = SemanticAnalyzer.init(allocator);
    defer analyzer.deinit();

    return analyzer.analyze(nodes);
}
