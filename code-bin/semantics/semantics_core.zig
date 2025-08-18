const std = @import("std");
pub const ast = @import("../ast.zig");
pub const typer = @import("../typer.zig");

// TODO: Stub implementations - replace with actual modules when backend is integrated
const comptime_eval = struct {
    pub const ComptimeEvaluator = struct {
        pub fn init(allocator: std.mem.Allocator) ComptimeEvaluator {
            _ = allocator;
            return ComptimeEvaluator{};
        }

        pub fn deinit(self: *ComptimeEvaluator) void {
            _ = self;
        }
    };
};

const static_verifier = struct {
    pub const StaticVerifier = struct {
        pub fn init(allocator: std.mem.Allocator) StaticVerifier {
            _ = allocator;
            return StaticVerifier{};
        }

        pub fn deinit(self: *StaticVerifier) void {
            _ = self;
        }
    };
};

const formal_verifier = struct {
    pub const FormalVerifier = struct {
        pub fn init(allocator: std.mem.Allocator) FormalVerifier {
            _ = allocator;
            return FormalVerifier{};
        }

        pub fn deinit(self: *FormalVerifier) void {
            _ = self;
        }
    };
};

const optimizer = struct {
    pub const Optimizer = struct {
        pub fn init(allocator: std.mem.Allocator) Optimizer {
            _ = allocator;
            return Optimizer{};
        }

        pub fn deinit(self: *Optimizer) void {
            _ = self;
        }
    };
};

// Import other semantics modules
const semantics_errors = @import("semantics_errors.zig");
const semantics_state = @import("semantics_state.zig");
const semantics_memory_safety = @import("semantics_memory_safety.zig");
const semantics_contract_analyzer = @import("semantics_contract_analyzer.zig");
const semantics_function_analyzer = @import("semantics_function_analyzer.zig");
const semantics_expression_analyzer = @import("semantics_expression_analyzer.zig");
const semantics_import_analyzer = @import("semantics_import_analyzer.zig");
const semantics_type_integration = @import("semantics_type_integration.zig");

// Re-export types from other modules
pub const SemanticError = semantics_errors.SemanticError;
pub const Diagnostic = semantics_errors.Diagnostic;
pub const DiagnosticContext = semantics_errors.DiagnosticContext;
pub const AnalysisState = semantics_state.AnalysisState;
pub const ValidationCoverage = semantics_state.ValidationCoverage;
pub const ContractContext = semantics_state.ContractContext;

/// Main semantic analyzer for Ora
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
    in_assignment_target: bool,

    // Error propagation tracking
    in_error_propagation_context: bool,
    current_function_returns_error_union: bool,

    // Immutable variable tracking
    current_contract: ?*ast.ContractNode,
    immutable_variables: std.HashMap([]const u8, ImmutableVarInfo, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),
    in_constructor: bool,

    // Memory safety and error recovery
    error_recovery_mode: bool,
    analysis_state: AnalysisState,
    validation_coverage: ValidationCoverage,

    // Analysis modules
    import_analyzer: semantics_import_analyzer.ImportAnalyzer,
    type_integration: semantics_type_integration.TypeSystemIntegration,

    /// Information about an immutable variable
    pub const ImmutableVarInfo = struct {
        name: []const u8,
        declared_span: ast.SourceSpan,
        initialized: bool,
        init_span: ?ast.SourceSpan,
    };

    pub fn init(allocator: std.mem.Allocator) SemanticAnalyzer {
        const analyzer = SemanticAnalyzer{
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
            .import_analyzer = semantics_import_analyzer.ImportAnalyzer.init(allocator),
            .type_integration = semantics_type_integration.TypeSystemIntegration.init(allocator, undefined), // Will be set in initSelfReferences
        };
        return analyzer;
    }

    /// Initialize self-references after the struct is in its final location
    pub fn initSelfReferences(self: *SemanticAnalyzer) void {
        self.type_checker.fixSelfReferences();
        self.type_integration.type_checker = &self.type_checker;
    }

    pub fn deinit(self: *SemanticAnalyzer) void {
        // Free diagnostic messages before freeing the diagnostics array
        for (self.diagnostics.items) |diagnostic| {
            self.allocator.free(diagnostic.message);
        }

        self.type_checker.deinit();
        self.comptime_evaluator.deinit();
        self.static_verifier.deinit();
        self.formal_verifier.deinit();
        self.optimizer.deinit();
        self.diagnostics.deinit();
        self.immutable_variables.deinit();
        self.validation_coverage.deinit();
        self.import_analyzer.deinit();
        self.type_integration.deinit();
    }

    /// Perform complete semantic analysis on AST nodes
    pub fn analyze(self: *SemanticAnalyzer, nodes: []ast.AstNode) SemanticError![]Diagnostic {
        // Phase 0: Pre-initialize type checker with contract symbols
        for (nodes) |*node| {
            if (node.* == .Contract) {
                try self.preInitializeContractSymbols(&node.Contract);
            }
        }

        // Phase 1: Type checking
        self.type_checker.typeCheck(nodes) catch |err| {
            switch (err) {
                typer.TyperError.UndeclaredVariable => {
                    try self.addErrorStatic("Undeclared variable", ast.SourceSpan{ .line = 0, .column = 0, .length = 0 });
                },
                typer.TyperError.TypeMismatch => {
                    try self.addErrorStatic("Type mismatch", ast.SourceSpan{ .line = 0, .column = 0, .length = 0 });
                },
                typer.TyperError.UndeclaredFunction => {
                    try self.addErrorStatic("Undeclared function", ast.SourceSpan{ .line = 0, .column = 0, .length = 0 });
                },
                typer.TyperError.ArgumentCountMismatch => {
                    try self.addErrorStatic("Function argument count mismatch", ast.SourceSpan{ .line = 0, .column = 0, .length = 0 });
                },
                typer.TyperError.InvalidOperation => {
                    try self.addErrorStatic("Invalid type operation", ast.SourceSpan{ .line = 0, .column = 0, .length = 0 });
                },
                typer.TyperError.InvalidMemoryRegion => {
                    try self.addErrorStatic("Invalid memory region", ast.SourceSpan{ .line = 0, .column = 0, .length = 0 });
                },
                typer.TyperError.OutOfMemory => {
                    return SemanticError.OutOfMemory;
                },
            }
        };

        // Phase 2: Semantic analysis with safety checks
        self.analysis_state.phase = .SemanticAnalysis;
        for (nodes) |*node| {
            try self.safeAnalyzeNode(node);
        }

        return try self.diagnostics.toOwnedSlice();
    }

    // Forward declarations for methods that will be implemented in other modules
    pub fn addError(self: *SemanticAnalyzer, message: []const u8, span: ast.SourceSpan) SemanticError!void {
        return semantics_errors.addError(self, message, span);
    }

    pub fn addErrorStatic(self: *SemanticAnalyzer, message: []const u8, span: ast.SourceSpan) SemanticError!void {
        return semantics_errors.addErrorStatic(self, message, span);
    }

    pub fn addWarning(self: *SemanticAnalyzer, message: []const u8, span: ast.SourceSpan) SemanticError!void {
        return semantics_errors.addWarning(self, message, span);
    }

    pub fn addWarningStatic(self: *SemanticAnalyzer, message: []const u8, span: ast.SourceSpan) SemanticError!void {
        return semantics_errors.addWarningStatic(self, message, span);
    }

    pub fn addInfo(self: *SemanticAnalyzer, message: []const u8, span: ast.SourceSpan) SemanticError!void {
        return semantics_errors.addInfo(self, message, span);
    }

    pub fn safeAnalyzeNode(self: *SemanticAnalyzer, node: *ast.AstNode) SemanticError!void {
        return semantics_memory_safety.safeAnalyzeNode(self, node);
    }

    pub fn preInitializeContractSymbols(self: *SemanticAnalyzer, contract: *ast.ContractNode) SemanticError!void {
        return semantics_contract_analyzer.preInitializeContractSymbols(self, contract);
    }

    // Additional method stubs that will be implemented in specialized modules
    pub fn analyzeNode(self: *SemanticAnalyzer, node: *ast.AstNode) SemanticError!void {
        switch (node.*) {
            .Contract => |*contract| {
                try semantics_contract_analyzer.analyzeContract(self, contract);
            },
            .Function => |*function| {
                try semantics_function_analyzer.analyzeFunction(self, function);
            },
            .Expression => |expr| {
                try semantics_expression_analyzer.analyzeExpression(self, expr);
            },
            .Import => |*import| {
                try semantics_import_analyzer.analyzeImport(self, import);
            },
            else => {
                // For other node types, just mark as analyzed for coverage
                self.validation_coverage.visited_node_types.insert(@as(std.meta.Tag(ast.AstNode), node.*));
                self.validation_coverage.validation_stats.nodes_analyzed += 1;
            },
        }
    }
};
