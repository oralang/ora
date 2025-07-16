const std = @import("std");
const ora = @import("ora_lib");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== Ora Formal Verification Demo ===\n\n", .{});

    // Initialize formal verifier
    var formal_verifier = ora.formal_verifier.FormalVerifier.init(allocator);
    defer formal_verifier.deinit();

    // Demonstrate different proof strategies
    try demonstrateProofStrategies(&formal_verifier, allocator);

    // Demonstrate mathematical domains
    try demonstrateMathematicalDomains(&formal_verifier, allocator);

    // Demonstrate quantified conditions
    try demonstrateQuantifiedConditions(&formal_verifier, allocator);

    // Demonstrate symbolic execution
    try demonstrateSymbolicExecution(&formal_verifier, allocator);

    // Demonstrate theory integration
    try demonstrateTheoryIntegration(&formal_verifier, allocator);

    // Demonstrate complex condition analysis
    try demonstrateComplexConditionAnalysis(&formal_verifier, allocator);

    // Generate verification report
    try generateVerificationReport(&formal_verifier, allocator);

    std.debug.print("=== Demo Complete ===\n", .{});
}

fn demonstrateProofStrategies(formal_verifier: *ora.formal_verifier.FormalVerifier, _: std.mem.Allocator) !void {
    std.debug.print("1. Proof Strategy Demonstration\n", .{});
    std.debug.print("================================\n", .{});

    const strategies = [_]ora.formal_verifier.ProofStrategy{
        .DirectProof,
        .ProofByContradiction,
        .StructuralInduction,
        .MathematicalInduction,
        .CaseAnalysis,
        .SymbolicExecution,
        .BoundedModelChecking,
        .AbstractInterpretation,
    };

    for (strategies) |strategy| {
        std.debug.print("Testing strategy: {s}\n", .{@tagName(strategy)});

        // Create a simple condition for testing
        var true_literal = ora.ast.LiteralNode{ .Bool = ora.ast.BoolLiteral{ .value = true, .span = ora.ast.SourceSpan{ .line = 1, .column = 1, .length = 4 } } };
        var true_expr = ora.ast.ExprNode{ .Literal = true_literal };

        var condition = ora.formal_verifier.FormalCondition{
            .expression = &true_expr,
            .domain = ora.formal_verifier.MathDomain.Integer,
            .quantifiers = &[_]ora.formal_verifier.FormalCondition.Quantifier{},
            .axioms = &[_]ora.formal_verifier.FormalCondition.Axiom{},
            .proof_strategy = strategy,
            .complexity_bound = 100,
            .timeout_ms = 5000,
        };

        const result = formal_verifier.verify(&condition) catch |err| {
            std.debug.print("  Error: {s}\n", .{@errorName(err)});
            continue;
        };

        std.debug.print("  Result: {s} (confidence: {d:.2}%)\n", .{
            if (result.proven) "PROVEN" else "UNPROVEN",
            result.confidence_level * 100,
        });

        if (result.proof) |proof| {
            std.debug.print("  Proof steps: {}\n", .{proof.steps.items.len});
            std.debug.print("  Verification time: {}ms\n", .{proof.verification_time_ms});
            std.debug.print("  Complexity score: {d:.2}\n", .{proof.complexity_score});
        }
    }

    std.debug.print("\n", .{});
}

fn demonstrateMathematicalDomains(formal_verifier: *ora.formal_verifier.FormalVerifier, allocator: std.mem.Allocator) !void {
    _ = allocator;
    std.debug.print("2. Mathematical Domain Demonstration\n", .{});
    std.debug.print("====================================\n", .{});

    const domains = [_]ora.formal_verifier.MathDomain{
        .Integer,
        .Real,
        .BitVector,
        .Array,
        .Set,
        .Function,
        .Algebraic,
    };

    for (domains) |domain| {
        std.debug.print("Testing domain: {s}\n", .{@tagName(domain)});

        // Create a domain-specific condition
        const true_literal = ora.ast.LiteralNode{ .Bool = ora.ast.BoolLiteral{ .value = true, .span = ora.ast.SourceSpan{ .line = 1, .column = 1, .length = 4 } } };
        const true_expr = ora.ast.ExprNode{ .Literal = true_literal };

        var condition = ora.formal_verifier.FormalCondition{
            .expression = &true_expr,
            .domain = domain,
            .quantifiers = &[_]ora.formal_verifier.FormalCondition.Quantifier{},
            .axioms = &[_]ora.formal_verifier.FormalCondition.Axiom{},
            .proof_strategy = ora.formal_verifier.ProofStrategy.DirectProof,
            .complexity_bound = 100,
            .timeout_ms = 5000,
        };

        const result = formal_verifier.verify(&condition) catch |err| {
            std.debug.print("  Error: {s}\n", .{@errorName(err)});
            continue;
        };

        std.debug.print("  Domain complexity: {s}\n", .{
            switch (domain) {
                .Integer => "Low",
                .Real => "Medium",
                .BitVector => "Low",
                .Array => "High",
                .Set => "High",
                .Function => "Very High",
                .Algebraic => "Very High",
            },
        });

        std.debug.print("  Verification result: {s}\n", .{
            if (result.proven) "PROVEN" else "UNPROVEN",
        });
    }

    std.debug.print("\n", .{});
}

fn demonstrateQuantifiedConditions(formal_verifier: *ora.formal_verifier.FormalVerifier, allocator: std.mem.Allocator) !void {
    _ = allocator;
    std.debug.print("3. Quantified Condition Demonstration\n", .{});
    std.debug.print("=====================================\n", .{});

    const quantifier_types = [_]ora.formal_verifier.QuantifierType{
        .Forall,
        .Exists,
        .Unique,
    };

    for (quantifier_types) |qtype| {
        std.debug.print("Testing quantifier: {s}\n", .{@tagName(qtype)});

        // Create a quantified condition
        const true_literal = ora.ast.LiteralNode{ .Bool = ora.ast.BoolLiteral{ .value = true, .span = ora.ast.SourceSpan{ .line = 1, .column = 1, .length = 4 } } };
        const true_expr = ora.ast.ExprNode{ .Literal = true_literal };

        const quantifier = ora.formal_verifier.FormalCondition.Quantifier{
            .type = qtype,
            .variable = "x",
            .domain_constraint = null,
            .range = null,
        };

        const quantifiers = [_]ora.formal_verifier.FormalCondition.Quantifier{quantifier};

        var condition = ora.formal_verifier.FormalCondition{
            .expression = &true_expr,
            .domain = ora.formal_verifier.MathDomain.Integer,
            .quantifiers = &quantifiers,
            .axioms = &[_]ora.formal_verifier.FormalCondition.Axiom{},
            .proof_strategy = ora.formal_verifier.ProofStrategy.MathematicalInduction,
            .complexity_bound = 500,
            .timeout_ms = 10000,
        };

        const result = formal_verifier.verify(&condition) catch |err| {
            std.debug.print("  Error: {s}\n", .{@errorName(err)});
            continue;
        };

        std.debug.print("  Quantifier complexity: {s}\n", .{
            switch (qtype) {
                .Forall => "High",
                .Exists => "Medium",
                .Unique => "High",
            },
        });

        std.debug.print("  Verification result: {s}\n", .{
            if (result.proven) "PROVEN" else "UNPROVEN",
        });
    }

    std.debug.print("\n", .{});
}

fn demonstrateSymbolicExecution(formal_verifier: *ora.formal_verifier.FormalVerifier, _: std.mem.Allocator) !void {
    std.debug.print("4. Symbolic Execution Demonstration\n", .{});
    std.debug.print("===================================\n", .{});

    // Create a simple function for symbolic execution
    const function_name = "test_function";
    const param_name = "x";
    const param_type = ora.ast.TypeRef{ .U32 = {} };

    const function_body = ora.ast.BlockNode{
        .statements = &[_]ora.ast.StmtNode{},
        .span = ora.ast.SourceSpan{ .line = 1, .column = 1, .length = 10 },
    };

    const param = ora.ast.ParamNode{
        .name = param_name,
        .typ = param_type,
        .span = ora.ast.SourceSpan{ .line = 1, .column = 1, .length = 10 },
    };

    const function = ora.ast.FunctionNode{
        .pub_ = false,
        .name = function_name,
        .parameters = &[_]ora.ast.ParamNode{param},
        .return_type = null,
        .requires_clauses = &[_]ora.ast.ExprNode{},
        .ensures_clauses = &[_]ora.ast.ExprNode{},
        .body = function_body,
        .span = ora.ast.SourceSpan{ .line = 1, .column = 1, .length = 50 },
    };

    // Perform symbolic execution
    const paths = formal_verifier.symbolic_executor.executeSymbolically(&function) catch |err| {
        std.debug.print("  Error in symbolic execution: {s}\n", .{@errorName(err)});
        return;
    };
    defer paths.deinit();

    std.debug.print("  Function: {s}\n", .{function.name});
    std.debug.print("  Parameters: {}\n", .{function.parameters.len});
    std.debug.print("  Symbolic paths explored: {}\n", .{paths.items.len});

    // Analyze each path
    for (paths.items, 0..) |path, i| {
        std.debug.print("  Path {}: {} variables, {} constraints\n", .{
            i,
            path.variables.count(),
            path.path_condition.items.len,
        });
    }

    std.debug.print("\n", .{});
}

fn demonstrateTheoryIntegration(formal_verifier: *ora.formal_verifier.FormalVerifier, allocator: std.mem.Allocator) !void {
    _ = allocator;
    std.debug.print("5. Theory Database Integration\n", .{});
    std.debug.print("==============================\n", .{});

    // Display theory database statistics
    std.debug.print("  Arithmetic axioms: {}\n", .{formal_verifier.theory_db.arithmetic_axioms.items.len});
    std.debug.print("  Algebraic theorems: {}\n", .{formal_verifier.theory_db.algebraic_theorems.items.len});
    std.debug.print("  Set theory axioms: {}\n", .{formal_verifier.theory_db.set_theory_axioms.items.len});
    std.debug.print("  Array theory: {}\n", .{formal_verifier.theory_db.array_theory.items.len});
    std.debug.print("  Bit vector theory: {}\n", .{formal_verifier.theory_db.bit_vector_theory.items.len});
    std.debug.print("  Custom lemmas: {}\n", .{formal_verifier.theory_db.custom_lemmas.items.len});

    // Display proof cache statistics
    std.debug.print("  Proof cache hits: {}\n", .{formal_verifier.proof_cache.cache_hits});
    std.debug.print("  Proof cache misses: {}\n", .{formal_verifier.proof_cache.cache_misses});

    const cache_efficiency = if (formal_verifier.proof_cache.cache_hits + formal_verifier.proof_cache.cache_misses > 0)
        (@as(f64, @floatFromInt(formal_verifier.proof_cache.cache_hits)) / @as(f64, @floatFromInt(formal_verifier.proof_cache.cache_hits + formal_verifier.proof_cache.cache_misses))) * 100.0
    else
        0.0;

    std.debug.print("  Cache efficiency: {d:.1}%\n", .{cache_efficiency});

    std.debug.print("\n", .{});
}

fn demonstrateComplexConditionAnalysis(formal_verifier: *ora.formal_verifier.FormalVerifier, _: std.mem.Allocator) !void {
    std.debug.print("6. Complex Condition Analysis\n", .{});
    std.debug.print("=============================\n", .{});

    // Create complex expressions for analysis
    const x_literal = ora.ast.LiteralNode{ .Integer = ora.ast.IntegerLiteral{ .value = "10", .span = ora.ast.SourceSpan{ .line = 1, .column = 1, .length = 2 } } };
    const x_expr = ora.ast.ExprNode{ .Literal = x_literal };

    const y_literal = ora.ast.LiteralNode{ .Integer = ora.ast.IntegerLiteral{ .value = "20", .span = ora.ast.SourceSpan{ .line = 1, .column = 4, .length = 2 } } };
    const y_expr = ora.ast.ExprNode{ .Literal = y_literal };

    const binary_expr = ora.ast.BinaryExpr{
        .lhs = &x_expr,
        .operator = ora.ast.BinaryOp.Plus,
        .rhs = &y_expr,
        .span = ora.ast.SourceSpan{ .line = 1, .column = 1, .length = 7 },
    };

    const complex_expr = ora.ast.ExprNode{ .Binary = binary_expr };

    // Analyze complexity
    const complexity = formal_verifier.analyzeComplexity(&ora.formal_verifier.FormalCondition{
        .expression = &complex_expr,
        .domain = ora.formal_verifier.MathDomain.Integer,
        .quantifiers = &[_]ora.formal_verifier.FormalCondition.Quantifier{},
        .axioms = &[_]ora.formal_verifier.FormalCondition.Axiom{},
        .proof_strategy = ora.formal_verifier.ProofStrategy.DirectProof,
        .complexity_bound = 1000,
        .timeout_ms = 10000,
    });

    std.debug.print("  Expression complexity: {}\n", .{complexity});
    std.debug.print("  Complexity threshold: {}\n", .{formal_verifier.verification_config.max_complexity});
    std.debug.print("  Suitable for verification: {s}\n", .{
        if (complexity <= formal_verifier.verification_config.max_complexity) "YES" else "NO",
    });

    // Choose appropriate strategy
    const strategy = formal_verifier.chooseProofStrategy(&ora.formal_verifier.FormalCondition{
        .expression = &complex_expr,
        .domain = ora.formal_verifier.MathDomain.Integer,
        .quantifiers = &[_]ora.formal_verifier.FormalCondition.Quantifier{},
        .axioms = &[_]ora.formal_verifier.FormalCondition.Axiom{},
        .proof_strategy = ora.formal_verifier.ProofStrategy.DirectProof,
        .complexity_bound = 1000,
        .timeout_ms = 10000,
    });

    std.debug.print("  Recommended strategy: {s}\n", .{@tagName(strategy)});

    std.debug.print("\n", .{});
}

fn generateVerificationReport(formal_verifier: *ora.formal_verifier.FormalVerifier, allocator: std.mem.Allocator) !void {
    std.debug.print("7. Verification Report Generation\n", .{});
    std.debug.print("=================================\n", .{});

    // Create some mock verification results
    const results = [_]ora.formal_verifier.FormalVerificationResult{
        .{
            .proven = true,
            .proof = null,
            .counterexample = null,
            .confidence_level = 0.95,
            .verification_method = ora.formal_verifier.ProofStrategy.DirectProof,
        },
        .{
            .proven = false,
            .proof = null,
            .counterexample = null,
            .confidence_level = 0.0,
            .verification_method = ora.formal_verifier.ProofStrategy.ProofByContradiction,
        },
        .{
            .proven = true,
            .proof = null,
            .counterexample = null,
            .confidence_level = 0.87,
            .verification_method = ora.formal_verifier.ProofStrategy.MathematicalInduction,
        },
    };

    const report = formal_verifier.generateVerificationReport(&results) catch |err| {
        std.debug.print("  Error generating report: {s}\n", .{@errorName(err)});
        return;
    };
    defer allocator.free(report);

    std.debug.print("  Report generated successfully\n", .{});
    std.debug.print("  Report length: {} bytes\n", .{report.len});
    std.debug.print("  Report preview:\n", .{});
    std.debug.print("  {s}\n", .{report});
}
