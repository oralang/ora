const std = @import("std");
const ora = @import("ora_lib");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("Testing Formal Verification System...\n", .{});

    // Initialize formal verifier
    var formal_verifier = ora.formal_verifier.FormalVerifier.init(allocator);
    defer formal_verifier.deinit();

    // Create a simple condition
    const true_literal = ora.ast.LiteralNode{ .Bool = ora.ast.BoolLiteral{ .value = true, .span = ora.ast.SourceSpan{ .line = 1, .column = 1, .length = 4 } } };
    const true_expr = ora.ast.ExprNode{ .Literal = true_literal };

    // Create a mutable copy for the formal verifier
    var mutable_expr = true_expr;

    var condition = ora.formal_verifier.FormalCondition{
        .expression = &mutable_expr,
        .domain = ora.formal_verifier.MathDomain.Integer,
        .quantifiers = &[_]ora.formal_verifier.FormalCondition.Quantifier{},
        .axioms = &[_]ora.formal_verifier.FormalCondition.Axiom{},
        .proof_strategy = ora.formal_verifier.ProofStrategy.DirectProof,
        .complexity_bound = 100,
        .timeout_ms = 5000,
    };

    var result = formal_verifier.verify(&condition) catch |err| {
        std.debug.print("Error: {s}\n", .{@errorName(err)});
        return;
    };
    defer result.deinit(); // Clean up allocated memory

    std.debug.print("Verification result: {s}\n", .{if (result.proven) "PROVEN" else "UNPROVEN"});
    std.debug.print("Confidence: {d:.2}%\n", .{result.confidence_level * 100});
    std.debug.print("Method: {s}\n", .{@tagName(result.verification_method)});

    std.debug.print("Formal verification system is working!\n", .{});
}
