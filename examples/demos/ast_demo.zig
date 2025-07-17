const std = @import("std");
const ast = @import("../src/ast.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const stdout = std.io.getStdOut().writer();

    try stdout.print("ZigOra AST Demo\n", .{});
    try stdout.print("===============\n\n", .{});

    // Create a simple ZigOra contract AST
    try demonstrateAstConstruction(allocator, stdout);

    // Show AST pretty printing
    try stdout.print("\n", .{});
    try demonstrateAstPrinting(allocator, stdout);
}

fn demonstrateAstConstruction(allocator: std.mem.Allocator, writer: anytype) !void {
    try writer.print("1. Constructing a ZigOra Contract AST\n", .{});
    try writer.print("=====================================\n", .{});

    const span = ast.SourceSpan{ .line = 1, .column = 1, .length = 8 };

    // Create a storage variable declaration: storage var balance: u256
    const balance_var = ast.VariableDeclNode{
        .name = "balance",
        .region = .Storage,
        .mutable = true,
        .locked = false,
        .typ = .U256,
        .value = null,
        .span = span,
    };

    // Create a transient storage variable: tstore var counter: u32
    const counter_var = ast.VariableDeclNode{
        .name = "counter",
        .region = .TStore,
        .mutable = true,
        .locked = false,
        .typ = .U32,
        .value = null,
        .span = span,
    };

    // Create function parameters
    const to_param = ast.ParamNode{
        .name = "to",
        .typ = .Address,
        .span = span,
    };

    const amount_param = ast.ParamNode{
        .name = "amount",
        .typ = .U256,
        .span = span,
    };

    const params = [_]ast.ParamNode{ to_param, amount_param };

    // Create a transfer function
    const transfer_func = ast.FunctionNode{
        .pub_ = true,
        .name = "transfer",
        .parameters = &params,
        .return_type = .Bool,
        .requires_clauses = &[_]ast.ExprNode{}, // Empty for demo
        .ensures_clauses = &[_]ast.ExprNode{}, // Empty for demo
        .body = ast.BlockNode{
            .statements = &[_]ast.StmtNode{},
            .span = span,
        },
        .span = span,
    };

    try writer.print("Created variable: {s} ({s}, mutable: {})\n", .{ balance_var.name, @tagName(balance_var.region), balance_var.mutable });
    try writer.print("Created variable: {s} ({s}, mutable: {})\n", .{ counter_var.name, @tagName(counter_var.region), counter_var.mutable });
    try writer.print("Created function: {s} (pub: {}, params: {})\n", .{ transfer_func.name, transfer_func.pub_, transfer_func.parameters.len });

    // Create contract body (would normally contain more nodes)
    var body = std.ArrayList(ast.AstNode).init(allocator);
    defer body.deinit();

    try body.append(.{ .VariableDecl = balance_var });
    try body.append(.{ .VariableDecl = counter_var });
    try body.append(.{ .Function = transfer_func });

    // Create the contract
    const contract = ast.ContractNode{
        .name = "SimpleToken",
        .body = try body.toOwnedSlice(),
        .span = span,
    };
    defer allocator.free(contract.body);

    try writer.print("Created contract: {s} with {} declarations\n", .{ contract.name, contract.body.len });
}

fn demonstrateAstPrinting(allocator: std.mem.Allocator, writer: anytype) !void {
    try writer.print("2. AST Pretty Printing\n", .{});
    try writer.print("======================\n", .{});

    const span = ast.SourceSpan{ .line = 1, .column = 1, .length = 5 };

    // Create a simple binary expression: balance + amount
    const balance_ident = try ast.createIdentifier(allocator, "balance", span);
    defer allocator.destroy(balance_ident);

    const amount_ident = try ast.createIdentifier(allocator, "amount", span);
    defer allocator.destroy(amount_ident);

    const binary_expr = try ast.createBinaryExpr(allocator, balance_ident, .Plus, amount_ident, span);
    defer allocator.destroy(binary_expr);

    // Print the expression components
    switch (binary_expr.*) {
        .Binary => |bin| {
            const left_name = switch (bin.lhs.*) {
                .Identifier => |id| id.name,
                else => "unknown",
            };
            const right_name = switch (bin.rhs.*) {
                .Identifier => |id| id.name,
                else => "unknown",
            };
            try writer.print("Binary Expression: {s} {} {s}\n", .{ left_name, @tagName(bin.operator), right_name });
        },
        else => {},
    }

    // Demonstrate type system
    try writer.print("\nType System Demonstration:\n", .{});
    try writer.print("- U256: {s}\n", .{@tagName(ast.TypeRef.U256)});
    try writer.print("- Address: {s}\n", .{@tagName(ast.TypeRef.Address)});
    try writer.print("- Bool: {s}\n", .{@tagName(ast.TypeRef.Bool)});

    // Demonstrate memory regions
    try writer.print("\nMemory Regions:\n", .{});
    try writer.print("- Stack: {s}\n", .{@tagName(ast.MemoryRegion.Stack)});
    try writer.print("- Storage: {s}\n", .{@tagName(ast.MemoryRegion.Storage)});
    try writer.print("- TStore: {s}\n", .{@tagName(ast.MemoryRegion.TStore)});
    try writer.print("- Memory: {s}\n", .{@tagName(ast.MemoryRegion.Memory)});
}
