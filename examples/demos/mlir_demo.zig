const std = @import("std");
const lib = @import("ora_lib");

const c = @cImport({
    @cInclude("mlir-c/IR.h");
    @cInclude("mlir-c/Support.h");
    @cInclude("mlir-c/RegisterEverything.h");
});

fn writeToFile(str: c.MlirStringRef, user_data: ?*anyopaque) callconv(.C) void {
    const file: *std.fs.File = @ptrCast(@alignCast(user_data.?));
    _ = file.writeAll(str.data[0..str.length]) catch {};
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    if (args.len < 2) {
        std.debug.print("Usage: mlir_demo <input.ora> [output.mlir]\n", .{});
        return;
    }
    const input = args[1];
    const output = if (args.len >= 3) args[2] else "output.mlir";

    // Frontend: lex + parse to AST
    const source = try std.fs.cwd().readFileAlloc(allocator, input, 10 * 1024 * 1024);
    defer allocator.free(source);

    var lexer = lib.Lexer.init(allocator, source);
    defer lexer.deinit();
    const tokens = try lexer.scanTokens();
    defer allocator.free(tokens);

    var arena = lib.ast_arena.AstArena.init(allocator);
    defer arena.deinit();
    var parser = lib.Parser.init(tokens, &arena);
    parser.setFileId(1);
    const ast_nodes = try parser.parse();
    _ = ast_nodes; // Placeholder: real lowering would traverse AST

    // MLIR: create empty module and print to file
    const ctx = c.mlirContextCreate();
    defer c.mlirContextDestroy(ctx);
    const registry = c.mlirDialectRegistryCreate();
    defer c.mlirDialectRegistryDestroy(registry);
    c.mlirRegisterAllDialects(registry);
    c.mlirContextAppendDialectRegistry(ctx, registry);
    c.mlirContextLoadAllAvailableDialects(ctx);

    const loc = c.mlirLocationUnknownGet(ctx);
    const module = c.mlirModuleCreateEmpty(loc);
    defer c.mlirModuleDestroy(module);

    var file = try std.fs.cwd().createFile(output, .{});
    defer file.close();

    const op = c.mlirModuleGetOperation(module);
    c.mlirOperationPrint(op, writeToFile, @ptrCast(&file));

    std.debug.print("Wrote MLIR to {s}\n", .{output});
}
