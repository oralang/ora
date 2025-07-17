const std = @import("std");
const print = std.debug.print;
const ora_lib = @import("ora_lib");

const yul_bindings = ora_lib.yul_bindings;
const codegen_yul = ora_lib.codegen_yul;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    print("=== Yul Compiler Integration Test ===\n\n", .{});

    // Test 1: Direct Yul compilation
    print("Test 1: Direct Yul Compilation\n", .{});
    print("--------------------------------\n", .{});

    const simple_yul =
        \\{
        \\    let x := 42
        \\    let y := add(x, 1)
        \\    mstore(0, y)
        \\    return(0, 32)
        \\}
    ;

    print("Yul Compiler Version: {s}\n", .{yul_bindings.YulCompiler.getVersion()});
    print("Compiling Yul source:\n{s}\n", .{simple_yul});

    var result = yul_bindings.YulCompiler.compile(allocator, simple_yul) catch |err| {
        print("Failed to compile: {}\n", .{err});
        return;
    };
    defer result.deinit(allocator);

    if (result.success) {
        if (result.bytecode) |bytecode| {
            print("✓ Compilation successful!\n", .{});
            print("Bytecode: {s}\n", .{bytecode});
        } else {
            print("✓ Compilation successful but no bytecode generated\n", .{});
        }
    } else {
        print("✗ Compilation failed\n", .{});
        if (result.error_message) |error_msg| {
            print("Error: {s}\n", .{error_msg});
        }
    }

    print("\n", .{});

    // Test 2: HIR to Yul to Bytecode pipeline
    print("Test 2: HIR -> Yul -> Bytecode Pipeline\n", .{});
    print("---------------------------------------\n", .{});

    try codegen_yul.test_yul_codegen();

    print("\n=== Tests Complete ===\n", .{});
}
