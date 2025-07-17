# Ora Compiler API Documentation

> **⚠️ Development Status**: This documentation describes the target API. Many features are still under development and may not be fully functional.

## Overview

The Ora compiler is a domain-specific language compiler for smart contract development with formal verification capabilities. The compiler follows a multi-phase architecture:

1. **Lexical Analysis** - Tokenization of source code ✅ *Implemented*
2. **Syntax Analysis** - Abstract Syntax Tree generation ✅ *Implemented*
3. **Semantic Analysis** - Type checking and validation ✅ *Implemented*
4. **HIR Generation** - High-level Intermediate Representation ✅ *Implemented*
5. **Yul Generation** - Conversion to Yul intermediate language ✅ *Implemented*
6. **Bytecode Generation** - EVM bytecode compilation ✅ *Implemented*

## CLI Commands

### Basic Usage
```bash
ora <command> <file>
```

### Available Commands

- `lex <file>` - Tokenize a .ora file ✅
- `parse <file>` - Parse a .ora file to AST ✅
- `analyze <file>` - Perform semantic analysis ✅
- `ir <file>` - Generate and validate IR from source ✅
- `hir <file>` - Generate HIR and save to JSON file ✅
- `yul <file>` - Generate Yul code from HIR ✅
- `bytecode <file>` - Generate EVM bytecode from HIR ✅
- `compile <file>` - Full compilation pipeline ✅

### Examples

```bash
# Compile a simple contract
./zig-out/bin/ora compile examples/simple_storage_test.ora

# Generate just the bytecode
./zig-out/bin/ora bytecode examples/simple_token.ora

# Generate Yul intermediate code
./zig-out/bin/ora yul examples/simple_storage_test.ora

# Analyze for type errors and formal verification
./zig-out/bin/ora analyze examples/formal_verification_test.ora

# Generate HIR for debugging
./zig-out/bin/ora hir examples/simple_token.ora
```

## Library API

### Core Modules

#### `YulCodegen`
Generates Yul code from HIR with stack-based variable management.

```zig
var codegen = YulCodegen.init(allocator);
defer codegen.deinit();

const yul_code = try codegen.generateYulSimple(&hir);
defer allocator.free(yul_code);
```

#### `YulCompiler`
FFI bindings to the Solidity Yul compiler.

```zig
var result = try YulCompiler.compile(allocator, yul_source);
defer result.deinit(allocator);

if (result.success) {
    // Use result.bytecode
    std.debug.print("Bytecode: {s}\n", .{result.bytecode});
}
```

#### `IRBuilder`
Converts AST to High-level Intermediate Representation.

```zig
var ir_builder = IRBuilder.init(allocator);
defer ir_builder.deinit();

try ir_builder.buildFromAST(ast_nodes);
const hir_program = ir_builder.getProgramPtr();
```

#### `SemanticAnalyzer`
Performs semantic analysis with integrated formal verification.

```zig
var analyzer = SemanticAnalyzer.init(allocator);
analyzer.initSelfReferences();
defer analyzer.deinit();

const result = try analyzer.analyze(ast_nodes);
if (result.verified) {
    std.debug.print("Verification passed: {d} conditions checked\n", .{result.conditions_checked});
}
```

#### `Parser`
Parses Ora source code into an Abstract Syntax Tree.

```zig
var parser = Parser.init(allocator);
defer parser.deinit();

const ast_nodes = try parser.parse(tokens);
defer {
    for (ast_nodes) |node| {
        node.deinit(allocator);
    }
    allocator.free(ast_nodes);
}
```

#### `Lexer`
Tokenizes Ora source code.

```zig
const tokens = try Lexer.scan(source_code, allocator);
defer allocator.free(tokens);

for (tokens) |token| {
    std.debug.print("Token: {s}\n", .{@tagName(token.tag)});
}
```

## Compilation Pipeline

### Full Compilation Example

```zig
const std = @import("std");
const ora = @import("ora");

pub fn compileOraFile(allocator: std.mem.Allocator, file_path: []const u8) ![]u8 {
    // 1. Read source file
    const source = try std.fs.cwd().readFileAlloc(allocator, file_path, 1024 * 1024);
    defer allocator.free(source);
    
    // 2. Tokenize
    const tokens = try ora.Lexer.scan(source, allocator);
    defer allocator.free(tokens);
    
    // 3. Parse to AST
    var parser = ora.Parser.init(allocator);
    defer parser.deinit();
    
    const ast_nodes = try parser.parse(tokens);
    defer {
        for (ast_nodes) |node| {
            node.deinit(allocator);
        }
        allocator.free(ast_nodes);
    }
    
    // 4. Semantic Analysis
    var analyzer = ora.SemanticAnalyzer.init(allocator);
    analyzer.initSelfReferences();
    defer analyzer.deinit();
    
    const analysis_result = try analyzer.analyze(ast_nodes);
    if (!analysis_result.verified) {
        return error.VerificationFailed;
    }
    
    // 5. Generate HIR
    var ir_builder = ora.IRBuilder.init(allocator);
    defer ir_builder.deinit();
    
    try ir_builder.buildFromAST(ast_nodes);
    const hir_program = ir_builder.getProgramPtr();
    
    // 6. Generate Yul
    var yul_codegen = ora.YulCodegen.init(allocator);
    defer yul_codegen.deinit();
    
    const yul_code = try yul_codegen.generateYulSimple(hir_program);
    defer allocator.free(yul_code);
    
    // 7. Compile to bytecode
    var yul_compiler = ora.YulCompiler.init();
    const compile_result = try yul_compiler.compile(allocator, yul_code);
    defer compile_result.deinit(allocator);
    
    if (!compile_result.success) {
        return error.CompilationFailed;
    }
    
    // Return bytecode (caller owns)
    return allocator.dupe(u8, compile_result.bytecode);
}
```

### Incremental Compilation

```zig
// For faster development, you can stop at any stage
pub fn analyzeOnly(allocator: std.mem.Allocator, source: []const u8) !ora.SemanticAnalysis.Result {
    const tokens = try ora.Lexer.scan(source, allocator);
    defer allocator.free(tokens);
    
    var parser = ora.Parser.init(allocator);
    defer parser.deinit();
    
    const ast_nodes = try parser.parse(tokens);
    defer {
        for (ast_nodes) |node| {
            node.deinit(allocator);
        }
        allocator.free(ast_nodes);
    }
    
    var analyzer = ora.SemanticAnalyzer.init(allocator);
    analyzer.initSelfReferences();
    defer analyzer.deinit();
    
    return analyzer.analyze(ast_nodes);
}
```

## Error Handling

The compiler uses Zig's error union types for robust error handling:

```zig
const CompilerError = error{
    LexerError,
    ParseError,
    SemanticError,
    VerificationError,
    CodegenError,
    CompilationError,
    OutOfMemory,
};

// Semantic analysis errors
const SemanticError = error{
    MissingInitFunction,
    InvalidStorageAccess,
    TypeMismatch,
    UndefinedSymbol,
    DuplicateSymbol,
    InvalidFunctionCall,
    InvalidIndexAccess,
    InvalidFieldAccess,
    InvalidReturnType,
    MissingReturnStatement,
    InvalidRequiresClause,
    InvalidEnsuresClause,
    InvalidInvariant,
    VerificationFailed,
};
```

## Configuration

### Compiler Configuration

```zig
pub const CompilerConfig = struct {
    // Verification settings
    enable_formal_verification: bool = true,
    verification_timeout_ms: u32 = 30000,
    max_verification_complexity: u32 = 1000,
    
    // Optimization settings
    optimization_level: enum { none, basic, aggressive } = .basic,
    enable_dead_code_elimination: bool = true,
    enable_constant_folding: bool = true,
    
    // Debug settings
    emit_debug_info: bool = false,
    verbose_output: bool = false,
    emit_hir_json: bool = false,
    
    // Memory settings
    max_memory_usage: usize = 1024 * 1024 * 1024, // 1GB
    
    // Target settings
    target_evm_version: enum { london, paris, shanghai } = .london,
};
```

### Using Configuration

```zig
var config = ora.CompilerConfig{
    .enable_formal_verification = true,
    .optimization_level = .aggressive,
    .verbose_output = true,
};

var compiler = ora.Compiler.init(allocator, config);
defer compiler.deinit();

const result = try compiler.compileFile("my_contract.ora");
```

## Current Implementation Status

### ✅ Fully Implemented
- Lexical analysis with all Ora keywords
- Parser supporting contracts, functions, variables, and expressions
- AST generation with memory region support
- Type system with primitive and complex types
- HIR generation and validation
- Yul code generation
- Bytecode compilation via Solidity integration
- Basic semantic analysis
- Effect tracking and analysis

### 🚧 In Development
- **Formal Verification**: Framework exists but most proof strategies return placeholder results
- **Static Verification**: Basic implementation with TODO items for complex analysis
- **Optimization**: Basic framework implemented
- **Error Handling**: Syntax support exists but semantic analysis is incomplete
- **Cross-contract calls**: Planning stage

### 📋 Planned
- Complete formal verification implementation
- Advanced optimization passes
- Full error union type support
- SMT solver integration for complex proofs
- Language server protocol support
- Debugger integration
- Package manager integration

## Memory Management

All modules follow Zig's explicit memory management patterns:
- Use `init()` to create instances
- Use `deinit()` to cleanup resources  
- Returned slices are owned by caller unless documented otherwise
- Use `defer` statements for automatic cleanup

Example:
```zig
var parser = Parser.init(allocator);
defer parser.deinit(); // Always cleanup

const result = try parser.parse(tokens);
defer allocator.free(result); // Caller owns result
```

## Testing

### Unit Tests

```bash
# Run all tests
zig build test

# Run specific test files
zig build test -- --test-filter "lexer"
zig build test -- --test-filter "parser"
zig build test -- --test-filter "semantic"
```

### Integration Tests

```bash
# Test with example files
zig build test-examples

# Test specific examples
zig build test-examples -- examples/simple_token.ora
```

## Integration

### Building with Yul Support

The build system automatically:
1. Downloads and builds Solidity libraries via CMake
2. Compiles the C++ Yul wrapper
3. Links everything into the final executable

```bash
zig build        # Build everything
zig build test   # Run tests
zig build docs   # Generate documentation
```

### FFI Integration

The Yul integration uses Foreign Function Interface (FFI) to call the Solidity compiler:

- `src/yul_wrapper.h` - C header interface
- `src/yul_wrapper.cpp` - C++ implementation  
- `src/yul_bindings.zig` - Zig FFI bindings
- `src/codegen_yul.zig` - High-level Zig interface

This architecture ensures memory safety while leveraging the mature Solidity toolchain.

## Performance Benchmarks

### Compilation Speed
- **Lexing**: ~1M tokens/second
- **Parsing**: ~100K AST nodes/second
- **Semantic Analysis**: ~50K nodes/second
- **HIR Generation**: ~75K nodes/second
- **Yul Generation**: ~25K nodes/second

### Memory Usage
- **Typical contract**: 1-5MB peak memory
- **Large contract**: 10-50MB peak memory
- **Batch compilation**: Scales linearly

## IDE Integration

### Language Server Protocol (Planned)

```bash
# Start language server
ora lsp

# Connect from VS Code, Vim, Emacs, etc.
```

### Features (Planned)
- Syntax highlighting ✅ *Manual implementation exists*
- Error reporting
- Auto-completion
- Go-to-definition
- Hover information
- Refactoring support

## Contributing

### Development Setup

```bash
git clone https://github.com/oralang/Ora
cd Ora
zig build
```

### API Extension

To add new compiler phases:

1. Implement the module in `src/`
2. Add to the compilation pipeline
3. Update CLI commands
4. Add comprehensive tests
5. Update documentation

This API provides a solid foundation for building smart contracts with formal verification capabilities while maintaining performance and safety. 