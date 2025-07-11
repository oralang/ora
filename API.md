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

### 🚧 In Development
- **Formal Verification**: Framework exists but most proof strategies return placeholder results
- **Static Verification**: Basic implementation with TODO items for complex analysis
- **Optimization**: Basic framework implemented
- **Error Handling**: Syntax support exists but semantic analysis is incomplete

### 📋 Planned
- Complete formal verification implementation
- Advanced optimization passes
- Full error union type support
- SMT solver integration for complex proofs

## Memory Management

All modules follow Zig's explicit memory management patterns:
- Use `init()` to create instances
- Use `deinit()` to cleanup resources  
- Returned slices are owned by caller unless documented otherwise
- Use `defer` statements for automatic cleanup

## Error Handling

The compiler uses Zig's error union types for robust error handling:

```zig
const SemanticError = error{
    MissingInitFunction,
    InvalidStorageAccess,
    TypeMismatch,
    // ... other errors
};
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
```

### FFI Integration

The Yul integration uses Foreign Function Interface (FFI) to call the Solidity compiler:

- `src/yul_wrapper.h` - C header interface
- `src/yul_wrapper.cpp` - C++ implementation  
- `src/yul_bindings.zig` - Zig FFI bindings
- `src/codegen_yul.zig` - High-level Zig interface

This architecture ensures memory safety while leveraging the mature Solidity toolchain. 