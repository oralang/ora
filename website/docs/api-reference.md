---
sidebar_position: 5
---

# API Reference

The Ora compiler and standard library API documentation is automatically generated from the source code using Zig's built-in documentation system.

## Interactive API Documentation

The complete API reference is available as an interactive documentation website:

**[View API Reference →](/api-docs/)**

This includes:

- **Compiler API**: Functions and types for lexing, parsing, semantic analysis, and code generation
- **Standard Library**: Built-in functions, constants, and utilities available in Ora contracts
- **Type Definitions**: Complete type information for all public APIs
- **Source Code Links**: Direct links to implementation code

## Key Modules

### Compiler Components

- **`lexer.zig`**: Tokenization and lexical analysis
- **`parser.zig`**: Abstract Syntax Tree (AST) generation
- **`semantics.zig`**: Type checking and semantic analysis
- **`ir.zig`**: High-level Intermediate Representation (HIR)
- **`codegen_yul.zig`**: Yul code generation
- **`comptime_eval.zig`**: Compile-time evaluation engine

### Standard Library

- **`std.transaction`**: Transaction context (`sender`, `value`, etc.)
- **`std.block`**: Block information (`timestamp`, `number`, etc.)
- **`std.constants`**: Common constants (`ZERO_ADDRESS`, etc.)
- **`std.crypto`**: Cryptographic functions
- **`std.math`**: Mathematical operations with overflow checking
- **Division Operations**: `@divTrunc`, `@divFloor`, `@divCeil`, `@divExact`, `@divmod` with comprehensive error handling

### Formal Verification

- **`static_verifier.zig`**: Formal verification engine
- **`proof_engine.zig`**: Mathematical proof generation
- **`smt_interface.zig`**: SMT solver integration

## Mathematical Operations

### Division Functions

All division operations support comprehensive error handling and compile-time evaluation:

- **`@divTrunc(a, b)`**: Truncating division toward zero
- **`@divFloor(a, b)`**: Floor division toward negative infinity  
- **`@divCeil(a, b)`**: Ceiling division toward positive infinity
- **`@divExact(a, b)`**: Exact division, errors if remainder is non-zero
- **`@divmod(a, b)`**: Returns quotient and remainder as tuple

#### Error Types

- `DivisionByZero`: Raised when divisor is zero
- `InexactDivision`: Raised by `@divExact` when division has remainder

#### Type Support

All division operations work with signed and unsigned integers: `u8`, `u16`, `u32`, `u64`, `u128`, `i8`, `i16`, `i32`, `i64`, `i128`.

## Using the API Documentation

The generated documentation is fully interactive:

1. **Search**: Use the search box to find specific functions or types
2. **Navigation**: Browse by module or use the sidebar
3. **Source Links**: Click on items to view the source code
4. **Type Information**: Hover over types to see detailed information
5. **Cross-references**: Click on types to navigate to their definitions

## Regenerating Documentation

To update the API documentation after code changes:

```bash
# Generate and copy to website
./scripts/generate-docs.sh

# Or use the npm script
cd website
npm run docs:generate
```

The documentation is automatically regenerated during the build process.

## Development Workflow

For contributors working on the compiler:

1. **Document your code**: Use Zig doc comments (`///`) for public APIs
2. **Generate docs locally**: Run `zig build docs` to check formatting
3. **Update website**: Run the generation script to update the website
4. **Review changes**: Check the interactive docs for accuracy

## API Stability

> **⚠️ Development Status**: The Ora compiler API is under active development. Internal APIs may change between versions.

- **Public APIs**: Documented with stability guarantees
- **Internal APIs**: May change without notice  
- **Experimental**: Marked clearly in documentation

## Further Reading

- [Language Basics](./language-basics) - Core language syntax and concepts
- [Examples](./examples) - Real code examples using the APIs
- [Getting Started](./getting-started) - Development environment setup 