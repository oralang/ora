---
sidebar_position: 2
---

# Getting Started

Set up the Ora development environment and try the current implementation.

> Ora is an experimental project and NOT ready for production use. Syntax and features change frequently.

## Prerequisites

- Zig 0.15.x
- Git (for submodules)

## Installation

### Clone and Build

```bash
# Clone the repository
git clone https://github.com/oralang/Ora.git
cd Ora

# Initialize submodules (includes sensei-ir)
git submodule update --init --recursive

# Build the compiler
zig build

# Run tests to verify installation
zig build test
```

### Verify Installation

```bash
# View CLI help and available commands
./zig-out/bin/ora --help

# Compile an example contract
./zig-out/bin/ora ora-example/smoke.ora

# Format Ora source code
./zig-out/bin/ora fmt contract.ora

# Analyze storage access patterns
./zig-out/bin/ora --analyze-state ora-example/smoke.ora

# Generate MLIR intermediate representation
./zig-out/bin/ora --emit-mlir ora-example/smoke.ora
```

## Try Your First Contract (current syntax)

```ora
contract SimpleStorage {
    storage var value: u256;

    pub fn set(new_value: u256) {
        value = new_value;
    }

    pub fn get() -> u256 {
        return value;
    }
}
```

Parse and inspect:

```bash
./zig-out/bin/ora parse simple_test.ora
```

Format your code:

```bash
# Format a file in-place
./zig-out/bin/ora fmt simple_test.ora

# Check if code is formatted (useful for CI)
./zig-out/bin/ora fmt --check simple_test.ora

# Show diff of formatting changes
./zig-out/bin/ora fmt --diff simple_test.ora

# Output formatted code to stdout
./zig-out/bin/ora fmt --stdout simple_test.ora
```

> 📖 For complete formatter documentation, see [Code Formatter](./code-formatter.md)

## Current Implementation Status

**✅ What Works:** (79% success rate - 76/96 examples)
- Contract parsing and AST generation
- Complete type checking and semantic analysis
- Storage, memory, and transient storage operations
- Switch statements (expression and statement forms)
- Struct declarations, instantiation, and field operations
- Enum declarations with explicit values
- Arithmetic operations (add, sub, mul, div, rem, power)
- Control flow (if/else, switch)
- Map operations (get/store)
- Memory operations (mload, mstore, mload8, mstore8)
- MLIR lowering with 81 operations
- **Code formatter** (`ora fmt`) - Canonical, deterministic code formatting

**🚧 In Development:**
- Complete sensei-ir (SIR) lowering and EVM code generation
- Advanced for-loop syntax
- Standard library
- Formal verification (`requires`/`ensures` full implementation)

## Exploring Examples

Use repo fixtures and examples to explore:

- Semantics fixtures: `tests/fixtures/semantics/{valid,invalid}`
- Parser fixtures: `tests/fixtures/parser/{...}`
- Example snippets: `ora-example/` (reference language constructs; some may be experimental)

## Development Tips

- Run the full suite: `zig build test`
- For quick inspection, prefer `ora lex|parse|ast` on small files
- Grammar references: `GRAMMAR.bnf`, `GRAMMAR.ebnf`

## Building Variants

```bash
# Clean build
zig build clean && zig build

# Debug build
zig build -Doptimize=Debug

# Release build
zig build -Doptimize=ReleaseFast
```

## Current Limitations

**Not Yet Available:**
- Standard library (in development)
- Complete Yul code generation
- Advanced for-loop capture syntax
- Full formal verification

**Important Notes:**
- Syntax may change before ASUKA release
- Not ready for production use
- Some examples demonstrate planned features
- See [roadmap](./roadmap-to-asuka.md) for development timeline 