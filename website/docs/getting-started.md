---
sidebar_position: 2
---

# Getting Started

Set up the Ora development environment and try the current implementation.

> Ora is an experimental project and NOT ready for production use. Syntax and features change frequently.

## Prerequisites

- Zig 0.15.x
- CMake (for Solidity library integration)
- Git (for submodules)

## Installation

### Clone and Build

```bash
# Clone the repository
git clone https://github.com/oralang/Ora.git
cd Ora

# Initialize submodules (required for Solidity integration)
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

## Current Implementation Status

**✅ What Works:**
- Contract parsing and AST generation (23/29 examples pass)
- Type checking and semantic analysis
- Error unions and return type checking
- Region transition validation (storage/memory/tstore)
- Switch statements (expression and statement forms)
- Struct and enum declarations

**🚧 In Development:**
- Complete Yul/EVM code generation
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