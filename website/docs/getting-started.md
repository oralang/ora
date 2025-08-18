---
sidebar_position: 2
---

# Getting Started

Set up the Ora development environment and try the current implementation.

> Ora is an experimental project and NOT ready for production use. Syntax and features change frequently.

## Prerequisites

- Zig 0.14.1 or later
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
# CLI help
./zig-out/bin/ora --help

# Lex/parse a simple, known-good fixture
./zig-out/bin/ora lex tests/fixtures/semantics/valid/storage_region_moves.ora
./zig-out/bin/ora parse tests/fixtures/semantics/valid/storage_region_moves.ora

# Generate AST JSON
./zig-out/bin/ora -o build ast tests/fixtures/semantics/valid/storage_region_moves.ora
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

What works now:
- Basic contract parsing and AST generation
- Error unions and return checking (partial)
- Region transition checks (storage/stack/memory/tstore basics)

In development:
- Formal verification (`requires`/`ensures`/`invariant` checks)
- Full operator typing and switch typing
- Yul/EVM backend stabilization

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

## Limitations

- No standard library
- Diagnostics and typing are evolving
- Backend codegen is experimental
- Syntax may change without notice 