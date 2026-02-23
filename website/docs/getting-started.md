---
sidebar_position: 2
---

# Getting Started

Set up the Ora development environment and run the current compiler.

> Ora is experimental and not ready for production use. Syntax and features
> change frequently.

## Prerequisites

- Zig 0.15.x
- CMake
- Git
- Z3
- MLIR

## Installation

```bash
# Clone the repository
git clone https://github.com/oralang/Ora.git
cd Ora

# Run the setup helper
./setup.sh

# Build the compiler
zig build

# Run tests
zig build test
```

## Verify the install

```bash
# View CLI help
./zig-out/bin/ora --help

# Compile an example contract
./zig-out/bin/ora ora-example/smoke.ora

# Format Ora source code
./zig-out/bin/ora fmt contract.ora

# Emit Ora MLIR
./zig-out/bin/ora emit-mlir ora-example/smoke.ora
```

## Start a new project

```bash
# Scaffold a project in a new directory
./zig-out/bin/ora init my-project
cd my-project
```

This generates an `ora.toml`, `contracts/main.ora`, and a `README.md`. You can
also run `ora init` in an existing empty directory or `ora init .` for the
current directory.

## Try your first contract

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

Build it:

```bash
./zig-out/bin/ora build contracts/main.ora
```

## Formatting

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

See [Code Formatter](./code-formatter.md) for details.

## Multi-file projects

Ora supports namespace-qualified imports for splitting code across files:

```ora
const math = @import("./math.ora");

contract Calculator {
    pub fn run() -> u256 {
        return math.add(40, 2);
    }
}
```

See [Imports and Modules](./imports) for the full import system, including
package imports, `ora.toml` configuration, and resolution rules.

## Exploring the repo

- `ora-example/` contains runnable samples.
- `examples/imports_simple/` contains multi-file import examples.
- `tests/fixtures/` contains parser and semantics fixtures.
- `GRAMMAR.bnf` and `GRAMMAR.ebnf` describe the current grammar.
- [Sensei-IR (SIR)](./specifications/sensei-ir.md) describes the backend IR.

## Status

- Examples in `ora-example/` are aligned with the current compiler behavior.
- Expect breaking changes before ASUKA.
