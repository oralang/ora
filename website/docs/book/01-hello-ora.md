---
title: "Chapter 1: Hello, Ora"
description: Install the Ora compiler, write your first contract, and understand the basics.
sidebar_position: 1
---

# Hello, Ora

Ora is a smart contract language for the EVM. It compiles through MLIR to bytecode, with explicit memory regions, error unions, refinement types, and built-in formal verification. The compiler is written in Zig.

This book teaches Ora from first contract to production-grade vault. Each chapter adds features to a running example.

> **Note:** This book was written with AI assistance and reviewed by the Ora team. If you find inaccuracies, please open an issue or PR — the compiler is the source of truth, not this text.

## Install

```bash
git clone https://github.com/oralang/Ora.git
cd Ora
./setup.sh
zig build
```

Verify the install:

```bash
./zig-out/bin/ora --help
```

## Your first contract

Create a file called `counter.ora`:

```ora
contract Counter {
    storage var count: u256 = 0;

    pub fn increment() {
        count += 1;
    }

    pub fn get() -> u256 {
        return count;
    }
}
```

Build it:

```bash
./zig-out/bin/ora build counter.ora
```

Format it:

```bash
./zig-out/bin/ora fmt counter.ora
```

## What just happened

- `contract Counter` declares a contract — the top-level unit in Ora.
- `storage var count: u256 = 0` declares a persistent storage variable. The `storage` keyword is explicit — Ora never hides where data lives.
- `pub fn increment()` is a public function, callable from outside the contract. Functions without `pub` are internal.
- `count += 1` modifies storage. The compiler tracks this as a write effect.
- `fn get() -> u256` returns a value. The `->` arrow separates parameters from return type.

## Ora vs Solidity at a glance

| Concept | Solidity | Ora |
|---------|----------|-----|
| Contract | `contract C { }` | `contract C { }` |
| Storage variable | `uint256 count;` | `storage var count: u256;` |
| Function visibility | `function f() public` | `pub fn f()` |
| Return type | `returns (uint256)` | `-> u256` |
| Variable declaration | `uint256 x = 5;` | `let x: u256 = 5;` |
| Error handling | `require(...)` / `revert` | Error unions (`!T \| Error`) |
| Memory annotation | `memory` keyword on params | Region system (`storage`, `memory`, `calldata`, `transient`) |

The syntax differences are deliberate. Ora favors explicitness: types are always annotated, regions are always visible, and errors are always in the type system.

## Project scaffolding

For larger projects, use `ora init`:

```bash
./zig-out/bin/ora init my-project
cd my-project
```

This creates an `ora.toml` config, a `contracts/main.ora` starter file, and a `README.md`.

## CLI quick reference

| Command | Purpose |
|---------|---------|
| `ora build file.ora` | Compile to bytecode |
| `ora fmt file.ora` | Format a single file |
| `ora fmt src/` | Format all `.ora` files in a directory (recursive) |
| `ora fmt --check file.ora` | Check formatting (CI) |
| `ora emit --emit-mlir file.ora` | Emit Ora MLIR |
| `ora emit --emit-ast file.ora` | Emit parsed AST |
| `ora emit --emit-bytecode file.ora` | Emit EVM bytecode |
| `ora init name` | Scaffold a new project |

## Further reading

- [Getting Started](../getting-started) — full installation details
- [Code Formatter](../code-formatter) — formatter options and configuration
