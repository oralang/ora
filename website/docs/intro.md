---
sidebar_position: 1
---

# Introduction to Ora

Welcome to the **Ora Development Notebook** - documentation for an experimental smart contract language with formal verification capabilities.

> **🚧 EXPERIMENTAL PROJECT**: Ora is in active development and is NOT ready for production use. This documentation serves as an open notebook documenting the language design and implementation progress. Features, syntax, and APIs are subject to change without notice.

## What is Ora?

Ora is an experimental smart contract language that compiles to Yul (Ethereum's intermediate language) and EVM bytecode. Built with Zig, it aims to provide safety guarantees through formal verification while maintaining high performance and developer productivity.

### Development Status

**✅ Currently Functional:**
- Core compilation pipeline: Lexical analysis → Syntax analysis → Semantic analysis → HIR → Yul → Bytecode
- Basic smart contract syntax and compilation
- Yul code generation and EVM bytecode output
- Error handling foundations

**🚧 In Active Development:**
- **Formal Verification**: Mathematical proof capabilities with `requires`, `ensures`, and `invariant` statements
- **Advanced Safety**: Memory safety guarantees and overflow protection
- **Comprehensive Error Handling**: Full `!T` error union implementation
- **Standard Library**: Core utilities and common patterns

**📋 Planned Features:**
- Compile-time evaluation optimizations
- Advanced type system features
- IDE integration and tooling
- Comprehensive testing frameworks

## Language Design Philosophy

Ora is built on the principle that **correctness should be the default**. The language design encourages:

- **Compile-time computation**: Maximize work done at compile time
- **Explicit error handling**: Using `!T` error unions for robust error management
- **Memory region awareness**: Clear distinction between `storage`, `immutable`, and compile-time constants
- **Formal verification**: Mathematical proofs for contract correctness (in development)

## Current Language Sample

> **Note**: Syntax is experimental and subject to change

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

## Planned Advanced Features

### Formal Verification (In Development)

```ora
function transfer(to: address, amount: u256) -> bool
    requires balances[sender] >= amount
    ensures balances[sender] + balances[to] == old(balances[sender]) + old(balances[to])
{
    balances[sender] -= amount;
    balances[to] += amount;
    return true;
}
```

### Error Handling (In Development)

```ora
fn transfer(to: address, amount: u256) -> !u256 {
    if (to == std.constants.ZERO_ADDRESS) {
        return error.InvalidAddress;
    }
    
    if (balance < amount) {
        return error.InsufficientBalance;
    }
    
    balance -= amount;
    return balance;
}
```

### Memory Regions (Partially Implemented)

```ora
contract Token {
    storage var total_supply: u256;        // Persistent storage
    storage var balances: map[address, u256]; // Mapping storage
    immutable owner: address;              // Set once at deployment
    storage const MAX_SUPPLY: u256 = 1000000; // Compile-time constant
}
```

## Getting Started

Ready to explore? Check out our [Getting Started](./getting-started) guide to set up the development environment and try the current implementation.

Browse our [Examples](./examples) to see working code patterns from the repository.

## Development Notes

This is an **experimental project** serving as:
- Language design exploration
- Implementation learning exercise
- Formal verification research
- Smart contract safety research

**Not suitable for:**
- Production smart contracts
- Financial applications
- Critical infrastructure
- Stable API requirements

## Contributing & Community

Ora is an open-source research project. Follow development:

- **Source Code**: [oralang/Ora](https://github.com/oralang/Ora)
- **Issues**: [Report bugs or discuss features](https://github.com/oralang/Ora/issues)
- **Discussions**: [GitHub Discussions](https://github.com/oralang/Ora/discussions)

---

*Last updated: December 2024*
