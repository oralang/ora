---
sidebar_position: 1
---

# Introduction to Ora

> **Pre-ASUKA Alpha** | **Contributors Welcome** | **Active Development**

Welcome to the **Ora Development Notebook** - documentation for an experimental smart contract language targeting EVM/Yul with explicit semantics and clean compilation pipeline.

## Project Status

Ora is in **pre-release alpha**, working toward the first ASUKA release. The core compiler infrastructure is functional, with active development on Yul code generation and standard library.

**Not ready for production.** Syntax and features may change without notice.

## What is Ora?

Ora is an experimental smart contract language that compiles to Yul (Ethereum's intermediate language) and EVM bytecode. Built with Zig, it aims to provide safety guarantees through formal verification while maintaining high performance and developer productivity.

### What Works Now

✅ **Compiler Pipeline**
- Full lexer and parser (23/29 examples pass)
- Type checking and semantic analysis
- AST generation and validation
- MLIR lowering for optimization

✅ **Language Features**
- Storage, memory, and transient storage regions
- Error unions (`!T | E1 | E2`)
- Switch statements (expression and statement forms)
- Structs, enums, and custom types
- Function preconditions (`requires`)
- Contract declarations and event logs

### In Development

🚧 **Yul Backend**: Complete code generation for EVM bytecode
🚧 **Standard Library**: Core utilities and common patterns
🚧 **For Loops**: Advanced capture syntax
🚧 **Formal Verification**: Full `requires`/`ensures` implementation

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
    requires(balances[sender] >= amount)
    ensures(balances[sender] + balances[to] == old(balances[sender]) + old(balances[to]))
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

## Contributing

**We welcome contributors!** Ora is a great project for learning compiler development and language design.

**Ways to contribute:**
- 🐛 Report bugs or unexpected behavior
- 📝 Improve documentation and examples
- ✅ Add test cases
- 🔧 Implement language features
- 💡 Participate in design discussions

See [CONTRIBUTING.md](https://github.com/oralang/Ora/blob/main/CONTRIBUTING.md) for setup and guidelines.

**Good first issues:**
- Add test cases to `tests/fixtures/`
- Improve parser error messages  
- Create more examples in `ora-example/`
- Update documentation

## Community

- **GitHub**: [oralang/Ora](https://github.com/oralang/Ora)
- **Issues**: [Report bugs](https://github.com/oralang/Ora/issues)
- **Discussions**: [Join conversations](https://github.com/oralang/Ora/discussions)

---

*Last updated: October 2025*
