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
- Contract declarations and event logs
- **Inline functions** with automatic complexity analysis

✅ **Advanced Compiler Features**
- **State Analysis**: Automatic storage access tracking with dead store detection
- **Formal Verification**: Z3 integration for mathematical proofs (AST complete)
- **MLIR Optimization**: Industry-standard compiler infrastructure
- **Gas Insights**: Built-in warnings for inefficient patterns

### In Development

🚧 **Yul Backend**: Complete code generation for EVM bytecode  
🚧 **Standard Library**: Core utilities and common patterns  
🚧 **Z3 Verification**: Verification condition generation (grammar & AST complete)  
🚧 **For Loops**: Advanced capture syntax

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

## Advanced Compiler Features

### State Analysis (✅ Complete)

Automatic storage access tracking with actionable warnings:

```bash
$ ora contract.ora

⚠️  State Analysis Warnings for MyContract (2):

💡 [DeadStore] Storage variable 'unusedData' is written but never read
   💬 Remove unused storage variable or add read logic

ℹ️  [MissingCheck] Function 'approve' modifies storage without validation
   💬 Add validation checks before modifying storage
```

[Learn more about State Analysis →](./state-analysis)

### Formal Verification (🚧 In Progress)

Mathematical proofs for contract correctness using Z3 SMT solver:

```ora
pub fn transfer(to: address, amount: u256) -> bool
    requires amount > 0
    requires balances[std.msg.sender()] >= amount
    ensures balances[std.msg.sender()] == old(balances[std.msg.sender()]) - amount
    ensures balances[to] == old(balances[to]) + amount
{
    let sender = std.msg.sender();
    balances[sender] -= amount;
    balances[to] += amount;
    return true;
}
```

[Learn more about Formal Verification →](./formal-verification)

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

Ready to explore? Check out our guides:

- 📚 [Getting Started](./getting-started) - Set up the compiler and try examples
- 🔍 [State Analysis](./state-analysis) - Automatic storage optimization
- 🔬 [Formal Verification](./formal-verification) - Mathematical proofs with Z3
- 📖 [Examples](./examples) - Working code patterns

Try the compiler features:

```bash
# Compile with automatic state analysis
ora contract.ora

# Detailed storage access analysis
ora --analyze-state contract.ora

# Generate optimized MLIR
ora --emit-mlir contract.ora
```

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
