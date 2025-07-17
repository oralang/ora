# Ora Development Notebook

An experimental smart contract language with formal verification capabilities.

> **🚧 EXPERIMENTAL PROJECT**: Ora is NOT ready for production use. This repository serves as an open notebook documenting language design and implementation progress. Features, syntax, and APIs are subject to change without notice.

## Overview

Ora is an experimental smart contract language that compiles to Yul (Ethereum's intermediate language) and EVM bytecode. Built with Zig, it aims to provide safety guarantees through formal verification while maintaining high performance and developer productivity.

## Development Status

### ✅ Currently Functional
- Core compilation pipeline: Lexical analysis → Syntax analysis → Semantic analysis → HIR → Yul → Bytecode
- Basic smart contract syntax and compilation
- Yul code generation and EVM bytecode output
- Error handling foundations

### 🚧 In Active Development
- **Formal Verification**: Mathematical proof capabilities for complex conditions and quantifiers
- **Advanced Safety**: Memory safety, type safety, and overflow protection
- **Comprehensive Error Handling**: Full `!T` error union implementation
- **Standard Library**: Core utilities and common patterns

### 📋 Planned Features
- Compile-time evaluation optimizations
- Advanced type system features
- IDE integration and tooling
- Comprehensive testing frameworks

## Quick Start

### Prerequisites

- Zig 0.14.1 or later
- CMake (for Solidity library integration)
- Git (for submodules)

### Building

```bash
# Clone the repository
git clone https://github.com/oralang/Ora.git
cd Ora

# Initialize submodules
git submodule update --init --recursive

# Build the compiler
zig build

# Run tests
zig build test
```

### Try Current Implementation

Create a simple storage contract (`test.ora`):

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

Compile it:

```bash
./zig-out/bin/ora test.ora
```

## Language Features (Current)

### Basic Contract Structure
```ora
contract MyContract {
    storage var balance: u256;
    immutable owner: address;
    storage const MAX_SUPPLY: u256 = 1000000;
    
    pub fn transfer(to: address, amount: u256) -> bool {
        // Implementation
        return true;
    }
}
```

### Error Handling (In Development)
```ora
error InsufficientBalance;
error InvalidAddress;

fn transfer(to: address, amount: u256) -> !u256 {
    if (balance < amount) {
        return error.InsufficientBalance;
    }
    
    balance -= amount;
    return balance;
}
```

### Formal Verification (Planned)
```ora
fn transfer(to: address, amount: u256) -> bool
    requires balances[sender] >= amount
    ensures balances[sender] + balances[to] == old(balances[sender]) + old(balances[to])
{
    balances[sender] -= amount;
    balances[to] += amount;
    return true;
}
```

## Project Structure

```
Ora/
├── src/                    # Compiler implementation (Zig)
│   ├── ast.zig            # Abstract Syntax Tree
│   ├── parser.zig         # Syntax analysis
│   ├── semantic.zig       # Semantic analysis
│   ├── hir.zig            # High-level IR
│   └── codegen_yul.zig    # Yul code generation
├── examples/              # Working code examples
│   ├── core/              # Basic functionality
│   ├── advanced/          # Advanced features (experimental)
│   └── tokens/            # Token contract patterns
├── docs/                  # Technical specifications
│   ├── GRAMMAR.bnf        # Language grammar
│   ├── HIR_SPEC.md        # IR specification
│   └── formal-verification.md
├── website/               # Documentation site
└── vendor/solidity/       # Solidity integration
```

## Documentation

- **Website**: [ora-lang.org](https://ora-lang.org) - Development notebook and documentation
- **Examples**: Browse `examples/` directory for working code patterns
- **Grammar**: See `GRAMMAR.bnf` for current syntax specification
- **API**: Check `API.md` for compiler interface

## Development Notes

### Technical Decisions

- **Zig as Implementation Language**: Leverages compile-time capabilities for meta-programming
- **Yul Backend**: Compiles to Ethereum's intermediate language for optimal bytecode generation
- **Multi-Phase Compilation**: Separate lexical, syntax, semantic, and code generation phases
- **Formal Verification Focus**: Designing for mathematical proof capabilities from the ground up

### Current Limitations

- No standard library implementation yet
- Limited error messages and debugging information
- Incomplete type system (basic types only)
- No formal verification execution (syntax defined but not implemented)
- Minimal testing framework

### Not Suitable For

- Production smart contracts
- Financial applications
- Critical infrastructure
- Projects requiring stable APIs
- Applications needing comprehensive tooling

## Contributing

This is an experimental research project. Ways to contribute:

1. **Report Issues**: File bugs for unexpected behavior
2. **Suggest Improvements**: Discuss language design decisions
3. **Submit Examples**: Share interesting contract patterns
4. **Improve Documentation**: Help expand this development notebook

## Community

- **Source Code**: [GitHub Repository](https://github.com/oralang/Ora)
- **Issues**: [Bug Reports & Feature Requests](https://github.com/oralang/Ora/issues)
- **Discussions**: [GitHub Discussions](https://github.com/oralang/Ora/discussions)

## License

This project is licensed under the GNU License - see the [LICENSE](LICENSE) file for details.

---

*Last updated: December 2024 - Reflects current development status*