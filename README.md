# Ora

A domain-specific language for smart contract development with formal verification capabilities.

> **⚠️ Development Status**: This project is under active development. Many features are still being implemented and the API may change.

## Overview

Ora is a modern smart contract language that compiles to Yul (Ethereum's intermediate language) and EVM bytecode. Built with Zig, it provides safety guarantees through formal verification while maintaining high performance and developer productivity.

## Key Features

- **Formal Verification**: Built-in mathematical proof capabilities for complex conditions and quantifiers *(in development)*
- **Multi-Phase Compilation**: Lexical analysis → Syntax analysis → Semantic analysis → HIR → Yul → Bytecode
- **Safe by Design**: Memory safety, type safety, and overflow protection *(in development)*
- **Ethereum Integration**: Direct compilation to EVM bytecode via Yul intermediate representation
- **Modern Syntax**: Clean, readable syntax inspired by Rust and Zig

> **🚧 Implementation Status**: Core compilation pipeline is functional. Advanced features like formal verification and comprehensive safety checks are being actively developed.

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

### Your First Contract

Create a simple storage contract (`storage.ora`):

```ora
contract SimpleStorage {
    var value: u256;
    
    pub fn set(new_value: u256) {
        value = new_value;
    }
    
    pub fn get() -> u256 {
        return value;
    }
}
```

Compile to bytecode:

```bash
./zig-out/bin/ora compile storage.ora
```

## Language Features

### Storage and State Management

```ora
contract Token {
    let name: string;
    var total_supply: u256;
    var balances: mapping[address, u256];
}
```

### Formal Verification

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

> **⚠️ Note**: Formal verification syntax is still being finalized and may not be fully functional yet.

### Error Handling

```ora
function safe_divide(a: u256, b: u256) -> u256 | DivisionError {
    if (b == 0) {
        return DivisionError.DivisionByZero;
    }
    return a / b;
}
```

## CLI Usage

```bash
# Full compilation pipeline
ora compile contract.ora

# Individual phases
ora lex contract.ora          # Tokenization
ora parse contract.ora        # AST generation
ora analyze contract.ora      # Semantic analysis
ora hir contract.ora          # HIR generation
ora yul contract.ora          # Yul generation
ora bytecode contract.ora     # Bytecode generation
```

## Examples

The `examples/` directory contains various contract examples:

- **Simple Storage**: Basic state management
- **ERC20 Token**: Standard token implementation
- **Formal Verification**: Advanced proof examples
- **Error Handling**: Comprehensive error management
- **Optimization**: Performance optimization patterns

## Architecture

### Compilation Pipeline

1. **Lexer** (`src/lexer.zig`) - Tokenizes source code
2. **Parser** (`src/parser.zig`) - Generates Abstract Syntax Tree
3. **Semantic Analyzer** (`src/semantics.zig`) - Type checking and validation
4. **HIR Builder** (`src/ir.zig`) - High-level Intermediate Representation
5. **Yul Codegen** (`src/codegen_yul.zig`) - Yul code generation
6. **Bytecode Generation** - EVM bytecode via Solidity integration

### Formal Verification

The formal verification system supports:
- **Proof Strategies**: Direct proof, contradiction, induction, case analysis
- **Mathematical Domains**: Integer, real, bit-vector, array, set operations
- **Quantifiers**: Universal (∀) and existential (∃) quantification
- **SMT Integration**: Z3, CVC4, Yices solver support

## Development

### Project Structure

```
oralang/
├── src/                    # Compiler source code
├── examples/              # Example contracts
├── vendor/solidity/       # Solidity libraries (submodule)
├── build.zig             # Zig build configuration
└── README.md             # This file
```

### Building with Debug Info

```bash
zig build -Doptimize=Debug
```

### Running Examples

```bash
# Parser demo
zig build parser-demo

# Yul integration test
zig build yul-test

# Optimization demo
zig build optimization-demo

# Formal verification demo
zig build formal-verification-demo
```

## License

[License information here]

## Documentation

- [API Documentation](API.md)
- [Formal Verification Guide](formal-verification.md)
- [Syntax Guide](syntax-guide.md)

> **📝 Note**: Some documentation is still being written.