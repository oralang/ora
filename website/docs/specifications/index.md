# Specifications

This section contains the formal specifications and technical documentation for the Ora smart contract language.

## Language Specifications

### [Grammar](./grammar.md)
Complete BNF and EBNF grammar specifications for the Ora language, including:
- Formal syntax rules
- Operator precedence
- Language constructs
- Reserved keywords
- Memory region annotations

### [MLIR Integration](./mlir.md)
Comprehensive documentation of Ora's MLIR lowering system, covering:
- LLVM MLIR integration
- Type mapping strategies
- Memory region semantics
- Expression and statement lowering
- Pass management and optimization

### [Formal Verification](./formal-verification.md)
Comprehensive documentation of Ora's formal verification system:
- Proof strategies
- Mathematical domains
- Quantifier support
- SMT solver integration
- Verification examples

### [API Documentation](./api.md)
Complete API reference for the Ora compiler:
- CLI commands
- Library interfaces
- Compilation pipeline
- Error handling
- Performance benchmarks

## Implementation Status

Each specification includes implementation status indicators:
- ✅ **Complete**: Feature is fully implemented and tested (23/29 examples pass)
- 🚧 **In Progress**: Partially implemented, actively being developed
- 📋 **Planned**: Designed but not yet started

### Current Status Summary

| Component | Status | Notes |
|-----------|--------|-------|
| Lexer | ✅ Complete | All tokens, trivia support |
| Parser | ✅ Complete | 79% example validation |
| Type System | ✅ Complete | Full inference and checking |
| Semantics | ✅ Complete | Region and error validation |
| MLIR | ✅ Complete | Lowering and optimization |
| Yul Backend | 🚧 In Progress | Core generation working |
| Standard Lib | 🚧 In Progress | Basic utilities |
| Verification | 📋 Planned | Full formal verification |

## Contributing

These specifications are living documents that evolve with the language. Contributions are welcome:

1. **Grammar improvements**: Help refine language syntax
2. **HIR enhancements**: Extend the intermediate representation
3. **Verification advances**: Improve formal verification capabilities
4. **API extensions**: Add new compiler features

## Quick Navigation

- **New to Ora?** Start with the [Grammar](./grammar.md) specification
- **Compiler development?** Check the [MLIR Integration](./mlir.md) and [API](./api.md) docs
- **Formal verification?** See the [Formal Verification](./formal-verification.md) guide
- **Language implementation?** All specifications work together

These specifications provide the foundation for understanding and extending the Ora language. 