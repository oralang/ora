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

### [Formal Verification](../formal-verification.md)
Z3 SMT solver integration for mathematical proof of contract properties:
- Preconditions and postconditions (`requires`, `ensures`)
- Contract and loop invariants
- Quantified expressions (`forall`, `exists`)
- Ghost code for specification
- Verification condition generation

### [API Documentation](./api.md)
Complete API reference for the Ora compiler:
- CLI commands and flags
- Library interfaces
- Compilation pipeline
- Error handling
- Performance benchmarks

### [State Analysis](../state-analysis.md)
Automatic storage access tracking and optimization:
- Function property detection (stateless, readonly, state-modifying)
- Dead store detection (contract-level analysis)
- Missing validation warnings
- Constructor-specific checks
- Gas optimization insights

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
| State Analysis | ✅ Complete | Automatic tracking & warnings |
| Yul Backend | 🚧 In Progress | Core generation working |
| Standard Lib | 🚧 In Progress | Basic utilities |
| Z3 Verification | 🚧 In Progress | Grammar & AST complete, VC generation in progress |

## Contributing

These specifications are living documents that evolve with the language. Contributions are welcome:

1. **Grammar improvements**: Help refine language syntax
2. **HIR enhancements**: Extend the intermediate representation
3. **Verification advances**: Improve formal verification capabilities
4. **API extensions**: Add new compiler features

## Quick Navigation

- **New to Ora?** Start with the [Grammar](./grammar.md) specification
- **Writing contracts?** Check out [State Analysis](../state-analysis.md) for automatic optimization
- **Formal verification?** See the [Formal Verification](../formal-verification.md) guide for Z3 integration
- **Compiler development?** Explore [MLIR Integration](./mlir.md) and [API](./api.md) docs
- **Language implementation?** All specifications work together

These specifications provide the foundation for understanding and extending the Ora language. 