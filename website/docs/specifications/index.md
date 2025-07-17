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

### [HIR (High-level Intermediate Representation)](./hir.md)
Detailed specification of Ora's intermediate representation, covering:
- Memory model and regions
- Effect system
- Type system
- Node structures
- Optimization framework

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
- ✅ **Fully Implemented**: Feature is complete and tested
- 🚧 **In Development**: Framework exists but under active development
- 📋 **Planned**: Feature designed but not yet implemented

## Contributing

These specifications are living documents that evolve with the language. Contributions are welcome:

1. **Grammar improvements**: Help refine language syntax
2. **HIR enhancements**: Extend the intermediate representation
3. **Verification advances**: Improve formal verification capabilities
4. **API extensions**: Add new compiler features

## Quick Navigation

- **New to Ora?** Start with the [Grammar](./grammar.md) specification
- **Compiler development?** Check the [HIR](./hir.md) and [API](./api.md) docs
- **Formal verification?** See the [Formal Verification](./formal-verification.md) guide
- **Language implementation?** All specifications work together

These specifications provide the foundation for understanding and extending the Ora language. 