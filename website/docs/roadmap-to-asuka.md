---
sidebar_position: 5
---

# Roadmap to ASUKA Release

The ASUKA release will be Ora's first stable alpha, marking completion of the core compiler infrastructure and basic language features.

## Release Goals

**ASUKA** aims to provide a complete, testable smart contract language implementation suitable for experimentation and early adoption.

### Core Requirements

✅ **Lexer & Parser** (Complete)
- Token generation
- AST building
- Error recovery
- 79% example validation rate

✅ **Type System** (Complete)
- Type checking
- Type inference
- Region validation
- Error unions

🚧 **Code Generation** (In Progress)
- Yul IR generation
- EVM bytecode output
- Optimization passes

🚧 **Standard Library** (In Progress)
- Core utilities
- Common patterns
- Type conversions

📋 **Documentation** (Planned)
- Complete language specification
- Comprehensive examples
- API reference
- Tutorial guides

## Current Progress

### Completed (Q3-Q4 2025)

- ✅ Complete lexer with all tokens
- ✅ Full parser for current grammar
- ✅ Type checking and semantic analysis
- ✅ Memory region validation
- ✅ Error union implementation
- ✅ Switch statement support
- ✅ Struct and enum declarations
- ✅ MLIR lowering infrastructure
- ✅ Example validation framework

### In Progress (Q4 2025)

- 🚧 Yul code generation completion
- 🚧 Standard library basics
- 🚧 Advanced loop syntax (for loops with captures)
- 🚧 Documentation overhaul
- 🚧 Example expansion

### Planned for ASUKA (Q1 2025)

- 📋 Complete Yul backend
- 📋 EVM bytecode generation
- 📋 Basic standard library (10+ functions)
- 📋 50+ working examples
- 📋 Language specification v1.0
- 📋 Comprehensive test suite (>1000 tests)

## Feature Status

### Language Features

| Feature | Status | Notes |
|---------|--------|-------|
| Contracts | ✅ Complete | Full declaration support |
| Functions | ✅ Complete | Including visibility and `requires` |
| Storage regions | ✅ Complete | `storage`, `memory`, `tstore` |
| Types | ✅ Complete | Primitives, structs, enums, maps |
| Error unions | ✅ Complete | `!T \| E1 \| E2` syntax |
| Switch | ✅ Complete | Statement and expression forms |
| While loops | ✅ Complete | Basic iteration |
| For loops | 🚧 In Progress | Capture syntax incomplete |
| Try-catch | ✅ Complete | Basic error handling |
| Event logs | ✅ Complete | Log declarations |
| Imports | ✅ Complete | Module system |

### Compiler Features

| Feature | Status | Notes |
|---------|--------|-------|
| Lexer | ✅ Complete | All tokens, trivia support |
| Parser | ✅ Complete | 79% example pass rate |
| Type checker | ✅ Complete | Full type inference |
| Semantic analysis | ✅ Complete | Region and error validation |
| MLIR lowering | ✅ Complete | Optimization infrastructure |
| Yul generation | 🚧 In Progress | Core functionality implemented |
| Bytecode output | 🚧 In Progress | EVM backend |
| Optimization | 📋 Planned | Advanced passes |

## Release Timeline

**Target**: Q1 2026

### Milestones

1. **M1: Yul Backend Complete** (November 2025)
   - Generate Yul for all valid Ora programs
   - Pass Yul validation
   - Basic optimization

2. **M2: Standard Library** (December 2025)
   - Core utilities (10+ functions)
   - Type conversions
   - Common patterns

3. **M3: Documentation** (December 2025 - January 2026)
   - Language specification v1.0
   - 50+ examples
   - Tutorial guides
   - API reference

4. **M4: Testing & Stabilization** (January 2026)
   - 1000+ test cases
   - Bug fixes
   - Performance tuning
   - Release preparation

5. **ASUKA Release** (February 2026)
   - Public announcement
   - Release notes
   - Migration guide
   - Community launch

## Contributing to ASUKA

Want to help get Ora to ASUKA? Here's how:

### High Priority

- **Yul Generation**: Implement missing Yul output for language constructs
- **Standard Library**: Write core utility functions
- **Examples**: Create comprehensive example programs
- **Testing**: Add test cases for edge cases
- **Documentation**: Write tutorials and guides

### Medium Priority

- **Optimization**: Implement optimization passes
- **Error Messages**: Improve compiler diagnostics
- **Tooling**: Build development tools (LSP, formatter)
- **Benchmarks**: Create performance benchmarks

### Open Issues

Check [GitHub Issues](https://github.com/oralang/Ora/issues) for tasks tagged with:
- `asuka-release` - Critical for first release
- `good-first-issue` - Perfect for new contributors
- `help-wanted` - Community contributions welcome

## Post-ASUKA

After ASUKA, development will focus on:

- **Formal Verification**: Complete `requires`/`ensures` implementation
- **Advanced Features**: Generics, traits, advanced types
- **Tooling**: IDE integration, debugger, profiler
- **Optimization**: Advanced compiler passes
- **Ecosystem**: Package manager, testing framework

## Questions?

- Open a [GitHub Discussion](https://github.com/oralang/Ora/discussions)
- Check the [Contributing Guide](https://github.com/oralang/Ora/blob/main/CONTRIBUTING.md)
- Join development conversations in Issues

---

*Last updated: October 2025*

