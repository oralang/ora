---
sidebar_position: 5
---

# Roadmap to ASUKA Release

The ASUKA release will be Ora's first stable alpha, marking completion of the core compiler infrastructure and basic language features.

## Release Goals

**ASUKA** aims to provide a complete, testable smart contract language implementation suitable for experimentation and early adoption.

### Core Requirements

✅ **Lexer & Parser** (Complete)
- Token generation with trivia support
- AST building
- Error recovery
- **79% example validation rate** (76/96 examples passing)

✅ **Type System** (Complete)
- Type checking and validation
- Region validation (storage, memory, transient)
- Error unions
- ⚠️ Type inference (requires explicit types in some cases)

🚧 **Code Generation** (In Progress)
- MLIR lowering: ✅ Complete (81 operations)
- sensei-ir (SIR) lowering: 🚧 In Progress
- EVM bytecode output via sensei-ir: 🚧 In Progress
- Optimization passes: 📋 Planned

🚧 **Standard Library** (In Progress)
- Core utilities
- Common patterns
- Type conversions

🚧 **Documentation** (In Progress)
- ✅ Type System Specification v0.11 (PDF)
- ✅ ABI Specification v0.1
- 🚧 Complete language specification v1.0
- 🚧 Comprehensive examples (expanding to 80+)
- ✅ API reference (in progress)
- 📋 Tutorial guides

## Current Progress

### Completed (Q3-Q4 2025)

- ✅ Complete lexer with all tokens and trivia support
- ✅ Full parser for current grammar (79% success rate - 76/96 examples)
- ✅ Type checking and semantic analysis
- ✅ Memory region validation (storage, memory, transient)
- ✅ Error union implementation
- ✅ Switch statement support (expression and statement forms)
- ✅ Struct and enum declarations with full operations
- ✅ MLIR lowering infrastructure (81 operations)
- ✅ Arithmetic operations (add, sub, mul, div, rem, power)
- ✅ Control flow (if/else, switch, while loops)
- ✅ Map operations (get/store)
- ✅ Memory operations (mload, mstore, mload8, mstore8)
- ✅ Transient storage operations (tload, tstore)
- ✅ Example validation framework

### In Progress (Q4 2025 - Q1 2026)

- 🚧 sensei-ir (SIR) lowering and integration
- 🚧 EVM bytecode generation via sensei-ir
- 🚧 For loops with capture syntax
- 🚧 Enhanced error handling (try-catch improvements)
- 🚧 Type inference improvements (currently requires explicit types)
- 🚧 Standard library basics
- 🚧 Documentation overhaul
- 🚧 Playground development (syntax validator)

### Planned for ASUKA (Q1-Q2 2026)

- 📋 Complete sensei-ir (SIR) lowering for all constructs
- 📋 EVM bytecode generation via sensei-ir debug-backend
- 📋 Basic standard library (10+ functions)
- 📋 80+ working examples (target: 90%+ success rate)
- 📋 Language specification v1.0
- 📋 Comprehensive test suite (>1000 tests)
- 📋 Interactive playground (syntax validation)
- 📋 Improved error messages and diagnostics

## Feature Status

### Language Features

| Feature | Status | Notes |
|---------|--------|-------|
| Contracts | ✅ Complete | Full declaration support |
| Functions | ✅ Complete | Including visibility and `requires` |
| Storage regions | ✅ Complete | `storage`, `memory`, `transient` |
| Types | ✅ Complete | Primitives, structs, enums, maps |
| Error unions | ✅ Complete | `!T \| E1 \| E2` syntax |
| Switch | ✅ Complete | Statement and expression forms |
| Control flow | ✅ Complete | if/else, switch, while loops |
| Structs | ✅ Complete | Declaration, instantiation, field operations |
| Enums | ✅ Complete | Declaration with explicit values |
| Maps | ✅ Complete | Map get/store operations |
| Memory ops | ✅ Complete | mload, mstore, mload8, mstore8 |
| Transient storage | ✅ Complete | tload, tstore operations |
| Arithmetic | ✅ Complete | All operations (add, sub, mul, div, rem, power) |
| For loops | 🚧 In Progress | Capture syntax incomplete |
| Try-catch | ⚠️ Partial | Error declarations work, try-catch needs improvement |
| Event logs | ✅ Complete | Log declarations |
| Imports | ✅ Complete | Module system |
| Type inference | 🚧 In Progress | Currently requires explicit type annotations |

### Compiler Features

| Feature | Status | Notes |
|---------|--------|-------|
| Lexer | ✅ Complete | All tokens, trivia support, 79% success rate |
| Parser | ✅ Complete | 79% example pass rate (76/96 examples) |
| Type checker | ✅ Complete | Full type checking and validation |
| Semantic analysis | ✅ Complete | Region and error validation |
| MLIR lowering | ✅ Complete | 81 operations, optimization infrastructure |
| State analysis | ✅ Complete | Automatic storage tracking & warnings |
| sensei-ir lowering | 🚧 In Progress | Integration with sensei-ir in development |
| Bytecode output | 🚧 In Progress | EVM backend via sensei-ir |
| Optimization | 📋 Planned | Advanced passes |
| Playground | 🚧 In Progress | Syntax validator (WASM-based) |

## Release Timeline

**Target**: Q2 2026

### Milestones

1. **M1: sensei-ir Backend Complete** (Q1 2026)
   - Lower MLIR to sensei-ir (SIR) for all valid Ora programs
   - Generate EVM bytecode via sensei-ir debug-backend
   - Basic optimization passes
   - Integration testing

2. **M2: Standard Library & Examples** (Q1-Q2 2026)
   - Core utilities (10+ functions)
   - Type conversions
   - Common patterns
   - Expand examples to 80+ (target 90%+ success rate)

3. **M3: Playground & Tooling** (Q1 2026)
   - Interactive playground (syntax validator)
   - WASM-based validation
   - Example library in playground
   - Feedback collection system

4. **M4: Documentation** (Q1-Q2 2026)
   - Language specification v1.0
   - Complete API reference
   - Tutorial guides
   - Migration documentation

5. **M5: Testing & Stabilization** (Q2 2026)
   - 1000+ test cases
   - Bug fixes and error message improvements
   - Performance tuning
   - Release preparation

6. **ASUKA Release** (Q2 2026)
   - Public announcement
   - Release notes
   - Migration guide
   - Community launch

## Contributing to ASUKA

Want to help get Ora to ASUKA? Here's how:

### High Priority

- **sensei-ir Lowering**: Complete MLIR to sensei-ir (SIR) lowering for all language constructs
- **EVM Bytecode Generation**: Integrate sensei-ir debug-backend for bytecode output
- **For Loops**: Complete capture syntax implementation
- **Error Handling**: Improve try-catch error handling
- **Type Inference**: Reduce need for explicit type annotations
- **Standard Library**: Write core utility functions
- **Examples**: Expand to 80+ examples (target 90%+ success rate)
- **Playground**: Complete interactive syntax validator

### Medium Priority

- **Testing**: Add test cases for edge cases and improve coverage
- **Documentation**: Write tutorials, guides, and complete API reference
- **Error Messages**: Improve compiler diagnostics and suggestions
- **Optimization**: Implement optimization passes
- **Tooling**: Build development tools (LSP, formatter)
- **Benchmarks**: Create performance benchmarks

### Open Issues

Check [GitHub Issues](https://github.com/oralang/Ora/issues) for tasks tagged with:
- `asuka-release` - Critical for first release
- `good-first-issue` - Perfect for new contributors
- `help-wanted` - Community contributions welcome

## Post-ASUKA

After ASUKA, development will focus on:

- **Formal Verification**: Complete `requires`/`ensures` implementation with Z3 integration
- **Advanced Features**: Generics, traits, advanced types
- **Tooling**: IDE integration (LSP), debugger, profiler
- **Optimization**: Advanced compiler passes and gas optimization
- **Ecosystem**: Package manager, testing framework
- **Playground**: Full compilation support, sharing features
- **Performance**: Compiler speed improvements, optimization passes

## Questions?

- Open a [GitHub Discussion](https://github.com/oralang/Ora/discussions)
- Check the [Contributing Guide](https://github.com/oralang/Ora/blob/main/CONTRIBUTING.md)
- Join development conversations in Issues

## Current Statistics

- **Success Rate**: 79% (76/96 examples passing)
- **MLIR Operations**: 81 operations implemented
- **Language Features**: Most core features complete
- **Compiler Pipeline**: Lexer → Parser → Type Check → MLIR (complete)
- **Backend**: sensei-ir integration in progress

## Recent Achievements

- ✅ Reached 79% example success rate (76/96 examples)
- ✅ Complete MLIR lowering with 81 operations
- ✅ Full struct and enum support
- ✅ Complete memory operations (storage, memory, transient)
- ✅ Comprehensive arithmetic and control flow support
- ✅ Migrated from Yul to sensei-ir backend
- ✅ Established Specs section (Type System v0.11, ABI v0.1)

---

*Last updated: December 2025*

