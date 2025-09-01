# MLIR Lowering System

## Overview

The MLIR (Multi-Level Intermediate Representation) lowering system provides a comprehensive intermediate representation for the Ora smart contract language. This system leverages the existing MLIR infrastructure from the LLVM project (https://mlir.llvm.org) to enable advanced analysis, optimization, and potential alternative code generation paths while maintaining compatibility with the existing Yul backend.

**Note:** We are using the existing MLIR framework developed by the LLVM community, not implementing MLIR itself. Our contribution is the Ora-specific lowering logic that translates Ora language constructs into MLIR operations and types.

## Architecture

The MLIR lowering system follows a modular architecture with clear separation of concerns:

```
CST → AST → Semantics → MLIR Module → MLIR Passes → Analysis/Optimization
                                   ↘
                                    Yul Backend (Primary)
```

### Core Components

1. **Main Orchestrator** (`src/mlir/lower.zig`) - Coordinates the lowering process
2. **Type Mapping** (`src/mlir/types.zig`) - Maps Ora types to MLIR types
3. **Expression Lowering** (`src/mlir/expressions.zig`) - Handles expression constructs
4. **Statement Lowering** (`src/mlir/statements.zig`) - Handles statement constructs
5. **Declaration Lowering** (`src/mlir/declarations.zig`) - Handles top-level declarations
6. **Memory Management** (`src/mlir/memory.zig`) - Manages storage/memory/tstore semantics
7. **Symbol Table** (`src/mlir/symbols.zig`) - Tracks variable bindings and scopes
8. **Location Tracking** (`src/mlir/locations.zig`) - Preserves source location information
9. **Error Handling** (`src/mlir/error_handling.zig`) - Comprehensive error reporting
10. **Pass Manager** (`src/mlir/pass_manager.zig`) - MLIR optimization passes

## Type Mapping Strategy

The type mapping system provides comprehensive support for all Ora types:

### Primitive Types
- Integer types (`u8`-`u256`, `i8`-`i256`) → MLIR integer types
- `bool` → `i1`
- `address` → `i160` with `ora.address` attribute
- `string` → `!ora.string` dialect type
- `bytes` → `!ora.bytes` dialect type
- `void` → MLIR void type

### Complex Types
- Arrays `[T; N]` → `memref<NxT, space>` with region attributes
- Slices `slice[T]` → `!ora.slice<T>` dialect types
- Maps `map[K, V]` → `!ora.map<K, V>` dialect types
- Double maps `doublemap[K1, K2, V]` → `!ora.doublemap<K1, K2, V>` dialect types
- Structs → `!llvm.struct<...>` (migrating to `!ora.struct<fields>`)
- Enums → `!ora.enum<name, repr>` with underlying integer representation
- Error types `!T` → `!ora.error<T>` dialect types
- Error unions `!T1 | T2` → `!ora.error_union<T1, T2, ...>` dialect types

## Memory Region Semantics

The memory management system distinguishes between different memory regions:

- **Storage (space 1):** Persistent contract state with `ora.region = "storage"` attributes
- **Memory (space 0):** Transient execution memory with `ora.region = "memory"` attributes  
- **TStore (space 2):** Transient storage window with `ora.region = "tstore"` attributes

Memory operations are generated with appropriate space semantics:
- `memref.alloca` operations in correct memory spaces
- `memref.store` and `memref.load` with memory space constraints
- Region validation for memory access patterns

## Expression Lowering

The expression lowering system handles all Ora expression types:

### Binary Operations
- Arithmetic operators (`+`, `-`, `*`, `/`, `%`) → `arith` dialect operations
- Comparison operators (`==`, `!=`, `<`, `<=`, `>`, `>=`) → `arith.cmpi` operations
- Logical operators (`&&`, `||`) → Short-circuit evaluation with `scf.if`
- Bitwise operators (`&`, `|`, `^`, `<<`, `>>`) → `arith` dialect operations

### Unary Operations
- Logical not (`!`) → `arith.xori` with constant true
- Arithmetic negation (`-`) → `arith.subi` from zero
- Unary plus (`+`) → Identity operation

### Complex Expressions
- Field access → `llvm.extractvalue` or `llvm.getelementptr`
- Array/map indexing → `memref.load` with computed indices
- Function calls → `func.call` with proper argument passing
- Cast operations → Appropriate conversion operations
- Switch expressions → `scf.if` chains or `cf.switch` with result values
- Anonymous structs → Struct construction operations
- Array literals → Array initialization operations

### Verification Expressions
- Old expressions `old(expr)` → Operations with `ora.old = true` attributes
- Quantified expressions → Verification constructs with `ora.quantified` attributes
- Comptime expressions → Compile-time evaluation or `ora.comptime` operations

## Statement Lowering

The statement lowering system handles all control flow and Ora-specific statements:

### Control Flow
- If statements → `scf.if` operations with then/else regions
- While loops → `scf.while` operations with condition and body regions
- For loops → `scf.for` operations with iteration variables
- Switch statements → `cf.switch` operations with case blocks
- Labeled blocks → `scf.execute_region` operations

### Jump Statements
- Return statements → `func.return` with proper value handling
- Break/continue → Appropriate control flow transfers with label support

### Ora-Specific Statements
- Move statements → Atomic transfer operations with `ora.move` attributes
- Try-catch → Exception handling with error propagation
- Log statements → `ora.log` operations with indexed parameters
- Lock/unlock → `ora.lock` and `ora.unlock` operations
- Destructuring assignments → Field extraction operations

## Declaration Lowering

The declaration lowering system handles top-level constructs:

### Functions
- Function declarations with visibility modifiers → `ora.visibility` attributes
- Inline functions → `ora.inline = true` attributes
- Function contracts → `ora.requires` and `ora.ensures` attributes
- Init functions → `ora.init = true` attributes

### Types
- Struct declarations → Type definitions with field information
- Enum declarations → Enum type definitions with variant information
- Error declarations → Error type definitions

### Contracts
- Contract declarations → Module-level constructs with contract metadata
- Member functions and storage variables
- Contract inheritance support (if applicable)

### Globals
- Import declarations → Module import constructs
- Const declarations → Global constant definitions
- Immutable declarations → Immutable globals with initialization constraints

## Symbol Table Management

The symbol table system provides hierarchical scope management:

```zig
const SymbolEntry = struct {
    name: []const u8,
    value: MlirValue,
    type: MlirType,
    region: ?MemoryRegion,
    is_mutable: bool,
    scope_level: u32,
};
```

Features:
- Nested scope support with `pushScope`/`popScope`
- Variable binding with type and region information
- Function and type symbol tracking
- Parameter mapping for function calls

## Location Tracking

Source location information is preserved throughout the lowering process:

- File location information using `mlirLocationFileLineColGet`
- Byte offset and length preservation in location attributes
- All operations have proper source location metadata
- Original source text preservation where available

## Error Handling

The error handling system provides comprehensive error reporting:

### Error Categories
- **Type Errors:** Mismatched types during lowering
- **Symbol Errors:** Undefined variables or functions
- **Memory Errors:** Invalid memory region usage
- **Structural Errors:** Malformed AST nodes
- **MLIR Errors:** Invalid MLIR operations or types

### Error Recovery
- Graceful degradation with placeholder operations
- Error accumulation before failing
- Context preservation with source locations
- Actionable error messages with suggestions

## Pass Management

The pass manager integrates with MLIR's optimization infrastructure:

- Custom Ora-specific verification passes
- Standard MLIR optimization passes
- Pass pipeline configuration
- Pass result reporting and analysis

## API Reference

### Main Entry Points

```zig
// Main lowering function with error handling
pub fn lowerFunctionsToModuleWithErrors(
    ctx: c.MlirContext, 
    nodes: []lib.AstNode, 
    allocator: std.mem.Allocator
) !LoweringResult

// Legacy entry point for compatibility
pub fn lowerFunctionsToModule(
    ctx: c.MlirContext, 
    nodes: []lib.AstNode, 
    allocator: std.mem.Allocator
) c.MlirModule
```

### Type Mapping

```zig
// Convert Ora type to MLIR type
pub fn toMlirType(
    ctx: c.MlirContext, 
    ora_type: lib.ast.type_info.TypeInfo
) c.MlirType

// Create Ora dialect types
pub fn createOraDialectType(
    ctx: c.MlirContext, 
    type_name: []const u8, 
    params: []c.MlirType
) c.MlirType
```

### Expression Lowering

```zig
// Lower expression to MLIR value
pub fn lowerExpression(
    self: *const ExpressionLowerer, 
    expr: *const lib.ast.Expressions.ExprNode
) c.MlirValue

// Lower binary operation
pub fn lowerBinary(
    self: *const ExpressionLowerer, 
    binary: *const lib.ast.Expressions.BinaryNode
) c.MlirValue
```

### Statement Lowering

```zig
// Lower statement to MLIR operations
pub fn lowerStatement(
    self: *const StatementLowerer, 
    stmt: *const lib.ast.Statements.StmtNode
) void

// Lower control flow statements
pub fn lowerIf(
    self: *const StatementLowerer, 
    if_stmt: *const lib.ast.Statements.IfNode
) void
```

## MLIR Dependencies and Setup

### LLVM MLIR Integration

The Ora MLIR lowering system depends on the LLVM MLIR framework:

- **MLIR Version:** Compatible with LLVM 17.x and later
- **Required Components:** MLIR Core, Standard Dialects, Pass Infrastructure
- **C API:** Uses MLIR's C API for cross-language compatibility
- **Build Integration:** Links against MLIR libraries during compilation

### Installation Requirements

To build the Ora compiler with MLIR support:

1. **Install MLIR Development Libraries**
   ```bash
   # Ubuntu/Debian
   apt-get install mlir-dev libmlir-dev
   
   # macOS with Homebrew
   brew install llvm
   
   # From source
   git clone https://github.com/llvm/llvm-project.git
   cd llvm-project
   cmake -S llvm -B build -DLLVM_ENABLE_PROJECTS=mlir
   cmake --build build
   ```

2. **Configure Build System**
   ```zig
   // build.zig
   exe.linkSystemLibrary("MLIR");
   exe.linkSystemLibrary("MLIRSupport");
   exe.addIncludePath("/usr/include/mlir");
   ```

3. **Verify Installation**
   ```bash
   mlir-opt --version
   mlir-translate --version
   ```

## Integration with Compiler Pipeline

The MLIR lowering system integrates seamlessly with the existing compiler pipeline:

1. **Input:** Semantically analyzed AST nodes
2. **Processing:** Modular lowering to MLIR operations using LLVM MLIR APIs
3. **Output:** Validated MLIR module with source locations
4. **Integration:** Optional MLIR passes and analysis using MLIR pass infrastructure
5. **Backend:** Continues to existing Yul backend

## CLI Integration

The system provides several CLI flags for MLIR functionality:

- `--emit-mlir` - Generate MLIR output
- `--mlir-verify` - Validate MLIR IR correctness
- `--mlir-passes` - Run custom pass pipeline
- `--mlir-dump-after` - Debug pass effects

## Performance Characteristics

The MLIR lowering system is designed for:

- **Scalability:** Handles large smart contracts efficiently
- **Memory Usage:** Minimal memory overhead during lowering
- **Compilation Time:** Reasonable compilation times for development
- **Deterministic Output:** Consistent MLIR generation for testing

## Testing Infrastructure

Comprehensive testing ensures correctness and prevents regressions:

- **Unit Tests:** Individual component testing
- **Integration Tests:** End-to-end lowering validation
- **FileCheck Tests:** Pattern-based MLIR output validation
- **Regression Tests:** Prevent feature regressions
- **Performance Tests:** Compilation time and memory usage

## Troubleshooting

### Common Issues

1. **Type Mismatch Errors**
   - Check type compatibility between Ora and MLIR representations
   - Verify type mapping for complex types
   - Ensure proper type conversion operations

2. **Memory Region Violations**
   - Validate memory region usage patterns
   - Check storage/memory/tstore constraints
   - Verify region attribute attachment

3. **Symbol Resolution Failures**
   - Check symbol table scope management
   - Verify variable binding and lookup
   - Ensure proper parameter mapping

4. **MLIR Verification Failures**
   - Validate MLIR operation construction
   - Check type consistency in operations
   - Verify proper block and region structure

### Debug Strategies

1. **Enable Verbose Logging**
   - Use debug builds for detailed error information
   - Enable MLIR diagnostic output
   - Check pass manager results

2. **Incremental Testing**
   - Test individual components in isolation
   - Use minimal test cases for debugging
   - Validate intermediate representations

3. **Source Location Tracking**
   - Use preserved source locations for error context
   - Check span information accuracy
   - Verify location attachment to operations

## Future Enhancements

Planned improvements to the MLIR lowering system:

1. **Enhanced Dialect Support**
   - Complete Ora dialect implementation
   - Custom operation definitions
   - Dialect-specific optimizations

2. **Advanced Analysis**
   - Data flow analysis
   - Control flow analysis
   - Memory safety verification

3. **Optimization Passes**
   - Ora-specific optimizations
   - Smart contract gas optimization
   - Dead code elimination

4. **Alternative Backends**
   - LLVM backend integration
   - WebAssembly target support
   - Native code generation

## MLIR Resources and References

### Official MLIR Documentation
- **MLIR Website:** https://mlir.llvm.org
- **MLIR Language Reference:** https://mlir.llvm.org/docs/LangRef/
- **MLIR Dialects:** https://mlir.llvm.org/docs/Dialects/
- **MLIR Pass Infrastructure:** https://mlir.llvm.org/docs/PassManagement/
- **MLIR C API:** https://mlir.llvm.org/docs/CAPI/

### LLVM Project Resources
- **LLVM Project:** https://llvm.org
- **MLIR Source Code:** https://github.com/llvm/llvm-project/tree/main/mlir
- **MLIR Tutorials:** https://mlir.llvm.org/docs/Tutorials/
- **MLIR Community:** https://discourse.llvm.org/c/mlir/31

### Ora-Specific MLIR Implementation
- **Source Code:** `src/mlir/` directory in the Ora repository
- **Tests:** `tests/mlir_*_test.zig` files
- **Examples:** `ora-example/` directory with MLIR output examples

## Contributing

When contributing to the MLIR lowering system:

1. **Follow Architecture Patterns**
   - Maintain modular component separation
   - Use consistent error handling patterns
   - Preserve source location information
   - Follow MLIR best practices and conventions

2. **Add Comprehensive Tests**
   - Include unit tests for new components
   - Add integration tests for complex features
   - Create FileCheck patterns for MLIR output
   - Test against MLIR verification passes

3. **Document Changes**
   - Update API documentation
   - Add troubleshooting information
   - Include usage examples
   - Reference relevant MLIR documentation

4. **Maintain Compatibility**
   - Preserve existing API contracts
   - Ensure backward compatibility with MLIR versions
   - Test integration with existing pipeline
   - Validate against MLIR C API changes