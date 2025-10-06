# MLIR Lowering Completion TODO List

This document tracks missing features, incomplete implementations, and necessary improvements for complete MLIR lowering support in the Ora compiler.

## 🔴 Critical Missing Features

### 1. Ora Dialect Definition & Registration
**Priority: Critical**
- [ ] **Custom Ora MLIR Dialect**: Create proper `ora` dialect with TableGen definitions
- [ ] **Dialect Registration**: Register dialect with MLIR context during initialization
- [ ] **Operation Definitions**: Define all Ora-specific operations with proper syntax
- [ ] **Type Definitions**: Define custom Ora types in MLIR dialect
- [ ] **Attribute Definitions**: Define custom attributes for Ora semantics

**Current Status**: Using placeholder `ora.*` operations that aren't properly registered
**Files**: `src/mlir/dialect.zig`, need new `OraDialect.td` file

### 2. Complex Type System Support
**Priority: Critical**
- [ ] **Slice Types**: `slice[T]` → `!ora.slice<T>` or `memref<?xT, space>`
- [ ] **Map Types**: `map[K, V]` → `!ora.map<K, V>` 
- [ ] **Double Map Types**: `doublemap[K1, K2, V]` → `!ora.doublemap<K1, K2, V>`
- [ ] **Tuple Types**: `(T1, T2, ...)` → proper MLIR tuple or struct types
- [ ] **Function Types**: Function signatures with proper MLIR function types
- [ ] **Error Union Types**: `!T | E1 | E2` → `!ora.error_union<T, E1, E2>`
- [ ] **Union Types**: `T1 | T2 | ...` → `!ora.union<T1, T2, ...>`
- [ ] **Enum Types**: Named enums → `!ora.enum<name, repr>`
- [ ] **Struct Types**: Named structs → `!ora.struct<name>`
- [ ] **Anonymous Struct Types**: `.{ field: T, ... }` → anonymous struct types
- [ ] **Array Types with Memory Spaces**: `[T; N]` → `memref<NxT, space>` with proper memory space

**Current Status**: Most complex types fall back to `i256` placeholders
**Files**: `src/mlir/types.zig`

### 3. Memory Region Support
**Priority: Critical**
- [ ] **Memory Space Attributes**: Proper MLIR memory space encoding (0=memory, 1=storage, 2=tstore, 3=stack)
- [ ] **Region Transition Verification**: MLIR passes to verify valid region transitions
- [ ] **Storage Operations**: Complete `ora.sload`/`ora.sstore` with proper typing
- [ ] **Memory Operations**: `ora.mload`/`ora.mstore` for memory region
- [ ] **Transient Storage**: `ora.tload`/`ora.tstore` for transient storage
- [ ] **Stack Operations**: Local variable allocation and access
- [ ] **Region-aware Type System**: Types carry memory region information

**Current Status**: Basic storage operations only, no proper memory space handling
**Files**: `src/mlir/memory.zig`, `src/mlir/types.zig`

## 🟠 Major Missing Language Features

### 4. Control Flow & Statements
**Priority: High**
- [ ] **For Loops**: `for` loops with proper iteration semantics
- [ ] **While Loops**: `while` loops with condition evaluation
- [ ] **Switch Statements**: Pattern matching with proper lowering
- [ ] **Switch Expressions**: Switch as expression with value yielding
- [ ] **Break/Continue**: Loop control flow with labeled breaks
- [ ] **Try/Catch Blocks**: Error handling constructs
- [ ] **Move Semantics**: `move` statements for explicit ownership transfer
- [ ] **Destructuring Assignment**: Pattern matching in assignments
- [ ] **Labeled Blocks**: Block expressions with labels and break values

**Current Status**: Only basic if statements and function calls supported
**Files**: `src/mlir/statements.zig`

### 5. Expression Types
**Priority: High**
- [ ] **Try Expressions**: `try expr` for error propagation
- [ ] **Error Return**: `error.SomeError` expressions
- [ ] **Error Cast**: `value as !T` for error union casting
- [ ] **Shift Expressions**: `mapping from source -> dest : amount`
- [ ] **Struct Instantiation**: `StructName { field1: value1, ... }`
- [ ] **Anonymous Struct Literals**: `.{ field1: value1, ... }`
- [ ] **Range Expressions**: `1...1000` or `0..periods`
- [ ] **Labeled Block Expressions**: `label: { ... break :label value; }`
- [ ] **Destructuring Expressions**: Pattern matching in expressions
- [ ] **Quantified Expressions**: `forall`/`exists` for formal verification
- [ ] **Tuple Expressions**: `(a, b, c)` tuple creation
- [ ] **Switch Expressions**: `switch (x) { case y: ... }`

**Current Status**: Only basic arithmetic, comparison, and field access
**Files**: `src/mlir/expressions.zig`

### 6. Declaration Types
**Priority: High**
- [ ] **Enum Declarations**: Named enums with variants
- [ ] **Struct Declarations**: Named structs with fields
- [ ] **Error Declarations**: Custom error types
- [ ] **Import Declarations**: Module import system
- [ ] **Type Aliases**: `type NewName = OldType`
- [ ] **Const Declarations**: Compile-time constants
- [ ] **Immutable Declarations**: Runtime immutable values
- [ ] **Generic Declarations**: Generic functions and types

**Current Status**: Only basic variable declarations and functions
**Files**: `src/mlir/declarations.zig`

## 🟡 Verification & Analysis Features

### 7. Formal Verification Support
**Priority: Medium**
- [ ] **Requires Clauses**: Function preconditions
- [ ] **Ensures Clauses**: Function postconditions  
- [ ] **Invariant Clauses**: Loop and contract invariants
- [ ] **Old Expressions**: `old(x)` in postconditions
- [ ] **Quantified Expressions**: Universal/existential quantification
- [ ] **Verification Attributes**: Contract verification metadata
- [ ] **Assertion Statements**: Runtime and compile-time assertions

**Current Status**: Placeholder support only
**Files**: Need new `src/mlir/verification.zig`

### 8. Error Handling System
**Priority: Medium**
- [ ] **Error Union Types**: Complete error union support
- [ ] **Error Propagation**: Automatic error bubbling
- [ ] **Try/Catch Semantics**: Proper exception handling
- [ ] **Error Recovery**: Graceful error handling in MLIR
- [ ] **Error Type Checking**: Static error analysis

**Current Status**: Basic error reporting only
**Files**: `src/mlir/error_handling.zig`

## 🟢 Optimization & Advanced Features

### 9. MLIR Pass Infrastructure
**Priority: Medium**
- [ ] **Custom Ora Passes**: Ora-specific optimization passes
- [ ] **Verification Passes**: Contract and invariant verification
- [ ] **Region Analysis Passes**: Memory region validation
- [ ] **Type Inference Passes**: Advanced type inference
- [ ] **Dead Code Elimination**: Ora-specific DCE
- [ ] **Constant Folding**: Ora constant evaluation
- [ ] **Function Inlining**: Smart inlining for Ora functions

**Current Status**: Basic pass manager with standard MLIR passes only
**Files**: `src/mlir/pass_manager.zig`

### 10. Advanced Type Features
**Priority: Low**
- [ ] **Generic Type Instantiation**: Generic type specialization
- [ ] **Type Inference**: Advanced type inference algorithms
- [ ] **Trait System**: If/when added to Ora language
- [ ] **Associated Types**: Advanced type relationships
- [ ] **Higher-Kinded Types**: Advanced type system features

**Current Status**: Basic type mapping only
**Files**: `src/mlir/types.zig`

## 🔧 Infrastructure & Tooling

### 11. MLIR C API Extensions
**Priority: Medium**
- [ ] **Missing C API Functions**: Implement missing MLIR C bindings
- [ ] **Location API**: Complete source location tracking
- [ ] **Diagnostic API**: Better error reporting integration
- [ ] **Pass API**: Custom pass registration and execution
- [ ] **Dialect API**: Custom dialect registration

**Current Status**: Using basic C API subset only
**Files**: `src/mlir/c.zig`, `src/mlir/locations.zig`

### 12. Testing & Validation
**Priority: Medium**
- [ ] **MLIR Unit Tests**: Comprehensive MLIR lowering tests
- [ ] **Round-trip Tests**: Ora → MLIR → Ora consistency
- [ ] **Optimization Tests**: Pass pipeline validation
- [ ] **Error Recovery Tests**: Graceful error handling tests
- [ ] **Performance Benchmarks**: MLIR compilation performance

**Current Status**: Basic validation script only
**Files**: Need new test files in `tests/mlir/`

### 13. Documentation & Examples
**Priority: Low**
- [ ] **MLIR Lowering Guide**: Complete technical documentation
- [ ] **Dialect Reference**: Ora MLIR dialect documentation
- [ ] **Pass Reference**: Custom pass documentation
- [ ] **Example Gallery**: Comprehensive Ora → MLIR examples
- [ ] **Performance Guide**: Optimization best practices

**Current Status**: Basic documentation only
**Files**: `docs/mlir-lowering.md`

## 📊 Implementation Priority Matrix

### Phase 1: Foundation (Critical)
1. Ora Dialect Definition & Registration
2. Complex Type System Support  
3. Memory Region Support

### Phase 2: Language Features (Major)
4. Control Flow & Statements
5. Expression Types
6. Declaration Types

### Phase 3: Advanced Features (Important)
7. Formal Verification Support
8. Error Handling System
9. MLIR Pass Infrastructure

### Phase 4: Polish (Nice-to-have)
10. Advanced Type Features
11. MLIR C API Extensions
12. Testing & Validation
13. Documentation & Examples

## 🎯 Success Criteria

### Milestone 1: Basic Language Support
- [ ] All basic Ora constructs compile to valid MLIR
- [ ] Memory regions properly represented
- [ ] Type system correctly mapped

### Milestone 2: Complete Language Support  
- [ ] All Ora language features supported
- [ ] Advanced control flow working
- [ ] Error handling functional

### Milestone 3: Production Ready
- [ ] Optimization passes working
- [ ] Verification features complete
- [ ] Performance competitive with direct compilation

## 📝 Notes

### Current MLIR Output Quality
- ✅ Basic arithmetic and comparison operations
- ✅ Function definitions and calls
- ✅ Simple control flow (if statements)
- ✅ Basic storage operations
- ❌ Complex types (maps, slices, error unions)
- ❌ Advanced control flow (loops, switch)
- ❌ Error handling constructs
- ❌ Verification features

### Technical Debt
- Many type mappings use `i256` placeholders
- No proper Ora dialect registration
- Limited error recovery
- Missing optimization passes
- Incomplete source location tracking

### Dependencies
- MLIR C API limitations may require upstream contributions
- TableGen dialect definitions needed
- Custom pass development required
- Advanced type system design needed

---

**Last Updated**: December 2024
**Total Tasks**: 100+ individual implementation tasks
**Estimated Effort**: 6-12 months for complete implementation
