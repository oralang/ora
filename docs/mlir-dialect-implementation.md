# MLIR Dialect Implementation Guide

## Overview

This document explains the implementation of the Ora MLIR dialect, the role of TableGen files (`.td`), generated C++ code, and the overall architecture. This serves as a technical guide for understanding and continuing the dialect implementation.

## Table of Contents

1. [What is an MLIR Dialect?](#what-is-an-mlir-dialect)
2. [TableGen (.td files) Explained](#tablegen-td-files-explained)
3. [Architecture Overview](#architecture-overview)
4. [File Structure and Purpose](#file-structure-and-purpose)
5. [Code Generation Process](#code-generation-process)
6. [Current Implementation Status](#current-implementation-status)
7. [Next Steps](#next-steps)

## What is an MLIR Dialect?

### Basic Concept

An MLIR **dialect** is a collection of operations, types, and attributes that extend MLIR's core functionality. Think of it as a "vocabulary" that MLIR understands for a specific domain.

```mlir
// Without dialect (unregistered operations)
"ora.sload"() {global = "counter"} : () -> i256  // MLIR treats this as text

// With dialect (registered operations)  
%0 = ora.sload "counter" : i256  // MLIR understands this semantically
```

### Why Dialects Matter

| Unregistered Operations | Registered Dialect Operations |
|------------------------|-------------------------------|
| ❌ Text-based parsing only | ✅ Semantic understanding |
| ❌ No type checking | ✅ Full type validation |
| ❌ No optimization | ✅ Optimization opportunities |
| ❌ Limited error messages | ✅ Rich error diagnostics |
| ❌ No custom passes | ✅ Domain-specific passes |

### Ora Dialect Operations

Our minimal dialect currently defines 4 core operations:

```mlir
// Contract declaration
ora.contract @SimpleContract {
  ora.global "counter" : i256 = 0 : i256
  // functions...
}

// Storage variable declaration  
ora.global "counter" : i256 = 0 : i256

// Load from storage
%value = ora.sload "counter" : i256

// Store to storage
ora.sstore %value, "counter" : i256
```

## TableGen (.td files) Explained

### What is TableGen?

**TableGen** is LLVM/MLIR's domain-specific language for generating C++ code. Instead of writing repetitive C++ boilerplate, you write declarative `.td` files that describe your dialect.

### Why TableGen?

```cpp
// Without TableGen: Manual C++ (hundreds of lines per operation)
class OraGlobalOp : public Op<OraGlobalOp, OpTrait::Symbol> {
  static StringRef getOperationName() { return "ora.global"; }
  static ParseResult parse(OpAsmParser &parser, OperationState &result) {
    // 50+ lines of parsing logic
  }
  void print(OpAsmPrinter &p) {
    // 20+ lines of printing logic  
  }
  LogicalResult verify() {
    // 30+ lines of verification logic
  }
  // ... many more methods
};
```

```tablegen
// With TableGen: Declarative definition (10 lines)
def Ora_GlobalOp : Ora_Op<"global", [Symbol]> {
  let summary = "Global storage variable declaration";
  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttr:$type, AnyAttr:$init);
  // TableGen generates all the C++ boilerplate automatically
}
```

### Our TableGen Files

#### 1. `OraDialect.td` - Dialect Definition

```tablegen
def Ora_Dialect : Dialect {
  let name = "ora";
  let summary = "Ora smart contract language dialect";
  let cppNamespace = "::mlir::ora";
}
```

**Purpose**: Defines the dialect itself, its namespace, and basic properties.

#### 2. `OraOps.td` - Operation Definitions

```tablegen
def Ora_SLoadOp : Ora_Op<"sload", [Pure]> {
  let summary = "Load value from storage";
  let arguments = (ins StrAttr:$global);
  let results = (outs AnyType:$result);
}
```

**Purpose**: Defines each operation's signature, traits, and behavior.

#### 3. `OraTypes.td` - Type Definitions

```tablegen
// Currently minimal - placeholder for future Ora-specific types
class Ora_Type<string name, string typeMnemonic> : TypeDef<Ora_Dialect, name> {
  let mnemonic = typeMnemonic;
}
```

**Purpose**: Defines custom types (currently minimal, uses standard MLIR types).

#### 4. `OraAttributes.td` - Attribute Definitions

```tablegen
// Currently minimal - placeholder for future Ora-specific attributes
class Ora_Attr<string name, string attrMnemonic> : AttrDef<Ora_Dialect, name> {
  let mnemonic = attrMnemonic;
}
```

**Purpose**: Defines custom attributes (currently minimal, uses standard MLIR attributes).

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Ora Compiler                            │
├─────────────────────────────────────────────────────────────────┤
│  Ora Source Code (.ora)                                        │
│           ↓                                                     │
│  Parser → AST                                                   │
│           ↓                                                     │
│  MLIR Lowering (Zig code)                                      │
│           ↓                                                     │
│  MLIR IR with ora.* operations                                 │
├─────────────────────────────────────────────────────────────────┤
│                      MLIR System                               │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐                   │
│  │ TableGen Files  │───▶│ Generated C++   │                   │
│  │ (.td)          │    │ Code            │                   │
│  │                │    │                │                   │
│  │ OraDialect.td   │    │ OraDialect.h    │                   │
│  │ OraOps.td       │    │ OraOps.h.inc    │                   │
│  │ OraTypes.td     │    │ OraDialect.cpp  │                   │
│  │ OraAttributes.td│    │ OraOps.cpp.inc  │                   │
│  └─────────────────┘    └─────────────────┘                   │
│           ↓                       ↓                            │
│  ┌─────────────────────────────────────────────────────────────┤
│  │                Ora MLIR Dialect                            │
│  │  • Semantic understanding of ora.* operations             │
│  │  • Type checking and validation                           │
│  │  • Optimization opportunities                             │
│  │  • Custom passes                                          │
│  └─────────────────────────────────────────────────────────────┤
│           ↓                                                     │
│  Standard MLIR Passes & Optimizations                          │
│           ↓                                                     │
│  Target Code Generation                                         │
└─────────────────────────────────────────────────────────────────┘
```

## File Structure and Purpose

```
src/mlir/
├── OraDialect.td           # Dialect definition (input)
├── OraOps.td              # Operations definition (input)  
├── OraTypes.td            # Types definition (input)
├── OraAttributes.td       # Attributes definition (input)
├── generated/             # Generated code directory
│   ├── OraDialect.h.inc   # Generated dialect declarations
│   ├── OraDialect.cpp.inc # Generated dialect definitions
│   ├── OraOps.h.inc       # Generated operation declarations
│   ├── OraOps.cpp.inc     # Generated operation definitions
│   ├── OraDialect.h       # Hand-written dialect header
│   ├── OraDialect.cpp     # Hand-written dialect implementation
│   └── CMakeLists.txt     # Build configuration (future use)
├── dialect.zig           # Current Zig dialect wrapper
├── context.zig           # MLIR context management
└── [other existing files...]
```

### Why the `generated/` Directory?

1. **Separation of Concerns**: 
   - `src/mlir/*.td` = Source definitions (version controlled)
   - `src/mlir/generated/*` = Generated code (can be regenerated)

2. **Build Process Clarity**:
   - Input: `.td` files
   - Process: `mlir-tblgen` compilation  
   - Output: `.inc` and `.cpp` files

3. **Integration Strategy**:
   - Current: Zig-based dialect (`dialect.zig`)
   - Future: C++-based dialect (`generated/OraDialect.cpp`)
   - Transition: Both can coexist during migration

## Code Generation Process

### Step-by-Step Process

```bash
# 1. TableGen Compilation (what we did)
mlir-tblgen -gen-dialect-decls -I vendor/mlir/include -dialect=ora \
    src/mlir/OraDialect.td -o src/mlir/generated/OraDialect.h.inc

mlir-tblgen -gen-dialect-defs -I vendor/mlir/include -dialect=ora \
    src/mlir/OraDialect.td -o src/mlir/generated/OraDialect.cpp.inc

mlir-tblgen -gen-op-decls -I vendor/mlir/include -I src/mlir \
    src/mlir/OraOps.td -o src/mlir/generated/OraOps.h.inc

mlir-tblgen -gen-op-defs -I vendor/mlir/include -I src/mlir \
    src/mlir/OraOps.td -o src/mlir/generated/OraOps.cpp.inc
```

### What Gets Generated

#### From `OraDialect.td` → `OraDialect.h.inc`:
```cpp
namespace mlir {
namespace ora {
class OraDialect : public ::mlir::Dialect {
  explicit OraDialect(::mlir::MLIRContext *context);
  void initialize();
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("ora");
  }
  // ... more methods
};
} // namespace ora  
} // namespace mlir
```

#### From `OraOps.td` → `OraOps.h.inc`:
```cpp
namespace mlir {
namespace ora {
class GlobalOp : public ::mlir::Op<GlobalOp, /* traits */> {
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("ora.global");
  }
  // Generated methods for parsing, printing, verification, etc.
};
} // namespace ora
} // namespace mlir
```

### Hand-Written Integration Code

#### `OraDialect.h` (we wrote this):
```cpp
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

// Include generated declarations
#include "OraDialect.h.inc"
#define GET_OP_CLASSES  
#include "OraOps.h.inc"
```

#### `OraDialect.cpp` (we wrote this):
```cpp
// Include generated implementations
#include "OraDialect.cpp.inc"
#define GET_OP_CLASSES
#include "OraOps.cpp.inc"

void OraDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "OraOps.cpp.inc"
  >();
}
```

## Current Implementation Status

### ✅ Completed

1. **TableGen Definitions**: All 4 core operations defined
2. **Code Generation**: Successfully generated C++ code
3. **Integration Framework**: Hand-written glue code ready
4. **Validation**: System still works with `--allow-unregistered-dialect`

### 🔄 Current State

- **Zig Dialect** (`dialect.zig`): Active, creates unregistered operations
- **C++ Dialect** (`generated/`): Ready, but not integrated into build
- **MLIR Validation**: Uses `--allow-unregistered-dialect` flag

### ❌ Pending

1. **Build Integration**: Link C++ dialect into MLIR build system
2. **Dialect Registration**: Replace Zig wrapper with C++ dialect
3. **Remove Unregistered Flag**: Enable full MLIR semantic checking
4. **Advanced Features**: Custom passes, optimizations, etc.

## Next Steps

### Phase 1: Build System Integration
```bash
# Add Ora dialect to MLIR build
# Compile C++ dialect as library
# Link with existing Zig code
```

### Phase 2: Registration  
```cpp
// Replace current Zig registration
mlirContextRegisterDialect(context, oraDialect);
```

### Phase 3: Remove Unregistered Flag
```bash
# Change from:
mlir-opt --allow-unregistered-dialect

# To:
mlir-opt  # Ora operations now fully registered
```

### Phase 4: Advanced Features
- Custom optimization passes
- Type inference improvements  
- Domain-specific verifications
- Integration with Solidity backend

## Key Concepts Summary

### TableGen Benefits
- **Declarative**: Describe what you want, not how to implement it
- **Code Generation**: Eliminates boilerplate C++ code  
- **Consistency**: Ensures uniform operation behavior
- **Maintainability**: Changes to `.td` files regenerate all code

### Why C++ Code?
- **MLIR Core**: MLIR's core is C++, dialects must integrate at C++ level
- **Performance**: C++ provides optimal performance for compiler operations
- **Ecosystem**: Access to full MLIR optimization and pass infrastructure  
- **Interoperability**: Seamless integration with existing MLIR dialects

### Why Generated Directory?
- **Build Clarity**: Clear separation between source and generated code
- **Version Control**: Only `.td` files need versioning, `.inc` files are artifacts
- **Development Flow**: Modify `.td` → regenerate → test → commit `.td` only

This architecture provides a solid foundation for evolving the Ora compiler from a basic MLIR generator to a full-featured compiler with semantic understanding, optimization, and verification capabilities.
