# MLIR-to-Yul Lowering TODO

## ✅ Implemented Operations

### Core Infrastructure
- `ora.contract` - Contract declarations
- `ora.global` - Storage variable declarations
- `ora.sload` - Load from storage
- `ora.sstore` - Store to storage
- `arith.constant` - Integer constants
- `arith.addi` - Integer addition
- `arith.cmpi` - Integer comparison
- `scf.if` - Conditional execution
- `func.func` - Function definitions
- `func.return` - Function returns

### Control Flow ✅
- `ora.if` - Conditional execution
- `ora.while` - While loops
- `ora.for` - For loops
- `ora.switch` - Switch statements
- `ora.break` - Break statements
- `ora.continue` - Continue statements
- `ora.return` - Return statements

### Arithmetic & Math ✅
- `ora.power` - Exponentiation
- `ora.cast` - Type conversions
- `arith.subi` - Integer subtraction
- `arith.muli` - Integer multiplication
- `arith.divi` - Integer division
- `arith.remi` - Integer remainder
- `arith.andi` - Bitwise AND
- `arith.ori` - Bitwise OR
- `arith.xori` - Bitwise XOR
- `arith.shli` - Left shift
- `arith.shrsi` - Right shift (signed)
- `arith.shrui` - Right shift (unsigned)

### Memory Operations ✅
- `ora.mload` - Load from memory
- `ora.mstore` - Store to memory
- `ora.tload` - Load from transient storage
- `ora.tstore` - Store to transient storage

### Data Structures ✅
- `ora.struct.decl` - Struct declarations
- `ora.struct.field.store` - Struct field assignment
- `ora.struct.instantiate` - Struct instantiation
- `ora.struct.init` - Struct initialization
- `ora.enum.decl` - Enum declarations
- `ora.enum.constant` - Enum constants
- `ora.map.get` - Map lookup
- `ora.map.store` - Map assignment

### Constants & Literals ✅
- `ora.string.constant` - String constants
- `ora.hex.constant` - Hexadecimal constants
- `ora.binary.constant` - Binary constants
- `ora.address.constant` - Address constants

### Variable Declarations ✅
- `ora.const` - Constant variables
- `ora.immutable` - Immutable variables

### Formal Verification ✅
- `ora.requires` - Preconditions
- `ora.ensures` - Postconditions
- `ora.invariant` - Invariants
- `ora.old` - Old value references

### Events & Logging ✅
- `ora.log` - Event logging

### Financial Operations ✅
- `ora.move` - Financial transfers

### Error Handling ✅
- `ora.try_catch` - Try-catch blocks
- `ora.yield` - Yield operation for control flow regions
- `ora.error.decl` - Error declarations
- `ora.error` - Error type operations
- `ora.error_union` - Error union types

### Advanced Features ✅
- `ora.destructure` - Pattern matching
- `ora.lock` - Resource locking
- `ora.unlock` - Resource unlocking

## ❌ Missing Operations (Critical)


### Switch & Pattern Matching
- `ora.switch` - Switch statements (basic implementation exists)
- Switch case pattern matching (range patterns, enum variants)
- Switch expression handling
- Pattern destructuring in switch cases

### Union & Error Types
- `!ora.error<T>` - Error type system
- `!ora.error_union<T1, T2, ...>` - Error union types
- Error type conversions and handling
- Union type operations

### Advanced Control Flow
- `ora.yield` - Yield operations for regions
- Proper switch case handling with pattern matching
- Range pattern matching in switch cases
- Enum variant pattern matching

### Type System Operations
- Error type mapping and conversion
- Union type operations
- Type casting for error types
- Error propagation handling

## ✅ Implemented Operations

### Core Infrastructure
- `ora.contract` - Contract declarations
- `ora.global` - Storage variable declarations
- `ora.sload`/`ora.sstore` - Storage operations
- `func.func`/`func.return` - Function definitions
- `arith.constant`/`arith.addi`/`arith.cmpi` - Basic arithmetic

### Control Flow (Basic)
- `ora.if` - Conditional execution
- `ora.while` - While loops
- `ora.for` - For loops
- `ora.switch` - Switch statements (basic)
- `ora.break`/`ora.continue` - Loop control
- `ora.return` - Return statements

### Arithmetic & Math
- `ora.power` - Exponentiation
- `ora.cast` - Type conversions
- `arith.subi`/`arith.muli`/`arith.divi`/`arith.remi` - Basic arithmetic
- `arith.andi`/`arith.ori`/`arith.xori` - Bitwise operations
- `arith.shli`/`arith.shrsi`/`arith.shrui` - Shift operations

### Memory Operations
- `ora.mload`/`ora.mstore` - Memory operations
- `ora.tload`/`ora.tstore` - Transient storage

### Data Structures
- `ora.struct.*` - Struct operations
- `ora.enum.*` - Enum operations
- `ora.map.*` - Map operations

### Constants & Literals
- `ora.string.constant` - String constants
- `ora.hex.constant` - Hexadecimal constants
- `ora.binary.constant` - Binary constants
- `ora.address.constant` - Address constants

### Variable Declarations
- `ora.const` - Constant variables
- `ora.immutable` - Immutable variables

### Formal Verification
- `ora.requires` - Preconditions
- `ora.ensures` - Postconditions
- `ora.invariant` - Invariants
- `ora.old` - Old value references

### Events & Logging
- `ora.log` - Event logging

### Financial Operations
- `ora.move` - Financial transfers

### Advanced Features
- `ora.destructure` - Pattern matching
- `ora.lock`/`ora.unlock` - Resource locking

## Priority Order (Updated)

### **CRITICAL MISSING** (Must implement first)
1. **Error Handling & Management** - `ora.yield`, `ora.error.*`, `ora.try_catch`
2. **Switch & Pattern Matching** - Complete switch implementation with pattern matching
3. **Union & Error Types** - `!ora.error<T>`, `!ora.error_union<T1, T2, ...>`
4. **Advanced Control Flow** - Proper yield operations and region handling

### **COMPLETED** ✅
5. **Core Infrastructure** - Contracts, storage, functions
6. **Basic Control Flow** - if, while, for, basic switch, break, continue, return
7. **Arithmetic & Math** - All arithmetic operations, bitwise ops, shifts
8. **Memory Operations** - Memory and transient storage
9. **Data Structures** - Structs, enums, maps
10. **Constants & Literals** - All constant types
11. **Variable Declarations** - const, immutable
12. **Formal Verification** - requires, ensures, invariants, old
13. **Events & Logging** - Event logging
14. **Financial Operations** - Transfers
15. **Advanced Features** - Destructuring, locking

### **IMPLEMENTATION STATUS**
- ✅ **Completed**: 15/19 categories (79%)
- ❌ **Missing**: 4/19 categories (21%) - **CRITICAL**
