# MLIR Intermediate Representation

## Overview

The Ora compiler includes a comprehensive MLIR (Multi-Level Intermediate Representation) lowering system that provides an intermediate representation for advanced analysis, optimization, and alternative code generation paths. This system complements the primary Yul backend and enables sophisticated compiler transformations.

## What is MLIR?

MLIR (Multi-Level Intermediate Representation) is a compiler infrastructure developed by the LLVM community (https://mlir.llvm.org) that provides:
- **Multi-level representation:** Support for different abstraction levels
- **Extensible dialects:** Custom operation and type definitions
- **Optimization passes:** Pluggable transformation framework
- **Analysis infrastructure:** Data flow and control flow analysis

The Ora compiler integrates with the existing MLIR framework to provide Ora-specific lowering and analysis capabilities.

## Ora MLIR Integration

The Ora compiler integrates with the LLVM MLIR framework (https://mlir.llvm.org) to:
1. **Represent Ora semantics** in a structured intermediate form using MLIR operations and types
2. **Enable advanced analysis** for verification and optimization using MLIR's pass infrastructure
3. **Support alternative backends** beyond Yul by leveraging MLIR's code generation capabilities
4. **Provide debugging information** with source location preservation using MLIR's location tracking

Our implementation provides Ora-specific dialects, operations, and lowering passes that work within the existing MLIR ecosystem.

## Type System Mapping

### Primitive Types

Ora primitive types are mapped to MLIR as follows:

| Ora Type | MLIR Type | Notes |
|----------|-----------|-------|
| `u8` - `u256` | `iN` | N-bit unsigned integers |
| `i8` - `i256` | `iN` | N-bit signed integers |
| `bool` | `i1` | Single bit boolean |
| `address` | `i160` | With `ora.address` attribute |
| `string` | `!ora.string` | Ora dialect type |
| `bytes` | `!ora.bytes` | Ora dialect type |
| `void` | `()` | MLIR void type |

### Complex Types

| Ora Type | MLIR Type | Description |
|----------|-----------|-------------|
| `[T; N]` | `memref<NxT, space>` | Fixed-size arrays with memory space |
| `slice[T]` | `!ora.slice<T>` | Dynamic slices |
| `map[K, V]` | `!ora.map<K, V>` | Key-value mappings |
| `doublemap[K1, K2, V]` | `!ora.doublemap<K1, K2, V>` | Nested mappings |
| `struct { ... }` | `!llvm.struct<...>` | Struct types |
| `enum` | `!ora.enum<name, repr>` | Enumeration types |
| `!T` | `!ora.error<T>` | Error types |
| `!T1 \| T2` | `!ora.error_union<T1, T2>` | Error unions |

## Memory Region Semantics

Ora's memory regions are represented in MLIR using memory spaces:

### Storage (Space 1)
```mlir
%storage_var = memref.alloca() {ora.region = "storage"} : memref<1xi256, 1>
```
- Persistent contract state
- Transactional semantics
- Gas costs for access

### Memory (Space 0)
```mlir
%memory_var = memref.alloca() : memref<1xi256, 0>
```
- Transient execution memory
- Cleared between calls
- Lower gas costs

### TStore (Space 2)
```mlir
%tstore_var = memref.alloca() {ora.region = "tstore"} : memref<1xi256, 2>
```
- Transient storage window
- Temporary persistence
- Intermediate gas costs

## Expression Lowering

### Arithmetic Operations

Ora arithmetic expressions are lowered to MLIR `arith` dialect operations:

```ora
let result = a + b * c;
```

```mlir
%mul = arith.muli %b, %c : i256
%add = arith.addi %a, %mul : i256
```

### Comparison Operations

```ora
let is_greater = x > y;
```

```mlir
%cmp = arith.cmpi sgt, %x, %y : i256
```

### Logical Operations

Logical operators use short-circuit evaluation:

```ora
let result = condition1 && condition2;
```

```mlir
%result = scf.if %condition1 -> i1 {
  scf.yield %condition2 : i1
} else {
  %false = arith.constant false
  scf.yield %false : i1
}
```

### Field Access

Struct field access uses LLVM operations:

```ora
let value = my_struct.field;
```

```mlir
%value = llvm.extractvalue %my_struct[0] : !llvm.struct<(i256, i256)>
```

### Function Calls

```ora
let result = my_function(arg1, arg2);
```

```mlir
%result = func.call @my_function(%arg1, %arg2) : (i256, i256) -> i256
```

## Statement Lowering

### Control Flow

#### If Statements

```ora
if (condition) {
    // then block
} else {
    // else block
}
```

```mlir
scf.if %condition {
    // then region
} else {
    // else region
}
```

#### While Loops

```ora
while (condition) {
    // body
}
```

```mlir
scf.while (%arg0 = %initial) : (i256) -> () {
    %cond = // evaluate condition
    scf.condition(%cond)
} do {
    // loop body
    scf.yield %next_value : i256
}
```

#### For Loops

```ora
for (array) |item| {
    // body
}
```

```mlir
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%len = // array length
scf.for %i = %c0 to %len step %c1 {
    %item = memref.load %array[%i] : memref<?xi256>
    // loop body
}
```

### Switch Statements

```ora
switch (value) {
    case 1 => // handle 1
    case 2...5 => // handle range
    else => // default case
}
```

```mlir
cf.switch %value : i256, [
    default: ^default,
    1: ^case1,
    2: ^case2_5,
    3: ^case2_5,
    4: ^case2_5,
    5: ^case2_5
]
```

## Ora-Specific Features

### Move Statements

```ora
move 100 from sender to receiver;
```

```mlir
%amount = arith.constant 100 : i256
%move_op = ora.move %amount from %sender to %receiver {ora.move = true}
```

### Log Statements

```ora
log Transfer(from: sender, to: receiver, amount: value);
```

```mlir
ora.log "Transfer"(%sender, %receiver, %value) {
    indexed = [true, true, false]
} : (i160, i160, i256) -> ()
```

### Try-Catch Blocks

```ora
try {
    risky_operation();
} catch (error) {
    handle_error(error);
}
```

```mlir
%result = scf.execute_region -> !ora.error<()> {
    %success = func.call @risky_operation() : () -> !ora.error<()>
    scf.yield %success : !ora.error<()>
}
%is_error = ora.is_error %result : !ora.error<()>
scf.if %is_error {
    %error = ora.unwrap_error %result : !ora.error<()>
    func.call @handle_error(%error) : (()) -> ()
}
```

## Verification Features

### Old Expressions

```ora
ensures balance == old(balance) + amount;
```

```mlir
%old_balance = // captured at function entry
%postcond = arith.cmpi eq, %balance, %expected : i256
ora.assert %postcond {ora.ensures = true, ora.old = %old_balance}
```

### Quantified Expressions

```ora
requires forall(i in 0...array.length) array[i] > 0;
```

```mlir
%all_positive = ora.forall %i in %range {
    %elem = memref.load %array[%i] : memref<?xi256>
    %zero = arith.constant 0 : i256
    %positive = arith.cmpi sgt, %elem, %zero : i256
    ora.yield %positive : i1
} : (index) -> i1
ora.assert %all_positive {ora.requires = true}
```

## Function Contracts

### Requires Clauses

```ora
fn transfer(amount: u256) 
    requires(amount > 0)
    requires(balance >= amount)
{
    // function body
}
```

```mlir
func.func @transfer(%amount: i256) {
    %zero = arith.constant 0 : i256
    %amount_positive = arith.cmpi sgt, %amount, %zero : i256
    ora.assert %amount_positive {ora.requires = true}
    
    %balance = // load balance
    %sufficient = arith.cmpi uge, %balance, %amount : i256
    ora.assert %sufficient {ora.requires = true}
    
    // function body
}
```

### Ensures Clauses

```ora
fn transfer(amount: u256)
ensures balance == old(balance) - amount
{
    // function body
}
```

```mlir
func.func @transfer(%amount: i256) {
    %old_balance = memref.load %balance_ref : memref<1xi256>
    
    // function body
    
    %new_balance = memref.load %balance_ref : memref<1xi256>
    %expected = arith.subi %old_balance, %amount : i256
    %postcond = arith.cmpi eq, %new_balance, %expected : i256
    ora.assert %postcond {ora.ensures = true}
}
```

## Compiler Integration

### CLI Usage

Generate MLIR output:
```bash
ora compile --emit-mlir contract.ora
```

Verify MLIR correctness:
```bash
ora compile --mlir-verify contract.ora
```

Run optimization passes:
```bash
ora compile --mlir-passes="canonicalize,cse" contract.ora
```

### Build Integration

The MLIR lowering integrates with the existing compilation pipeline:

1. **Lexing and Parsing:** Source code → AST
2. **Semantic Analysis:** Type checking and validation
3. **MLIR Lowering:** AST → MLIR module
4. **MLIR Passes:** Optimization and analysis
5. **Backend Selection:** MLIR → Yul (primary) or other targets

## Debugging and Analysis

### Source Location Preservation

All MLIR operations preserve source location information:

```mlir
%result = arith.addi %a, %b : i256 loc("contract.ora":15:8)
```

### Error Reporting

MLIR lowering provides detailed error messages with source context:

```
error: Type mismatch in binary operation
  --> contract.ora:15:8
   |
15 |     let result = balance + "invalid";
   |                          ^ expected numeric type, found string
```

### Verification Passes

Custom verification passes check Ora-specific constraints:

- Memory region usage validation
- Contract invariant checking
- Gas usage analysis
- Security pattern detection

## Performance Characteristics

The MLIR lowering system is designed for:

- **Fast compilation:** Efficient lowering algorithms
- **Memory efficiency:** Minimal overhead during compilation
- **Scalability:** Handles large smart contracts
- **Deterministic output:** Consistent results for testing

## Future Enhancements

Planned improvements include:

1. **Enhanced Dialect:** Complete Ora dialect with custom operations
2. **Advanced Analysis:** Data flow and control flow analysis
3. **Optimization Passes:** Ora-specific optimizations
4. **Alternative Backends:** LLVM, WebAssembly, and native targets
5. **Formal Verification:** Integration with verification tools

## Examples

### Complete Contract Example

```ora
contract Token {
    storage balance: map[address, u256];
    storage total_supply: u256;
    
    fn transfer(to: address, amount: u256)
        requires(balance[msg.sender] >= amount)
        ensures(balance[to] == old(balance[to]) + amount)
        ensures(balance[msg.sender] == old(balance[msg.sender]) - amount)
    {
        move amount from msg.sender to to;
        log Transfer(from: msg.sender, to: to, amount: amount);
    }
}
```

This generates MLIR with:
- Memory space annotations for storage variables
- Precondition and postcondition assertions
- Move operation with transfer semantics
- Event logging with indexed parameters
- Source location preservation throughout

The MLIR representation enables advanced analysis and optimization while maintaining the semantic integrity of the original Ora code.

## MLIR Resources

### Learn More About MLIR
- **MLIR Official Website:** https://mlir.llvm.org
- **MLIR Documentation:** https://mlir.llvm.org/docs/
- **MLIR Tutorials:** https://mlir.llvm.org/docs/Tutorials/
- **LLVM Project:** https://llvm.org

### Ora MLIR Implementation
- **Source Code:** Available in the `src/mlir/` directory of the Ora repository
- **Technical Documentation:** See `docs/mlir-lowering.md` for implementation details
- **API Reference:** See `docs/mlir-api.md` for developer documentation
- **Troubleshooting:** See `docs/mlir-troubleshooting.md` for common issues

The Ora MLIR integration leverages the powerful MLIR framework developed by the LLVM community to provide advanced compiler capabilities for smart contract development.