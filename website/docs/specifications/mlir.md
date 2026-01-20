# MLIR Intermediate Representation

> Mixed implementation notes with design intent. For the current
> Ora op set, see `src/mlir/ora/td/OraOps.td` in the repo. Examples are
> illustrative and may require explicit type annotations in the current
> compiler.

## Overview

The Ora compiler includes a comprehensive MLIR (Multi-Level Intermediate Representation) lowering system that provides an intermediate representation for advanced analysis, optimization, and alternative code generation paths. This system lowers to Sensei-IR (SIR) for EVM bytecode generation and enables sophisticated compiler transformations.

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
3. **Lower to Sensei-IR (SIR)** for EVM bytecode generation via the Sensei-IR backend
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

The Ora compiler uses a modern, streamlined command interface:

**Compile and emit bytecode:**
```bash
ora contract.ora
ora --emit-bytecode contract.ora
```

**View MLIR IR:**
```bash
ora --emit-mlir contract.ora
```

**View Sensei-IR (SIR) intermediate code:**
```bash
ora --emit-sir contract.ora
```

**Advanced options:**
```bash
# Disable automatic MLIR validation (not recommended)
ora --no-validate-mlir contract.ora

# Use custom MLIR optimization passes
ora --mlir-passes="canonicalize,cse,mem2reg" contract.ora
```

### Automatic MLIR Validation

The compiler automatically validates MLIR correctness before Sensei-IR lowering:

```
$ ora contract.ora
Parsing contract.ora...
Performing semantic analysis...
Lowering to MLIR...
Validating MLIR before Sensei-IR lowering...
✅ MLIR validation passed
Lowering to Sensei-IR (SIR)...
Compiling to EVM bytecode via Sensei-IR...
Successfully compiled to EVM bytecode!
```

If validation fails, compilation stops immediately:
```
❌ MLIR validation failed with 3 error(s):
  - [TypeMismatch] Expected i256, found i1
  - [MalformedAst] Missing operand for arith.addi
  - [InvalidRegion] Block has no terminator
```

Validation is automatic to catch errors early in the compilation pipeline.

### Build Integration

The MLIR lowering pipeline:

1. **Lexing and Parsing**: Source code → AST
2. **Semantic Analysis**: Type checking, builtin validation
3. **MLIR Lowering**: AST → MLIR module (with source locations)
4. **MLIR Validation**: Structural and semantic checks (automatic)
5. **MLIR Optimization**: CSE, canonicalization, mem2reg, SCCP, LICM
6. **Sensei-IR Lowering**: MLIR → Sensei-IR (SIR)
7. **Bytecode Generation**: Sensei-IR → EVM bytecode via Sensei-IR debug-backend

## SSA Transformation

### Ora to SSA

While Ora allows mutable local variables, the MLIR representation uses **Static Single Assignment (SSA)** form internally:

**Ora Code (Mutable Variables)**:
```ora
pub fn calculate(x: u256) -> u256 {
    let result = x;      // Mutable local variable
    result = result + 10;
    result = result * 2;
    return result;
}
```

**MLIR Representation (SSA with Memory)**:
```mlir
func.func @calculate(%arg0: i256) -> i256 {
    %0 = memref.alloca() : memref<1xi256>    // Stack allocation
    memref.store %arg0, %0[] : memref<1xi256> // Store initial value
    
    %1 = memref.load %0[] : memref<1xi256>    // Load for +10
    %c10 = arith.constant 10 : i256
    %2 = arith.addi %1, %c10 : i256
    memref.store %2, %0[] : memref<1xi256>    // Store back
    
    %3 = memref.load %0[] : memref<1xi256>    // Load for *2
    %c2 = arith.constant 2 : i256
    %4 = arith.muli %3, %c2 : i256
    memref.store %4, %0[] : memref<1xi256>    // Store back
    
    %5 = memref.load %0[] : memref<1xi256>    // Load final result
    return %5 : i256
}
```

**After mem2reg Optimization Pass**:
```mlir
func.func @calculate(%arg0: i256) -> i256 {
    %c10 = arith.constant 10 : i256
    %0 = arith.addi %arg0, %c10 : i256       // Pure SSA!
    %c2 = arith.constant 2 : i256
    %1 = arith.muli %0, %c2 : i256           // Pure SSA!
    return %1 : i256
}
```

### SSA Benefits

1. **Optimization**: Standard MLIR passes (CSE, SCCP, dead code elimination) work directly
2. **Analysis**: Data flow analysis is straightforward in SSA form
3. **Type Safety**: Each SSA value has a single, well-defined type
4. **Gas Efficiency**: `mem2reg` eliminates redundant loads/stores

### Memory Regions vs SSA

**Local Variables** (SSA via mem2reg):
- Function-scope variables
- `memref.alloca` → SSA values
- Optimized away in final code

**Storage Variables** (NOT SSA):
- Contract state (persistent)
- `ora.sload` / `ora.sstore` operations
- Direct EVM `SLOAD`/`SSTORE` opcodes
- Cannot be optimized away

```mlir
// Local variable (SSA):
%local = arith.constant 100 : i256

// Storage variable (NOT SSA):
%slot = arith.constant 0 : i256
%value = ora.sload %slot : i256
ora.sstore %slot, %value : i256
```

---

## Standard Library Integration

### Built-in Lowering

Ora's standard library built-ins are lowered to custom `ora.evm.*` MLIR operations:

**Ora Code**:
```ora
let timestamp = std.block.timestamp();
let sender = std.msg.sender();
```

**MLIR**:
```mlir
%timestamp = ora.evm.timestamp() : i256 loc("contract.ora":10:20)
%sender = ora.evm.caller() : i160 loc("contract.ora":11:17)
```

**Sensei-IR (SIR)**:
```sir
fn main:
  entry -> timestamp sender {
    timestamp = timestamp
    sender = caller
    iret
  }
```

### Zero-Overhead Guarantee

All standard library calls are **inlined** at the MLIR level - no function call overhead:

| Ora Built-in | MLIR Operation | Sensei-IR Operation | Overhead |
|--------------|----------------|-------------------|----------|
| `std.block.timestamp()` | `ora.evm.timestamp` | `timestamp` | **Zero** |
| `std.msg.sender()` | `ora.evm.caller` | `caller` | **Zero** |
| `std.constants.U256_MAX` | `arith.constant -1` | `0xfff...fff` | **Zero** |

### Semantic Validation

Built-in calls are validated during semantic analysis:

✅ **Valid**:
```ora
let sender = std.msg.sender();  // Correct usage
```

❌ **Invalid** (caught at compile time):
```ora
let sender = std.msg.sender;    // Error: missing ()
let invalid = std.block.fake(); // Error: unknown built-in
```

---

## Debugging and Analysis

### Source Location Preservation

All MLIR operations preserve source location information where available:

```mlir
%result = arith.addi %a, %b : i256 loc("contract.ora":15:8)
```

**What's tracked**:
- Filename (e.g., `contract.ora`)
- Line number (e.g., `15`)
- Column number (e.g., `8`)

**Use cases**:
- Error reporting with exact source position
- Debugging with source-level stepping
- Coverage analysis
- Profiling with source attribution

### Error Reporting

MLIR validation provides detailed error messages:

```
❌ MLIR validation failed with 2 error(s):
  - [TypeMismatch] Expected i256, found i1
    at contract.ora:42:15
  - [MalformedAst] Missing terminator in block
    at contract.ora:50:1
```

### Optimization Passes

Standard MLIR passes optimize the IR:

| Pass | Purpose | Benefit |
|------|---------|---------|
| **CSE** | Common Subexpression Elimination | Reduce duplicate computations |
| **Canonicalization** | Simplify IR patterns | Cleaner code |
| **mem2reg** | Convert memory to SSA values | Eliminate loads/stores |
| **SCCP** | Sparse Conditional Constant Propagation | Constant folding |
| **LICM** | Loop-Invariant Code Motion | Hoist invariants |

**Result**: Reduces redundant IR and simplifies lowering; exact impact
depends on backend maturity.

## Performance Characteristics

The MLIR lowering system targets:

- **Predictable compilation:** Deterministic output for tests
- **Modular passes:** Clear transformation stages
- **Traceability:** Source locations preserved where available

## Implementation Features

### Core Features

| Feature | Coverage |
|---------|----------|
| AST → MLIR Lowering | Implemented |
| Source Location Tracking | Implemented (coverage varies by pass) |
| Automatic Validation | Yes |
| Type System Mapping | Implemented |
| Standard Library | EVM-facing built-ins |
| SSA Transformation | mem2reg pass |
| Optimization Passes | Core passes (CSE, canonicalization, mem2reg, SCCP, LICM) |
| Storage Operations | sload, sstore, tload, tstore |
| Control Flow | if, while, for, switch |
| Function Calls | Implemented |
| Sensei-IR Lowering | Supported |

### Active Areas

- Formal verification integration
- Custom Ora dialect registration
- Advanced type features (generics, constraints)

### Future

- Advanced analysis (loop, alias, escape)
- Gas cost modeling
- Alternative backends (LLVM IR, WebAssembly)
- IDE integration

## Examples

### ERC20 Token Contract

```ora
contract SimpleToken {
    storage totalSupply: u256;
    storage balances: map[address, u256];
    storage allowances: doublemap[address, address, u256];
    
    pub fn initialize(initialSupply: u256) -> bool {
        var deployer: address = std.msg.sender();
        totalSupply = initialSupply;
        balances[deployer] = initialSupply;
        return true;
    }
    
    pub fn transfer(recipient: address, amount: u256) -> bool {
        let sender = std.msg.sender();
        let senderBalance = balances[sender];
        
        if (recipient == std.constants.ZERO_ADDRESS) {
            return false;
        }
        
        if (senderBalance < amount) {
            return false;
        }
        
        balances[sender] = senderBalance - amount;
        let recipientBalance = balances[recipient];
        balances[recipient] = recipientBalance + amount;
        
        return true;
    }
}
```

This example demonstrates:

- Zero-overhead built-ins (`std.msg.sender()` → `CALLER` opcode)
- Storage operations (`balances[sender]` → `ora.sload` with keccak256)
- Type safety (all operations type-checked)
- SSA form (local variables via mem2reg)
- Source locations (every operation tagged with source position)

The MLIR representation enables analysis and optimization while maintaining semantic integrity.

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
