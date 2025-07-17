# HIR (High-level Intermediate Representation) Specification

## Overview

The Ora HIR bridges the AST and codegen phases. It models validated, effect-aware, and optimizable program structure.

### Key Features
- **Explicit effect tracking**
- **Memory region modeling** 
- **Strong type system**
- **Formal verification support**
- **Optimizer-friendly**
- **Error union/result support**
- **Dynamic symbol tracking**

## Design Principles
1. **Explicit behavior modeling**
2. **Immutable HIR nodes**
3. **Must validate before codegen**
4. **All effects are tracked**
5. **Memory safety enforced**
6. **Simplicity over complexity**
7. **Accurate program behavior modeling**

## Memory Model

```zig
enum Region {
  stack, memory, storage, tstore, const_, immutable
}
```

| Region | Lifetime | Mutability | Gas Cost | Use Case |
|--------|----------|------------|----------|----------|
| stack | Function | Variable | Low | Local vars |
| memory | Transaction | Variable | Medium | Temporary buffers |
| storage | Persistent | Variable | High | Contract state |
| tstore | Transaction | Variable | Medium | Transient state |
| const_ | Compile-time | Immutable | None | Constants |
| immutable | Deployment | Immutable | Low | Init-only contract vars |

## Type System

```zig
enum PrimitiveType {
  u8, u16, u32, u64, u128, u256,
  bool, address, string
}

union Type {
  primitive: PrimitiveType,
  mapping: MappingType,
  slice: SliceType,
  custom: CustomType,
  error_union: ErrorUnionType,
  result: ResultType
}
```

## Node Structure

### HIRProgram
```zig
struct HIRProgram {
  version: string,
  contracts: []Contract,
  allocator: Allocator
}
```

### Contract
```zig
struct Contract {
  name: string,
  storage: []StorageVariable,
  functions: []Function,
  events: []Event,
  allocator: Allocator
}
```

### Function
```zig
struct Function {
  name: string,
  visibility: Visibility,
  parameters: []Parameter,
  return_type: ?Type,
  requires: []Expression,
  ensures: []Expression,
  body: Block,
  state_effects: EffectSet,
  observable_effects: EffectSet,
  effects: FunctionEffects,
  location: SourceLocation,
  allocator: Allocator
}
```

### FunctionEffects
```zig
struct FunctionEffects {
  writes_storage: bool,
  reads_storage: bool,
  writes_transient: bool,
  reads_transient: bool,
  emits_logs: bool,
  calls_other: bool,
  modifies_state: bool,
  is_pure: bool
}
```

## Expressions
```zig
union Expression {
  binary, unary, call, index, field, transfer, shift,
  old, literal, identifier, try_expr, error_value, error_cast
}
```

## Statements
```zig
union Statement {
  variable_decl, assignment, compound_assignment, if_statement,
  while_statement, return_statement, expression_statement,
  lock_statement, unlock_statement, error_decl,
  try_statement, error_return
}
```

## Effect System

### Effect
```zig
struct Effect {
  type: EffectType,
  path: AccessPath,
  condition: ?Expression
}
```

### AccessPath
```zig
struct AccessPath {
  base: string,
  selectors: []PathSelector,
  region: Region
}
```

### PathSelector
```zig
union PathSelector {
  field: { name: string },
  index: { index: Expression }
}
```

## Effect Analysis
- **Symbol tables are rebuilt per contract**
- **All expressions and statements are walked**
- **Effects are computed recursively**
- **Computed metadata:**

```zig
modifies_state = writes_storage || writes_transient || emits_logs
is_pure = !(writes_storage || reads_storage || writes_transient || reads_transient || emits_logs || calls_other)
```

## Validation Rules
- **All expressions must be typed**
- **Assignments must be type-compatible**
- **Function args must match params**
- **Index ops must match container types**
- **All effects must be consistent with requires/ensures**
- **Return paths must be valid**
- **Invariants must be stated on loops**
- **Error branches must be handled or propagated**

## Optimization Framework

```zig
struct OptimizationPass {
  name: string,
  run: fn(*HIRProgram, Allocator) -> anyerror!void
}
```

**Standard passes:**
1. **Dead code elimination**
2. **Constant folding**
3. **Effect optimization**
4. **Gas optimization**

## JSON Serialization
- **HIR serializes to JSON for tooling/debugging**
- **Effects included as simple flags**
- **Example:**

```json
{
  "name": "transfer",
  "effects": {
    "writes_storage": true,
    "emits_logs": true,
    "is_pure": false
  }
}
```

## Examples

### Simple Function HIR
```ora
pub fn transfer(to: address, amount: u256) -> bool
    requires(balances[std.transaction.sender] >= amount)
    ensures(balances[std.transaction.sender] + balances[to] == 
            old(balances[std.transaction.sender]) + old(balances[to]))
{
    @lock(balances[to]);
    balances from std.transaction.sender -> to : amount;
    log Transfer(std.transaction.sender, to, amount);
    return true;
}
```

**HIR Effects:**
- `writes_storage: true` (modifies balances)
- `emits_logs: true` (emits Transfer event)
- `is_pure: false` (has side effects)

### Complex Expression HIR
```ora
let result = balances[sender] + balances[receiver];
```

**AccessPath Analysis:**
- `balances[sender]` → `{base: "balances", selectors: [index(sender)], region: storage}`
- `balances[receiver]` → `{base: "balances", selectors: [index(receiver)], region: storage}`

## Implementation Notes
- **All HIR nodes use allocators**
- **Every node includes source location**
- **Effects tracked both structurally and via summary flags**
- **Per-contract symbol isolation required**

## Extensions
- **Future: cross-contract effects, verification output, gas metrics**
- **Lock/Transfer annotations may decorate functions with semantic guarantees**

## Usage in Compilation Pipeline

1. **AST → HIR**: Convert abstract syntax tree to high-level IR
2. **HIR Validation**: Ensure type safety and effect consistency
3. **HIR Optimization**: Apply optimization passes
4. **HIR → Yul**: Generate Yul code from optimized HIR
5. **Yul → Bytecode**: Compile to EVM bytecode

The HIR serves as the central representation for analysis, optimization, and code generation in the Ora compiler. 