# Struct Types

Ora provides powerful struct types that combine fields into custom data structures with automatic memory layout optimization and advanced semantics.

## Basic Struct Declaration

Define custom types by grouping related fields together:

```ora
struct Point {
    x: u32,
    y: u32,
}

struct User {
    id: u256,
    balance: u256,
    active: bool,
}
```

## Usage with Memory Regions

Structs work seamlessly with Ora's memory regions, automatically optimizing layout for gas efficiency:

```ora
contract Example {
    // Storage structs use optimized field packing
    storage var origin: Point;
    storage let config: User;
    
    // Memory structs optimize for access patterns
    memory var temp_user: User;
    
    // Immutable structs for deployment-time constants
    immutable default_point: Point;
    
    pub fn init() {
        origin = Point { x: 0, y: 0 };
        temp_user = User { 
            id: 1, 
            balance: 1000000, 
            active: true 
        };
    }
}
```

## Memory Layout Optimization

The compiler automatically optimizes struct memory layout based on usage context. For storage variables, fields are reordered and packed to minimize gas costs. Smaller fields are grouped together within 32-byte EVM storage slots when possible.

```ora
struct OptimizedExample {
    // These fields will be packed together efficiently
    flag1: bool,      // 1 byte
    flag2: bool,      // 1 byte  
    counter: u32,     // 4 bytes
    flag3: bool,      // 1 byte
    // Large fields get their own slots
    balance: u256,    // 32 bytes
    data: bytes,      // 32 bytes
}
```

## Field Access and Operations

Access struct fields using dot notation. The compiler generates optimized code based on the field's memory layout:

```ora
pub fn updateUser(new_balance: u256) {
    temp_user.balance = new_balance;
    
    if (temp_user.active) {
        temp_user.id = temp_user.id + 1;
    }
}

pub fn getUserInfo() -> (u256, bool) {
    return (temp_user.balance, temp_user.active);
}
```

## Advanced Semantics

Structs support sophisticated lifecycle management including copy semantics, move semantics, and automatic cleanup. The compiler handles memory management transparently while providing control over resource ownership when needed.

Local structs automatically clean up when going out of scope, while storage structs persist between function calls. Memory structs exist for the transaction duration, and immutable structs are embedded in the contract bytecode at deployment time.

## Integration with Formal Verification

Struct types work seamlessly with Ora's formal verification system. Invariants can be expressed over struct fields, and the compiler generates appropriate verification conditions:

```ora
struct BankAccount {
    balance: u256,
    frozen: bool,
}

storage var account: BankAccount;

pub fn withdraw(amount: u256) 
    requires: !account.frozen && account.balance >= amount
    ensures: account.balance == old(account.balance) - amount
{
    account.balance = account.balance - amount;
}
```

## Compilation Pipeline

Struct definitions flow through Ora's complete compilation pipeline. The parser creates AST nodes, the semantic analyzer validates field types and access patterns, HIR optimizes struct operations, and the Yul generator produces efficient EVM bytecode with optimized memory operations.

This seamless integration ensures that struct types maintain both high-level expressiveness and low-level efficiency, making them ideal for complex smart contract development while preserving gas optimization and formal verification capabilities. 