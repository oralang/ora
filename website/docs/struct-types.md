---
sidebar_position: 3.5
---

# Struct Types

Ora provides powerful struct types that combine fields into custom data structures with automatic memory layout optimization and gas-efficient storage packing.

## Basic Struct Declaration

Define custom types by grouping related fields together:

```ora
struct Point {
    x: u32,
    y: u32,
}

struct User {
    name: string,
    age: u8,
    balance: u256,
    location: Point,
}
```

## Memory Regions and Storage Optimization

Structs work seamlessly with Ora's memory regions, with the compiler automatically optimizing layout based on usage patterns:

### Storage Memory Region
Storage structs use optimized field packing to minimize gas costs. Fields are automatically reordered and packed into 32-byte EVM storage slots:

```ora
contract TokenContract {
    storage user_data: User;
    storage origin: Point;
    
    pub fn init() {
        // Storage assignments are optimized for minimal gas usage
        user_data = User {
            name: "Alice",
            age: 25,
            balance: 1000000,
            location: Point { x: 100, y: 200 }
        };
    }
}
```

### Memory Layout Analysis

The compiler provides detailed analysis of struct memory usage:

- **Storage Slots**: Number of 32-byte EVM storage slots required
- **Field Packing**: How fields are grouped within storage slots
- **Gas Optimization**: Warnings about field ordering for better efficiency

Example compiler output:
```
Info: Struct 'User' uses 4 storage slots (97 bytes) - consider field ordering for gas optimization
Warning: Struct 'User' contains complex types - actual gas costs will depend on usage patterns
```

## Field Ordering and Gas Optimization

The compiler analyzes field types and suggests optimizations:

### Automatic Packing Strategy
```ora
struct OptimizedData {
    // Small fields packed together (1 storage slot)
    active: bool,     // 1 byte
    level: u8,        // 1 byte
    flags: u16,       // 2 bytes
    counter: u32,     // 4 bytes
    
    // Large fields get dedicated slots
    balance: u256,    // 32 bytes (1 slot)
    metadata: string, // Dynamic size
}
```

### Manual Optimization
For critical contracts, consider field ordering:

```ora
struct GasOptimized {
    // Group small fields first
    flag1: bool,
    flag2: bool,
    small_counter: u32,
    
    // Place large fields last
    primary_balance: u256,
    secondary_balance: u256,
}
```

## Nested Structs and Cross-References

Structs can contain other struct types, enabling complex data modeling:

```ora
struct Address {
    street: string,
    city: string,
    postal_code: u32,
}

struct Employee {
    id: u256,
    name: string,
    home_address: Address,
    work_address: Address,
}

contract Company {
    storage employees: map[u256, Employee];
    
    pub fn add_employee(id: u256, name: string) {
        employees[id] = Employee {
            id: id,
            name: name,
            home_address: Address { 
                street: "", 
                city: "", 
                postal_code: 0 
            },
            work_address: Address { 
                street: "Main St", 
                city: "Tech City", 
                postal_code: 12345 
            }
        };
    }
}
```

## Field Access and Operations

Access struct fields using dot notation with compiler-optimized assembly generation:

```ora
pub fn update_user_balance(new_balance: u256) {
    user_data.balance = new_balance;
    
    // Nested field access
    user_data.location.x = user_data.location.x + 10;
    
    // Conditional logic with struct fields
    if (user_data.age >= 18) {
        user_data.balance = user_data.balance * 2;
    }
}

pub fn get_user_location() -> (u32, u32) {
    return (user_data.location.x, user_data.location.y);
}
```

## Memory Management

Ora handles struct memory management automatically:

### Lifecycle Management
- **Local structs**: Automatically cleaned up when scope ends
- **Storage structs**: Persist between transactions
- **Memory structs**: Exist for transaction duration
- **Immutable structs**: Embedded in contract bytecode

### Resource Cleanup
The compiler ensures proper cleanup of:
- Dynamic arrays within structs
- String fields
- Nested struct references
- Complex type allocations

### Memory Safety
- No dangling pointers or use-after-free errors
- Automatic bounds checking for array fields
- Type-safe field access validation

## Performance Considerations

### Storage Access Patterns
```ora
// Efficient: Single storage read/write per struct
pub fn efficient_update(user: User) {
    storage temp: User = user;
    temp.balance = temp.balance + 100;
    user_data = temp;
}

// Less efficient: Multiple storage operations
pub fn inefficient_update() {
    user_data.balance = user_data.balance + 100;  // Storage read + write
    user_data.age = user_data.age + 1;            // Another read + write
}
```

### Dynamic Fields Impact
Structs with dynamic fields (strings, arrays) have variable gas costs:

```ora
struct VariableSize {
    id: u256,           // Fixed cost
    name: string,       // Variable cost based on length
    tags: u32[],        // Variable cost based on array size
}
```

## Integration with Formal Verification

Struct types work seamlessly with Ora's formal verification system:

```ora
struct BankAccount {
    balance: u256,
    frozen: bool,
    owner: address,
}

storage account: BankAccount;

pub fn withdraw(amount: u256) 
    requires: !account.frozen && account.balance >= amount
    ensures: account.balance == old(account.balance) - amount
{
    account.balance = account.balance - amount;
}

pub fn freeze_account()
    requires: account.owner == msg.sender
    ensures: account.frozen == true
{
    account.frozen = true;
}
```

## Compilation Pipeline Integration

Struct definitions flow through Ora's complete compilation pipeline:

1. **Parsing**: Creates structured AST nodes for struct declarations
2. **Semantic Analysis**: Validates field types, cross-references, and memory region compatibility
3. **Type Checking**: Ensures type safety and proper field access patterns
4. **HIR Generation**: Optimizes struct operations and memory layout
5. **Yul Generation**: Produces efficient EVM bytecode with optimized memory operations