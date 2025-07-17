---
sidebar_position: 3
---

# Language Basics

Core language features and syntax in the current implementation.

> **🚧 EXPERIMENTAL LANGUAGE**: This documentation describes the current state of Ora's syntax and features. Language design is evolving rapidly and syntax may change without notice.

## Overview

Ora is designed as a smart contract language with formal verification capabilities. The current implementation focuses on basic compilation functionality while advanced features are being developed.

### Current Implementation Status

**✅ Implemented:**
- Basic contract structure
- Function definitions
- Storage variables
- Basic types (u256, address, bool)
- Simple control flow

**🚧 In Development:**
- Formal verification syntax
- Advanced error handling
- Memory safety features
- Standard library functions

**📋 Planned:**
- Comprehensive type system
- Advanced compile-time features
- IDE integration

## Contract Structure

Every Ora program is organized around contracts:

```ora
contract MyContract {
    // Contract contents
}
```

## Variables and Storage

### Storage Variables

Persistent state that survives between function calls:

```ora
contract Counter {
    storage var count: u256;
    storage var owner: address;
    storage var active: bool;
}
```

### Immutable Variables

Set once during contract deployment:

```ora
contract Token {
    immutable name: string;
    immutable symbol: string;
    immutable decimals: u8;
}
```

### Compile-Time Constants

Values computed at compile time:

```ora
contract Config {
    storage const MAX_SUPPLY: u256 = 1000000;
    storage const RATE: u256 = 100;
}
```

## Types

### Basic Types

```ora
// Unsigned integers
var small: u8 = 255;
var medium: u32 = 4294967295;
var large: u256 = 115792089237316195423570985008687907853269984665640564039457584007913129639935;

// Signed integers (in development)
var signed: i256 = -1000;

// Boolean
var flag: bool = true;

// Address
var addr: address = 0x742d35Cc6634C0532925a3b8D0C5e0E0f8d7D2;

// Strings (basic support)
var text: string = "Hello, Ora!";
```

### Complex Types

```ora
// Arrays (in development)
var numbers: [10]u256;

// Mappings
var balances: map[address, u256];
var approved: map[address, map[address, u256]];

// Structs (planned)
struct User {
    name: string,
    balance: u256,
    active: bool,
}
```

## Functions

### Function Declaration

```ora
contract Math {
    // Public function
    pub fn add(a: u256, b: u256) -> u256 {
        return a + b;
    }
    
    // Private function
    fn internal_calc(x: u256) -> u256 {
        return x * 2;
    }
    
    // Function with no return value
    pub fn reset() {
        // Implementation
    }
}
```

### Function Visibility

- `pub`: Public, callable from outside the contract
- (no modifier): Private, internal use only

## Control Flow

### Conditional Statements

```ora
fn check_balance(amount: u256) -> bool {
    if (balance >= amount) {
        return true;
    } else {
        return false;
    }
}
```

### Loops (Basic Implementation)

```ora
fn sum_range(n: u256) -> u256 {
    var result: u256 = 0;
    var i: u256 = 0;
    
    while (i < n) {
        result = result + i;
        i = i + 1;
    }
    
    return result;
}
```

## Error Handling (In Development)

### Error Unions

```ora
// Error declarations
error InsufficientBalance;
error InvalidAddress;
error Overflow;

// Function returning error union
fn transfer(to: address, amount: u256) -> !u256 {
    if (to == std.constants.ZERO_ADDRESS) {
        return error.InvalidAddress;
    }
    
    if (balance < amount) {
        return error.InsufficientBalance;
    }
    
    // Success case
    balance = balance - amount;
    return balance;
}
```

### Try-Catch (Planned)

```ora
fn safe_transfer(to: address, amount: u256) {
    try {
        let new_balance = transfer(to, amount);
        // Success handling
    } catch (error.InsufficientBalance) {
        // Handle insufficient balance
    } catch (error.InvalidAddress) {
        // Handle invalid address
    }
}
```

## Memory Regions

### Storage Region

Persistent contract state:

```ora
contract Token {
    storage var total_supply: u256;
    storage var balances: map[address, u256];
}
```

### Memory Region (Planned)

Temporary data for function execution:

```ora
fn process_data() {
    memory var temp_array: [100]u256;
    memory var calculation: u256;
    // Process data
}
```

## Events and Logging

### Event Declaration

```ora
contract Token {
    log Transfer(from: address, to: address, amount: u256);
    log Approval(owner: address, spender: address, amount: u256);
}
```

### Event Emission

```ora
fn transfer(to: address, amount: u256) {
    // Transfer logic
    
    log Transfer(std.transaction.sender, to, amount);
}
```

## Standard Library (In Development)

### Transaction Context

```ora
fn get_sender() -> address {
    return std.transaction.sender;
}

fn get_value() -> u256 {
    return std.transaction.value;
}
```

### Constants

```ora
fn check_zero_address(addr: address) -> bool {
    return addr == std.constants.ZERO_ADDRESS;
}
```

## Formal Verification (Planned)

### Preconditions and Postconditions

```ora
fn transfer(to: address, amount: u256) -> bool
    requires balances[std.transaction.sender] >= amount
    requires to != std.constants.ZERO_ADDRESS
    ensures balances[std.transaction.sender] + balances[to] == 
            old(balances[std.transaction.sender]) + old(balances[to])
{
    balances[std.transaction.sender] -= amount;
    balances[to] += amount;
    return true;
}
```

### Invariants

```ora
contract Token {
    storage var total_supply: u256;
    storage var balances: map[address, u256];
    
    invariant sum_balances_equals_total_supply() {
        // Sum of all balances equals total supply
    }
}
```

## Comments

```ora
// Single-line comment

/*
   Multi-line comment
   Can span multiple lines
*/

contract Example {
    /// Documentation comment for functions
    pub fn documented_function() {
        // Implementation
    }
}
```

## Compilation Phases

Understanding how Ora processes your code:

1. **Lexical Analysis**: Source code → Token stream
2. **Syntax Analysis**: Tokens → Abstract Syntax Tree (AST)
3. **Semantic Analysis**: AST → Validated AST with type information
4. **HIR Generation**: AST → High-level Intermediate Representation
5. **Yul Generation**: HIR → Yul code
6. **Bytecode Generation**: Yul → EVM bytecode

## Best Practices (Current)

### Code Organization

```ora
contract WellOrganized {
    // 1. Constants first
    storage const MAX_USERS: u256 = 1000;
    
    // 2. State variables
    storage var user_count: u256;
    storage var users: map[address, bool];
    
    // 3. Events
    log UserAdded(user: address);
    
    // 4. Functions (public first, then private)
    pub fn add_user(user: address) {
        // Implementation
    }
    
    fn validate_user(user: address) -> bool {
        // Implementation
        return true;
    }
}
```

### Error Handling

```ora
// Use descriptive error names
error UserNotFound;
error UserAlreadyExists;
error ExceedsMaxUsers;

// Check preconditions early
fn remove_user(user: address) -> !bool {
    if (!users[user]) {
        return error.UserNotFound;
    }
    
    // Main logic
    users[user] = false;
    user_count = user_count - 1;
    
    return true;
}
```

## Current Limitations

### Not Yet Implemented

- Advanced type system features
- Comprehensive standard library
- Formal verification execution
- Advanced memory management
- Optimization passes

### Syntax Subject to Change

- Error handling syntax
- Formal verification syntax
- Memory region declarations
- Advanced type annotations

## Next Steps

1. **Try the Examples**: See [Examples](./examples) for working code patterns
2. **Read the Specifications**: Check [Technical Specifications](./specifications/) for detailed design
3. **Experiment**: Modify existing examples to understand current capabilities
4. **Report Issues**: Help improve the language by reporting bugs

---

*Last updated: December 2024 - Reflects current implementation status* 