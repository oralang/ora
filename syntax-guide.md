# Ora Language Syntax Guide

> A comprehensive reference for the Ora smart contract programming language

> **⚠️ Development Status**: This guide describes the target syntax. Some features are still being implemented and may not be fully functional yet.

## Table of Contents

1. [Basic Contract Structure](#basic-contract-structure)
2. [Memory Regions](#memory-regions)
3. [Data Types](#data-types)
4. [Functions](#functions)
5. [Error Handling](#error-handling)
6. [Compile-Time Evaluation](#compile-time-evaluation)
7. [Annotations](#annotations)
8. [Transfer Operations](#transfer-operations)
9. [Formal Verification](#formal-verification)
10. [Events and Logging](#events-and-logging)
11. [Control Flow](#control-flow)
12. [Advanced Features](#advanced-features)

---

## Basic Contract Structure

Every Ora program is organized around **contracts**, which are similar to classes in object-oriented programming.

```ora
contract MyContract {
    // Storage variables
    var balance: u256;
    let owner: address;
    
    // Events
    log Transfer(from: address, to: address, amount: u256);
    
    // Constructor
    pub fn init(initial_balance: u256) {
        balance = initial_balance;
        owner = msg.sender;
    }
    
    // Public function with return type
    pub fn getBalance() -> u256 {
        return balance;
    }
}
```

---

## Memory Regions

Ora provides explicit memory region annotations to optimize gas usage and ensure correct data persistence.

### Storage Region
Persistent data stored on the blockchain:

```ora
var balance: u256;           // Mutable storage
let name: string;            // Immutable storage
var balances: mapping[address, u256];  // Mapping storage
var allowances: doublemap[address, address, u256];  // Double mapping
```

### Immutable Region
Set once during contract deployment:

```ora
immutable name: string;
immutable symbol: string;
immutable decimals: u8;
```

### Compile-Time Constants
Evaluated at compile time:

```ora
storage const MAX_SUPPLY: u256 = 1000000;
storage const HALF_SUPPLY: u256 = MAX_SUPPLY / 2;
storage const CONTRACT_NAME: string = "MyToken";
```

### Memory Layout Examples

```ora
contract MemoryDemo {
    // Different memory regions
    var total_supply: u256;          // Persistent storage
    immutable owner: address;        // Set once at deployment
    let MAX_TOKENS: u256 = 1000;     // Compile-time constant
    
    // Local variables use stack/memory automatically
    fn calculate() -> u256 {
        let temp: u256 = 100;  // Stack variable
        return temp * MAX_TOKENS;
    }
}
```

---

## Data Types

### Primitive Types

```ora
// Unsigned integers
let small: u8 = 255;
let medium: u32 = 4294967295;
let large: u256 = 115792089237316195423570985008687907853269984665640564039457584007913129639935;

// Boolean
let active: bool = true;
let disabled: bool = false;

// Address
let owner: address = 0x742d35Cc6634C0532925a3b8D4e6f69d2d1e7CE8;
let zero_addr: address = address(0);

// String
let name: string = "MyToken";
```

### Complex Types

```ora
// Arrays
let numbers: u256[] = [1, 2, 3, 4, 5];
let addresses: address[] = [addr1, addr2, addr3];

// Slices
let slice: slice[u256] = numbers[1..3];  // Elements 1 and 2

// Mappings
storage balances: mapping[address, u256];
storage allowances: doublemap[address, address, u256];  // Double mapping

// Structs (custom types)
struct TokenInfo {
    name: string,
    symbol: string,
    decimals: u8
}
```

---

## Functions

### Function Declaration

```ora
// Public function (externally callable)
pub fn transfer(to: address, amount: u256) -> bool {
    // Function body
    return true;
}

// Private function (internal only)
fn internal_helper() -> u256 {
    return 42;
}

// Function with multiple return values
pub fn getTokenInfo() -> (string, string, u8) {
    return ("MyToken", "MTK", 18);
}
```

### Function Visibility

```ora
contract VisibilityDemo {
    // Public functions - callable from outside
    pub fn publicFunction() -> u256 {
        return 1;
    }
    
    // Private functions - internal only
    fn privateFunction() -> u256 {
        return 2;
    }
    
    // Constructor - special function
    pub fn init() {
        // Initialization code
    }
}
```

---

## Error Handling

Ora uses **Zig-style error handling** with error unions (`!T`) for robust error management.

### Error Declarations

```ora
// Declare custom errors
error InsufficientBalance;
error InvalidAddress;
error TransferFailed;
error AccessDenied;
```

### Error Union Types

```ora
// Function that can return u256 or an error
fn transfer(to: address, amount: u256) -> !u256 {
    // Check for errors
    if (to == address(0)) {
        return error.InvalidAddress;
    }
    
    if (balance < amount) {
        return error.InsufficientBalance;
    }
    
    // Success case
    balance -= amount;
    return balance;
}
```

### Try Expressions

```ora
fn safeTransfer(to: address, amount: u256) -> !bool {
    // Try to execute, unwrap success value or return error
    let current_balance = try getBalance(msg.sender);
    
    if (current_balance < amount) {
        return error.InsufficientBalance;
    }
    
    // Update balances
    balances[msg.sender] = current_balance - amount;
    balances[to] = balances[to] + amount;
    
    return true;
}
```

### Result Type Alternative

```ora
// Alternative Result[T, E] syntax
fn safeDivide(a: u256, b: u256) -> Result[u256, u8] {
    if (b == 0) {
        return Result.error(1);  // Division by zero
    }
    
    return Result.ok(a / b);
}
```

---

## Compile-Time Evaluation

**We want 90% of Ora is compile-time evaluation** - this is a core feature that makes Ora extremely efficient by computing as much as possible at compile time.

### Compile-Time Constants

```ora
contract Constants {
    // Compile-time arithmetic
    storage const MAX_SUPPLY: u256 = 1000000;
    storage const HALF_SUPPLY: u256 = MAX_SUPPLY / 2;
    storage const COMPLEX_CALC: u256 = (100 + 50) * 2 - 25;
    
    // Compile-time bitwise operations
    storage const BITWISE_RESULT: u256 = 0xFF & 0x0F;
    storage const SHIFT_RESULT: u256 = 8 << 2;
    
    // Boolean constants
    storage const IS_ENABLED: bool = true;
    
    // Address constants
    storage const OWNER: address = 0x742d35Cc6634C0532925a3b8D4e6f69d2d1e7CE8;
}
```

### Compile-Time Evaluation of Expressions

```ora
contract ComptimeDemo {
    // Most expressions are evaluated at compile time
    storage const DECIMALS: u8 = 18;
    storage const SCALE: u256 = 10**DECIMALS;  // Computed at compile time
    
    pub fn init() {
        // These calculations happen at compile time
        let initial_amount: u256 = 1000 * SCALE;
        let fee_amount: u256 = initial_amount * 3 / 100;  // 3% fee
        
        // Even complex expressions are compile-time evaluated
        let complex_result: u256 = (SCALE * 2) + (SCALE / 4) - 1;
    }
}
```

### Compile-Time Control Flow

```ora
contract ComptimeControl {
    // Compile-time conditionals
    storage const DEBUG_MODE: bool = false;
    
    fn process() -> u256 {
        // This branch is eliminated at compile time
        if (DEBUG_MODE) {
            return 42;  // Dead code elimination
        } else {
            return 100;
        }
    }
    
    // Compile-time loop unrolling
    fn calculateSum() -> u256 {
        let sum: u256 = 0;
        // This loop is unrolled at compile time
        for (i in 0..10) {
            sum += i;
        }
        return sum;  // Returns 45 (computed at compile time)
    }
}
```

---

## Annotations

Ora uses `@` annotations to provide metadata and optimize code generation.

### Function Annotations

```ora
contract AnnotationDemo {
    storage mut balances: mapping[address, u256];
    
    // Mark function as transferable (optimizes for token transfers)
    @transferable
    pub fn transfer(to: address, amount: u256) -> bool {
        // Lock annotation to protect against state changes during the transaction
        @lock(balances[to]);
        
        // Special transfer syntax
        balances from std.transaction.sender -> to : amount;
        
        return true;
    }
    
    @lock(balances[msg.sender])
    pub fn withdraw(amount: u256) -> bool {
        balances[msg.sender] -= amount;
        return true;
    }
}
```

### Available Annotations

```ora
// Function-level annotations
@transferable        // Optimizes for token transfers
@lock(path)         // Protects storage path from changes during transaction execution
```

---

## Transfer Operations

Ora has special syntax for token transfers that's optimized for gas efficiency.

### Transfer Syntax

```ora
contract TransferDemo {
    storage mut balances: mapping[address, u256];
    
    pub fn transfer(to: address, amount: u256) -> bool {
        // Special transfer syntax: balances from sender -> recipient : amount
        balances from std.transaction.sender -> to : amount;
        
        return true;
    }
    
    pub fn transferFrom(from: address, to: address, amount: u256) -> bool {
        // Transfer between arbitrary addresses
        balances from from -> to : amount;
        
        return true;
    }
}
```

### Transfer with Locks

```ora
contract SafeTransfer {
    storage mut balances: mapping[address, u256];
    
    @transferable
    pub fn safeTransfer(to: address, amount: u256) -> bool {
        // Lock protects against changes to recipient balance during transaction
        @lock(balances[to]);
        
        // Perform the transfer
        balances from std.transaction.sender -> to : amount;
        
        return true;
    }
}
```

---

## Formal Verification

Ora includes built-in formal verification capabilities with mathematical specifications.

### Preconditions and Postconditions

```ora
fn transfer(to: address, amount: u256) -> bool
    requires to != address(0)           // Precondition
    requires balanceOf(msg.sender) >= amount
    ensures balanceOf(msg.sender) == old(balanceOf(msg.sender)) - amount  // Postcondition
    ensures balanceOf(to) == old(balanceOf(to)) + amount
{
    // Implementation
    balances[msg.sender] -= amount;
    balances[to] += amount;
    return true;
}
```

### Invariants

```ora
contract TokenWithInvariants {
    storage balances: mapping[address, u256];
    storage total_supply: u256;
    
    // Contract invariant - always true
    invariant total_supply == sum(balances);
    invariant total_supply <= MAX_SUPPLY;
    
    // Loop invariant example
    fn calculateSum(values: u256[]) -> u256 {
        let sum: u256 = 0;
        let i: u256 = 0;
        
        while (i < values.length) {
            invariant i <= values.length;
            invariant sum >= 0;
            
            sum += values[i];
            i += 1;
        }
        
        return sum;
    }
}
```

### Quantifiers

```ora
// Universal quantifier (forall)
fn batchTransfer(recipients: address[], amounts: u256[]) -> bool
    requires recipients.length == amounts.length
    requires forall i: u256 where i < recipients.length => recipients[i] != address(0)
    requires forall i: u256 where i < amounts.length => amounts[i] > 0
{
    // Implementation
}

// Existential quantifier (exists)
fn hasPositiveBalance(accounts: address[]) -> bool
    ensures result == true => exists i: u256 where i < accounts.length && balanceOf(accounts[i]) > 0
{
    // Implementation
}
```

---

## Events and Logging

Events are used to log important state changes for off-chain monitoring.

### Event Declaration

```ora
// Event declaration
log Transfer(from: address, to: address, amount: u256);
log Approval(owner: address, spender: address, amount: u256);
log OwnershipTransferred(previousOwner: address, newOwner: address);
```

### Emitting Events

```ora
fn transfer(to: address, amount: u256) -> bool {
    // Update state
    balances[msg.sender] -= amount;
    balances[to] += amount;
    
    // Emit event
    log Transfer(msg.sender, to, amount);
    
    return true;
}
```

---

## Control Flow

### Conditional Statements

```ora
fn processPayment(amount: u256) -> bool {
    if (amount == 0) {
        return false;
    } else if (amount > balance) {
        return false;
    } else {
        balance -= amount;
        return true;
    }
}
```

### Loops

```ora
fn sumArray(values: u256[]) -> u256 {
    let sum: u256 = 0;
    
    // For loop
    for (i in 0..values.length) {
        sum += values[i];
    }
    
    return sum;
}

fn countDown(start: u256) -> u256 {
    let counter: u256 = start;
    
    // While loop
    while (counter > 0) {
        counter -= 1;
    }
    
    return counter;
}
```

### Error Handling with Try

```ora
fn handleResult(result: !u256) -> string {
    let value = try result;  // Unwrap or return error
    return "Success";
}
```

---

## Advanced Features

### Generic Functions (Planned)

```ora
// Generic function example (future feature)
fn swap<T>(a: T, b: T) -> (T, T) {
    return (b, a);
}
```

### Operator Overloading (Planned)

```ora
// Custom operators for user-defined types
struct Point {
    x: u256,
    y: u256
}

impl Point {
    fn add(self, other: Point) -> Point {
        return Point { x: self.x + other.x, y: self.y + other.y };
    }
}
```

### Module System (Planned)

```ora
// Import other contracts/modules
import std.token.ERC20;
import utils.math;

contract MyToken : ERC20 {
    // Implementation
}
```

---

## Standard Library

Ora provides a standard library with common utilities:

```ora
// Standard library usage
fn example() {
    let sender = std.transaction.sender;
    let block_number = std.block.number;
    let zero_address = std.constants.ZERO_ADDRESS;
    
    // Math utilities
    let max_val = std.math.max(a, b);
    let min_val = std.math.min(a, b);
}
```

---

## Best Practices

### 1. Use Explicit Memory Regions

```ora
// Good: Explicit memory regions
storage mut balance: u256;
immutable owner: address;

// Avoid: Implicit memory usage
// let balance: u256;  // Unclear where this lives
```

### 2. Handle Errors Properly (Zig-style)

```ora
// Good: Use error unions
fn transfer(to: address, amount: u256) -> !bool {
    if (to == address(0)) {
        return error.InvalidAddress;
    }
    // ... rest of function
}

// Good: Use try expressions
fn safeOperation() -> !u256 {
    let result = try riskyOperation();
    return result;
}
```

### 3. Use Formal Verification

```ora
// Good: Add preconditions and postconditions
fn transfer(to: address, amount: u256) -> bool
    requires to != address(0)
    requires balanceOf(msg.sender) >= amount
    ensures balanceOf(msg.sender) == old(balanceOf(msg.sender)) - amount
{
    // Implementation
}
```

### 4. Leverage Compile-Time Evaluation

```ora
// Good: Use compile-time constants
storage const MAX_SUPPLY: u256 = 1000000;
storage const DECIMALS: u8 = 18;

// Use in code - computed at compile time
fn init() {
    total_supply = MAX_SUPPLY;
    let scale = 10**DECIMALS;  // Computed at compile time
}
```

### 5. Use Annotations for Optimization

```ora
// Good: Use annotations for optimization
@transferable
pub fn transfer(to: address, amount: u256) -> bool {
    @lock(balances[to]);
    balances from msg.sender -> to : amount;
    return true;
}
```

---

## Complete Example

Here's a complete ERC-20 token implementation showcasing most Ora features:

```ora
contract OraToken {
    // Error declarations
    error InsufficientBalance;
    error InvalidAddress;
    error InsufficientAllowance;
    
    // Constants (compile-time evaluated)
    storage const MAX_SUPPLY: u256 = 1000000 * 10**18;
    storage const DECIMALS: u8 = 18;
    immutable name: string;
    immutable symbol: string;
    
    // Storage
    storage mut total_supply: u256;
    storage mut balances: mapping[address, u256];
    storage mut allowances: doublemap[address, address, u256];
    
    // Events
    log Transfer(from: address, to: address, amount: u256);
    log Approval(owner: address, spender: address, amount: u256);
    
    // Constructor
    pub fn init(_name: string, _symbol: string, _initial_supply: u256) -> !bool {
        if (_initial_supply > MAX_SUPPLY) {
            return error.InvalidAddress;
        }
        
        name = _name;
        symbol = _symbol;
        total_supply = _initial_supply;
        balances[msg.sender] = _initial_supply;
        
        log Transfer(address(0), msg.sender, _initial_supply);
        return true;
    }
    
    // View functions
    pub fn balanceOf(account: address) -> u256 {
        return balances[account];
    }
    
    pub fn totalSupply() -> u256 {
        return total_supply;
    }
    
    pub fn allowance(owner: address, spender: address) -> u256 {
        return allowances[owner, spender];
    }
    
    // Transfer function with formal verification and annotations
    @transferable
    pub fn transfer(to: address, amount: u256) -> !bool
        requires to != address(0)
        requires balanceOf(msg.sender) >= amount
        ensures result == true => balanceOf(msg.sender) == old(balanceOf(msg.sender)) - amount
        ensures result == true => balanceOf(to) == old(balanceOf(to)) + amount
    {
        if (to == address(0)) {
            return error.InvalidAddress;
        }
        
        let sender_balance = try balanceOf(msg.sender);
        if (sender_balance < amount) {
            return error.InsufficientBalance;
        }
        
        @lock(balances[to]);
        balances from msg.sender -> to : amount;
        
        log Transfer(msg.sender, to, amount);
        return true;
    }
    
    // Approval function
    pub fn approve(spender: address, amount: u256) -> !bool {
        if (spender == address(0)) {
            return error.InvalidAddress;
        }
        
        allowances[msg.sender, spender] = amount;
        log Approval(msg.sender, spender, amount);
        return true;
    }
    
    // Transfer from function
    pub fn transferFrom(from: address, to: address, amount: u256) -> !bool {
        if (from == address(0) || to == address(0)) {
            return error.InvalidAddress;
        }
        
        let current_allowance = allowances[from, msg.sender];
        if (current_allowance < amount) {
            return error.InsufficientAllowance;
        }
        
        let from_balance = balances[from];
        if (from_balance < amount) {
            return error.InsufficientBalance;
        }
        
        @lock(balances[to]);
        balances from from -> to : amount;
        allowances[from, msg.sender] = current_allowance - amount;
        
        log Transfer(from, to, amount);
        return true;
    }
}
```