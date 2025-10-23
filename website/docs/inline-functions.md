# Inline Functions

Ora provides the `inline` keyword as a compiler hint to inline small, frequently-called functions for better performance.

## Overview

The `inline` keyword suggests to the compiler that a function should be inlined at the call site, potentially reducing function call overhead and improving gas efficiency in smart contracts.

```ora
// Simple inline function
inline fn add(a: u256, b: u256) -> u256 {
    return a + b;
}

// Use it like any other function
pub fn calculate_total(x: u256, y: u256) -> u256 {
    return add(x, y);  // Will be inlined
}
```

## When to Use `inline`

### ✅ Good Candidates for Inline

- **Simple helper functions** (< 20 statements)
- **Validation functions** that are called frequently
- **Getters and setters** for common operations
- **Mathematical utilities** with minimal logic

```ora
// Perfect for inline - 1 statement
inline fn is_zero_address(addr: address) -> bool {
    return addr == std.constants.ZERO_ADDRESS;
}

// Good for inline - 3 statements
inline fn within_limit(amount: u256, max: u256) -> bool {
    if (amount > max) {
        return false;
    }
    return true;
}
```

### ❌ Avoid Inlining

- **Complex functions** (> 100 statements)
- **Functions with deep nesting** (> 4 levels)
- **Functions called rarely**
- **Recursive functions**

```ora
// ❌ BAD: Too complex for inline
inline fn complex_calculation(data: u256) -> u256 {
    // 150 statements of complex logic
    // Multiple nested loops and conditions
    // This will trigger a compiler warning!
}
```

## Complexity Analysis

Ora includes a built-in complexity analyzer to help you make informed decisions about which functions to inline.

### Running Complexity Analysis

```bash
ora --analyze-complexity contract.ora
```

### Understanding the Output

The analyzer categorizes functions into three levels:

#### ✓ Simple (< 20 statements)
```
inline fn only_owner() -> bool
  Complexity: ✓ Simple
  Nodes:      1
  Statements: 1
  Max Depth:  1
```
**Perfect for inline!** These functions are lightweight and benefit most from inlining.

#### ○ Moderate (20-100 statements)
```
pub fn calculate_interest_rate() -> u256
  Complexity: ○ Moderate
  Nodes:      25
  Statements: 25
  Max Depth:  2
```
**Generally avoid inlining** unless performance-critical. Consider refactoring if needed.

#### ✗ Complex (> 100 statements)
```
pub fn rebalance_portfolio() -> bool
  Complexity: ✗ Complex
  Nodes:      120
  Statements: 120
  Max Depth:  7
  💡 Recommendation: Consider breaking into smaller functions
```
**Never inline these!** If marked as `inline`, the compiler will warn you. Consider refactoring into smaller functions.

## Practical Examples

### Example 1: Validation Helpers

```ora
contract Token {
    storage var balances: map[address, u256];
    
    // Inline validation - called frequently, very simple
    inline fn is_valid_amount(amount: u256) -> bool
        requires(amount > 0)
    {
        return amount <= std.constants.U256_MAX;
    }
    
    inline fn has_sufficient_balance(user: address, amount: u256) -> bool {
        return balances[user] >= amount;
    }
    
    pub fn transfer(to: address, amount: u256) -> bool {
        if (!is_valid_amount(amount)) {
            return false;
        }
        
        if (!has_sufficient_balance(std.transaction.sender, amount)) {
            return false;
        }
        
        balances[std.transaction.sender] -= amount;
        balances[to] += amount;
        return true;
    }
}
```

### Example 2: Mathematical Utilities

```ora
contract DeFi {
    storage var base_rate: u256;
    
    // Simple math - good for inline
    inline fn calculate_percentage(amount: u256, bps: u256) -> u256 {
        return (amount * bps) / 10000;
    }
    
    inline fn min(a: u256, b: u256) -> u256 {
        if (a < b) {
            return a;
        }
        return b;
    }
    
    inline fn max(a: u256, b: u256) -> u256 {
        if (a > b) {
            return a;
        }
        return b;
    }
    
    pub fn calculate_fee(amount: u256) -> u256 {
        let fee_bps: u256 = base_rate;
        return calculate_percentage(amount, fee_bps);
    }
}
```

### Example 3: Access Control

```ora
contract Vault {
    storage var owner: address;
    storage var admins: map[address, bool];
    
    // Inline access checks
    inline fn is_owner() -> bool {
        return std.transaction.sender == owner;
    }
    
    inline fn is_admin() -> bool {
        return admins[std.transaction.sender];
    }
    
    inline fn is_authorized() -> bool {
        return is_owner() || is_admin();
    }
    
    pub fn sensitive_operation() -> bool
        requires(is_authorized())
    {
        // Protected operation
        return true;
    }
}
```

## How Switch Statements Affect Complexity

Ora's analyzer understands switch statements and counts them intelligently:

### Simple Switch Expression (1 statement)
```ora
// Counted as: 1 statement
inline fn get_discount(tier: UserTier) -> u256 {
    return switch (tier) {
        UserTier.Basic => 0,
        UserTier.Silver => 10,
        UserTier.Gold => 25,
        UserTier.Platinum => 50,
    };
}
```
**Analysis Output:**
```
inline fn get_discount()
  Complexity: ✓ Simple
  Nodes:      1
  Statements: 1
  Max Depth:  1
```

### Switch Statement with Blocks (counts each statement)
```ora
// Counted as: 1 (switch) + statements in each block
fn process_operation(op: OperationType, amount: u256) -> u256 {
    var result: u256 = amount;
    
    switch (op) {
        OperationType.Deposit => {
            let bonus: u256 = amount / 100;
            result = amount + bonus;
        },
        OperationType.Withdraw => {
            result = amount;
        },
    }
    
    return result;
}
```
**Analysis Output:**
```
fn process_operation()
  Complexity: ✓ Simple
  Nodes:      7
  Statements: 7
  Max Depth:  2
```

## Best Practices

### 1. Analyze Before Marking Inline

Always run complexity analysis before adding `inline`:

```bash
# Analyze your contract
ora --analyze-complexity contract.ora

# Look for functions marked "✓ Simple"
# These are good inline candidates
```

### 2. Start Small

Begin by inlining only the simplest functions:
- Single statement returns
- Basic validation checks
- Simple arithmetic

### 3. Measure Impact

In performance-critical code:
1. Profile without `inline`
2. Add `inline` to hot paths
3. Measure the difference
4. Keep changes that improve performance

### 4. Listen to Warnings

If the compiler warns about inline complexity, **remove the inline keyword** or refactor:

```ora
// Compiler warning: "Function 'complex_calc' is marked inline but has high complexity"
inline fn complex_calc() -> u256 {  // ❌ 120 statements
    // Refactor into smaller functions!
}
```

### 5. Profile in Context

What's inline-worthy depends on usage:
- **Called once** → Don't inline
- **Called in tight loops** → Consider inlining
- **Called from multiple places** → Inline if simple

## Real-World Example

See our complete DeFi lending pool example demonstrating inline usage:

```bash
ora --analyze-complexity ora-example/defi_lending_pool.ora
```

This 700+ line contract shows:
- ✓ **76% simple functions** - optimal for inline candidates
- ○ **19% moderate** - well-structured, not for inline
- ✗ **4% complex** - needs refactoring

Example functions from the contract:

```ora
// ✓ Simple: Perfect for inline (1 statement)
inline fn only_owner() -> bool {
    return std.transaction.sender == owner;
}

// ✓ Simple: Good for inline (2 statements)
inline fn when_not_paused() -> bool {
    return !is_paused;
}

// ○ Moderate: Don't inline (22 statements)
pub fn borrow(amount: u256) -> bool {
    // Complex business logic
    // Better as a regular function
}

// ✗ Complex: Never inline! (120 statements)
pub fn rebalance_portfolio() -> bool {
    // Way too complex
    // Needs refactoring into smaller functions
}
```

## Compiler Behavior

### Current Implementation (ASUKA)

In the current ASUKA release:
- `inline` is recognized and stored in the AST
- Complexity analysis warns about complex inline functions
- MLIR attribute is set for future optimization passes

### Future Implementation

Post-ASUKA, the compiler will:
- Automatically inline simple functions at MLIR level
- Apply cost-benefit analysis for inlining decisions
- Generate optimized bytecode based on inline hints
- Provide detailed gas cost comparisons

## Summary

| Complexity | Statement Count | Inline? | Example |
|------------|----------------|---------|---------|
| ✓ Simple | < 20 | ✅ Yes | Getters, validators, simple math |
| ○ Moderate | 20-100 | ⚠️ Rarely | Business logic, calculations |
| ✗ Complex | > 100 | ❌ Never | Multi-step algorithms, complex flows |

**Key Takeaway:** Use complexity analysis to guide your inline decisions. When in doubt, profile and measure!

## See Also

- [Complexity Analysis CLI](../CLI_GUIDE.md#analyze-complexity) - Full CLI reference
- [Performance Optimization](./performance.md) - General optimization tips
- [Examples](../examples.md) - More code examples

