# State Analysis

Ora's compiler includes sophisticated **state change tracking** that automatically analyzes storage access patterns in your smart contracts, helping you write more efficient, secure, and maintainable code.

## Overview

State analysis runs automatically during compilation and provides insights about:
- 📊 **Storage reads and writes**: Which functions access which variables
- 🔍 **Function properties**: Stateless, readonly, or state-modifying
- ⚠️ **Potential issues**: Dead stores, missing validation, unused variables
- ⚡ **Gas optimization**: Identify expensive storage operations

## Automatic Analysis

Every compilation includes state analysis with **zero overhead**:

```bash
ora contract.ora

⚠️  State Analysis Warnings for MyContract (2):

💡 [DeadStore] Storage variable 'unusedData' is written by 'setData' 
               but never read by any function in the contract
   💬 Remove unused storage variable or add read logic

ℹ️  [MissingCheck] Function 'approve' modifies storage without reading any state
   💬 Add validation checks before modifying storage (e.g., require conditions)
```

**Clean output**: Only warnings are shown. No noise during compilation!

## Detailed Analysis Mode

For comprehensive analysis, use the `--analyze-state` flag:

```bash
ora --analyze-state contract.ora
```

This shows complete details for every function:

```
=== State Analysis: SimpleToken ===

Function: transfer
├─ Stateless: false
├─ Readonly: false
├─ Modifies State: true
├─ Reads: balances
└─ Writes: balances

Function: balanceOf
├─ Stateless: true
├─ Readonly: true
├─ Modifies State: false
├─ Reads: balances
└─ Writes: (none)

Function: getTotalSupply
├─ Stateless: false
├─ Readonly: true
├─ Modifies State: false
├─ Reads: totalSupply
└─ Writes: (none)
```

## Function Properties

### Stateless Functions

Pure computation with no storage access:

```ora
pub fn add(a: u256, b: u256) -> u256 {
    return a + b;  // ✅ Stateless: No storage access
}
```

**Benefits**:
- Can be inlined for gas savings
- Safe to call from any context
- Predictable behavior

### Readonly Functions

Only read storage, never write:

```ora
pub fn balanceOf(account: address) -> u256 {
    return balances[account];  // ✅ Readonly: Only reads
}
```

**Benefits**:
- Safe for view-like queries
- Can be called off-chain
- No state changes

### State-Modifying Functions

Write to storage:

```ora
pub fn transfer(recipient: address, amount: u256) {
    balances[std.msg.sender()] -= amount;  // ⚠️ Modifies state
    balances[recipient] += amount;
}
```

**Note**: The compiler checks if validation is missing!

## Warning Types

### 1. Dead Store (⚠️ Warning)

Storage variable written but never read by **any function**:

```ora
contract Example {
    storage unusedData: u256;  // ⚠️ Problem!
    
    pub fn writeData(value: u256) {
        unusedData = value;  // Written but never read
    }
    
    // No function reads unusedData!
}
```

**Warning**:
```
⚠️  [DeadStore] Storage variable 'unusedData' is written by 'writeData' 
               but never read by any function in the contract
   💬 Remove unused storage variable or add read logic in another function
```

**Fix**: Either use it or remove it!

### 2. Missing Check (ℹ️ Info)

Function modifies storage without validation:

```ora
pub fn approve(spender: address, amount: u256) {
    allowances[std.msg.sender()][spender] = amount;  // ℹ️ No validation!
}
```

**Warning**:
```
ℹ️  [MissingCheck] Function 'approve' modifies storage without reading any state
   💬 Add validation checks before modifying storage (e.g., require conditions)
```

**Good practice**:
```ora
pub fn approve(spender: address, amount: u256) {
    require(spender != address(0), "Invalid spender");  // ✅ Validation added
    require(amount > 0, "Amount must be positive");
    allowances[std.msg.sender()][spender] = amount;
}
```

### 3. Unvalidated Constructor Parameter (ℹ️ Info)

Constructor stores parameters without validation:

```ora
pub fn init(initialSupply: u256) {
    totalSupply = initialSupply;  // ℹ️ No validation!
}
```

**Warning**:
```
ℹ️  [UnvalidatedConstructorParam] Constructor 'init' stores parameters without validation
   💬 Add validation for constructor parameters (e.g., require amount > 0, address != 0)
```

**Good practice**:
```ora
pub fn init(initialSupply: u256) {
    require(initialSupply > 0, "Supply must be positive");  // ✅ Validation
    totalSupply = initialSupply;
}
```

## Real-World Example: ERC20

Here's what state analysis shows for an ERC20 token:

```ora
contract SimpleToken {
    storage totalSupply: u256;
    storage balances: map[address, u256];
    storage allowances: doublemap[address, address, u256];
    
    pub fn init(initialSupply: u256) {
        totalSupply = initialSupply;
        balances[std.msg.sender()] = initialSupply;
    }
    
    pub fn balanceOf(account: address) -> u256 {
        return balances[account];
    }
    
    pub fn transfer(recipient: address, amount: u256) -> bool {
        let sender = std.msg.sender();
        let senderBalance = balances[sender];
        
        if (senderBalance < amount) {
            return false;
        }
        
        balances[sender] = senderBalance - amount;
        balances[recipient] = balances[recipient] + amount;
        
        return true;
    }
    
    pub fn approve(spender: address, amount: u256) -> bool {
        allowances[std.msg.sender()][spender] = amount;
        return true;
    }
}
```

**Analysis Results**:

```
Function: balanceOf
├─ Stateless: false
├─ Readonly: true        ✅ Only reads, never writes
├─ Reads: balances
└─ Writes: (none)

Function: transfer
├─ Stateless: false
├─ Readonly: false
├─ Modifies State: true
├─ Reads: balances       ✅ Reads before writing (good!)
└─ Writes: balances

Function: approve
├─ Stateless: false
├─ Readonly: false
├─ Modifies State: true
├─ Reads: (none)         ⚠️ No validation checks
└─ Writes: allowances

Warnings:
ℹ️  [MissingCheck] Function 'approve' modifies storage without reading
```

## How It Works

### Contract-Level Analysis

State analysis tracks reads and writes **across all functions**:

```ora
contract Example {
    storage counter: u256;
    
    pub fn increment() {
        counter += 1;        // Writes to 'counter'
    }
    
    pub fn getCounter() -> u256 {
        return counter;      // Reads from 'counter'
    }
}
```

**Result**: ✅ No dead store warning because `getCounter` reads what `increment` writes.

### Smart Detection

The analyzer understands:
- Direct access: `counter = 5`
- Map access: `balances[addr] = 100`
- Compound assignments: `counter += 1`
- Multiple operations: Reading then writing same variable

### False Positive Prevention

**Old approach** (function-level):
```
❌ Warning: 'approve' writes 'allowances' but doesn't read it
   (False positive! transferFrom reads it!)
```

**New approach** (contract-level):
```
✅ No warning: 'allowances' is written by approve() and read by transferFrom()
```

## Benefits

### For Developers
- 🔍 **Instant insights**: See what each function does to storage
- 🐛 **Catch bugs early**: Find unused variables before deployment
- 📖 **Self-documenting**: Function behavior is explicit
- ⚡ **Gas awareness**: Identify expensive operations

### For Security
- 🔐 **Audit trail**: Know exactly what modifies state
- 🎯 **Attack surface**: Identify state-changing entry points
- ✅ **Missing checks**: Detect unvalidated writes
- 🗑️ **Dead stores**: Find unused or redundant operations

### For Optimization
- 💰 **Gas savings**: Remove unnecessary storage writes
- 📦 **Caching**: Identify readonly functions
- 🎯 **Inlining**: Mark stateless functions for optimization

## Best Practices

1. **Review warnings**: Don't ignore state analysis output
2. **Validate inputs**: Add checks before storage writes
3. **Remove dead stores**: Clean up unused variables
4. **Use readonly**: Make functions readonly when possible
5. **Keep it simple**: Fewer storage operations = lower gas costs

## Advanced Usage

### CI/CD Integration

Fail builds on warnings:

```bash
ora --analyze-state contract.ora | grep "⚠️" && exit 1
```

### Comparison Across Versions

```bash
# Before changes
ora --analyze-state contract.ora > before.txt

# After changes  
ora --analyze-state contract.ora > after.txt

# Compare
diff before.txt after.txt
```

## Technical Details

**Zero Runtime Overhead**:
- Analysis runs after parsing, before MLIR generation
- No impact on compilation speed
- No changes to generated bytecode

**Comprehensive Tracking**:
- All storage operations (SLOAD/SSTORE)
- Direct variable access
- Map and array operations
- Nested structures

**Smart Heuristics**:
- Identifies storage variables by declaration
- Tracks reads and writes separately
- Handles complex expressions correctly

## Why This Matters

Smart contracts interact with expensive storage operations. State analysis helps you:

1. **Avoid waste**: Don't store values you never read
2. **Add validation**: Check before you write
3. **Optimize gas**: Minimize storage operations
4. **Improve security**: Validate all state changes
5. **Better code**: Clear understanding of side effects

State analysis is **always on**, helping you write better Ora contracts automatically! 🚀

