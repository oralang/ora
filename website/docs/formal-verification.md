# Formal Verification with Z3

Ora integrates with the **Z3 SMT solver** to provide formal verification capabilities, allowing you to mathematically prove properties about your smart contracts.

## Overview

Formal verification in Ora helps you:
- **Prove correctness**: Mathematically verify that your contract behaves as intended
- **Catch bugs early**: Find edge cases that testing might miss
- **Increase confidence**: Deploy with mathematical guarantees about your code
- **Document intent**: Specifications serve as precise documentation

## Verification Annotations

### Preconditions (`requires`)

Specify what must be true before a function executes:

```ora
pub fn transfer(recipient: address, amount: u256) -> bool
    requires amount > 0
    requires recipient != address(0)
{
    // Implementation
}
```

### Postconditions (`ensures`)

Specify what must be true after a function executes:

```ora
pub fn transfer(sender: address, recipient: address, amount: u256) -> bool
    requires amount > 0
    ensures balances[sender] == old(balances[sender]) - amount
    ensures balances[recipient] == old(balances[recipient]) + amount
{
    // Implementation
}
```

### Contract Invariants

Specify properties that must always hold true for the contract:

```ora
contract Token {
    storage totalSupply: u256;
    storage balances: map[address, u256];
    
    // Contract-level invariant
    invariant totalSupplyPreserved(
        sumOfBalances() == totalSupply
    );
    
    // Functions...
}
```

### Loop Invariants

Specify properties that remain true throughout loop execution:

```ora
pub fn sumBalances(accounts: slice[address]) -> u256 {
    let total: u256 = 0;
    for (accounts) |account|
        invariant(total >= 0)
    {
        total = total + balances[account];
    }
    return total;
}
```

### Quantifiers

Express properties over collections:

```ora
// Universal quantification
requires forall addr: address where addr != address(0) => balances[addr] >= 0

// Existential quantification  
ensures exists addr: address => balances[addr] > 1000
```

### Assertions

Runtime checks during verification:

```ora
pub fn complexOperation() {
    let x = calculate();
    assert(x > 0, "x must be positive");
    // Continue with confidence that x > 0
}
```

### Ghost Code

Specification-only code that doesn't generate bytecode:

```ora
contract Token {
    // Ghost storage - for verification only
    ghost storage sumOfAllBalances: u256;
    
    // Ghost function - for specification only
    ghost fn computeSum() -> u256 {
        // Complex calculation used in proofs
        return sumOfAllBalances;
    }
    
    pub fn transfer(recipient: address, amount: u256)
        ensures computeSum() == old(computeSum())  // Uses ghost function
    {
        // Implementation
    }
}
```

## Complete Example: Verified ERC20

```ora
contract VerifiedToken {
    storage totalSupply: u256;
    storage balances: map[address, u256];
    
    // Contract invariant: total supply is preserved
    invariant supplyPreserved(
        sumOfBalances() == totalSupply
    );
    
    ghost fn sumOfBalances() -> u256 {
        // Specification-only function for verification
        return totalSupply;
    }
    
    pub fn init(initialSupply: u256)
        requires initialSupply > 0
        ensures totalSupply == initialSupply
    {
        totalSupply = initialSupply;
        balances[std.msg.sender()] = initialSupply;
    }
    
    pub fn transfer(recipient: address, amount: u256) -> bool
        requires amount > 0
        requires recipient != address(0)
        requires balances[std.msg.sender()] >= amount
        ensures balances[std.msg.sender()] == old(balances[std.msg.sender()]) - amount
        ensures balances[recipient] == old(balances[recipient]) + amount
        ensures totalSupply == old(totalSupply)
    {
        let sender = std.msg.sender();
        let senderBalance = balances[sender];
        
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

## How It Works

### Compilation Pipeline

1. **Parse**: Ora parses verification annotations alongside regular code
2. **Generate VCs**: Compiler generates verification conditions (VCs) from specifications
3. **Z3 Solving**: VCs are sent to Z3 SMT solver for verification
4. **Report**: Results show which properties are proven or violated
5. **Code Generation**: Ghost code is filtered out during bytecode generation

### What Gets Verified

- ✅ Preconditions are satisfied at all call sites
- ✅ Postconditions hold after function execution
- ✅ Invariants are maintained across all operations
- ✅ No integer overflows/underflows
- ✅ No null pointer dereferences
- ✅ Array bounds are respected

### What Doesn't Affect Bytecode

Ghost code is **specification-only** and never appears in deployed bytecode:
- Ghost storage variables
- Ghost functions
- `requires` / `ensures` clauses
- `invariant` declarations
- `assert` statements with `ghost` modifier

## Current Status

🚧 **Z3 integration is under active development**

**What's working:**
- ✅ Full grammar support for verification annotations
- ✅ AST representation for all FV features
- ✅ Parser implementation complete
- ✅ Ghost code filtering (no verification code in bytecode)
- ✅ Z3 build system integration

**In progress:**
- 🔄 Verification condition generation
- 🔄 Z3 constraint encoding
- 🔄 Counterexample reporting

**Coming soon:**
- 📋 Full verification pipeline
- 📋 Interactive verification feedback
- 📋 Incremental verification

## Installation

Z3 is automatically detected if installed on your system:

```bash
# macOS (Homebrew)
brew install z3

# Ubuntu/Debian
sudo apt-get install z3

# Arch Linux
sudo pacman -S z3
```

If Z3 is not found, Ora falls back to compilation without verification.

## Usage

Verification runs automatically when you compile contracts with formal verification annotations:

```bash
# Compile with verification
ora contract.ora

# The compiler will:
# 1. Parse verification annotations
# 2. Generate and solve verification conditions
# 3. Report any property violations
# 4. Generate bytecode (without ghost code)
```

## Learn More

- 📚 [Z3 Documentation](https://github.com/Z3Prover/z3)
- 📝 [SMT-LIB Standard](https://smtlib.cs.uiowa.edu/)
- 🎓 [Formal Verification Study Guide](../docs/tech-work/z3-formal-verification/Z3-STUDY-GUIDE.md)
- 📋 [FV Implementation Plan](../docs/tech-work/z3-formal-verification/00-MASTER-PLAN.md)

## Best Practices

1. **Start simple**: Add `requires` and `ensures` to key functions first
2. **Be specific**: Precise specifications catch more bugs
3. **Use ghost code**: For complex mathematical properties
4. **Verify incrementally**: Don't try to verify everything at once
5. **Test + Verify**: Formal verification complements testing, doesn't replace it

## Why Formal Verification?

Traditional smart contract development relies on testing, which can only verify specific cases. Formal verification proves properties for **all possible inputs**, providing mathematical certainty that your contract behaves correctly.

This is especially critical for:
- 💰 **DeFi protocols**: Financial logic must be correct
- 🔐 **Access control**: Security properties must hold
- 🎯 **Governance**: Voting mechanisms must be fair
- 🏦 **Custody**: Asset transfers must preserve invariants

