---
sidebar_position: 4
---

# Examples

Explore real Ora smart contracts and language features from the repository.

## Basic Examples

### Simple Storage Contract

From `examples/core/simple_storage_test.ora`:

```ora
contract SimpleTest {
    storage const name: string;
    storage var balance: u256;
    
    pub fn init() {
        name = "TestToken";
        balance = 1000;
    }
    
    pub fn getName() -> string {
        return name;
    }
}
```

### Control Flow Demo

From `examples/core/control_flow_test.ora`:

```ora
contract ControlFlowTest {
    storage var counter: u256;
    
    pub fn init() {
        counter = 0;
    }
    
    pub fn testIfElse(value: u256) -> bool {
        if (value > 10) {
            counter += 1;
            return true;
        } else {
            counter = 0;
            return false;
        }
    }
    
    pub fn testWhileLoop(limit: u256) {
        var i: u256 = 0;
        while (i < limit) {
            counter += 1;
            i += 1;
        }
    }
    
    pub fn testBreakContinue(items: slice[u256]) {
        for (items) |item| {
            if (item == 0) {
                continue;
            }
            if (item > 100) {
                break;
            }
            counter += item;
        }
    }
}
```

## Token Contract

From `examples/tokens/simple_token_basic.ora`:

```ora
contract SimpleToken {
    // Storage state
    storage var totalSupply: u256;
    storage var balances: map[address, u256];
    
    // Events
    log Transfer(from: address, to: address, amount: u256);
    
    // Constructor
    pub fn init(_supply: u256) {
        totalSupply = _supply;
        balances[std.transaction.sender] = _supply;
    }
    
    // Balance query
    pub fn balanceOf(owner: address) -> u256 {
        return balances[owner];
    }
}
```

## Advanced Error Handling

From `examples/advanced/error_union_demo.ora`:

```ora
contract ErrorUnionDemo {
    // Error declarations
    error InsufficientBalance;
    error InvalidAddress;
    error TransferFailed;
    error AccessDenied;
    error AmountTooLarge;

    // Storage variables
    storage balances: map[address, u256];
    storage owner: address;

         // Transfer function with error union return
     fn transfer(to: address, amount: u256) -> !u256 {
         // Check for valid address
         if (to == std.constants.ZERO_ADDRESS) {
             return error.InvalidAddress;
         }

        // Check amount limit
        if (amount > 500000) {
            return error.AmountTooLarge;
        }

        // Get current balance (with error handling)
        let balance_result = try getBalance(std.transaction.sender);
        
        // Use try-catch for error handling
        try {
            let current_balance = balance_result;
            
            // Check sufficient balance
            if (current_balance < amount) {
                return error.InsufficientBalance;
            }
            
            // Update balances
            balances[std.transaction.sender] = current_balance - amount;
            balances[to] = balances[to] + amount;
            
            // Return new balance
            return balances[std.transaction.sender];
        } catch(e) {
            // Propagate error
            return error.TransferFailed;
        }
    }

         // Get balance function with error union return
     fn getBalance(account: address) -> !u256 {
         if (account == std.constants.ZERO_ADDRESS) {
             return error.InvalidAddress;
         }
        
        return balances[account];
    }

    // Batch transfer with error union handling
    fn batchTransfer(recipients: slice[address], amounts: slice[u256]) -> !u256 {
        // Check input lengths match
        if (recipients.len != amounts.len) {
            return error.InvalidAddress;
        }
        
        let total_transferred: u256 = 0;
        
        // Process each transfer
        for (i in 0..recipients.len) {
            let transfer_result = try transfer(recipients[i], amounts[i]);
            
            // Handle individual transfer results
            try {
                let new_balance = transfer_result;
                total_transferred = total_transferred + amounts[i];
            } catch(transfer_error) {
                // If any transfer fails, return error
                return error.TransferFailed;
            }
        }
        
        return total_transferred;
    }
}
```

## Mathematical Operations

From `examples/demos/division_test.ora`:

```ora
contract DivisionDemo {
    error DivisionByZero;
    error InexactDivision;
    
    function demonstrateDivision() public {
        let a: u32 = 10;
        let b: u32 = 3;
        
        // Truncating division (toward zero)
        let trunc_result = @divTrunc(a, b);  // 3
        
        // Floor division (toward negative infinity)
        let floor_result = @divFloor(a, b);  // 3
        
        // Ceiling division (toward positive infinity)
        let ceil_result = @divCeil(a, b);    // 4
        
        // Exact division (errors if remainder != 0)
        let exact_result = @divExact(12, 4); // 3
        
        // Division with remainder (returns tuple)
        let (quotient, remainder) = @divmod(a, b);  // (3, 1)
    }
    
    function testSignedDivision() public {
        let a: i32 = -7;
        let b: i32 = 3;
        
        // Different rounding behaviors for negative numbers
        let trunc_result = @divTrunc(a, b);  // -2 (toward zero)
        let floor_result = @divFloor(a, b);  // -3 (toward -∞)
        let ceil_result = @divCeil(a, b);    // -2 (toward +∞)
    }
    
    function testErrorHandling() public {
        try {
            let result = @divExact(10, 3);  // Will error - not exact
        } catch (error.InexactDivision) {
            log("Division is not exact");
        }
        
        try {
            let result = @divTrunc(10, 0);  // Will error - division by zero
        } catch (error.DivisionByZero) {
            log("Cannot divide by zero");
        }
    }
    
    function safeDivide(a: u32, b: u32) -> !u32 {
        if (b == 0) {
            return error.DivisionByZero;
        }
        return @divTrunc(a, b);
    }
}
```

## Formal Verification

From `examples/advanced/formal_verification_test.ora`:

```ora
contract MathematicalProofs {
    // Storage for mathematical operations
    storage values: u256[];
    
    // Complex mathematical invariant with quantifiers
    invariant forall i: u256 where i < values.length => values[i] > 0;
    invariant exists j: u256 where j < values.length && values[j] % 2 == 0;
    
    // Function demonstrating complex preconditions and postconditions
    function fibonacci(n: u256) -> u256 
        requires n >= 0 && n < 100  // Prevent overflow
        ensures result >= 0
        ensures n <= 1 || result == fibonacci(n-1) + fibonacci(n-2)
        ensures n >= 2 => result > fibonacci(n-1) && result > fibonacci(n-2)
    {
        if (n <= 1) {
            return n;
        }
        
        let prev1 = fibonacci(n - 1);
        let prev2 = fibonacci(n - 2);
        
        invariant prev1 >= 0 && prev2 >= 0;
        invariant prev1 + prev2 >= prev1 && prev1 + prev2 >= prev2;
        
        return prev1 + prev2;
    }
    
    // Function with complex mathematical conditions
    function gcd(a: u256, b: u256) -> u256
        requires a > 0 && b > 0
        ensures result > 0
        ensures a % result == 0 && b % result == 0
        ensures forall d: u256 where d > 0 && a % d == 0 && b % d == 0 => d <= result
    {
        if (b == 0) {
            return a;
        }
        
        invariant a > 0 && b > 0;
        invariant gcd(a, b) == gcd(b, a % b);
        
        return gcd(b, a % b);
    }
}
```

## Voting System with Formal Verification

From `examples/advanced/formal_verification_test.ora`:

```ora
contract VotingSystem {
    storage proposals: map[u256, Proposal];
    storage voters: map[address, Voter];
    storage proposal_count: u256;
    
    struct Proposal {
        description: string;
        vote_count: u256;
        deadline: u256;
        executed: bool;
    }
    
    struct Voter {
        has_voted: map[u256, bool];
        voting_power: u256;
    }
    
    // Complex invariants for voting system
    invariant forall p: u256 where p < proposal_count => 
        proposals[p].vote_count <= totalVotingPower();
    invariant forall p: u256 where p < proposal_count => 
        proposals[p].executed => proposals[p].vote_count > totalVotingPower() / 2;
    
    // Function demonstrating complex voting logic verification
    function vote(proposal_id: u256, support: bool) -> bool
        requires proposal_id < proposal_count
        requires !voters[std.transaction.sender].has_voted[proposal_id]
        requires std.block.timestamp < proposals[proposal_id].deadline
        requires !proposals[proposal_id].executed
        requires voters[std.transaction.sender].voting_power > 0
        ensures result == true => voters[std.transaction.sender].has_voted[proposal_id]
        ensures result == true => 
            support => proposals[proposal_id].vote_count == old(proposals[proposal_id].vote_count) + voters[std.transaction.sender].voting_power
        ensures result == false => proposals[proposal_id].vote_count == old(proposals[proposal_id].vote_count)
    {
        let proposal = proposals[proposal_id];
        let voter = voters[std.transaction.sender];
        
        if (voter.has_voted[proposal_id]) {
            return false;
        }
        
        if (std.block.timestamp >= proposal.deadline) {
            return false;
        }
        
        if (proposal.executed) {
            return false;
        }
        
        invariant voter.voting_power > 0;
        invariant !voter.has_voted[proposal_id];
        invariant proposal.vote_count <= totalVotingPower();
        
        voters[std.transaction.sender].has_voted[proposal_id] = true;
        
        if (support) {
            proposals[proposal_id].vote_count += voter.voting_power;
        }
        
        return true;
    }
    
    // Helper function
    function totalVotingPower() -> u256 {
        return 10000; // Placeholder implementation
    }
}
```

## Compile-Time Evaluation

Ora emphasizes compile-time computation for optimal performance:

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
        let complex_calc: u256 = (100 + 50) * 2 - 25;
    }
    
    // Compile-time bitwise operations
    storage const BITWISE_RESULT: u256 = 0xFF & 0x0F;
    storage const SHIFT_RESULT: u256 = 8 << 2;
}
```

## Key Features Demonstrated

These examples showcase Ora's unique features:

### 1. **Error Unions (`!T`)**
- Explicit error handling with `!T` return types
- `try` and `catch` blocks for error management
- Multiple error types in single function

### 2. **Memory Regions**
- `storage var` for mutable persistent state
- `storage const` for immutable persistent state
- `immutable` for deployment-time constants

### 3. **Formal Verification**
- `requires` and `ensures` clauses for function contracts
- `invariant` statements for loop and contract invariants
- Mathematical quantifiers (`forall`, `exists`)

### 4. **Compile-Time Evaluation**
- 90% of computation happens at compile time
- Constant folding and expression evaluation
- Optimal gas usage through pre-computation

### 5. **Modern Syntax**
- Clean, readable code structure
- Explicit type annotations
- Zig-inspired language design

## Building and Running Examples

All examples are included in the repository. To build and test them:

```bash
# Build the compiler
zig build

# Run parser demo
zig build parser-demo

# Run optimization demo
zig build optimization-demo

# Run formal verification demo
zig build formal-verification-demo
```

## Next Steps

- **Study the Repository**: Browse the full `/examples` directory for more patterns
- **Language Reference**: Check [Language Basics](./language-basics) for complete syntax
- **Get Started**: Follow the [Getting Started](./getting-started) guide to build your own contracts
- **Documentation**: Read the [Formal Verification Guide](https://github.com/oralang/Ora/blob/main/formal-verification.md) and [Syntax Guide](https://github.com/oralang/Ora/blob/main/syntax-guide.md)

## Community

Join our community to discuss these examples and share your own:
- [GitHub Discussions](https://github.com/oralang/Ora/discussions)
- [Issues](https://github.com/oralang/Ora/issues) 