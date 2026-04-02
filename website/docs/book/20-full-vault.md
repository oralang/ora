---
title: "Chapter 20: Putting It All Together"
description: The complete vault contract using every feature from the book.
sidebar_position: 20
---

# Putting It All Together

Over 19 chapters, we built a vault contract feature by feature. Here's the complete version, using:

- Storage variables and maps (Ch. 2)
- Functions with return types (Ch. 3)
- Control flow (Ch. 4)
- Structs and enums (Ch. 5)
- Error unions (Ch. 6)
- Explicit memory regions (Ch. 7)
- Refinement types (Ch. 8)
- Logs (Ch. 9)
- Specification clauses (Ch. 10)
- Ghost state (Ch. 11)
- Locks (Ch. 12)
- Standard library (Ch. 13)

## The complete vault

```ora
error InsufficientBalance(required: u256, available: u256);

comptime const std = @import("std");

contract Vault {
    // Storage state (Ch. 2, 7)
    storage var totalDeposits: u256 = 0;
    storage var balances: map<address, u256>;
    storage var reentrancyGuard: u256 = 0;

    // Ghost state for verification (Ch. 11)
    ghost storage spec_sum: u256 = 0;

    // Events (Ch. 9)
    log Deposit(account: address, amount: u256);
    log Withdrawal(account: address, amount: u256);

    // Deposit with all safety features
    pub fn deposit(amount: MinValue<u256, 1>)                              // Refinement type (Ch. 8)
        requires(amount > 0)                                                // Precondition (Ch. 10)
        requires(totalDeposits <= std.constants.U256_MAX - amount)          // Overflow guard (Ch. 13)
        requires(balances[std.msg.sender()] <= std.constants.U256_MAX - amount)
        ensures(totalDeposits == old(totalDeposits) + amount)               // Postcondition (Ch. 10)
        ensures(balances[std.msg.sender()] == old(balances[std.msg.sender()]) + amount)
        ensures(spec_sum == old(spec_sum) + amount)                         // Ghost invariant (Ch. 11)
    {
        let sender: NonZeroAddress = std.msg.sender();                      // Refinement (Ch. 8)
        @lock(reentrancyGuard);                                             // Reentrancy lock (Ch. 12)
        balances[sender] += amount;                                         // Compound assignment (Ch. 3)
        totalDeposits += amount;
        spec_sum += amount;                                                 // Ghost update (Ch. 11)
        @unlock(reentrancyGuard);
        log Deposit(sender, amount);                                        // Event (Ch. 9)
    }

    // Withdraw with error handling
    pub fn withdraw(amount: MinValue<u256, 1>) -> !bool | InsufficientBalance  // Error union (Ch. 6)
        requires(amount > 0)
        requires(balances[std.msg.sender()] >= amount)
        ensures(totalDeposits == old(totalDeposits) - amount)
        ensures(balances[std.msg.sender()] == old(balances[std.msg.sender()]) - amount)
        ensures(spec_sum == old(spec_sum) - amount)
    {
        let sender: NonZeroAddress = std.msg.sender();
        @lock(reentrancyGuard);
        let current: u256 = balances[sender];                               // Storage → memory (Ch. 7)
        if (current < amount) {                                             // Control flow (Ch. 4)
            @unlock(reentrancyGuard);
            return InsufficientBalance(amount, current);                    // Error return (Ch. 6)
        }
        balances[sender] = current - amount;
        totalDeposits -= amount;
        spec_sum -= amount;
        @unlock(reentrancyGuard);
        log Withdrawal(sender, amount);
        return true;
    }

    pub fn balanceOf(account: address) -> u256 {
        return balances[account];
    }

    pub fn getTotalDeposits() -> u256 {
        return totalDeposits;
    }
}
```

## What this contract guarantees

The compiler and verifier enforce:

1. **No zero deposits or withdrawals** — `MinValue<u256, 1>` makes them type errors
2. **No zero-address interactions** — `NonZeroAddress` on sender
3. **No overflow** — `requires(totalDeposits <= U256_MAX - amount)` prevents arithmetic overflow
4. **Correct accounting** — `ensures(totalDeposits == old(totalDeposits) + amount)` verified by Z3
5. **Conservation** — `spec_sum` tracks total flow; verified to match `totalDeposits`
6. **No reentrancy** — `@lock`/`@unlock` on `reentrancyGuard`
7. **Explicit errors** — `InsufficientBalance` with context, not a silent `false`
8. **Auditability** — `Deposit`/`Withdrawal` events for off-chain tracking

## What Ora adds over a conventional approach

The equivalent contract in a language without these features would use runtime `require` checks instead of type-level refinements, modifier patterns instead of compiler-checked locks, and manual auditing instead of machine-verified postconditions.

The Ora version makes five things *compiler-checked* that would otherwise be convention:

| Guarantee | How Ora enforces it | Conventional alternative |
|-----------|--------------------|-----------------------|
| No zero amounts | `MinValue<u256, 1>` type | `require(amount > 0)` per function |
| Correct accounting | `ensures(total == old(total) + amount)` | Manual audit |
| Conservation proof | `ghost storage spec_sum` | Not expressible |
| No reentrancy | `@lock`/`@unlock` with compiler checking | Modifier pattern |
| Overflow safety | `requires(total <= U256_MAX - amount)` | Implicit runtime check |

## Where to go from here

- **Reference docs** — each feature has a dedicated reference page linked from its chapter
- **Formal specification** — [`docs/formal-specs/ora-2.md`](https://github.com/oralang/Ora/blob/main/docs/formal-specs/ora-2.md) for the type system calculus
- **Compiler Field Guide** — [Field Guide](../compiler/field-guide) for contributing to the compiler
- **Examples** — `ora-example/` in the repository for more contract patterns
