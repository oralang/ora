---
title: "Chapter 8: Refinement Types"
description: Types that carry constraints — NonZero, MinValue, InRange, and more.
sidebar_position: 8
---

# Refinement Types

A refinement type is a base type with a constraint. Instead of checking `require(amount > 0)` at runtime, you declare `amount: MinValue<u256, 1>` — the compiler enforces the constraint and can eliminate the runtime check entirely when it can prove the value satisfies it.

## Built-in refinements

Short examples in this section are inside a function.

### NonZero and NonZeroAddress

```ora
let divisor: NonZero<u256> = 42;
let owner: NonZeroAddress = 0x742d35Cc6634C0532925a3b8D0C5e0E0f8d7D2aB;
```

`NonZero<T>` guarantees the value is not zero. `NonZeroAddress` guarantees the address is not the zero address. The compiler rejects `let x: NonZero<u256> = 0` at compile time.

### MinValue and MaxValue

```ora
let deposit: MinValue<u256, 1> = 100;            // value >= 1
let fee: MaxValue<u256, 10000> = 250;             // value <= 10000
```

### InRange

```ora
let percentage: InRange<u256, 0, 100> = 75;       // 0 <= value <= 100
```

### BasisPoints

```ora
let rate: BasisPoints<u256> = 250;                 // 0 <= value <= 10000
```

`BasisPoints` is a shorthand for `InRange<u256, 0, 10000>` — common in DeFi for fee rates where 10000 = 100%.

### Scaled

```ora
let amount: Scaled<u256, 18> = 1_000_000_000_000_000_000;
```

`Scaled<T, D>` annotates that the value represents a fixed-point number with `D` decimal places. This is primarily for documentation and verification — it helps the compiler and auditors understand the intended precision.

## Refinements on function parameters

The most common use is constraining function inputs:

```ora
pub fn deposit(amount: MinValue<u256, 1>) {
    // The compiler guarantees amount >= 1 here.
    // No need for: if (amount == 0) revert;
    balances[sender] += amount;
}
```

Callers must provide a value that satisfies the refinement. If the compiler can't prove it statically, it inserts a runtime guard.

## Refinement subtyping

Refined types are subtypes of their base types. Inside a contract:

```ora
fn takeAddress(addr: address) { }

fn useNonZero() {
    let nz: NonZeroAddress = 0x742d35Cc6634C0532925a3b8D0C5e0E0f8d7D2aB;
    takeAddress(nz);   // OK: NonZeroAddress is a subtype of address
}
```

A `MinValue<u256, 1000>` is a subtype of `MinValue<u256, 500>` — a tighter constraint implies a looser one.

## Guard elimination

When the compiler can prove a refinement holds, it removes the runtime check. Inside a contract:

```ora
pub fn safeDeposit(amount: u256) {
    if (amount == 0) { return; }
    // After the check above, the compiler knows amount > 0.
    // A MinValue<u256, 1> refinement would be satisfied here
    // without an additional runtime guard.
    let safe_amount: MinValue<u256, 1> = amount;
    deposit(safe_amount);
}
```

The Z3 SMT solver proves that `amount > 0` implies `amount >= 1`. No runtime guard is emitted.

## The vault with refinements

```ora
error InsufficientBalance(required: u256, available: u256);

comptime const std = @import("std");

contract Vault {
    storage var totalDeposits: u256 = 0;
    storage var balances: map<address, u256>;

    log Deposit(account: address, amount: u256);
    log Withdrawal(account: address, amount: u256);

    pub fn deposit(amount: MinValue<u256, 1>) {
        let sender: NonZeroAddress = std.msg.sender();
        balances[sender] += amount;
        totalDeposits += amount;
        log Deposit(sender, amount);
    }

    pub fn withdraw(amount: MinValue<u256, 1>) -> !bool | InsufficientBalance {
        let sender: NonZeroAddress = std.msg.sender();
        let current: u256 = balances[sender];
        if (current < amount) { return InsufficientBalance(amount, current); }
        balances[sender] = current - amount;
        totalDeposits -= amount;
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

Changes from the previous version:
- `amount: MinValue<u256, 1>` on both `deposit` and `withdraw` — zero-amount operations are now type errors
- `sender: NonZeroAddress` — the zero address can never deposit or withdraw
- Added `log` declarations and emissions (covered next chapter)

These constraints are checked by the compiler. If a caller passes an unvalidated `u256`, the compiler either proves the constraint or inserts a minimal runtime guard.

## Further reading

- [Refinement Types](../refinement-types) — full reference
- [SMT Verification](../research/smt-verification) — how Z3 proves refinements
