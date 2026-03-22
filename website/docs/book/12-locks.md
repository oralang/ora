---
title: "Chapter 12: Locks and Reentrancy"
description: Protecting against reentrancy with @lock and @unlock.
sidebar_position: 12
---

# Locks and Reentrancy

Reentrancy is the most exploited vulnerability in smart contract history. It occurs when an external call re-enters the contract before state updates complete.

Ora provides `@lock` and `@unlock` directives for compiler-checked reentrancy protection.

## The lock pattern

```ora
contract Vault {
    storage var reentrancyGuard: u256 = 0;

    pub fn withdraw(amount: u256) {
        @lock(reentrancyGuard);
        // Protected code — no reentrant call can reach here
        balances[sender] -= amount;
        @unlock(reentrancyGuard);
    }
}
```

- `@lock(guard)` acquires the lock. If the guard is already locked, the transaction reverts.
- `@unlock(guard)` releases the lock.
- The guard is a storage variable used as a mutex.

## Lock discipline

The compiler tracks lock state:

- A function that calls `@lock` must call `@unlock` on every exit path
- Attempting to `@lock` an already-locked guard reverts
- The lock is transaction-scoped — it's automatically released if the transaction reverts

## The vault with locks

```ora
error InsufficientBalance(required: u256, available: u256);

comptime const std = @import("std");

contract Vault {
    storage var totalDeposits: u256 = 0;
    storage var balances: map<address, u256>;
    storage var reentrancyGuard: u256 = 0;

    log Deposit(account: address, amount: u256);
    log Withdrawal(account: address, amount: u256);

    pub fn deposit(amount: MinValue<u256, 1>) {
        let sender: NonZeroAddress = std.msg.sender();
        @lock(reentrancyGuard);
        balances[sender] += amount;
        totalDeposits += amount;
        @unlock(reentrancyGuard);
        log Deposit(sender, amount);
    }

    pub fn withdraw(amount: MinValue<u256, 1>) -> !bool | InsufficientBalance {
        let sender: NonZeroAddress = std.msg.sender();
        @lock(reentrancyGuard);
        let current: u256 = balances[sender];
        if (current < amount) {
            @unlock(reentrancyGuard);
            return InsufficientBalance(amount, current);
        }
        balances[sender] = current - amount;
        totalDeposits -= amount;
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

Note that `@unlock` must be called before the error return path in `withdraw`. The compiler checks that every exit path releases the lock.

## Further reading

- [Formal Verification](../formal-verification) — how lock analysis integrates with verification
